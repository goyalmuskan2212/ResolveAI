"""FastAPI application — HTTP layer for the AI Support Agent.

Provides 5 endpoints:
- GET  /health               — liveness / readiness probe
- POST /ticket               — submit a support ticket, get agent response
- POST /ticket/{id}/clarify  — resume interrupted graph with user's clarification
- GET  /ticket/{id}/trace    — fetch Langfuse trace for a ticket
- GET  /metrics              — aggregated metrics from Langfuse
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import BaseModel

from app.agent.graph import build_graph
from app.tracing.langfuse_setup import (
    get_langfuse_handler,
    record_trace_scores,
    trace_context,
)
from app.tracing.metrics import get_aggregate_metrics, get_trace_by_id

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class TicketRequest(BaseModel):
    text: str
    thread_id: str | None = None


class ClarifyRequest(BaseModel):
    reply: str


class TicketResponse(BaseModel):
    ticket_id: str
    response: str
    category: str
    priority: str
    language: str = ""
    quality_score: float
    escalated: bool
    trace_id: str | None = None
    needs_clarification: bool = False
    clarifying_question: str | None = None


class MetricsResponse(BaseModel):
    total_traces: int
    avg_latency_ms: float
    avg_cost: float
    automation_pct: float = 0.0
    category_distribution: dict = {}
    error: str | None = None


# ---------------------------------------------------------------------------
# App lifespan — graph singleton
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.agent = build_graph(checkpointer=MemorySaver())
    yield


app = FastAPI(
    title="AI Support Agent API",
    description="Customer support agent powered by LangGraph, LlamaIndex, and Google Gemini.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health_check():
    """Liveness / readiness probe — confirms the API and graph are ready."""
    graph = getattr(app.state, "agent", None)
    return {
        "status": "ok",
        "graph_ready": graph is not None,
    }


@app.post("/ticket", response_model=TicketResponse)
def create_ticket(request: TicketRequest):
    """Submit a support ticket and receive the agent's response."""
    graph = app.state.agent
    thread_id = request.thread_id or str(uuid4())

    handler = get_langfuse_handler()
    callbacks = [handler] if handler else []
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": callbacks,
    }

    try:
        with trace_context("support-ticket", session_id=thread_id):
            result = graph.invoke({"ticket_text": request.text}, config)
    except Exception as exc:
        logger.error("Graph invocation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process ticket")

    trace_id = getattr(handler, "last_trace_id", None) if handler else None

    # Check for interrupt (clarify flow)
    state_snapshot = graph.get_state(config)
    if state_snapshot.tasks and any(t.interrupts for t in state_snapshot.tasks):
        interrupt_value = state_snapshot.tasks[0].interrupts[0].value
        return TicketResponse(
            ticket_id=thread_id,
            response="",
            category=result.get("category", "unclear"),
            priority=result.get("priority", "low"),
            language=result.get("language", ""),
            quality_score=0.0,
            escalated=False,
            trace_id=trace_id,
            needs_clarification=True,
            clarifying_question=str(interrupt_value),
        )

    # Normal completion — record scores and return
    record_trace_scores(
        handler,
        category=result.get("category"),
        quality_score=result.get("quality_score"),
        escalated=result.get("needs_escalation"),
    )

    return TicketResponse(
        ticket_id=thread_id,
        response=result.get("draft_response", ""),
        category=result.get("category", ""),
        priority=result.get("priority", ""),
        language=result.get("language", ""),
        quality_score=result.get("quality_score", 0.0),
        escalated=result.get("needs_escalation", False),
        trace_id=trace_id,
    )


@app.post("/ticket/{ticket_id}/clarify", response_model=TicketResponse)
def clarify_ticket(ticket_id: str, request: ClarifyRequest):
    """Resume an interrupted graph with the user's clarification reply."""
    graph = app.state.agent

    handler = get_langfuse_handler()
    callbacks = [handler] if handler else []
    config = {
        "configurable": {"thread_id": ticket_id},
        "callbacks": callbacks,
    }

    # Verify there is a pending interrupt for this thread
    try:
        state_snapshot = graph.get_state(config)
    except Exception:
        raise HTTPException(status_code=404, detail="Ticket not found")

    has_interrupt = (
        state_snapshot.tasks
        and any(t.interrupts for t in state_snapshot.tasks)
    )
    if not has_interrupt:
        raise HTTPException(
            status_code=404,
            detail="No pending clarification for this ticket",
        )

    try:
        with trace_context("support-ticket-clarify", session_id=ticket_id):
            result = graph.invoke(Command(resume=request.reply), config)
    except Exception as exc:
        logger.error("Clarify invocation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process ticket")

    trace_id = getattr(handler, "last_trace_id", None) if handler else None

    record_trace_scores(
        handler,
        category=result.get("category"),
        quality_score=result.get("quality_score"),
        escalated=result.get("needs_escalation"),
    )

    return TicketResponse(
        ticket_id=ticket_id,
        response=result.get("draft_response", ""),
        category=result.get("category", ""),
        priority=result.get("priority", ""),
        language=result.get("language", ""),
        quality_score=result.get("quality_score", 0.0),
        escalated=result.get("needs_escalation", False),
        trace_id=trace_id,
    )


@app.get("/ticket/{trace_id}/trace")
def get_trace(trace_id: str):
    """Fetch a Langfuse trace by ID — returns nodes, durations, and cost."""
    data = get_trace_by_id(trace_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return data


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics(
    from_date: str | None = Query(default=None, description="ISO date string, e.g. 2025-01-01"),
    to_date: str | None = Query(default=None, description="ISO date string, e.g. 2025-12-31"),
):
    """Return aggregated metrics from recent Langfuse traces."""
    from_ts: datetime | None = None
    to_ts: datetime | None = None
    try:
        if from_date:
            from_ts = datetime.fromisoformat(from_date)
        if to_date:
            to_ts = datetime.fromisoformat(to_date)
    except ValueError:
        pass  # ignore invalid dates, fall through to unfiltered query
    return get_aggregate_metrics(from_timestamp=from_ts, to_timestamp=to_ts)
