"""Langfuse tracing setup — handler factory, client factory, trace context, score recording.

Provides opt-in tracing with graceful fallback. When LANGFUSE_ENABLED is False
or Langfuse is unreachable, all functions degrade safely (return None / no-op).

Usage at call site:
    handler = get_langfuse_handler()
    callbacks = [handler] if handler else []
    with trace_context("support-ticket", session_id="s-1"):
        result = graph.invoke(state, config={"callbacks": callbacks})
    record_trace_scores(handler, category="question", quality_score=0.85)
"""

import logging

from langfuse import Langfuse, propagate_attributes
from langfuse.langchain import CallbackHandler

from app.config import settings

logger = logging.getLogger(__name__)

_langfuse_client: Langfuse | None = None


def _get_client() -> Langfuse | None:
    """Return the singleton Langfuse client, creating it on first call."""
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client

    if not settings.LANGFUSE_ENABLED:
        return None

    try:
        _langfuse_client = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_BASE_URL,
            timeout=15,
        )
        return _langfuse_client
    except Exception as exc:
        logger.warning("Langfuse init failed: %s", exc)
        return None


def get_langfuse_handler() -> CallbackHandler | None:
    """Create a Langfuse CallbackHandler for LangGraph tracing.

    Returns None if tracing is disabled or Langfuse is unreachable.
    The handler auto-instruments all LLM calls within the graph.
    """
    client = _get_client()
    if client is None:
        return None

    try:
        return CallbackHandler(public_key=settings.LANGFUSE_PUBLIC_KEY)
    except Exception as exc:
        logger.warning("Langfuse unreachable, tracing disabled for this request: %s", exc)
        return None


def trace_context(
    trace_name: str,
    session_id: str | None = None,
    user_id: str | None = None,
):
    """Context manager that sets trace-level attributes via OpenTelemetry.

    Wrap graph.invoke() in this to set the trace name, session ID, and
    user ID on all spans created within the context. These appear as
    top-level fields in Langfuse UI (not buried in metadata).
    """
    return propagate_attributes(
        trace_name=trace_name,
        session_id=session_id,
        user_id=user_id,
    )


def get_langfuse_client() -> Langfuse | None:
    """Get the initialized Langfuse client for direct API access (metrics, trace retrieval)."""
    return _get_client()


def record_trace_scores(handler: CallbackHandler | None, **kwargs) -> None:
    """Record domain results as Langfuse scores on the trace.

    Each kwarg becomes a separate score (e.g., category="question",
    quality_score=0.85). Numeric values are stored as NUMERIC scores,
    others as STRING. Calls flush() to ensure data is sent.
    No-op if handler is None or no trace ID is available.
    """
    if handler is None:
        return

    client = _get_client()
    if client is None:
        return

    trace_id = handler.last_trace_id
    if not trace_id:
        return

    try:
        for name, value in kwargs.items():
            if value is None:
                continue
            if isinstance(value, bool):
                score_value = str(value)
            elif isinstance(value, (int, float)):
                score_value = value
            else:
                score_value = str(value)
            client.create_score(
                name=name,
                value=score_value,
                trace_id=trace_id,
            )
        client.flush()
    except Exception:
        logger.warning("Failed to record trace scores", exc_info=True)
