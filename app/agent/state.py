from typing import Annotated, Literal, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # Dialog history (LangGraph message accumulator)
    messages: Annotated[list, add_messages]

    # Input
    ticket_text: str

    # Classification results
    category: Literal["question", "complaint", "unclear"]
    priority: Literal["low", "medium", "high", "critical"]
    language: str

    # RAG results
    retrieved_contexts: list[str]
    sources: list[dict]

    # Generation
    draft_response: str
    quality_score: float

    # Routing / escalation
    needs_escalation: bool
    escalation_reason: str
    iteration_count: int

    # Empathy enrichment (AR-1)
    tone_instructions: str
    empathy_preamble: str
