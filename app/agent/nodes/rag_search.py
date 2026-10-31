"""RAG search node — retrieves relevant knowledge base context for the ticket.

Implements AR-4 early escalation: if no relevant results are found
(best rerank score < 3.0 on a 1-10 scale, or no results at all), the node
sets needs_escalation=True to skip generate + quality_check and avoid
hallucinations.

Additionally performs LLM relevance verification: even when chunks score
highly, the LLM checks whether the context contains a DIRECT answer to
the specific question (not just general info on a related topic).
"""

import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from app.agent.state import AgentState
from app.config import settings
from app.rag.query_engine import get_query_engine

logger = logging.getLogger(__name__)

RAG_RELEVANCE_THRESHOLD = 3.0  # LLMRerank scores on 1-10 scale

RELEVANCE_CHECK_PROMPT = """\
You are a relevance verification system for customer support RAG.

Given a customer's question and the retrieved knowledge base context, \
determine whether the context contains a SPECIFIC, DIRECT answer to \
the customer's question.

Rules:
- Return has_direct_answer=true ONLY if the context contains concrete \
information that specifically addresses what the customer is asking about.
- Return has_direct_answer=false if the context only contains general or \
tangentially related information that does not directly answer the specific question.
- A context about a RELATED topic is NOT sufficient. The context must \
address the SPECIFIC issue or request the customer is describing.

Examples of false (related but not direct):
- Customer asks about changing bank account details, context discusses \
how to place a corporate order → false
- Customer asks about sending a technician, context discusses warranty \
terms and service centers → false
- Customer asks about bulk pricing for resellers, context discusses \
loyalty rewards program → false

Examples of true (direct answer):
- Customer asks how to return a product, context describes the return \
policy with steps and deadlines → true
- Customer asks about delivery times, context lists shipping timeframes \
by region → true"""


class RelevanceVerification(BaseModel):
    """Structured output for relevance check."""

    has_direct_answer: bool = Field(
        description=(
            "True if the context contains specific information that "
            "directly answers the customer's question"
        ),
    )
    reason: str = Field(
        description="Brief explanation of the verdict",
    )


def _verify_relevance(question: str, contexts: list[str]) -> bool:
    """Check whether retrieved contexts contain a direct answer to the question.

    Uses Gemini 2.5 Flash for a cheap structured evaluation.
    Returns True (proceed) on any error — fail-open design so that a
    verification failure does not block the pipeline.
    """
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.8-flash",
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.0,
        )
        structured_llm = llm.with_structured_output(RelevanceVerification)

        contexts_text = "\n\n---\n\n".join(contexts)
        human_message = (
            f"## Customer Question\n{question}\n\n"
            f"## Retrieved Knowledge Base Context\n{contexts_text}"
        )

        result: RelevanceVerification = structured_llm.invoke([
            ("system", RELEVANCE_CHECK_PROMPT),
            ("human", human_message),
        ])
        logger.info("Relevance check: %s — %s", result.has_direct_answer, result.reason)
        return result.has_direct_answer
    except Exception:
        logger.warning("Relevance verification failed, proceeding", exc_info=True)
        return True


def rag_search_node(state: AgentState) -> dict:
    """Query the knowledge base and return retrieved contexts and sources.

    Returns:
        dict with keys: retrieved_contexts, sources, iteration_count,
        needs_escalation, escalation_reason.
    """
    iteration_count = state.get("iteration_count", 0) + 1

    try:
        query_engine = get_query_engine()
        response = query_engine.query(state["ticket_text"])
    except Exception as exc:
        return {
            "retrieved_contexts": [],
            "sources": [],
            "iteration_count": iteration_count,
            "needs_escalation": True,
            "escalation_reason": f"RAG search failed: {exc}",
        }

    source_nodes = response.source_nodes

    if not source_nodes:
        return {
            "retrieved_contexts": [],
            "sources": [],
            "iteration_count": iteration_count,
            "needs_escalation": True,
            "escalation_reason": "No knowledge base results found for this query",
        }

    best_score = max(
        (node.score for node in source_nodes if node.score is not None),
        default=0.0,
    )

    if best_score < RAG_RELEVANCE_THRESHOLD:
        return {
            "retrieved_contexts": [],
            "sources": [],
            "iteration_count": iteration_count,
            "needs_escalation": True,
            "escalation_reason": (
                f"No relevant knowledge base articles found "
                f"(best score: {best_score:.2f})"
            ),
        }

    retrieved_contexts = [node.text for node in source_nodes]

    # LLM relevance verification — catch "high similarity but no real answer"
    if not _verify_relevance(state["ticket_text"], retrieved_contexts):
        return {
            "retrieved_contexts": [],
            "sources": [],
            "iteration_count": iteration_count,
            "needs_escalation": True,
            "escalation_reason": (
                "Retrieved context does not contain a direct answer "
                "to the customer's specific question"
            ),
        }

    sources = [
        {
            "text": node.text,
            "score": node.score,
            "metadata": node.metadata,
        }
        for node in source_nodes
    ]

    return {
        "retrieved_contexts": retrieved_contexts,
        "sources": sources,
        "iteration_count": iteration_count,
        "needs_escalation": False,
        "escalation_reason": "",
    }
