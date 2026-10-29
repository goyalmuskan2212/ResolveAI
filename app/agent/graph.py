"""LangGraph StateGraph assembly — compiles the 8-node support agent graph.

Wires all nodes, conditional edges (classification routing, AR-4 early
escalation, quality-based routing), and direct edges into a single
compiled graph.  Supports interrupt/resume via an optional checkpointer.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agent.nodes.clarify import clarify_node
from app.agent.nodes.classify import classify_node
from app.agent.nodes.empathy import empathy_node
from app.agent.nodes.escalate import escalate_node
from app.agent.nodes.generate import generate_response_node
from app.agent.nodes.quality_check import quality_check_node
from app.agent.nodes.rag_search import rag_search_node
from app.agent.nodes.send_reply import send_reply_node
from app.agent.routing import route_after_rag, route_by_category, route_by_quality
from app.agent.state import AgentState


def build_graph(checkpointer=None):
    """Build and compile the support agent StateGraph.

    Args:
        checkpointer: Optional LangGraph checkpointer for interrupt support.
            If None, the graph is compiled without checkpoint persistence.

    Returns:
        Compiled StateGraph ready for invocation.
    """
    builder = StateGraph(AgentState)

    # -- Add all 8 nodes --
    builder.add_node("classify", classify_node)
    builder.add_node("rag_search", rag_search_node)
    builder.add_node("empathy", empathy_node)
    builder.add_node("generate_response", generate_response_node)
    builder.add_node("quality_check", quality_check_node)
    builder.add_node("escalate", escalate_node)
    builder.add_node("send_reply", send_reply_node)
    builder.add_node("clarify", clarify_node)

    # -- Entry point --
    builder.add_edge(START, "classify")

    # -- Conditional edges --
    builder.add_conditional_edges(
        "classify",
        route_by_category,
        {
            "question": "rag_search",
            "complaint": "empathy",
            "unclear": "clarify",
        },
    )

    builder.add_conditional_edges(
        "rag_search",
        route_after_rag,
        {
            "escalate": "escalate",
            "generate_response": "generate_response",
        },
    )

    builder.add_conditional_edges(
        "quality_check",
        route_by_quality,
        {
            "pass": "send_reply",
            "retry": "rag_search",
            "fail": "escalate",
        },
    )

    # -- Direct edges --
    builder.add_edge("empathy", "rag_search")
    builder.add_edge("generate_response", "quality_check")
    builder.add_edge("send_reply", END)
    builder.add_edge("escalate", END)
    builder.add_edge("clarify", "classify")

    return builder.compile(checkpointer=checkpointer)


def get_agent():
    """Create the support agent with MemorySaver checkpointer.

    Convenience wrapper that builds the graph with interrupt support
    enabled (required for the clarify node's interrupt() call).
    """
    return build_graph(checkpointer=MemorySaver())
