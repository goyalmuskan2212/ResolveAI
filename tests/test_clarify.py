"""Tests for the clarify node and interrupt flow (Block 3.4).

Covers:
- ClarifyResult model validation
- CLARIFY_SYSTEM_PROMPT existence
- Mini-graph interrupt / resume cycle with mocked LLM
"""

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.agent.nodes.clarify import CLARIFY_SYSTEM_PROMPT, ClarifyResult, clarify_node
from app.agent.routing import route_by_category
from app.agent.state import AgentState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(**overrides) -> dict:
    """Create a minimal AgentState dict for testing."""
    base = {
        "ticket_text": "Не работает!!!",
        "messages": [],
        "category": "unclear",
        "priority": "low",
        "language": "ru",
        "retrieved_contexts": [],
        "sources": [],
        "draft_response": "",
        "quality_score": 0.0,
        "needs_escalation": False,
        "escalation_reason": "",
        "iteration_count": 0,
        "tone_instructions": "",
        "empathy_preamble": "",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Unit tests — model and prompt
# ---------------------------------------------------------------------------


class TestClarifyResultModel:
    """Validate the ClarifyResult Pydantic model."""

    def test_valid_result(self):
        result = ClarifyResult(clarifying_question="Could you describe the issue?")
        assert result.clarifying_question == "Could you describe the issue?"

    def test_empty_question_allowed(self):
        result = ClarifyResult(clarifying_question="")
        assert result.clarifying_question == ""


class TestClarifyPrompt:
    """Verify the system prompt is present and contains key instructions."""

    def test_prompt_exists(self):
        assert CLARIFY_SYSTEM_PROMPT
        assert len(CLARIFY_SYSTEM_PROMPT) > 50

    def test_prompt_mentions_language(self):
        assert "language" in CLARIFY_SYSTEM_PROMPT.lower()

    def test_prompt_mentions_one_question(self):
        assert "ONE" in CLARIFY_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Mini-graph interrupt / resume test
# ---------------------------------------------------------------------------


def _mock_classify_factory():
    """Return a classify mock that returns 'unclear' first, then 'question'."""
    call_count = {"n": 0}

    def _classify(state: AgentState) -> dict:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {"category": "unclear", "priority": "low", "language": "ru"}
        return {"category": "question", "priority": "medium", "language": "ru"}

    return _classify


def _done_node(state: AgentState) -> dict:
    """Terminal node that signals the graph reached the end."""
    return {}


class TestClarifyInterruptFlow:
    """Test the full interrupt / resume cycle using a mini-graph."""

    @patch("app.agent.nodes.clarify.ChatGoogleGenerativeAI")
    def test_interrupt_pauses_and_resume_continues(self, mock_llm_cls):
        # -- Arrange: mock LLM to return a clarifying question ----------------
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = ClarifyResult(
            clarifying_question="Опишите, пожалуйста, что именно не работает?",
        )
        mock_llm_cls.return_value.with_structured_output.return_value = mock_structured

        # -- Build mini-graph -------------------------------------------------
        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("classify", _mock_classify_factory())
        graph_builder.add_node("clarify", clarify_node)
        graph_builder.add_node("done", _done_node)

        graph_builder.add_edge(START, "classify")
        graph_builder.add_conditional_edges(
            "classify",
            route_by_category,
            {
                "unclear": "clarify",
                "question": "done",
                "complaint": "done",
            },
        )
        graph_builder.add_edge("clarify", "classify")
        graph_builder.add_edge("done", END)

        checkpointer = MemorySaver()
        graph = graph_builder.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "test-clarify-1"}}

        # -- Act: first invocation — should pause at interrupt ----------------
        result = graph.invoke(
            _make_state(ticket_text="Не работает!!!"),
            config,
        )

        # -- Assert: graph is interrupted -------------------------------------
        graph_state = graph.get_state(config)
        assert graph_state.tasks, "Graph should have pending tasks (interrupted)"

        interrupts = graph_state.tasks[0].interrupts
        assert len(interrupts) > 0, "There should be at least one interrupt"
        assert "не работает" in interrupts[0].value.lower() or "what" in interrupts[0].value.lower()

        # -- Act: resume with user's clarification ----------------------------
        resumed_result = graph.invoke(
            Command(resume="Не загружается страница оплаты"),
            config,
        )

        # -- Assert: state was enriched correctly -----------------------------
        assert "Не работает!!!" in resumed_result["ticket_text"]
        assert "Не загружается страница оплаты" in resumed_result["ticket_text"]
        assert "User clarification" in resumed_result["ticket_text"]

        # Category should be re-classified as "question" on second pass
        assert resumed_result["category"] == "question"

    @patch("app.agent.nodes.clarify.ChatGoogleGenerativeAI")
    def test_messages_contain_clarification_exchange(self, mock_llm_cls):
        """After resume, messages should include the AI question and user reply."""
        mock_structured = MagicMock()
        clarifying_q = "What exactly is not working?"
        mock_structured.invoke.return_value = ClarifyResult(
            clarifying_question=clarifying_q,
        )
        mock_llm_cls.return_value.with_structured_output.return_value = mock_structured

        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("classify", _mock_classify_factory())
        graph_builder.add_node("clarify", clarify_node)
        graph_builder.add_node("done", _done_node)

        graph_builder.add_edge(START, "classify")
        graph_builder.add_conditional_edges(
            "classify",
            route_by_category,
            {"unclear": "clarify", "question": "done", "complaint": "done"},
        )
        graph_builder.add_edge("clarify", "classify")
        graph_builder.add_edge("done", END)

        checkpointer = MemorySaver()
        graph = graph_builder.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "test-clarify-2"}}

        # First invoke — hits interrupt
        graph.invoke(_make_state(), config)

        # Resume
        user_reply = "The payment page does not load"
        result = graph.invoke(Command(resume=user_reply), config)

        # Check messages contain the exchange
        messages = result.get("messages", [])
        ai_messages = [m for m in messages if isinstance(m, AIMessage)]
        human_messages = [m for m in messages if isinstance(m, HumanMessage)]

        assert any(clarifying_q in m.content for m in ai_messages), (
            "AI clarifying question should be in messages"
        )
        assert any(user_reply in m.content for m in human_messages), (
            "User reply should be in messages"
        )
