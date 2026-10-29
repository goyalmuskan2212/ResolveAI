"""Tests for the response pipeline: rag_search, empathy, generate nodes and routing."""

from unittest.mock import MagicMock, patch

import pytest

from app.agent.nodes.empathy import EmpathyResult, empathy_node
from app.agent.nodes.generate import generate_response_node
from app.agent.nodes.rag_search import rag_search_node
from app.agent.routing import route_after_rag


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_source_node(text: str, score: float, metadata: dict | None = None):
    """Create a mock LlamaIndex NodeWithScore."""
    node = MagicMock()
    node.text = text
    node.score = score
    node.metadata = metadata or {}
    return node


def _make_state(**overrides) -> dict:
    """Create a minimal AgentState dict for testing."""
    base = {
        "ticket_text": "How do I return an item?",
        "messages": [],
        "category": "question",
        "priority": "low",
        "language": "en",
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
# RAG search node — unit tests
# ---------------------------------------------------------------------------


class TestRagSearchNode:
    """Unit tests for rag_search_node with mocked query engine."""

    @patch("app.agent.nodes.rag_search._verify_relevance", return_value=True)
    @patch("app.agent.nodes.rag_search.get_query_engine")
    def test_normal_retrieval(self, mock_get_qe, _mock_relevance):
        mock_response = MagicMock()
        mock_response.source_nodes = [
            _make_source_node("Return policy info", 9.0, {"file": "faq.md"}),
            _make_source_node("Delivery terms", 7.0),
            _make_source_node("Contact info", 5.0),
        ]
        mock_get_qe.return_value.query.return_value = mock_response

        result = rag_search_node(_make_state())

        assert len(result["retrieved_contexts"]) == 3
        assert result["retrieved_contexts"][0] == "Return policy info"
        assert len(result["sources"]) == 3
        assert result["sources"][0]["score"] == 9.0
        assert result["sources"][0]["metadata"] == {"file": "faq.md"}
        assert result["needs_escalation"] is False
        assert result["escalation_reason"] == ""
        assert result["iteration_count"] == 1

    @patch("app.agent.nodes.rag_search.get_query_engine")
    def test_escalation_on_low_score(self, mock_get_qe):
        mock_response = MagicMock()
        mock_response.source_nodes = [
            _make_source_node("Irrelevant chunk", 2.0),
            _make_source_node("Another irrelevant", 1.0),
        ]
        mock_get_qe.return_value.query.return_value = mock_response

        result = rag_search_node(_make_state())

        assert result["needs_escalation"] is True
        assert "2.00" in result["escalation_reason"]
        assert result["retrieved_contexts"] == []

    @patch("app.agent.nodes.rag_search.get_query_engine")
    def test_escalation_on_no_results(self, mock_get_qe):
        mock_response = MagicMock()
        mock_response.source_nodes = []
        mock_get_qe.return_value.query.return_value = mock_response

        result = rag_search_node(_make_state())

        assert result["needs_escalation"] is True
        assert "No knowledge base results" in result["escalation_reason"]

    @patch("app.agent.nodes.rag_search._verify_relevance", return_value=True)
    @patch("app.agent.nodes.rag_search.get_query_engine")
    def test_iteration_count_increments(self, mock_get_qe, _mock_relevance):
        mock_response = MagicMock()
        mock_response.source_nodes = [_make_source_node("text", 8.0)]
        mock_get_qe.return_value.query.return_value = mock_response

        result = rag_search_node(_make_state(iteration_count=1))

        assert result["iteration_count"] == 2

    @patch("app.agent.nodes.rag_search.get_query_engine")
    def test_query_engine_error_escalates(self, mock_get_qe):
        mock_get_qe.return_value.query.side_effect = ConnectionError("Qdrant unavailable")

        result = rag_search_node(_make_state())

        assert result["needs_escalation"] is True
        assert "RAG search failed" in result["escalation_reason"]
        assert "Qdrant unavailable" in result["escalation_reason"]
        assert result["iteration_count"] == 1

    @patch("app.agent.nodes.rag_search._verify_relevance", return_value=False)
    @patch("app.agent.nodes.rag_search.get_query_engine")
    def test_escalation_on_irrelevant_context(self, mock_get_qe, _mock_relevance):
        """High scores but LLM says context doesn't directly answer the question."""
        mock_response = MagicMock()
        mock_response.source_nodes = [
            _make_source_node("Corporate orders info", 9.0),
            _make_source_node("Invoicing details", 7.0),
        ]
        mock_get_qe.return_value.query.return_value = mock_response

        result = rag_search_node(_make_state(
            ticket_text="How do I change my bank account for billing?",
        ))

        assert result["needs_escalation"] is True
        assert "does not contain a direct answer" in result["escalation_reason"]
        assert result["retrieved_contexts"] == []

    @patch("app.agent.nodes.rag_search.get_query_engine")
    def test_handles_none_scores(self, mock_get_qe):
        """Scores can be None in some vector store backends."""
        mock_response = MagicMock()
        mock_response.source_nodes = [
            _make_source_node("text", None),
        ]
        mock_get_qe.return_value.query.return_value = mock_response

        result = rag_search_node(_make_state())

        # None scores → best_score defaults to 0.0 → escalation
        assert result["needs_escalation"] is True


# ---------------------------------------------------------------------------
# Empathy node — unit tests
# ---------------------------------------------------------------------------


class TestEmpathyNode:
    """Unit tests for empathy_node with mocked LLM."""

    def _mock_llm(self, mock_cls, result: EmpathyResult):
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = result
        mock_cls.return_value.with_structured_output.return_value = mock_structured

    @patch("app.agent.nodes.empathy.ChatGoogleGenerativeAI")
    def test_returns_tone_and_preamble(self, mock_llm_cls):
        self._mock_llm(
            mock_llm_cls,
            EmpathyResult(
                tone_instructions="Use a warm, apologetic tone.",
                empathy_preamble="We understand your frustration.",
            ),
        )

        result = empathy_node(_make_state(
            ticket_text="My order is late and I'm very upset!",
            category="complaint",
        ))

        assert result == {
            "tone_instructions": "Use a warm, apologetic tone.",
            "empathy_preamble": "We understand your frustration.",
        }

    @patch("app.agent.nodes.empathy.ChatGoogleGenerativeAI")
    def test_no_rag_fields_in_output(self, mock_llm_cls):
        """AR-1: empathy node must NOT return RAG-related fields."""
        self._mock_llm(
            mock_llm_cls,
            EmpathyResult(
                tone_instructions="Be empathetic.",
                empathy_preamble="We're sorry.",
            ),
        )

        result = empathy_node(_make_state(category="complaint"))

        assert "retrieved_contexts" not in result
        assert "sources" not in result
        assert set(result.keys()) == {"tone_instructions", "empathy_preamble"}


# ---------------------------------------------------------------------------
# Generate node — unit tests
# ---------------------------------------------------------------------------


class TestGenerateNode:
    """Unit tests for generate_response_node with mocked LLM."""

    def _mock_llm(self, mock_cls, content: str = "Generated response"):
        mock_response = MagicMock()
        mock_response.content = content
        mock_cls.return_value.invoke.return_value = mock_response

    @patch("app.agent.nodes.generate.ChatGoogleGenerativeAI")
    def test_returns_draft_response(self, mock_llm_cls):
        self._mock_llm(mock_llm_cls, "Here is how to return your item...")

        result = generate_response_node(_make_state(
            retrieved_contexts=["Return policy: 14 days."],
        ))

        assert result == {"draft_response": "Here is how to return your item..."}

    @patch("app.agent.nodes.generate.ChatGoogleGenerativeAI")
    def test_question_path_no_empathy(self, mock_llm_cls):
        """For question tickets, no tone section should be in the prompt."""
        self._mock_llm(mock_llm_cls)

        generate_response_node(_make_state(
            retrieved_contexts=["Some context"],
            tone_instructions="",
            empathy_preamble="",
        ))

        call_args = mock_llm_cls.return_value.invoke.call_args[0][0]
        system_prompt = call_args[0][1]
        assert "Tone Guidelines" not in system_prompt

    @patch("app.agent.nodes.generate.ChatGoogleGenerativeAI")
    def test_complaint_path_with_empathy(self, mock_llm_cls):
        """For complaint tickets, tone instructions and preamble must be in the prompt."""
        self._mock_llm(mock_llm_cls)

        generate_response_node(_make_state(
            retrieved_contexts=["Some context"],
            tone_instructions="Use a warm tone.",
            empathy_preamble="We understand your frustration.",
        ))

        call_args = mock_llm_cls.return_value.invoke.call_args[0][0]
        system_prompt = call_args[0][1]
        assert "Tone Guidelines" in system_prompt
        assert "Use a warm tone." in system_prompt
        assert "We understand your frustration." in system_prompt

    @patch("app.agent.nodes.generate.ChatGoogleGenerativeAI")
    def test_contexts_included_in_prompt(self, mock_llm_cls):
        """Retrieved contexts must appear in the system prompt."""
        self._mock_llm(mock_llm_cls)

        generate_response_node(_make_state(
            retrieved_contexts=["Return within 14 days", "Free shipping on returns"],
        ))

        call_args = mock_llm_cls.return_value.invoke.call_args[0][0]
        system_prompt = call_args[0][1]
        assert "Return within 14 days" in system_prompt
        assert "Free shipping on returns" in system_prompt


# ---------------------------------------------------------------------------
# Routing — pure logic tests
# ---------------------------------------------------------------------------


class TestRouteAfterRag:
    def test_escalate_when_needed(self):
        assert route_after_rag({"needs_escalation": True}) == "escalate"

    def test_generate_when_no_escalation(self):
        assert route_after_rag({"needs_escalation": False}) == "generate_response"

    def test_default_no_escalation(self):
        """Missing needs_escalation key should default to generate_response."""
        assert route_after_rag({}) == "generate_response"


# ---------------------------------------------------------------------------
# Integration tests — real APIs, requires GEMINI_API_KEY + running Qdrant
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestEmpathyIntegration:
    def test_generates_empathy_for_complaint(self):
        state = _make_state(
            ticket_text="Мой заказ не пришёл уже 10 дней! Это безобразие!",
            category="complaint",
        )
        result = empathy_node(state)

        assert result["tone_instructions"]
        assert result["empathy_preamble"]
        assert len(result["tone_instructions"]) > 10
        assert len(result["empathy_preamble"]) > 10


@pytest.mark.integration
class TestGenerateIntegration:
    def test_generates_response_with_context(self):
        state = _make_state(
            ticket_text="How do I return an item?",
            retrieved_contexts=[
                "Return policy: Items can be returned within 14 days of purchase. "
                "The item must be in original packaging and unused."
            ],
        )
        result = generate_response_node(state)

        assert result["draft_response"]
        assert len(result["draft_response"]) > 20
