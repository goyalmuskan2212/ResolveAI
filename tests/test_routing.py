"""Tests for routing functions and post-generation nodes (Block 3.3).

Covers:
- route_by_quality (all score/iteration/priority combinations)
- route_by_category (question, complaint, unclear)
- route_after_rag (escalation flag)
- quality_check_node (mocked LLM)
- escalate_node (pure formatting)
- send_reply_node (pure formatting)
"""

from unittest.mock import MagicMock, patch

import pytest

from app.agent.nodes.escalate import escalate_node
from app.agent.nodes.quality_check import QualityCheckResult, quality_check_node
from app.agent.nodes.send_reply import send_reply_node
from app.agent.routing import route_after_rag, route_by_category, route_by_quality


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
# route_by_quality — pure logic tests
# ---------------------------------------------------------------------------


class TestRouteByQuality:
    """Test all routing branches of route_by_quality."""

    def test_pass_high_score(self):
        state = _make_state(quality_score=0.9, iteration_count=1, priority="low")
        assert route_by_quality(state) == "pass"

    def test_pass_boundary_score(self):
        """Score exactly 0.7 should pass."""
        state = _make_state(quality_score=0.7, iteration_count=1, priority="low")
        assert route_by_quality(state) == "pass"

    def test_retry_low_score_first_attempt(self):
        state = _make_state(quality_score=0.5, iteration_count=1, priority="low")
        assert route_by_quality(state) == "retry"

    def test_retry_zero_score(self):
        state = _make_state(quality_score=0.0, iteration_count=0, priority="low")
        assert route_by_quality(state) == "retry"

    def test_fail_low_score_max_iterations(self):
        state = _make_state(quality_score=0.5, iteration_count=2, priority="low")
        assert route_by_quality(state) == "fail"

    def test_fail_above_max_iterations(self):
        state = _make_state(quality_score=0.3, iteration_count=3, priority="medium")
        assert route_by_quality(state) == "fail"

    def test_critical_always_fails_high_score(self):
        """Critical priority escalates even with a perfect score."""
        state = _make_state(quality_score=0.95, iteration_count=1, priority="critical")
        assert route_by_quality(state) == "fail"

    def test_critical_always_fails_low_score(self):
        state = _make_state(quality_score=0.3, iteration_count=0, priority="critical")
        assert route_by_quality(state) == "fail"

    def test_high_priority_not_critical_passes(self):
        """'high' priority (not 'critical') should follow normal score logic."""
        state = _make_state(quality_score=0.8, iteration_count=1, priority="high")
        assert route_by_quality(state) == "pass"

    def test_default_score_missing(self):
        """Missing quality_score key defaults to 0.0 → retry if iter < 2."""
        state = {"priority": "low", "iteration_count": 0}
        assert route_by_quality(state) == "retry"

    def test_just_below_boundary_score(self):
        """Score just below 0.7 should retry."""
        state = _make_state(quality_score=0.69, iteration_count=1, priority="low")
        assert route_by_quality(state) == "retry"

    def test_medium_priority_retries(self):
        state = _make_state(quality_score=0.4, iteration_count=1, priority="medium")
        assert route_by_quality(state) == "retry"


# ---------------------------------------------------------------------------
# route_by_quality — exhaustive priority x score x iteration matrix (Block 7.1)
# ---------------------------------------------------------------------------


class TestRouteByQualityMatrix:
    """Parametrized test covering all priority x score x iteration combinations."""

    @pytest.mark.parametrize(
        "priority, quality_score, iteration_count, expected",
        [
            # Critical always fails regardless of score or iteration
            ("critical", 1.0, 0, "fail"),
            ("critical", 0.7, 1, "fail"),
            ("critical", 0.0, 2, "fail"),
            ("critical", 0.5, 0, "fail"),
            # High priority follows normal logic
            ("high", 0.9, 1, "pass"),
            ("high", 0.7, 0, "pass"),
            ("high", 0.5, 1, "retry"),
            ("high", 0.5, 2, "fail"),
            ("high", 0.0, 0, "retry"),
            ("high", 0.0, 2, "fail"),
            # Medium priority follows normal logic
            ("medium", 0.85, 1, "pass"),
            ("medium", 0.7, 2, "pass"),
            ("medium", 0.69, 0, "retry"),
            ("medium", 0.69, 1, "retry"),
            ("medium", 0.69, 2, "fail"),
            ("medium", 0.0, 1, "retry"),
            # Low priority — normal logic + fine boundaries
            ("low", 1.0, 1, "pass"),
            ("low", 0.7, 0, "pass"),
            ("low", 0.71, 1, "pass"),
            ("low", 0.69, 0, "retry"),
            ("low", 0.69, 1, "retry"),
            ("low", 0.5, 2, "fail"),
            ("low", 0.0, 0, "retry"),
            ("low", 0.0, 2, "fail"),
        ],
    )
    def test_priority_score_iteration_matrix(
        self, priority, quality_score, iteration_count, expected
    ):
        state = _make_state(
            priority=priority,
            quality_score=quality_score,
            iteration_count=iteration_count,
        )
        assert route_by_quality(state) == expected


# ---------------------------------------------------------------------------
# route_by_category — pure logic tests
# ---------------------------------------------------------------------------


class TestRouteByCategory:
    def test_question(self):
        assert route_by_category({"category": "question"}) == "question"

    def test_complaint(self):
        assert route_by_category({"category": "complaint"}) == "complaint"

    def test_unclear(self):
        assert route_by_category({"category": "unclear"}) == "unclear"

    def test_unknown_category_passes_through(self):
        """route_by_category is a pure pass-through; unknown values should not raise."""
        assert route_by_category({"category": "refund"}) == "refund"


# ---------------------------------------------------------------------------
# route_after_rag — pure logic tests
# ---------------------------------------------------------------------------


class TestRouteAfterRag:
    def test_escalate_when_needed(self):
        assert route_after_rag({"needs_escalation": True}) == "escalate"

    def test_generate_when_no_escalation(self):
        assert route_after_rag({"needs_escalation": False}) == "generate_response"

    def test_default_no_escalation(self):
        assert route_after_rag({}) == "generate_response"


# ---------------------------------------------------------------------------
# quality_check_node — unit tests (mocked LLM)
# ---------------------------------------------------------------------------


class TestQualityCheckNode:
    """Unit tests for quality_check_node with mocked LLM."""

    def _mock_llm(self, mock_cls, result: QualityCheckResult):
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = result
        mock_cls.return_value.with_structured_output.return_value = mock_structured

    @patch("app.agent.nodes.quality_check.ChatGoogleGenerativeAI")
    def test_returns_quality_score(self, mock_llm_cls):
        self._mock_llm(
            mock_llm_cls,
            QualityCheckResult(
                relevance=0.9,
                completeness=0.8,
                tone=0.9,
                groundedness=0.85,
                overall_score=0.85,
                reason="Good response overall.",
            ),
        )

        result = quality_check_node(_make_state(
            draft_response="Your return window is 14 days.",
            retrieved_contexts=["Return policy: 14 days."],
        ))

        assert result == {"quality_score": 0.85}

    @patch("app.agent.nodes.quality_check.ChatGoogleGenerativeAI")
    def test_returns_only_quality_score_key(self, mock_llm_cls):
        """Node must return only quality_score, no sub-scores in state."""
        self._mock_llm(
            mock_llm_cls,
            QualityCheckResult(
                relevance=0.5,
                completeness=0.4,
                tone=0.6,
                groundedness=0.3,
                overall_score=0.42,
                reason="Poor groundedness.",
            ),
        )

        result = quality_check_node(_make_state(draft_response="Some response"))

        assert set(result.keys()) == {"quality_score"}

    @patch("app.agent.nodes.quality_check.ChatGoogleGenerativeAI")
    def test_contexts_in_prompt(self, mock_llm_cls):
        """Retrieved contexts must appear in the human message sent to LLM."""
        self._mock_llm(
            mock_llm_cls,
            QualityCheckResult(
                relevance=0.8, completeness=0.8, tone=0.8,
                groundedness=0.8, overall_score=0.8, reason="Fine.",
            ),
        )

        quality_check_node(_make_state(
            retrieved_contexts=["Return within 14 days", "Free shipping"],
            draft_response="Some draft",
        ))

        call_args = mock_llm_cls.return_value.with_structured_output.return_value.invoke.call_args[0][0]
        human_msg = call_args[1][1]
        assert "Return within 14 days" in human_msg
        assert "Free shipping" in human_msg

    @patch("app.agent.nodes.quality_check.ChatGoogleGenerativeAI")
    def test_draft_response_in_prompt(self, mock_llm_cls):
        """The draft response must appear in the human message."""
        self._mock_llm(
            mock_llm_cls,
            QualityCheckResult(
                relevance=0.8, completeness=0.8, tone=0.8,
                groundedness=0.8, overall_score=0.8, reason="Fine.",
            ),
        )

        quality_check_node(_make_state(
            draft_response="You can return items within 14 days.",
        ))

        call_args = mock_llm_cls.return_value.with_structured_output.return_value.invoke.call_args[0][0]
        human_msg = call_args[1][1]
        assert "You can return items within 14 days." in human_msg

    @patch("app.agent.nodes.quality_check.ChatGoogleGenerativeAI")
    def test_clamps_score_above_one(self, mock_llm_cls):
        """Score above 1.0 should be clamped to 1.0."""
        self._mock_llm(
            mock_llm_cls,
            QualityCheckResult(
                relevance=1.0, completeness=1.0, tone=1.0,
                groundedness=1.0, overall_score=1.5, reason="Overscored.",
            ),
        )

        result = quality_check_node(_make_state(draft_response="Draft"))
        assert result["quality_score"] == 1.0

    @patch("app.agent.nodes.quality_check.ChatGoogleGenerativeAI")
    def test_clamps_score_below_zero(self, mock_llm_cls):
        """Score below 0.0 should be clamped to 0.0."""
        self._mock_llm(
            mock_llm_cls,
            QualityCheckResult(
                relevance=0.0, completeness=0.0, tone=0.0,
                groundedness=0.0, overall_score=-0.5, reason="Negative.",
            ),
        )

        result = quality_check_node(_make_state(draft_response="Draft"))
        assert result["quality_score"] == 0.0


# ---------------------------------------------------------------------------
# escalate_node — pure formatting tests
# ---------------------------------------------------------------------------


class TestEscalateNode:
    """Test escalation package formatting."""

    def test_returns_expected_keys(self):
        result = escalate_node(_make_state())
        assert set(result.keys()) == {"needs_escalation", "escalation_reason", "draft_response"}

    def test_needs_escalation_true(self):
        result = escalate_node(_make_state())
        assert result["needs_escalation"] is True

    def test_preserves_existing_reason(self):
        """If escalation_reason is already set (AR-4 path), keep it."""
        state = _make_state(escalation_reason="No relevant knowledge base articles found (best score: 0.15)")
        result = escalate_node(state)
        assert result["escalation_reason"] == "No relevant knowledge base articles found (best score: 0.15)"

    def test_critical_priority_reason(self):
        """Critical priority with no prior reason should mention 'critical'."""
        state = _make_state(priority="critical", escalation_reason="")
        result = escalate_node(state)
        assert "critical" in result["escalation_reason"].lower()

    def test_low_quality_reason(self):
        """Low quality score path should mention score and attempts."""
        state = _make_state(quality_score=0.4, iteration_count=2, escalation_reason="")
        result = escalate_node(state)
        assert "0.40" in result["escalation_reason"]
        assert "2" in result["escalation_reason"]

    def test_package_contains_ticket(self):
        state = _make_state(ticket_text="My order is lost!")
        result = escalate_node(state)
        assert "My order is lost!" in result["draft_response"]

    def test_package_contains_agent_draft(self):
        state = _make_state(draft_response="We apologize for the delay...")
        result = escalate_node(state)
        assert "We apologize for the delay..." in result["draft_response"]

    def test_package_no_draft(self):
        """If no draft was generated, package should say so."""
        state = _make_state(draft_response="")
        result = escalate_node(state)
        assert "No draft generated" in result["draft_response"]

    def test_package_contains_category_and_priority(self):
        state = _make_state(category="complaint", priority="high")
        result = escalate_node(state)
        assert "complaint" in result["draft_response"]
        assert "high" in result["draft_response"]


# ---------------------------------------------------------------------------
# send_reply_node — pure formatting tests
# ---------------------------------------------------------------------------


class TestSendReplyNode:
    """Test final response formatting."""

    def test_returns_draft_response(self):
        result = send_reply_node(_make_state(draft_response="Your item can be returned."))
        assert "draft_response" in result

    def test_includes_response_body(self):
        result = send_reply_node(_make_state(draft_response="Your item can be returned within 14 days."))
        assert "Your item can be returned within 14 days." in result["draft_response"]

    def test_includes_sources(self):
        state = _make_state(
            draft_response="Some response",
            sources=[
                {"text": "chunk", "score": 0.9, "metadata": {"file_name": "faq.md"}},
                {"text": "chunk2", "score": 0.8, "metadata": {"file_name": "return_policy.md"}},
            ],
        )
        result = send_reply_node(state)
        assert "faq.md" in result["draft_response"]
        assert "return_policy.md" in result["draft_response"]

    def test_no_sources_when_empty(self):
        result = send_reply_node(_make_state(draft_response="Response", sources=[]))
        assert "Sources" not in result["draft_response"]

    def test_includes_signature(self):
        result = send_reply_node(_make_state(draft_response="Response"))
        assert "Support Team" in result["draft_response"]

    def test_no_double_greeting_complaints(self):
        """When empathy_preamble is set, skip the greeting."""
        state = _make_state(
            draft_response="We understand... here is the answer.",
            empathy_preamble="We sincerely apologize.",
            language="en",
        )
        result = send_reply_node(state)
        assert not result["draft_response"].startswith("Hello!")

    def test_greeting_for_english(self):
        state = _make_state(draft_response="Response", language="en")
        result = send_reply_node(state)
        assert "Hello!" in result["draft_response"]

    def test_greeting_for_russian(self):
        state = _make_state(draft_response="Response", language="ru")
        result = send_reply_node(state)
        assert "Здравствуйте!" in result["draft_response"]

    def test_deduplicates_sources(self):
        """Same file appearing in multiple sources should only be listed once."""
        state = _make_state(
            draft_response="Response",
            sources=[
                {"text": "chunk1", "score": 0.9, "metadata": {"file_name": "faq.md"}},
                {"text": "chunk2", "score": 0.8, "metadata": {"file_name": "faq.md"}},
                {"text": "chunk3", "score": 0.7, "metadata": {"file_name": "return_policy.md"}},
            ],
        )
        result = send_reply_node(state)
        assert result["draft_response"].count("faq.md") == 1
