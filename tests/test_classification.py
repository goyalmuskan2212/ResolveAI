"""Tests for the classification node and category routing."""

from unittest.mock import MagicMock, patch

import pytest

from app.agent.nodes.classify import ClassificationResult, classify_node
from app.agent.routing import route_by_category


# ---------------------------------------------------------------------------
# Unit tests — mocked LLM, no API key required
# ---------------------------------------------------------------------------


class TestClassifyNodeUnit:
    """Unit tests for classify_node with mocked LLM."""

    def _make_state(self, text: str = "Test ticket") -> dict:
        return {"ticket_text": text, "messages": []}

    def _mock_llm(self, mock_cls, result: ClassificationResult):
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = result
        mock_cls.return_value.with_structured_output.return_value = mock_structured

    @patch("app.agent.nodes.classify.ChatGoogleGenerativeAI")
    def test_returns_correct_keys(self, mock_llm_cls):
        self._mock_llm(
            mock_llm_cls,
            ClassificationResult(
                category="question",
                priority="low",
                language="ru",
                summary="Question about returns",
            ),
        )
        result = classify_node(self._make_state())
        assert set(result.keys()) == {"category", "priority", "language"}

    @patch("app.agent.nodes.classify.ChatGoogleGenerativeAI")
    def test_classify_question(self, mock_llm_cls):
        self._mock_llm(
            mock_llm_cls,
            ClassificationResult(
                category="question",
                priority="low",
                language="ru",
                summary="Question about return policy",
            ),
        )
        result = classify_node(self._make_state("Как вернуть товар?"))
        assert result == {"category": "question", "priority": "low", "language": "ru"}

    @patch("app.agent.nodes.classify.ChatGoogleGenerativeAI")
    def test_classify_complaint(self, mock_llm_cls):
        self._mock_llm(
            mock_llm_cls,
            ClassificationResult(
                category="complaint",
                priority="high",
                language="ru",
                summary="Complaint about delayed delivery",
            ),
        )
        result = classify_node(self._make_state("Заказ не пришёл!"))
        assert result["category"] == "complaint"
        assert result["priority"] == "high"

    @patch("app.agent.nodes.classify.ChatGoogleGenerativeAI")
    def test_classify_unclear(self, mock_llm_cls):
        self._mock_llm(
            mock_llm_cls,
            ClassificationResult(
                category="unclear",
                priority="low",
                language="ru",
                summary="Unclear message",
            ),
        )
        result = classify_node(self._make_state("!!!"))
        assert result["category"] == "unclear"

    @patch("app.agent.nodes.classify.ChatGoogleGenerativeAI")
    def test_summary_not_in_output(self, mock_llm_cls):
        """summary is used for LLM reasoning but must NOT be returned to state."""
        self._mock_llm(
            mock_llm_cls,
            ClassificationResult(
                category="question",
                priority="low",
                language="en",
                summary="Some summary text",
            ),
        )
        result = classify_node(self._make_state())
        assert "summary" not in result

    # --- Edge-case tests (Block 7.1) ---

    @patch("app.agent.nodes.classify.ChatGoogleGenerativeAI")
    def test_empty_ticket_text(self, mock_llm_cls):
        """Empty ticket text should still produce a well-formed result."""
        self._mock_llm(
            mock_llm_cls,
            ClassificationResult(
                category="unclear",
                priority="low",
                language="en",
                summary="Empty input",
            ),
        )
        result = classify_node(self._make_state(""))
        assert set(result.keys()) == {"category", "priority", "language"}
        assert result["category"] == "unclear"

    @patch("app.agent.nodes.classify.ChatGoogleGenerativeAI")
    def test_very_long_ticket_text(self, mock_llm_cls):
        """10 000-char input should not crash and full text must reach the LLM."""
        self._mock_llm(
            mock_llm_cls,
            ClassificationResult(
                category="question",
                priority="medium",
                language="en",
                summary="Long text",
            ),
        )
        long_text = "A" * 10_000
        result = classify_node(self._make_state(long_text))
        assert set(result.keys()) == {"category", "priority", "language"}

        # Verify the full text was sent to the LLM
        call_args = mock_llm_cls.return_value.with_structured_output.return_value.invoke.call_args[0][0]
        human_msg = call_args[-1][1]
        assert long_text in human_msg

    @patch("app.agent.nodes.classify.ChatGoogleGenerativeAI")
    def test_special_characters_ticket(self, mock_llm_cls):
        """Emojis, HTML tags, null bytes, newlines should not crash the node."""
        self._mock_llm(
            mock_llm_cls,
            ClassificationResult(
                category="unclear",
                priority="low",
                language="en",
                summary="Special chars",
            ),
        )
        weird_text = "<script>alert('xss')</script>\n\x00\U0001f4a9"
        result = classify_node(self._make_state(weird_text))
        assert set(result.keys()) == {"category", "priority", "language"}

    @patch("app.agent.nodes.classify.ChatGoogleGenerativeAI")
    def test_multilingual_mixed_input(self, mock_llm_cls):
        """Mixed-language input should detect the dominant language."""
        self._mock_llm(
            mock_llm_cls,
            ClassificationResult(
                category="question",
                priority="low",
                language="fr",
                summary="Mixed language ticket",
            ),
        )
        result = classify_node(
            self._make_state("Bonjour, I need help avec mon commande")
        )
        assert result["language"] == "fr"


# ---------------------------------------------------------------------------
# Routing tests — pure logic, no mocks needed
# ---------------------------------------------------------------------------


class TestRouteByCategory:
    def test_question(self):
        assert route_by_category({"category": "question"}) == "question"

    def test_complaint(self):
        assert route_by_category({"category": "complaint"}) == "complaint"

    def test_unclear(self):
        assert route_by_category({"category": "unclear"}) == "unclear"


# ---------------------------------------------------------------------------
# Integration tests — real Gemini API, requires GEMINI_API_KEY in .env
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestClassifyNodeIntegration:
    """Integration tests hitting the real Gemini API.

    Run with: pytest -m integration
    """

    def test_question_ticket(self):
        state = {
            "ticket_text": "Здравствуйте, подскажите, как оформить возврат товара?",
            "messages": [],
        }
        result = classify_node(state)
        assert result["category"] == "question"
        assert result["priority"] in ("low", "medium")
        assert result["language"] == "ru"

    def test_complaint_ticket(self):
        state = {
            "ticket_text": (
                "Мой заказ не пришёл уже 10 дней! "
                "Это безобразие, требую возврат денег немедленно!"
            ),
            "messages": [],
        }
        result = classify_node(state)
        assert result["category"] == "complaint"
        assert result["priority"] in ("high", "critical")
        assert result["language"] == "ru"

    def test_unclear_ticket(self):
        state = {
            "ticket_text": "Помогите",
            "messages": [],
        }
        result = classify_node(state)
        assert result["category"] == "unclear"

    def test_short_angry_ticket_is_complaint_or_unclear(self):
        """Short emotional messages like 'Не работает!!!' are borderline."""
        state = {
            "ticket_text": "Не работает!!!",
            "messages": [],
        }
        result = classify_node(state)
        assert result["category"] in ("complaint", "unclear")

    def test_english_ticket_detected(self):
        state = {
            "ticket_text": "I want to return my order, it arrived damaged.",
            "messages": [],
        }
        result = classify_node(state)
        assert result["language"] == "en"

    # --- Edge-case integration tests (Block 7.1) ---

    def test_empty_text_classified_as_unclear(self):
        """Real Gemini should classify an empty string as 'unclear'."""
        state = {"ticket_text": "", "messages": []}
        result = classify_node(state)
        assert result["category"] == "unclear"

    def test_long_text_does_not_error(self):
        """Real Gemini should handle a 5 000-char input without errors."""
        state = {
            "ticket_text": "Я хочу вернуть товар. " * 250,  # ~5500 chars
            "messages": [],
        }
        result = classify_node(state)
        assert set(result.keys()) == {"category", "priority", "language"}

    def test_mixed_language_detects_dominant(self):
        """Mixed Russian/English text should detect one of the two languages."""
        state = {
            "ticket_text": (
                "Привет, I ordered a product but оно не пришло. "
                "Когда будет delivery?"
            ),
            "messages": [],
        }
        result = classify_node(state)
        assert result["language"] in ("ru", "en")

    def test_german_ticket_detected(self):
        """German complaint text should be detected as 'de'."""
        state = {
            "ticket_text": (
                "Ich bin sehr enttäuscht! Meine Bestellung ist seit "
                "einer Woche nicht angekommen. Ich fordere eine Rückerstattung!"
            ),
            "messages": [],
        }
        result = classify_node(state)
        assert result["language"] == "de"
