"""Tests for Langfuse tracing setup and metrics (Block 4.2)."""

import logging
from unittest.mock import MagicMock, patch

import app.tracing.langfuse_setup as _lf_mod
import pytest

from app.tracing.langfuse_setup import (
    get_langfuse_handler,
    get_langfuse_client,
    trace_context,
    record_trace_scores,
)
from app.tracing.metrics import get_trace_by_id, get_aggregate_metrics


@pytest.fixture(autouse=True)
def _reset_langfuse_singleton():
    """Reset the module-level singleton before each test to avoid leaks."""
    _lf_mod._langfuse_client = None
    yield
    _lf_mod._langfuse_client = None


# ---------------------------------------------------------------------------
# get_langfuse_handler
# ---------------------------------------------------------------------------


class TestGetLangfuseHandler:
    """Tests for get_langfuse_handler factory."""

    @patch("app.tracing.langfuse_setup.Langfuse")
    @patch("app.tracing.langfuse_setup.CallbackHandler")
    @patch("app.tracing.langfuse_setup.settings")
    def test_returns_handler_when_enabled(self, mock_settings, mock_cb_cls, mock_lf_cls):
        mock_settings.LANGFUSE_ENABLED = True
        mock_settings.LANGFUSE_PUBLIC_KEY = "pk-test"

        handler = get_langfuse_handler()
        assert handler is not None
        mock_cb_cls.assert_called_once_with(public_key="pk-test")

    @patch("app.tracing.langfuse_setup.settings")
    def test_returns_none_when_disabled(self, mock_settings):
        mock_settings.LANGFUSE_ENABLED = False

        handler = get_langfuse_handler()
        assert handler is None

    @patch("app.tracing.langfuse_setup.Langfuse")
    @patch("app.tracing.langfuse_setup.CallbackHandler")
    @patch("app.tracing.langfuse_setup.settings")
    def test_returns_none_on_connection_error(self, mock_settings, mock_cb_cls, mock_lf_cls):
        mock_settings.LANGFUSE_ENABLED = True
        mock_settings.LANGFUSE_PUBLIC_KEY = "pk-test"
        mock_cb_cls.side_effect = Exception("Connection refused")

        handler = get_langfuse_handler()
        assert handler is None


# ---------------------------------------------------------------------------
# trace_context
# ---------------------------------------------------------------------------


class TestTraceContext:
    """Tests for trace_context context manager."""

    @patch("app.tracing.langfuse_setup.propagate_attributes")
    def test_passes_attributes_to_propagate(self, mock_propagate):
        trace_context(
            trace_name="support-ticket",
            session_id="s-1",
            user_id="u-1",
        )
        mock_propagate.assert_called_once_with(
            trace_name="support-ticket",
            session_id="s-1",
            user_id="u-1",
        )


# ---------------------------------------------------------------------------
# get_langfuse_client
# ---------------------------------------------------------------------------


class TestGetLangfuseClient:
    """Tests for get_langfuse_client factory."""

    @patch("app.tracing.langfuse_setup.Langfuse")
    @patch("app.tracing.langfuse_setup.settings")
    def test_returns_client_with_correct_params(self, mock_settings, mock_lf_cls):
        mock_settings.LANGFUSE_ENABLED = True
        mock_settings.LANGFUSE_PUBLIC_KEY = "pk-test"
        mock_settings.LANGFUSE_SECRET_KEY = "sk-test"
        mock_settings.LANGFUSE_BASE_URL = "http://localhost:3000"

        client = get_langfuse_client()
        mock_lf_cls.assert_called_once_with(
            public_key="pk-test",
            secret_key="sk-test",
            host="http://localhost:3000",
        )
        assert client is not None


# ---------------------------------------------------------------------------
# record_trace_scores
# ---------------------------------------------------------------------------


class TestRecordTraceScores:
    """Tests for record_trace_scores."""

    @patch("app.tracing.langfuse_setup.Langfuse")
    @patch("app.tracing.langfuse_setup.settings")
    def test_records_scores_and_flushes(self, mock_settings, mock_lf_cls):
        mock_settings.LANGFUSE_ENABLED = True
        mock_client = MagicMock()
        mock_lf_cls.return_value = mock_client

        mock_handler = MagicMock()
        mock_handler.last_trace_id = "trace-123"

        record_trace_scores(
            mock_handler,
            category="question",
            escalated=False,
            quality_score=0.85,
            priority="low",
        )

        # Should create one score per domain field
        assert mock_client.create_score.call_count == 4
        # Verify one of the calls
        calls = {c[1]["name"]: c[1] for c in mock_client.create_score.call_args_list}
        assert calls["category"]["value"] == "question"
        assert calls["category"]["trace_id"] == "trace-123"
        assert calls["quality_score"]["value"] == 0.85
        assert calls["escalated"]["value"] == "False"
        # Should flush after all scores
        mock_client.flush.assert_called_once()

    def test_noop_when_handler_is_none(self):
        # Should not raise
        record_trace_scores(None, category="question")

    @patch("app.tracing.langfuse_setup.Langfuse")
    @patch("app.tracing.langfuse_setup.settings")
    def test_noop_when_no_trace_id(self, mock_settings, mock_lf_cls):
        mock_settings.LANGFUSE_ENABLED = True
        mock_client = MagicMock()
        mock_lf_cls.return_value = mock_client

        mock_handler = MagicMock()
        mock_handler.last_trace_id = None

        # Should not raise, should not create scores
        record_trace_scores(mock_handler, category="question")
        mock_client.create_score.assert_not_called()


# ---------------------------------------------------------------------------
# get_trace_by_id
# ---------------------------------------------------------------------------


class TestGetTraceById:
    """Tests for get_trace_by_id."""

    @patch("app.tracing.metrics.get_langfuse_client")
    def test_returns_structured_dict(self, mock_get_client):
        mock_trace = MagicMock()
        mock_trace.id = "trace-abc"
        mock_trace.name = "support-ticket"
        mock_trace.latency = 3200
        mock_trace.total_cost = 0.015
        mock_trace.metadata = {"environment": "production"}
        mock_trace.scores = ["score-1"]

        mock_obs = MagicMock()
        mock_obs.name = "classify"
        mock_obs.latency = 800
        mock_obs.model = "gemini-2.5-flash"
        mock_obs.total_cost = 0.002
        mock_trace.observations = [mock_obs]

        mock_get_client.return_value.api.trace.get.return_value = mock_trace

        result = get_trace_by_id("trace-abc")

        assert result is not None
        assert result["trace_id"] == "trace-abc"
        assert result["name"] == "support-ticket"
        assert result["duration_ms"] == 3200
        assert result["total_cost"] == 0.015
        assert len(result["observations"]) == 1
        assert result["observations"][0]["name"] == "classify"

    @patch("app.tracing.metrics.get_langfuse_client")
    def test_returns_none_for_missing_trace(self, mock_get_client):
        mock_get_client.return_value.api.trace.get.side_effect = Exception("Not found")

        result = get_trace_by_id("nonexistent")
        assert result is None


# ---------------------------------------------------------------------------
# get_aggregate_metrics
# ---------------------------------------------------------------------------


class TestGetAggregateMetrics:
    """Tests for get_aggregate_metrics."""

    @patch("app.tracing.metrics.get_langfuse_client")
    def test_computes_correctly(self, mock_get_client):
        traces = []
        for lat, cost in [(2000, 0.01), (3000, 0.02), (4000, 0.03)]:
            t = MagicMock()
            t.latency = lat
            t.total_cost = cost
            traces.append(t)

        mock_response = MagicMock()
        mock_response.data = traces
        mock_get_client.return_value.api.trace.list.return_value = mock_response

        result = get_aggregate_metrics(limit=10)

        assert result["total_traces"] == 3
        assert result["avg_latency_ms"] == 3000.0
        assert abs(result["avg_cost"] - 0.02) < 0.001

    @patch("app.tracing.metrics.get_langfuse_client")
    def test_handles_empty(self, mock_get_client):
        mock_response = MagicMock()
        mock_response.data = []
        mock_get_client.return_value.api.trace.list.return_value = mock_response

        result = get_aggregate_metrics()

        assert result["total_traces"] == 0
        assert result["avg_latency_ms"] == 0.0
        assert result["avg_cost"] == 0.0

    @patch("app.tracing.metrics.get_langfuse_client")
    def test_handles_error(self, mock_get_client):
        mock_get_client.return_value.api.trace.list.side_effect = Exception("down")

        result = get_aggregate_metrics()

        assert result["total_traces"] == 0
        assert "error" in result
