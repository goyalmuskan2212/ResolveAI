"""Langfuse metrics — trace retrieval and aggregate statistics.

Queries the Langfuse API for individual traces and computes aggregate
metrics (latency, cost). Domain data (category, escalation) is stored
as Langfuse scores and available per-trace via get_trace_by_id().
"""

import logging
from datetime import datetime

from langfuse.api.core.request_options import RequestOptions

from app.tracing.langfuse_setup import get_langfuse_client

logger = logging.getLogger(__name__)

_REQ_OPTS = RequestOptions(timeout_in_seconds=15)


def get_trace_by_id(trace_id: str) -> dict | None:
    """Fetch a single trace by ID and return a structured dict.

    Returns None if the trace is not found or Langfuse is unreachable.
    """
    try:
        client = get_langfuse_client()
        trace = client.api.trace.get(trace_id, request_options=_REQ_OPTS)

        observations = []
        for obs in getattr(trace, "observations", []) or []:
            observations.append({
                "name": obs.name,
                "latency_ms": obs.latency,
                "model": getattr(obs, "model", None),
                "cost": getattr(obs, "total_cost", None),
            })

        return {
            "trace_id": trace.id,
            "name": trace.name,
            "duration_ms": trace.latency,
            "total_cost": trace.total_cost,
            "observations": observations,
            "metadata": trace.metadata or {},
            "scores": getattr(trace, "scores", []) or [],
        }
    except Exception:
        logger.warning("Failed to fetch trace %s", trace_id, exc_info=True)
        return None


def get_aggregate_metrics(
    limit: int = 100,
    from_timestamp: datetime | None = None,
    to_timestamp: datetime | None = None,
) -> dict:
    """Compute aggregate metrics from recent Langfuse traces.

    Returns a dict with total_traces, avg_latency_ms, avg_cost,
    automation_pct, and category_distribution.
    On error, returns a dict with zeros and an 'error' field.

    Args:
        limit: Maximum number of traces to fetch.
        from_timestamp: Only include traces after this datetime (UTC).
        to_timestamp: Only include traces before this datetime (UTC).
    """
    empty = {
        "total_traces": 0,
        "avg_latency_ms": 0.0,
        "avg_cost": 0.0,
        "automation_pct": 0.0,
        "category_distribution": {},
    }

    try:
        client = get_langfuse_client()
        list_kwargs: dict = {"limit": limit}
        if from_timestamp:
            list_kwargs["from_timestamp"] = from_timestamp
        if to_timestamp:
            list_kwargs["to_timestamp"] = to_timestamp
        response = client.api.trace.list(**list_kwargs, request_options=_REQ_OPTS)
        traces = response.data
    except Exception as exc:
        logger.warning("Failed to fetch traces for metrics", exc_info=True)
        return {**empty, "error": str(exc)}

    if not traces:
        return empty

    total = len(traces)
    total_latency = sum(t.latency or 0 for t in traces)
    total_cost = sum(t.total_cost or 0 for t in traces)

    # Escalation rate
    try:
        score_kwargs: dict = {"name": "escalated", "limit": limit}
        if from_timestamp:
            score_kwargs["from_timestamp"] = from_timestamp
        if to_timestamp:
            score_kwargs["to_timestamp"] = to_timestamp
        scores_resp = client.api.scores.get_many(**score_kwargs, request_options=_REQ_OPTS)
        escalation_scores = scores_resp.data or []
        escalated_count = sum(
            1 for s in escalation_scores
            if str(getattr(s, "value", "")).lower() == "true"
        )
        total_scored = len(escalation_scores)
        automation_pct = (
            round((1 - escalated_count / total_scored) * 100, 1)
            if total_scored > 0
            else 0.0
        )
    except Exception:
        logger.warning("Failed to fetch escalation scores", exc_info=True)
        automation_pct = 0.0

    # Category distribution
    category_distribution: dict[str, int] = {}
    try:
        cat_kwargs: dict = {"name": "category", "limit": limit}
        if from_timestamp:
            cat_kwargs["from_timestamp"] = from_timestamp
        if to_timestamp:
            cat_kwargs["to_timestamp"] = to_timestamp
        cat_resp = client.api.scores.get_many(**cat_kwargs, request_options=_REQ_OPTS)
        for s in cat_resp.data or []:
            val = str(getattr(s, "string_value", None) or getattr(s, "value", "unknown"))
            category_distribution[val] = category_distribution.get(val, 0) + 1
    except Exception:
        logger.warning("Failed to fetch category scores", exc_info=True)

    return {
        "total_traces": total,
        "avg_latency_ms": round(total_latency / total, 2),
        "avg_cost": round(total_cost / total, 6),
        "automation_pct": automation_pct,
        "category_distribution": category_distribution,
    }
