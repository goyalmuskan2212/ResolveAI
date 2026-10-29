"""streamlit_app.py — Streamlit UI for the AI Support Agent.

Two-column layout:
- Left  (~2/3): chat interface with session history and clarify flow
- Right (~1/3): live metrics dashboard with category distribution chart

Below main columns:
- Trace timeline (last response)
- Ticket history table

Communicates with FastAPI backend via httpx (AR-3).
Run: streamlit run streamlit_app.py
"""

from datetime import date

import httpx
import pandas as pd
import streamlit as st

API_BASE_URL_DEFAULT = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Section 1: Session state
# ---------------------------------------------------------------------------


def init_session_state() -> None:
    defaults = {
        "messages": [],
        "pending_clarification": None,
        "api_base_url": API_BASE_URL_DEFAULT,
        "metrics": None,
        "ticket_history": [],
        "last_trace": None,
        "date_from": None,
        "date_until": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ---------------------------------------------------------------------------
# Section 2: API client functions
# ---------------------------------------------------------------------------


def submit_ticket(text: str, base_url: str) -> dict:
    """POST /ticket — submit a support ticket."""
    try:
        resp = httpx.post(
            f"{base_url}/ticket",
            json={"text": text},
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        return {"error": f"API error {exc.response.status_code}: {detail}"}
    except httpx.RequestError as exc:
        return {"error": f"Cannot connect to backend at {base_url}. Is the API server running? ({exc})"}


def submit_clarification(ticket_id: str, reply: str, base_url: str) -> dict:
    """POST /ticket/{id}/clarify — resume interrupted clarify flow."""
    try:
        resp = httpx.post(
            f"{base_url}/ticket/{ticket_id}/clarify",
            json={"reply": reply},
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        return {"error": f"API error {exc.response.status_code}: {detail}"}
    except httpx.RequestError as exc:
        return {"error": f"Cannot connect to backend at {base_url}. ({exc})"}


def fetch_metrics(
    base_url: str,
    from_date: str | None = None,
    to_date: str | None = None,
) -> dict:
    """GET /metrics — fetch aggregated metrics with optional date range."""
    params: dict = {}
    if from_date:
        params["from_date"] = from_date
    if to_date:
        params["to_date"] = to_date
    try:
        resp = httpx.get(f"{base_url}/metrics", params=params, timeout=10.0)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as exc:
        return {"error": f"API error {exc.response.status_code}", "total_traces": 0, "avg_latency_ms": 0.0, "avg_cost": 0.0}
    except httpx.RequestError as exc:
        return {"error": f"Cannot reach backend: {exc}", "total_traces": 0, "avg_latency_ms": 0.0, "avg_cost": 0.0}


def fetch_trace(ticket_id: str, base_url: str) -> dict | None:
    """GET /ticket/{id}/trace — fetch Langfuse trace data for a ticket."""
    try:
        resp = httpx.get(f"{base_url}/ticket/{ticket_id}/trace", timeout=10.0)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Section 3: Rendering helpers
# ---------------------------------------------------------------------------


_PRIORITY_COLORS = {
    "low": "#28a745",
    "medium": "#ffc107",
    "high": "#fd7e14",
    "critical": "#dc3545",
}

_CATEGORY_COLOR = "#007bff"


def _badge(text: str, color: str, text_color: str = "#fff") -> str:
    return (
        f'<span style="background:{color};color:{text_color};padding:2px 8px;'
        f'border-radius:12px;font-size:0.75em;font-weight:600;margin-right:4px;">'
        f"{text}</span>"
    )


def render_metadata_badges(metadata: dict) -> None:
    """Render colored inline badges for category, priority, escalated, quality_score."""
    if not metadata:
        return

    parts = []

    category = metadata.get("category")
    if category:
        parts.append(_badge(category.capitalize(), _CATEGORY_COLOR))

    priority = metadata.get("priority")
    if priority:
        color = _PRIORITY_COLORS.get(priority.lower(), "#6c757d")
        text_color = "#000" if priority.lower() == "medium" else "#fff"
        parts.append(_badge(priority.capitalize(), color, text_color))

    language = metadata.get("language")
    if language:
        parts.append(_badge(language.upper(), "#17a2b8"))

    if metadata.get("escalated"):
        parts.append(_badge("Escalated", "#dc3545"))

    if metadata.get("needs_clarification"):
        parts.append(_badge("Needs Clarification", "#ffc107", "#000"))

    quality = metadata.get("quality_score", 0.0)
    if quality and quality > 0.0:
        parts.append(_badge(f"Q: {quality:.2f}", "#6c757d"))

    if parts:
        st.markdown("".join(parts), unsafe_allow_html=True)


def render_trace_timeline(trace_data: dict) -> None:
    """Render a horizontal timeline of LangGraph nodes with their durations."""
    observations = [
        o for o in (trace_data.get("observations") or [])
        if (o.get("latency_ms") or 0) > 0
    ]
    if not observations:
        return

    st.markdown("**Trace Timeline**")

    # Build a bar chart: node name → latency_ms
    series = {o["name"]: o["latency_ms"] for o in observations}
    df = pd.DataFrame.from_dict(series, orient="index", columns=["latency_ms"])
    st.bar_chart(df, height=180)

    total_ms = trace_data.get("duration_ms") or sum(o.get("latency_ms", 0) for o in observations)
    total_cost = trace_data.get("total_cost") or 0.0
    st.caption(
        f"Total: {total_ms:,.0f} ms  |  Cost: ${total_cost:.6f}"
    )


def render_category_chart(data: dict) -> None:
    """Render bar chart for category distribution."""
    dist = data.get("category_distribution") or {}
    if not dist:
        return

    st.markdown("**Category Distribution**")
    df = pd.DataFrame.from_dict(dist, orient="index", columns=["count"])
    st.bar_chart(df, height=180)


# ---------------------------------------------------------------------------
# Section 4: Panel functions
# ---------------------------------------------------------------------------


def render_sidebar() -> None:
    with st.sidebar:
        st.title("Settings")
        url = st.text_input(
            "API Base URL",
            value=st.session_state.api_base_url,
            help="URL of the FastAPI backend (e.g. http://localhost:8000)",
        )
        if url != st.session_state.api_base_url:
            st.session_state.api_base_url = url
            st.session_state.metrics = None

        st.divider()
        st.subheader("Metrics Filter")

        date_from = st.date_input(
            "From date",
            value=st.session_state.date_from,
            help="Filter metrics from this date (inclusive)",
        )
        date_until = st.date_input(
            "Until date",
            value=st.session_state.date_until,
            help="Filter metrics up to this date (inclusive)",
        )

        if st.button("Apply Filter"):
            st.session_state.date_from = date_from if isinstance(date_from, date) else None
            st.session_state.date_until = date_until if isinstance(date_until, date) else None
            from_str = st.session_state.date_from.isoformat() if st.session_state.date_from else None
            until_str = st.session_state.date_until.isoformat() if st.session_state.date_until else None
            st.session_state.metrics = fetch_metrics(
                st.session_state.api_base_url,
                from_date=from_str,
                to_date=until_str,
            )

        if st.button("Refresh Metrics"):
            from_str = st.session_state.date_from.isoformat() if st.session_state.date_from else None
            until_str = st.session_state.date_until.isoformat() if st.session_state.date_until else None
            st.session_state.metrics = fetch_metrics(
                st.session_state.api_base_url,
                from_date=from_str,
                to_date=until_str,
            )

        st.divider()
        st.caption("AI Support Agent — powered by LangGraph, LlamaIndex, and Gemini.")


def render_chat_panel() -> None:
    st.subheader("Support Chat")

    # Render full message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("metadata"):
                render_metadata_badges(msg["metadata"])

    # Trace timeline for the last response
    if st.session_state.last_trace:
        with st.expander("Trace Timeline", expanded=False):
            render_trace_timeline(st.session_state.last_trace)

    # Clarification state banner
    if st.session_state.pending_clarification:
        st.info("Please answer the clarifying question above to continue.")

    # Chat input — label adapts to state
    label = (
        "Your clarification..."
        if st.session_state.pending_clarification
        else "Type your support message..."
    )
    user_input = st.chat_input(label)

    if not user_input:
        return

    # Append user message to history
    st.session_state.messages.append({"role": "user", "content": user_input, "metadata": None})

    base_url = st.session_state.api_base_url

    with st.spinner("Processing..."):
        if st.session_state.pending_clarification:
            ticket_id = st.session_state.pending_clarification
            resp = submit_clarification(ticket_id, user_input, base_url)
            st.session_state.pending_clarification = None
        else:
            resp = submit_ticket(user_input, base_url)

    if "error" in resp:
        st.error(resp["error"])
        return

    # Fetch trace for timeline
    trace_id = resp.get("trace_id")
    if trace_id:
        st.session_state.last_trace = fetch_trace(trace_id, base_url)
    else:
        st.session_state.last_trace = None

    if resp.get("needs_clarification"):
        st.session_state.pending_clarification = resp["ticket_id"]
        st.session_state.messages.append({
            "role": "assistant",
            "content": resp.get("clarifying_question") or "Could you please clarify?",
            "metadata": {**resp},
        })
    else:
        st.session_state.messages.append({
            "role": "assistant",
            "content": resp.get("response", ""),
            "metadata": resp,
        })
        # Record to history
        import datetime as _dt
        st.session_state.ticket_history.append({
            "time": _dt.datetime.now().strftime("%H:%M:%S"),
            "preview": user_input[:60] + ("…" if len(user_input) > 60 else ""),
            "category": resp.get("category", ""),
            "priority": resp.get("priority", ""),
            "quality": f"{resp.get('quality_score', 0.0):.2f}",
            "status": "escalated" if resp.get("escalated") else "resolved",
            "ticket_id": resp.get("ticket_id", ""),
        })

    st.rerun()


def render_metrics_panel() -> None:
    st.subheader("Metrics")

    base_url = st.session_state.api_base_url

    # Auto-fetch on first render
    if st.session_state.metrics is None:
        from_str = st.session_state.date_from.isoformat() if st.session_state.date_from else None
        until_str = st.session_state.date_until.isoformat() if st.session_state.date_until else None
        st.session_state.metrics = fetch_metrics(base_url, from_date=from_str, to_date=until_str)

    data = st.session_state.metrics or {}

    if err := data.get("error"):
        st.warning(f"Metrics unavailable: {err}")

    st.metric("Total Traces", data.get("total_traces", 0))
    st.metric("Avg Latency", f"{data.get('avg_latency_ms', 0.0):.0f} ms")
    st.metric("Avg Cost", f"${data.get('avg_cost', 0.0):.6f}")
    st.metric("% Automatic", f"{data.get('automation_pct', 0.0):.1f}%")

    st.divider()
    render_category_chart(data)


def render_history_panel() -> None:
    """Render full-width ticket history table below the main columns."""
    history = st.session_state.ticket_history
    if not history:
        return

    st.divider()
    st.subheader("Ticket History")
    df = pd.DataFrame(history)
    st.dataframe(df, use_container_width=True)


# ---------------------------------------------------------------------------
# Section 5: Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="AI Support Agent",
        page_icon="🤖",
        layout="wide",
    )
    init_session_state()
    render_sidebar()

    st.title("AI Support Agent")
    st.caption("Powered by LangGraph · LlamaIndex · Gemini · Langfuse")

    col_chat, col_metrics = st.columns([2, 1])

    with col_chat:
        render_chat_panel()

    with col_metrics:
        render_metrics_panel()

    render_history_panel()


main()
