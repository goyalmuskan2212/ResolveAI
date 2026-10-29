"""Escalate node — formats an escalation package for a human operator.

Pure formatting node (no LLM calls). Collects ticket details, agent draft,
retrieved contexts, escalation reason, and conversation history into a
structured summary that helps the operator resolve the issue faster.
"""

from app.agent.state import AgentState


def escalate_node(state: AgentState) -> dict:
    """Format an escalation package for a human operator.

    Determines the escalation reason based on state (AR-4 early escalation,
    critical priority, or low quality score) and assembles all relevant
    information into a structured text package.

    Returns:
        dict with keys: needs_escalation, escalation_reason, draft_response.
    """
    # Determine escalation reason
    reason = state.get("escalation_reason", "")
    if not reason:
        if state.get("priority") == "critical":
            reason = "Critical priority ticket requires human review"
        else:
            score = state.get("quality_score", 0.0)
            iterations = state.get("iteration_count", 0)
            reason = (
                f"Low quality score ({score:.2f}) after {iterations} attempt(s)"
            )

    # Build escalation package
    ticket_text = state.get("ticket_text", "")
    category = state.get("category", "unknown")
    priority = state.get("priority", "unknown")
    language = state.get("language", "unknown")
    draft = state.get("draft_response", "")
    contexts = state.get("retrieved_contexts", [])
    messages = state.get("messages", [])

    # Format contexts
    if contexts:
        contexts_section = "\n".join(
            f"  {i}. {ctx[:200]}{'...' if len(ctx) > 200 else ''}"
            for i, ctx in enumerate(contexts, 1)
        )
    else:
        contexts_section = "  No contexts retrieved"

    # Format conversation history
    if messages:
        history_lines = []
        for msg in messages:
            role = getattr(msg, "type", "unknown")
            content = getattr(msg, "content", str(msg))
            history_lines.append(f"  [{role}]: {content}")
        history_section = "\n".join(history_lines)
    else:
        history_section = "  No conversation history"

    package = (
        f"=== ESCALATION PACKAGE ===\n\n"
        f"Ticket: {ticket_text}\n"
        f"Category: {category} | Priority: {priority} | Language: {language}\n\n"
        f"Escalation Reason: {reason}\n\n"
        f"--- Agent Draft ---\n"
        f"{draft or 'No draft generated'}\n\n"
        f"--- Retrieved Contexts ---\n"
        f"{contexts_section}\n\n"
        f"--- Conversation History ---\n"
        f"{history_section}\n\n"
        f"=== END ESCALATION ==="
    )

    return {
        "needs_escalation": True,
        "escalation_reason": reason,
        "draft_response": package,
    }
