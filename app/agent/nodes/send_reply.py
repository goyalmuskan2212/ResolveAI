"""Send reply node — formats the final customer-facing response.

Pure formatting node (no LLM calls). Assembles a greeting, the draft
response body, source references, and a closing signature into the
final reply ready to be sent to the customer.
"""

from app.agent.state import AgentState

GREETINGS = {
    "ru": "Здравствуйте!",
    "en": "Hello!",
    "de": "Hallo!",
    "fr": "Bonjour!",
    "es": "Hola!",
}

DEFAULT_GREETING = "Hello!"

SIGNATURE = "\nBest regards,\nSupport Team"


def send_reply_node(state: AgentState) -> dict:
    """Format the final customer-facing response.

    Assembles greeting (language-aware, skipped for complaint path),
    draft response body, source references (deduplicated), and a
    closing signature.

    Returns:
        dict with key: draft_response.
    """
    parts: list[str] = []

    # Greeting — skip for complaint path (empathy_preamble already provides opening)
    if not state.get("empathy_preamble"):
        language = state.get("language", "en")
        greeting = GREETINGS.get(language, DEFAULT_GREETING)
        parts.append(greeting)

    # Body
    draft = state.get("draft_response", "")
    parts.append(draft)

    # Source references
    sources = state.get("sources", [])
    if sources:
        filenames: list[str] = []
        seen: set[str] = set()
        for source in sources:
            metadata = source.get("metadata", {}) if isinstance(source, dict) else {}
            name = metadata.get("file_name", "Knowledge base")
            if name not in seen:
                seen.add(name)
                filenames.append(name)

        if filenames:
            sources_section = "---\nSources:\n" + "\n".join(
                f"- {fn}" for fn in filenames
            )
            parts.append(sources_section)

    # Signature
    parts.append(SIGNATURE)

    formatted_response = "\n\n".join(parts)

    return {"draft_response": formatted_response}
