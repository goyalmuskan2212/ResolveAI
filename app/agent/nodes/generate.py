"""Generate node — produces the draft customer response using RAG context.

Combines retrieved knowledge base contexts with optional empathy enrichment
(for complaint tickets) to generate a grounded, helpful response.
"""

from langchain_google_genai import ChatGoogleGenerativeAI

from app.agent.state import AgentState
from app.config import settings

GENERATE_SYSTEM_PROMPT = """\
You are a professional customer support agent.

Your task is to write a helpful, accurate response to the customer's message \
using ONLY the information provided in the knowledge base context below. \
Do NOT make up information that is not in the context.

## Knowledge Base Context
{contexts}

## Instructions
- Answer the customer's question accurately based on the context above.
- If the context does not contain enough information to fully answer, say so \
honestly and suggest the customer contact support for further assistance.
- Write the response in the same language as the customer's message.
- Be concise but thorough.
{tone_section}"""

TONE_SECTION_TEMPLATE = """
## Tone Guidelines
{tone_instructions}

## Opening (use this as the first paragraph)
{empathy_preamble}
"""


def generate_response_node(state: AgentState) -> dict:
    """Generate a draft response using RAG context and optional empathy enrichment.

    Uses Gemini Flash to produce a response grounded in the retrieved
    knowledge base contexts. If tone_instructions and empathy_preamble are
    present (complaint path), they are included in the prompt.

    Returns:
        dict with key: draft_response.
    """
    contexts_text = "\n\n---\n\n".join(state.get("retrieved_contexts", []))

    tone_section = ""
    if state.get("tone_instructions") and state.get("empathy_preamble"):
        tone_section = TONE_SECTION_TEMPLATE.format(
            tone_instructions=state["tone_instructions"],
            empathy_preamble=state["empathy_preamble"],
        )

    system_prompt = GENERATE_SYSTEM_PROMPT.format(
        contexts=contexts_text,
        tone_section=tone_section,
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.3,
    )

    response = llm.invoke([
        ("system", system_prompt),
        ("human", state["ticket_text"]),
    ])

    content = response.content
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )

    return {"draft_response": content}
