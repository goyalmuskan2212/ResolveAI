"""Empathy node — generates tone instructions for complaint tickets.

Per AR-1, this node does NOT perform RAG search. It only sets
tone_instructions and empathy_preamble in state, which the generate
node will use to craft an empathetic response.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from app.agent.state import AgentState
from app.config import settings

EMPATHY_SYSTEM_PROMPT = """\
You are an empathy specialist for customer support.

Given a customer complaint, generate two things:

1. **tone_instructions**: Brief internal instructions (1-2 sentences) for the \
response generator about what tone and approach to use. These are NOT shown to \
the customer — they guide the AI that writes the final response.

2. **empathy_preamble**: A short empathetic opening (2-3 sentences) that \
acknowledges the customer's feelings and situation. This WILL be shown to the \
customer as the opening of the response. Write it in the same language as the \
customer's message.

Focus on validation and acknowledgment. Do not offer solutions or promises — \
that comes from the knowledge base in the next step."""


class EmpathyResult(BaseModel):
    """Structured output schema for empathy generation."""

    tone_instructions: str = Field(
        description=(
            "Instructions for the response generator on tone and style, "
            "e.g. 'Use a warm, apologetic tone. Acknowledge the customer's "
            "frustration before providing a solution.'"
        ),
    )
    empathy_preamble: str = Field(
        description=(
            "An empathetic opening paragraph to prepend to the response, "
            "acknowledging the customer's feelings and situation. "
            "Written in the same language as the customer's message."
        ),
    )


def empathy_node(state: AgentState) -> dict:
    """Generate empathy-enriched tone instructions for complaint tickets.

    Uses Gemini Flash to produce tone_instructions and empathy_preamble
    based on the complaint text. Per AR-1, does NOT perform any RAG search.

    Returns:
        dict with keys: tone_instructions, empathy_preamble.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.4,
    )
    structured_llm = llm.with_structured_output(EmpathyResult)

    result: EmpathyResult = structured_llm.invoke([
        ("system", EMPATHY_SYSTEM_PROMPT),
        ("human", state["ticket_text"]),
    ])

    return {
        "tone_instructions": result.tone_instructions,
        "empathy_preamble": result.empathy_preamble,
    }
