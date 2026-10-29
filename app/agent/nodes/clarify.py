"""Clarify node — human-in-the-loop for unclear tickets.

When a ticket is classified as "unclear", this node generates a clarifying
question via Gemini 2.5 Flash and pauses the graph using LangGraph's
interrupt() mechanism.  On resume the user's reply is appended to the
ticket text so the graph can re-classify the enriched message.
"""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from langgraph.types import interrupt

from app.agent.state import AgentState
from app.config import settings

CLARIFY_SYSTEM_PROMPT = """\
You are a customer support specialist. The customer's message is too vague \
or incomplete to process.

Your task is to generate ONE clear, specific clarifying question to \
understand what the customer needs help with.

Rules:
- Write the question in the same language as the customer's message.
- Be polite and professional.
- Ask about the specific issue, product, or service they need help with.
- Keep it concise — one question, 1-2 sentences maximum.
- Do NOT try to answer or solve anything — only ask for clarification."""


class ClarifyResult(BaseModel):
    """Structured output schema for the clarifying question."""

    clarifying_question: str = Field(
        description=(
            "A polite, specific clarifying question for the customer, "
            "written in the same language as their message."
        ),
    )


def clarify_node(state: AgentState) -> dict:
    """Generate a clarifying question and wait for the user's reply.

    Uses Gemini 2.5 Flash to formulate the question, then calls
    ``interrupt()`` to pause the graph.  When the graph is resumed with
    the user's answer, the original ticket text is enriched and both the
    question and the answer are added to the message history.

    Returns:
        dict with keys: ticket_text, messages.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.3,
    )
    structured_llm = llm.with_structured_output(ClarifyResult)

    result: ClarifyResult = structured_llm.invoke([
        ("system", CLARIFY_SYSTEM_PROMPT),
        ("human", state["ticket_text"]),
    ])

    # Pause the graph — surface the clarifying question to the caller.
    # On resume, interrupt() returns the user's reply.
    user_reply = interrupt(result.clarifying_question)

    updated_text = f"{state['ticket_text']}\n\nUser clarification: {user_reply}"

    return {
        "ticket_text": updated_text,
        "messages": [
            AIMessage(content=result.clarifying_question),
            HumanMessage(content=str(user_reply)),
        ],
    }
