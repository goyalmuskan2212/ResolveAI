"""Classification node — entry point of the agent graph.

Calls Gemini 2.5 Flash with structured output to determine ticket category,
priority, language, and a brief summary.
"""

from typing import Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from app.agent.state import AgentState
from app.config import settings

CLASSIFICATION_SYSTEM_PROMPT = """\
You are a customer support ticket classifier.

Analyze the customer's message and determine:

1. **Category** (category):
   - "question" — the customer is asking a question or requesting information \
(delivery, returns, order status, etc.)
   - "complaint" — the customer is expressing dissatisfaction, complaining, \
or demanding a resolution / compensation
   - "unclear" — the message is too short, vague, or lacks sufficient \
information to classify

2. **Priority** (priority):
   - "critical" — mentions of lawsuits, legal threats, personal data leaks, \
security threats
   - "high" — complaint with an explicit refund demand, or a repeated \
contact about an unresolved issue
   - "medium" — standard complaint without a refund demand, or a complex \
multi-part question
   - "low" — simple informational question that can be answered from FAQ

3. **Language** (language): detect the language of the message (ISO 639-1 code).

4. **Summary** (summary): summarize the essence of the message in 1-2 sentences \
in the same language as the original message."""


class ClassificationResult(BaseModel):
    """Structured output schema for ticket classification."""

    category: Literal["question", "complaint", "unclear"] = Field(
        description=(
            "Ticket category: 'question' for informational requests, "
            "'complaint' for dissatisfaction or demands, "
            "'unclear' for vague or insufficient messages"
        ),
    )
    priority: Literal["low", "medium", "high", "critical"] = Field(
        description=(
            "Ticket priority: 'critical' for legal/security threats, "
            "'high' for refund demands or repeat contacts, "
            "'medium' for standard complaints or complex questions, "
            "'low' for simple FAQ questions"
        ),
    )
    language: str = Field(
        description="Language of the customer message as an ISO 639-1 code (e.g. 'ru', 'en', 'de')",
    )
    summary: str = Field(
        description="Brief 1-2 sentence summary of the ticket in the original language",
    )


def classify_node(state: AgentState) -> dict:
    """Classify an incoming support ticket.

    Uses Gemini 2.5 Flash with structured output (function calling) to extract
    category, priority, and language from the ticket text.

    Returns:
        dict with keys: category, priority, language.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.0,
    )
    structured_llm = llm.with_structured_output(ClassificationResult)

    result: ClassificationResult = structured_llm.invoke([
        ("system", CLASSIFICATION_SYSTEM_PROMPT),
        ("human", state["ticket_text"]),
    ])

    return {
        "category": result.category,
        "priority": result.priority,
        "language": result.language,
    }
