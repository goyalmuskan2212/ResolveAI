"""Run a test ticket through the graph with Langfuse tracing enabled.

Prerequisites:
    - Docker services running (Qdrant, Langfuse, PostgreSQL)
    - Knowledge base indexed (python scripts/index_knowledge_base.py)
    - .env configured with valid keys

Usage:
    python scripts/test_tracing.py
"""

from app.agent.graph import get_agent
from app.tracing.langfuse_setup import (
    get_langfuse_handler,
    record_trace_scores,
    trace_context,
)

TEST_TICKETS = [
    "Как вернуть товар, купленный неделю назад?",
    "Я очень недоволен! Мне привезли сломанный товар, требую компенсацию!",
    "аывоаыво непонятно что",
]


def main():
    agent = get_agent()

    for i, ticket_text in enumerate(TEST_TICKETS, start=1):
        print(f"\n{'='*60}")
        print(f"Ticket {i}: {ticket_text}")
        print("=" * 60)

        handler = get_langfuse_handler()
        callbacks = [handler] if handler else []

        state = {
            "ticket_text": ticket_text,
            "messages": [],
        }

        session_id = f"manual-test-{i:03d}"
        thread_id = f"test-{i:03d}"

        with trace_context("test-ticket", session_id=session_id):
            result = agent.invoke(
                state,
                config={
                    "callbacks": callbacks,
                    "configurable": {"thread_id": thread_id},
                },
            )

        record_trace_scores(
            handler,
            category=result.get("category"),
            quality_score=result.get("quality_score"),
            escalated=result.get("needs_escalation"),
        )

        print(f"  Category:  {result.get('category')}")
        print(f"  Priority:  {result.get('priority')}")
        print(f"  Quality:   {result.get('quality_score')}")
        print(f"  Escalated: {result.get('needs_escalation')}")
        print(f"  Response:  {result.get('draft_response', '')[:200]}")

    print("\n\nDone. Check Langfuse UI at http://localhost:3000 for traces.")


if __name__ == "__main__":
    main()
