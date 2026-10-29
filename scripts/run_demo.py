"""Run 3 demo scenarios through the support agent with full output.

Demonstrates all three agent paths:
  1. Question  — classify → rag_search → generate → quality_check → send_reply
  2. Complaint — classify → empathy → rag_search → generate → quality_check → send_reply
  3. Unclear   — classify → clarify → (resume) → classify → ... → send_reply

Prerequisites:
    - Docker services running (Qdrant, Langfuse, PostgreSQL)
    - Knowledge base indexed (python scripts/index_knowledge_base.py)
    - .env configured with valid keys

Usage:
    python scripts/run_demo.py
"""

import time

from langgraph.types import Command

from app.agent.graph import get_agent
from app.tracing.langfuse_setup import (
    get_langfuse_handler,
    record_trace_scores,
    trace_context,
)

DEMO_SCENARIOS = [
    {
        "label": "Question (return policy)",
        "text": (
            "Hello, I'd like to return an item I bought 3 days ago. "
            "What is the return procedure?"
        ),
        "expected_path": (
            "classify -> rag_search -> generate -> quality_check -> send_reply"
        ),
        "clarify_reply": None,
    },
    {
        "label": "Complaint (missing order)",
        "text": (
            "My order hasn't arrived in 5 days! "
            "This is unacceptable, I demand a refund!"
        ),
        "expected_path": (
            "classify -> empathy -> rag_search -> generate -> quality_check -> send_reply"
        ),
        "clarify_reply": None,
    },
    {
        "label": "Unclear ticket (needs clarification)",
        "text": "Help me pls",
        "expected_path": (
            "classify -> clarify -> [user reply] -> classify -> rag_search -> ... -> send_reply"
        ),
        "clarify_reply": (
            "I can't process a return on the website, the button doesn't respond"
        ),
    },
]


def _print_header(number: int, scenario: dict) -> None:
    print(f"\n{'=' * 70}")
    print(f"  DEMO {number}: {scenario['label']}")
    print(f"  Expected path: {scenario['expected_path']}")
    print(f"{'=' * 70}")
    print(f"\n  Input: \"{scenario['text']}\"")


def _print_result(result: dict, elapsed: float, trace_id: str | None) -> None:
    print(f"\n  --- Classification ---")
    print(f"  Category : {result.get('category')}")
    print(f"  Priority : {result.get('priority')}")
    print(f"  Language : {result.get('language')}")

    print(f"\n  --- Quality ---")
    print(f"  Score    : {result.get('quality_score')}")
    print(f"  Escalated: {result.get('needs_escalation', False)}")
    if result.get("escalation_reason"):
        print(f"  Reason   : {result['escalation_reason']}")

    print(f"\n  --- Response ---")
    print(f"  {result.get('draft_response', '(no response)')}")

    print(f"\n  --- Stats ---")
    print(f"  Time     : {elapsed:.1f}s")
    if trace_id:
        print(f"  Trace ID : {trace_id}")


def run_scenario(agent, number: int, scenario: dict) -> dict:
    """Run a single demo scenario and return the result dict augmented with timing."""
    _print_header(number, scenario)

    handler = get_langfuse_handler()
    callbacks = [handler] if handler else []
    thread_id = f"demo-{number:03d}"

    config = {
        "callbacks": callbacks,
        "configurable": {"thread_id": thread_id},
    }

    start = time.perf_counter()

    with trace_context("demo-ticket", session_id=thread_id):
        result = agent.invoke(
            {"ticket_text": scenario["text"], "messages": []},
            config=config,
        )

        # Handle interrupt (clarify scenario)
        state_snapshot = agent.get_state(config)
        if state_snapshot.tasks and any(
            hasattr(t, "interrupts") and t.interrupts
            for t in state_snapshot.tasks
        ):
            clarifying_question = state_snapshot.tasks[0].interrupts[0].value
            print(f"\n  --- Clarification Required ---")
            print(f"  Agent asks : \"{clarifying_question}\"")
            print(f"  User reply : \"{scenario['clarify_reply']}\"")

            result = agent.invoke(
                Command(resume=scenario["clarify_reply"]),
                config=config,
            )

    elapsed = time.perf_counter() - start

    trace_id = handler.trace_id if handler and hasattr(handler, "trace_id") else None

    record_trace_scores(
        handler,
        category=result.get("category"),
        quality_score=result.get("quality_score"),
        escalated=result.get("needs_escalation"),
    )

    _print_result(result, elapsed, trace_id)

    return {
        "label": scenario["label"],
        "category": result.get("category"),
        "priority": result.get("priority"),
        "quality": result.get("quality_score"),
        "escalated": result.get("needs_escalation", False),
        "time": elapsed,
    }


def print_summary(results: list[dict]) -> None:
    """Print a compact summary table of all demo results."""
    print(f"\n\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    print(
        f"  {'#':<4} {'Scenario':<35} {'Cat.':<12} {'Pri.':<9} "
        f"{'Quality':<9} {'Esc.':<6} {'Time':<6}"
    )
    print(f"  {'-' * 4} {'-' * 35} {'-' * 12} {'-' * 9} {'-' * 9} {'-' * 6} {'-' * 6}")
    for i, r in enumerate(results, 1):
        print(
            f"  {i:<4} {r['label']:<35} {r['category'] or '-':<12} "
            f"{r['priority'] or '-':<9} "
            f"{r['quality'] or 0:.2f}     "
            f"{'Yes' if r['escalated'] else 'No':<6} "
            f"{r['time']:.1f}s"
        )
    print(f"\n  Check Langfuse UI at http://localhost:3000 for detailed traces.\n")


def main():
    agent = get_agent()
    results = []

    for i, scenario in enumerate(DEMO_SCENARIOS, 1):
        summary = run_scenario(agent, i, scenario)
        results.append(summary)

    print_summary(results)


if __name__ == "__main__":
    main()
