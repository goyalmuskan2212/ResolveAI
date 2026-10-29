"""Smoke test — verify the full stack end-to-end via HTTP API.

Checks:
  1. API health endpoint
  2. Question flow   (classify → rag → generate → quality → send_reply)
  3. Trace endpoint   (GET /ticket/{trace_id}/trace)
  4. Complaint flow   (classify → empathy → rag → generate → quality → send_reply)
  5. Unclear flow     (classify → clarify → resume → ... → send_reply)
  6. Metrics endpoint

Prerequisites:
    - Infrastructure running (Qdrant, Langfuse — local or cloud)
    - Knowledge base indexed
    - API server running (uvicorn app.main:app)

Usage:
    python scripts/smoke_test.py [--base-url http://localhost:8000]
"""

import argparse
import sys
import time

import httpx

DEFAULT_BASE_URL = "http://localhost:8000"
TIMEOUT = 60.0  # seconds per request (LLM calls can be slow)

# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

SCENARIOS = {
    "question": {
        "text": "How do I return a product I bought 3 days ago?",
        "checks": {
            "category": "question",
            "escalated": False,
            "needs_clarification": False,
        },
    },
    "complaint": {
        "text": (
            "My order hasn't arrived in 5 days and nobody is answering! "
            "This is completely unacceptable!"
        ),
        "checks": {
            "category": "complaint",
            "needs_clarification": False,
        },
    },
    "unclear": {
        "text": "Help me pls",
        "checks": {
            "needs_clarification": True,
        },
        "clarify_reply": "I can't process a return on the website, the button doesn't work",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SmokeTestResult:
    def __init__(self):
        self.passed: list[str] = []
        self.failed: list[tuple[str, str]] = []

    def ok(self, name: str) -> None:
        self.passed.append(name)
        print(f"  PASS  {name}")

    def fail(self, name: str, reason: str) -> None:
        self.failed.append((name, reason))
        print(f"  FAIL  {name} — {reason}")

    @property
    def total(self) -> int:
        return len(self.passed) + len(self.failed)

    @property
    def success(self) -> bool:
        return len(self.failed) == 0


def check_response_fields(data: dict, checks: dict) -> list[str]:
    """Return a list of mismatch descriptions (empty = all good)."""
    errors = []
    for field, expected in checks.items():
        actual = data.get(field)
        if actual != expected:
            errors.append(f"{field}: expected {expected!r}, got {actual!r}")
    return errors


# ---------------------------------------------------------------------------
# Individual tests
# ---------------------------------------------------------------------------


def test_health(client: httpx.Client, results: SmokeTestResult) -> bool:
    """Check GET /health."""
    try:
        r = client.get("/health", timeout=10)
        data = r.json()
        if r.status_code == 200 and data.get("status") == "ok" and data.get("graph_ready"):
            results.ok("Health check")
            return True
        results.fail("Health check", f"status={r.status_code}, body={data}")
    except httpx.ConnectError:
        results.fail("Health check", "Cannot connect to API — is the server running?")
    except Exception as e:
        results.fail("Health check", str(e))
    return False


def test_ticket(
    client: httpx.Client,
    results: SmokeTestResult,
    name: str,
    scenario: dict,
) -> dict | None:
    """Submit a ticket and validate the response structure."""
    try:
        r = client.post("/ticket", json={"text": scenario["text"]}, timeout=TIMEOUT)
    except Exception as e:
        results.fail(name, f"Request failed: {e}")
        return None

    if r.status_code != 200:
        results.fail(name, f"HTTP {r.status_code}: {r.text[:200]}")
        return None

    data = r.json()

    # Validate required fields exist
    required = ["ticket_id", "response", "category", "priority", "quality_score", "escalated"]
    missing = [f for f in required if f not in data]
    if missing:
        results.fail(name, f"Missing fields: {missing}")
        return data

    # Validate expected values
    errors = check_response_fields(data, scenario["checks"])
    if errors:
        results.fail(name, "; ".join(errors))
        return data

    # For non-clarification flows, check that we got a real response
    if not data.get("needs_clarification"):
        if not data.get("response") or len(data["response"]) < 10:
            results.fail(name, f"Response too short: {data.get('response')!r}")
            return data
        if data.get("quality_score", 0) <= 0:
            results.fail(name, f"Quality score is {data.get('quality_score')}")
            return data

    results.ok(name)
    return data


def test_clarify_resume(
    client: httpx.Client,
    results: SmokeTestResult,
    ticket_id: str,
    reply: str,
) -> None:
    """Resume an interrupted clarify flow and validate the final response."""
    name = "Unclear -> clarify resume"
    try:
        r = client.post(
            f"/ticket/{ticket_id}/clarify",
            json={"reply": reply},
            timeout=TIMEOUT,
        )
    except Exception as e:
        results.fail(name, f"Request failed: {e}")
        return

    if r.status_code != 200:
        results.fail(name, f"HTTP {r.status_code}: {r.text[:200]}")
        return

    data = r.json()

    if not data.get("response") or len(data["response"]) < 10:
        results.fail(name, f"Response after clarify too short: {data.get('response')!r}")
        return

    if data.get("needs_clarification"):
        results.fail(name, "Still needs clarification after resume")
        return

    results.ok(name)


def test_trace(
    client: httpx.Client,
    results: SmokeTestResult,
    trace_id: str | None,
) -> None:
    """Check GET /ticket/{trace_id}/trace returns a valid trace structure."""
    name = "Trace endpoint"

    if not trace_id:
        results.ok(f"{name} (SKIP — no trace_id, Langfuse may not be configured)")
        return

    try:
        r = client.get(f"/ticket/{trace_id}/trace", timeout=30)
    except Exception as e:
        results.fail(name, f"Request failed: {e}")
        return

    if r.status_code == 404:
        results.ok(f"{name} (SKIP — trace not yet ingested, async delay)")
        return

    if r.status_code != 200:
        results.fail(name, f"HTTP {r.status_code}: {r.text[:200]}")
        return

    data = r.json()
    required = ["trace_id", "name", "observations"]
    missing = [f for f in required if f not in data]
    if missing:
        results.fail(name, f"Missing fields: {missing}")
        return

    if not isinstance(data["observations"], list):
        results.fail(name, f"observations is not a list: {type(data['observations'])}")
        return

    results.ok(name)


def test_metrics(client: httpx.Client, results: SmokeTestResult) -> None:
    """Check GET /metrics returns valid structure."""
    name = "Metrics endpoint"
    try:
        r = client.get("/metrics", timeout=30)
    except Exception as e:
        results.fail(name, f"Request failed: {e}")
        return

    if r.status_code != 200:
        results.fail(name, f"HTTP {r.status_code}: {r.text[:200]}")
        return

    data = r.json()
    required = ["total_traces", "avg_latency_ms", "avg_cost"]
    missing = [f for f in required if f not in data]
    if missing:
        results.fail(name, f"Missing fields: {missing}")
        return

    results.ok(name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_smoke_test(base_url: str) -> bool:
    results = SmokeTestResult()

    print(f"\n{'=' * 60}")
    print(f"  SMOKE TEST — {base_url}")
    print(f"{'=' * 60}\n")

    client = httpx.Client(base_url=base_url)
    start = time.perf_counter()

    # 1. Health
    if not test_health(client, results):
        print(f"\n  Aborting — API is not reachable.\n")
        return False

    # 2. Question flow
    question_data = test_ticket(client, results, "Question flow", SCENARIOS["question"])

    # 3. Trace endpoint (using trace_id from question flow)
    test_trace(client, results, question_data.get("trace_id") if question_data else None)

    # 4. Complaint flow
    test_ticket(client, results, "Complaint flow", SCENARIOS["complaint"])

    # 5. Unclear flow (ticket + clarify resume)
    unclear_data = test_ticket(client, results, "Unclear flow", SCENARIOS["unclear"])
    if unclear_data and unclear_data.get("needs_clarification"):
        test_clarify_resume(
            client,
            results,
            ticket_id=unclear_data["ticket_id"],
            reply=SCENARIOS["unclear"]["clarify_reply"],
        )

    # 6. Metrics
    test_metrics(client, results)

    elapsed = time.perf_counter() - start

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {len(results.passed)}/{results.total} passed  ({elapsed:.1f}s)")
    print(f"{'=' * 60}")

    if results.failed:
        print(f"\n  Failures:")
        for name, reason in results.failed:
            print(f"    - {name}: {reason}")
        print()

    if results.success:
        print(f"\n  All checks passed. The stack is working end-to-end.\n")
    else:
        print(f"\n  Some checks failed. See details above.\n")

    return results.success


def main():
    parser = argparse.ArgumentParser(description="Smoke test for AI Support Agent")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"API base URL (default: {DEFAULT_BASE_URL})",
    )
    args = parser.parse_args()

    success = run_smoke_test(args.base_url)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
