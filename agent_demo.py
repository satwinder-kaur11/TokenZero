"""
agent_demo.py — Simulates an AI agent calling the TokenZero route_request tool.

Usage:
    python agent_demo.py

Make sure the TokenZero API is running:
    poetry run python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

import httpx

BASE_URL = "http://localhost:8000"

PROMPTS = [
    ("What is 2 + 2?", "cheap"),
    ("Summarize the key differences between REST and GraphQL APIs.", "balanced"),
    (
        "Design a multi-agent orchestration architecture for a distributed "
        "financial fraud detection system with real-time stream processing, "
        "explainability requirements, and sub-100ms latency SLAs.",
        "quality",
    ),
]


def call_route_request(prompt: str, budget_hint: str) -> dict:
    response = httpx.post(
        f"{BASE_URL}/tools/route_request",
        json={"prompt": prompt, "budget_hint": budget_hint},
        timeout=10.0,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    print("=" * 65)
    print("  TokenZero — Agent Demo (MCP-Compatible Tool Call)")
    print("=" * 65)

    # Step 1: Fetch the tool schema (as an agent would on startup)
    schema_resp = httpx.get(f"{BASE_URL}/tools/schema", timeout=5.0)
    schema = schema_resp.json()
    print(f"\n[Agent] Discovered tool: '{schema['name']}'")
    print(f"[Agent] Description: {schema['description']}\n")
    print("-" * 65)

    # Step 2: Call the tool for each prompt
    for prompt, budget in PROMPTS:
        print(f"\n[Agent] Prompt   : {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        print(f"[Agent] Budget   : {budget}")

        result = call_route_request(prompt, budget)

        print(
            f"[Agent] Decision : Agent selected [{result['selected_model']}] "
            f"at [{result['tier']}] tier — "
            f"complexity {result['complexity_score']:.4f} — "
            f"estimated cost ${result['estimated_cost_usd']:.6f}"
        )

    print("\n" + "=" * 65)
    print("  All routing decisions complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()