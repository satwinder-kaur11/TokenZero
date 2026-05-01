from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import httpx
from rich.console import Console
from rich.table import Table

COST_PER_1K_BY_TIER = {
    "small": 0.0002,
    "medium": 0.0003,
    "large": 0.0009,
}


@dataclass(frozen=True)
class QueryCase:
    query_type: str
    prompt: str


@dataclass(frozen=True)
class BenchmarkRecord:
    mode: str
    query_type: str
    model_used: str
    tier: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    status: str
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run smart-router benchmark suite.")
    parser.add_argument("--router-url", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument(
        "--together-url", default="https://api.together.xyz/v1/chat/completions"
    )
    parser.add_argument("--together-api-key", default="")
    parser.add_argument("--baseline-model", default="meta-llama/Llama-2-70b-chat-hf")
    parser.add_argument("--simple-count", type=int, default=600)
    parser.add_argument("--medium-count", type=int, default=300)
    parser.add_argument("--complex-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--output-dir", default="benchmarks/results")
    return parser.parse_args()


def _extract_prompt(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        for key in ("prompt", "query", "text"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    raise ValueError(f"Unsupported query item: {item!r}")


def load_query_bank(path: str) -> list[str]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Query file must be a JSON array: {path}")
    prompts = [_extract_prompt(item) for item in raw]
    prompts = [p for p in prompts if p]
    if not prompts:
        raise ValueError(f"Query file has no usable prompts: {path}")
    return prompts


def sample_queries(
    pool: list[str],
    query_type: str,
    count: int,
    rng: random.Random,
) -> list[QueryCase]:
    if count <= 0:
        return []
    return [QueryCase(query_type=query_type, prompt=rng.choice(pool)) for _ in range(count)]


def build_benchmark_plan(
    simple_pool: list[str],
    medium_pool: list[str],
    complex_pool: list[str],
    simple_count: int,
    medium_count: int,
    complex_count: int,
    seed: int,
) -> list[QueryCase]:
    rng = random.Random(seed)
    plan = []
    plan.extend(sample_queries(simple_pool, "simple", simple_count, rng))
    plan.extend(sample_queries(medium_pool, "medium", medium_count, rng))
    plan.extend(sample_queries(complex_pool, "complex", complex_count, rng))
    rng.shuffle(plan)
    return plan


def percentile(values: Iterable[float], p: float) -> float:
    vals = sorted(v for v in values if v >= 0.0)
    if not vals:
        return 0.0
    idx = int(round((p / 100.0) * (len(vals) - 1)))
    return float(vals[idx])


def cost_for_tier(tier: str, prompt_tokens: int, completion_tokens: int) -> float:
    unit = COST_PER_1K_BY_TIER.get(tier, COST_PER_1K_BY_TIER["medium"])
    total_tokens = max(0, prompt_tokens) + max(0, completion_tokens)
    return round((total_tokens / 1000.0) * unit, 10)


def summarize_records(records: list[BenchmarkRecord]) -> dict[str, Any]:
    mode_groups: dict[str, list[BenchmarkRecord]] = {"baseline": [], "router": []}
    for row in records:
        mode_groups.setdefault(row.mode, []).append(row)

    def mode_stats(mode: str) -> dict[str, Any]:
        rows = mode_groups.get(mode, [])
        ok_rows = [r for r in rows if r.status == "ok"]
        latencies = [r.latency_ms for r in ok_rows]
        total_cost = sum(r.cost_usd for r in ok_rows)
        return {
            "requests": len(rows),
            "successful_requests": len(ok_rows),
            "failed_requests": len(rows) - len(ok_rows),
            "total_cost_usd": round(total_cost, 8),
            "avg_latency_ms": round((sum(latencies) / len(latencies)) if latencies else 0.0, 3),
            "p95_latency_ms": round(percentile(latencies, 95.0), 3),
        }

    baseline = mode_stats("baseline")
    router = mode_stats("router")
    savings_usd = baseline["total_cost_usd"] - router["total_cost_usd"]
    savings_pct = (savings_usd / baseline["total_cost_usd"] * 100.0) if baseline["total_cost_usd"] else 0.0

    model_distribution_router: dict[str, int] = {}
    for row in mode_groups.get("router", []):
        if row.status != "ok":
            continue
        model_distribution_router[row.model_used] = model_distribution_router.get(row.model_used, 0) + 1

    return {
        "baseline": baseline,
        "router": router,
        "savings_usd": round(savings_usd, 8),
        "savings_pct": round(savings_pct, 3),
        "model_distribution_router": model_distribution_router,
    }


async def run_baseline_case(
    client: httpx.AsyncClient,
    case: QueryCase,
    together_url: str,
    together_api_key: str,
    baseline_model: str,
) -> BenchmarkRecord:
    payload = {
        "model": baseline_model,
        "messages": [{"role": "user", "content": case.prompt}],
        "temperature": 0.2,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {together_api_key}",
        "Content-Type": "application/json",
    }
    start = time.monotonic()
    try:
        response = await client.post(together_url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage", {}) or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        latency_ms = (time.monotonic() - start) * 1000.0
        return BenchmarkRecord(
            mode="baseline",
            query_type=case.query_type,
            model_used=str(data.get("model", baseline_model)),
            tier="large",
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_for_tier("large", prompt_tokens, completion_tokens),
            status="ok",
        )
    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.monotonic() - start) * 1000.0
        return BenchmarkRecord(
            mode="baseline",
            query_type=case.query_type,
            model_used=baseline_model,
            tier="large",
            latency_ms=latency_ms,
            prompt_tokens=0,
            completion_tokens=0,
            cost_usd=0.0,
            status="error",
            error=str(exc),
        )


async def run_router_case(
    client: httpx.AsyncClient,
    case: QueryCase,
    router_url: str,
) -> BenchmarkRecord:
    payload = {
        "messages": [{"role": "user", "content": case.prompt}],
        "budget_hint": "balanced",
        "stream": False,
    }
    start = time.monotonic()
    try:
        response = await client.post(router_url, json=payload, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage", {}) or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        tier = response.headers.get("x-tier", "medium")
        model_used = response.headers.get("x-router-model") or str(data.get("model", "unknown"))
        latency_ms = (time.monotonic() - start) * 1000.0
        return BenchmarkRecord(
            mode="router",
            query_type=case.query_type,
            model_used=model_used,
            tier=tier,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_for_tier(tier, prompt_tokens, completion_tokens),
            status="ok",
        )
    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.monotonic() - start) * 1000.0
        return BenchmarkRecord(
            mode="router",
            query_type=case.query_type,
            model_used="unknown",
            tier="unknown",
            latency_ms=latency_ms,
            prompt_tokens=0,
            completion_tokens=0,
            cost_usd=0.0,
            status="error",
            error=str(exc),
        )


def print_summary(console: Console, summary: dict[str, Any]) -> None:
    table = Table(title="Smart Router Benchmark Summary")
    table.add_column("Metric")
    table.add_column("Baseline (Always 70B)", justify="right")
    table.add_column("Smart Router", justify="right")

    baseline = summary["baseline"]
    router = summary["router"]
    table.add_row("Successful Requests", str(baseline["successful_requests"]), str(router["successful_requests"]))
    table.add_row("Failed Requests", str(baseline["failed_requests"]), str(router["failed_requests"]))
    table.add_row("Total Cost (USD)", f'{baseline["total_cost_usd"]:.6f}', f'{router["total_cost_usd"]:.6f}')
    table.add_row("Avg Latency (ms)", f'{baseline["avg_latency_ms"]:.2f}', f'{router["avg_latency_ms"]:.2f}')
    table.add_row("P95 Latency (ms)", f'{baseline["p95_latency_ms"]:.2f}', f'{router["p95_latency_ms"]:.2f}')
    table.add_row("Savings (USD)", "-", f'{summary["savings_usd"]:.6f}')
    table.add_row("Savings (%)", "-", f'{summary["savings_pct"]:.2f}%')
    console.print(table)

    model_table = Table(title="Router Model Distribution")
    model_table.add_column("Model")
    model_table.add_column("Requests", justify="right")
    for model, count in sorted(summary["model_distribution_router"].items(), key=lambda x: x[1], reverse=True):
        model_table.add_row(model, str(count))
    if not summary["model_distribution_router"]:
        model_table.add_row("(none)", "0")
    console.print(model_table)


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    simple_pool = load_query_bank("benchmarks/query_sets/simple.json")
    medium_pool = load_query_bank("benchmarks/query_sets/medium.json")
    complex_pool = load_query_bank("benchmarks/query_sets/complex.json")
    plan = build_benchmark_plan(
        simple_pool=simple_pool,
        medium_pool=medium_pool,
        complex_pool=complex_pool,
        simple_count=args.simple_count,
        medium_count=args.medium_count,
        complex_count=args.complex_count,
        seed=args.seed,
    )

    records: list[BenchmarkRecord] = []
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        for case in plan:
            records.append(
                await run_baseline_case(
                    client=client,
                    case=case,
                    together_url=args.together_url,
                    together_api_key=args.together_api_key,
                    baseline_model=args.baseline_model,
                )
            )
        for case in plan:
            records.append(
                await run_router_case(
                    client=client,
                    case=case,
                    router_url=args.router_url,
                )
            )

    summary = summarize_records(records)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"benchmark_{timestamp}.json"

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "router_url": args.router_url,
            "together_url": args.together_url,
            "baseline_model": args.baseline_model,
            "simple_count": args.simple_count,
            "medium_count": args.medium_count,
            "complex_count": args.complex_count,
            "seed": args.seed,
        },
        "summary": summary,
        "records": [asdict(row) for row in records],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["output_path"] = str(output_path)
    return payload


async def main() -> None:
    args = parse_args()
    if not args.together_api_key:
        raise SystemExit("Missing --together-api-key (or provide via env wrapper).")
    result = await run_benchmark(args)
    console = Console()
    print_summary(console, result["summary"])
    console.print(f'[green]Saved benchmark artifact:[/green] {result["output_path"]}')


if __name__ == "__main__":
    asyncio.run(main())
