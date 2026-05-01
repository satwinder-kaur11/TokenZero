from __future__ import annotations

import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import aiosqlite

from db.schema import SCHEMA_SQL

LARGE_TIER_COST_PER_1K = 0.0009


async def init_db(sqlite_path: str) -> None:
    path = Path(sqlite_path)
    if path.parent and str(path.parent) not in {"", "."}:
        path.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(path) as conn:
        await conn.executescript(SCHEMA_SQL)
        await conn.commit()


async def insert_request(
    sqlite_path: str,
    *,
    ts: float,
    model: str,
    tier: str,
    complexity_score: float,
    latency_ms: float,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float,
    ab_variant: str,
) -> None:
    async with aiosqlite.connect(sqlite_path) as conn:
        await conn.execute(
            """
            INSERT INTO requests (
                ts, model, tier, complexity_score, latency_ms,
                prompt_tokens, completion_tokens, cost_usd, ab_variant
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                model,
                tier,
                complexity_score,
                latency_ms,
                prompt_tokens,
                completion_tokens,
                cost_usd,
                ab_variant,
            ),
        )
        await conn.commit()


class MetricsRecorder:
    def __init__(self, sqlite_path: str) -> None:
        self.sqlite_path = sqlite_path

    async def init(self) -> None:
        await init_db(self.sqlite_path)

    async def record(
        self,
        *,
        ts: float,
        model: str,
        tier: str,
        complexity_score: float,
        latency_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        ab_variant: str,
    ) -> None:
        await insert_request(
            self.sqlite_path,
            ts=ts,
            model=model,
            tier=tier,
            complexity_score=complexity_score,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            ab_variant=ab_variant,
        )


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil((p / 100.0) * len(ordered)))
    return float(ordered[rank - 1])


async def get_recent_requests(sqlite_path: str, hours: int = 24) -> list[dict[str, Any]]:
    cutoff = time.time() - (hours * 3600)
    async with aiosqlite.connect(sqlite_path) as conn:
        conn.row_factory = aiosqlite.Row
        cursor = await conn.execute(
            """
            SELECT
                ts, model, tier, complexity_score, latency_ms,
                prompt_tokens, completion_tokens, cost_usd, ab_variant
            FROM requests
            WHERE ts >= ?
            ORDER BY ts ASC
            """,
            (cutoff,),
        )
        rows = await cursor.fetchall()
    return [dict(row) for row in rows]


async def get_stats(sqlite_path: str, hours: int = 24) -> dict[str, Any]:
    rows = await get_recent_requests(sqlite_path=sqlite_path, hours=hours)
    if not rows:
        return {
            "total_requests": 0,
            "total_cost_usd": 0.0,
            "avg_latency_ms": 0.0,
            "cost_by_tier": {},
            "avg_latency_by_tier": {},
            "p95_latency_by_tier": {},
            "baseline_cost_if_always_70b": 0.0,
            "savings_usd": 0.0,
            "savings_pct": 0.0,
            "ab_comparison": {},
            "model_distribution": {},
            "complexity_scores": [],
        }

    total_requests = len(rows)
    total_cost = sum(float(r["cost_usd"] or 0.0) for r in rows)
    all_latencies = [float(r["latency_ms"] or 0.0) for r in rows]
    avg_latency_ms = sum(all_latencies) / total_requests if total_requests else 0.0

    costs_by_tier: dict[str, float] = defaultdict(float)
    latencies_by_tier: dict[str, list[float]] = defaultdict(list)
    model_distribution: dict[str, int] = defaultdict(int)
    complexity_scores: list[float] = []

    ab_cost: dict[str, float] = defaultdict(float)
    ab_latency: dict[str, list[float]] = defaultdict(list)
    ab_completion_tokens: dict[str, list[int]] = defaultdict(list)
    ab_counts: dict[str, int] = defaultdict(int)

    baseline_cost_if_always_70b = 0.0
    for row in rows:
        tier = str(row["tier"] or "unknown")
        model = str(row["model"] or "unknown")
        variant = str(row["ab_variant"] or "unknown")

        row_cost = float(row["cost_usd"] or 0.0)
        row_latency = float(row["latency_ms"] or 0.0)
        prompt_tokens = int(row["prompt_tokens"] or 0)
        completion_tokens = int(row["completion_tokens"] or 0)
        complexity = float(row["complexity_score"] or 0.0)

        costs_by_tier[tier] += row_cost
        latencies_by_tier[tier].append(row_latency)
        model_distribution[model] += 1
        complexity_scores.append(complexity)

        ab_cost[variant] += row_cost
        ab_latency[variant].append(row_latency)
        ab_completion_tokens[variant].append(completion_tokens)
        ab_counts[variant] += 1

        total_tokens = max(0, prompt_tokens) + max(0, completion_tokens)
        baseline_cost_if_always_70b += (total_tokens / 1000.0) * LARGE_TIER_COST_PER_1K

    avg_latency_by_tier = {
        tier: (sum(vals) / len(vals) if vals else 0.0) for tier, vals in latencies_by_tier.items()
    }
    p95_latency_by_tier = {tier: _percentile(vals, 95.0) for tier, vals in latencies_by_tier.items()}

    savings_usd = baseline_cost_if_always_70b - total_cost
    savings_pct = (savings_usd / baseline_cost_if_always_70b * 100.0) if baseline_cost_if_always_70b else 0.0

    ab_comparison: dict[str, dict[str, float | int]] = {}
    for variant, count in ab_counts.items():
        avg_variant_latency = sum(ab_latency[variant]) / count if count else 0.0
        avg_variant_cost = ab_cost[variant] / count if count else 0.0
        avg_completion_tokens = (
            sum(ab_completion_tokens[variant]) / count if count else 0.0
        )
        ab_comparison[variant] = {
            "requests": count,
            "avg_latency_ms": avg_variant_latency,
            "avg_cost_usd": avg_variant_cost,
            "avg_completion_tokens": avg_completion_tokens,
        }

    return {
        "total_requests": total_requests,
        "total_cost_usd": total_cost,
        "avg_latency_ms": avg_latency_ms,
        "cost_by_tier": dict(costs_by_tier),
        "avg_latency_by_tier": avg_latency_by_tier,
        "p95_latency_by_tier": p95_latency_by_tier,
        "baseline_cost_if_always_70b": baseline_cost_if_always_70b,
        "savings_usd": savings_usd,
        "savings_pct": savings_pct,
        "ab_comparison": ab_comparison,
        "model_distribution": dict(model_distribution),
        "complexity_scores": complexity_scores,
    }
