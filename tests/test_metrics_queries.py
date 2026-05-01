from __future__ import annotations

from pathlib import Path

import pytest

from db.queries import get_recent_requests, get_stats, init_db, insert_request


@pytest.mark.asyncio
async def test_metrics_stats_aggregation(tmp_path: Path) -> None:
    db_path = str(tmp_path / "router.db")
    await init_db(db_path)

    await insert_request(
        db_path,
        ts=2_000_000_000.0,
        model="small-model",
        tier="small",
        complexity_score=0.15,
        latency_ms=120.0,
        prompt_tokens=100,
        completion_tokens=50,
        cost_usd=0.00003,
        ab_variant="control",
    )
    await insert_request(
        db_path,
        ts=2_000_000_001.0,
        model="large-model",
        tier="large",
        complexity_score=0.90,
        latency_ms=900.0,
        prompt_tokens=100,
        completion_tokens=100,
        cost_usd=0.00018,
        ab_variant="variant",
    )

    stats = await get_stats(db_path, hours=10_000_000)
    assert stats["total_requests"] == 2
    assert round(stats["total_cost_usd"], 8) == 0.00021
    assert stats["cost_by_tier"]["small"] == 0.00003
    assert stats["cost_by_tier"]["large"] == 0.00018
    assert stats["p95_latency_by_tier"]["small"] == 120.0
    assert stats["p95_latency_by_tier"]["large"] == 900.0
    assert stats["model_distribution"]["small-model"] == 1
    assert stats["model_distribution"]["large-model"] == 1
    assert stats["ab_comparison"]["control"]["requests"] == 1
    assert stats["ab_comparison"]["variant"]["requests"] == 1
    assert stats["baseline_cost_if_always_70b"] > stats["total_cost_usd"]
    assert stats["savings_usd"] > 0.0


@pytest.mark.asyncio
async def test_recent_requests_returns_empty_when_no_data(tmp_path: Path) -> None:
    db_path = str(tmp_path / "router.db")
    await init_db(db_path)
    rows = await get_recent_requests(db_path, hours=1)
    assert rows == []
