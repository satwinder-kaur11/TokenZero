from __future__ import annotations

import json
from pathlib import Path

from benchmarks.run import BenchmarkRecord, build_benchmark_plan, load_query_bank, summarize_records


def test_load_query_bank_supports_strings_and_objects(tmp_path: Path) -> None:
    path = tmp_path / "queries.json"
    path.write_text(
        json.dumps(["hello", {"prompt": "world"}, {"query": "from-query"}, {"text": "from-text"}]),
        encoding="utf-8",
    )
    prompts = load_query_bank(str(path))
    assert prompts == ["hello", "world", "from-query", "from-text"]


def test_build_benchmark_plan_respects_counts() -> None:
    plan = build_benchmark_plan(
        simple_pool=["s1"],
        medium_pool=["m1"],
        complex_pool=["c1"],
        simple_count=6,
        medium_count=3,
        complex_count=1,
        seed=123,
    )
    assert len(plan) == 10
    assert sum(1 for item in plan if item.query_type == "simple") == 6
    assert sum(1 for item in plan if item.query_type == "medium") == 3
    assert sum(1 for item in plan if item.query_type == "complex") == 1


def test_summarize_records_computes_savings_and_distribution() -> None:
    rows = [
        BenchmarkRecord(
            mode="baseline",
            query_type="simple",
            model_used="70b",
            tier="large",
            latency_ms=900.0,
            prompt_tokens=100,
            completion_tokens=100,
            cost_usd=0.00018,
            status="ok",
        ),
        BenchmarkRecord(
            mode="router",
            query_type="simple",
            model_used="7b",
            tier="small",
            latency_ms=180.0,
            prompt_tokens=100,
            completion_tokens=100,
            cost_usd=0.00004,
            status="ok",
        ),
    ]
    summary = summarize_records(rows)
    assert summary["baseline"]["total_cost_usd"] == 0.00018
    assert summary["router"]["total_cost_usd"] == 0.00004
    assert summary["savings_usd"] == 0.00014
    assert summary["model_distribution_router"]["7b"] == 1
