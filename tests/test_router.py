from __future__ import annotations

import random
from dataclasses import dataclass

import pytest

from core.router import ModelRouter


@dataclass
class FakeSettings:
    small_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    medium_model: str = "meta-llama/Llama-2-13b-chat-hf"
    large_model: str = "meta-llama/Llama-2-70b-chat-hf"
    medium_threshold: float = 0.35
    large_threshold: float = 0.70
    ab_split_ratio: float = 0.10


def test_budget_cheap_always_returns_small() -> None:
    router = ModelRouter(settings=FakeSettings(), rng=random.Random(11))
    decision = router.pick_model(score=0.99, budget="cheap")
    assert decision.tier == "small"
    assert decision.model_id == "mistralai/Mistral-7B-Instruct-v0.2"


def test_budget_quality_always_returns_large() -> None:
    router = ModelRouter(settings=FakeSettings(), rng=random.Random(11))
    decision = router.pick_model(score=0.01, budget="quality")
    assert decision.tier == "large"
    assert decision.model_id == "meta-llama/Llama-2-70b-chat-hf"


def test_balanced_threshold_boundaries() -> None:
    router = ModelRouter(settings=FakeSettings(), rng=random.Random(11))
    assert router.pick_model(score=0.34).tier == "small"
    assert router.pick_model(score=0.35).tier == "medium"
    assert router.pick_model(score=0.69).tier == "medium"
    assert router.pick_model(score=0.70).tier == "large"


def test_invalid_budget_defaults_to_balanced() -> None:
    router = ModelRouter(settings=FakeSettings(), rng=random.Random(11))
    decision = router.pick_model(score=0.80, budget="speedrun")
    assert decision.tier == "large"


def test_ab_distribution_is_within_five_percent_of_ratio() -> None:
    ratio = 0.10
    runs = 10000
    router = ModelRouter(settings=FakeSettings(), rng=random.Random(7))
    variants = sum(1 for _ in range(runs) if router.ab_variant(ratio=ratio) == "variant")
    observed = variants / runs
    assert abs(observed - ratio) <= 0.05


def test_get_cost_uses_tier_pricing() -> None:
    router = ModelRouter(settings=FakeSettings(), rng=random.Random(11))
    cost = router.get_cost(tier="medium", prompt_tokens=1000, completion_tokens=500)
    assert cost == 0.00045


def test_get_cost_rejects_unknown_tier() -> None:
    router = ModelRouter(settings=FakeSettings(), rng=random.Random(11))
    with pytest.raises(ValueError):
        router.get_cost(tier="xlarge", prompt_tokens=100, completion_tokens=100)
