from __future__ import annotations

import random
from dataclasses import dataclass

from core.settings import Settings, get_settings


@dataclass(frozen=True)
class RouteDecision:
    model_id: str
    tier: str


@dataclass(frozen=True)
class TierConfig:
    model_id: str
    threshold: float
    cost_per_1k: float


class ModelRouter:
    def __init__(
        self,
        settings: Settings | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._rng = rng or random.Random()
        self._tiers: dict[str, TierConfig] = {
            "small": TierConfig(
                model_id=self.settings.small_model,
                threshold=0.0,
                cost_per_1k=0.0002,
            ),
            "medium": TierConfig(
                model_id=self.settings.medium_model,
                threshold=self.settings.medium_threshold,
                cost_per_1k=0.0003,
            ),
            "large": TierConfig(
                model_id=self.settings.large_model,
                threshold=self.settings.large_threshold,
                cost_per_1k=0.0009,
            ),
        }

    @staticmethod
    def _clamp_01(value: float) -> float:
        return max(0.0, min(1.0, value))

    def pick_model(self, score: float, budget: str = "balanced") -> RouteDecision:
        if budget not in {"cheap", "balanced", "quality"}:
            budget = "balanced"

        if budget == "cheap":
            tier = "small"
            return RouteDecision(model_id=self._tiers[tier].model_id, tier=tier)
        if budget == "quality":
            tier = "large"
            return RouteDecision(model_id=self._tiers[tier].model_id, tier=tier)

        clamped_score = self._clamp_01(score)
        if clamped_score >= self._tiers["large"].threshold:
            tier = "large"
        elif clamped_score >= self._tiers["medium"].threshold:
            tier = "medium"
        else:
            tier = "small"
        return RouteDecision(model_id=self._tiers[tier].model_id, tier=tier)

    def ab_variant(self, ratio: float | None = None) -> str:
        effective_ratio = self.settings.ab_split_ratio if ratio is None else ratio
        effective_ratio = self._clamp_01(effective_ratio)
        return "variant" if self._rng.random() < effective_ratio else "control"

    def get_cost(self, tier: str, prompt_tokens: int, completion_tokens: int) -> float:
        if tier not in self._tiers:
            raise ValueError(f"Unknown tier: {tier}")

        prompt = max(0, prompt_tokens)
        completion = max(0, completion_tokens)
        total_tokens = prompt + completion
        unit_cost = self._tiers[tier].cost_per_1k
        return round((total_tokens / 1000.0) * unit_cost, 10)
