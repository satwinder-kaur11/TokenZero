from dataclasses import dataclass


@dataclass(frozen=True)
class RouteDecision:
    model_id: str
    tier: str


class ModelRouter:
    def pick_model(self, score: float, budget: str = "balanced") -> RouteDecision:
        # Section 4 will implement threshold + budget-aware routing.
        _ = score
        if budget == "cheap":
            return RouteDecision(model_id="small-placeholder", tier="small")
        if budget == "quality":
            return RouteDecision(model_id="large-placeholder", tier="large")
        return RouteDecision(model_id="medium-placeholder", tier="medium")
