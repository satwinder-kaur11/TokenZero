from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from core.router import ModelRouter
from core.settings import Settings, get_settings

router = APIRouter(prefix="/tools", tags=["tools"])


# JSON Schema models                                                           #


class RouteRequestInput(BaseModel):
    prompt: str = Field(
        ...,
        description="The user prompt text to route.",
        examples=["Explain quantum entanglement in detail."],
    )
    budget_hint: Literal["cheap", "balanced", "quality"] = Field(
        default="balanced",
        description="Routing budget hint.",
    )


class RouteRequestOutput(BaseModel):
    tool: str = Field(default="route_request")
    selected_model: str
    tier: Literal["small", "medium", "large"]
    complexity_score: float
    estimated_cost_usd: float
    budget_hint: str


ROUTE_REQUEST_SCHEMA = {
    "name": "route_request",
    "description": (
        "Routes a prompt to the optimal LLM tier based on complexity scoring. "
        "Returns the selected model, tier, complexity score, and estimated cost."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "The user prompt text to route."},
            "budget_hint": {
                "type": "string",
                "enum": ["cheap", "balanced", "quality"],
                "default": "balanced",
            },
        },
        "required": ["prompt"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "tool": {"type": "string"},
            "selected_model": {"type": "string"},
            "tier": {"type": "string", "enum": ["small", "medium", "large"]},
            "complexity_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "estimated_cost_usd": {"type": "number"},
            "budget_hint": {"type": "string"},
        },
        "required": ["tool", "selected_model", "tier", "complexity_score", "estimated_cost_usd"],
    },
}


# --------------------------------------------------------------------------- #
# Endpoints                                                                    #
# --------------------------------------------------------------------------- #

@router.get("/schema", summary="Get JSON Schema for the route_request tool")
def get_tool_schema() -> dict:
    return ROUTE_REQUEST_SCHEMA


@router.post(
    "/route_request",
    response_model=RouteRequestOutput,
    summary="Route a prompt to the optimal LLM tier",
)
def route_request(
    body: RouteRequestInput,
    settings: Settings = Depends(get_settings),
) -> RouteRequestOutput:
    try:
        from core.classifier_factory import get_classifier

        classifier = get_classifier()
        model_router = ModelRouter(settings=settings)

        complexity_score = classifier.score(body.prompt)
        decision = model_router.pick_model(
            score=complexity_score,
            budget=body.budget_hint,
        )

        # Estimate cost using average 500 tokens
        estimated_cost = model_router.get_cost(
            tier=decision.tier,
            prompt_tokens=250,
            completion_tokens=250,
        )

        return RouteRequestOutput(
            selected_model=decision.model_id,
            tier=decision.tier,
            complexity_score=round(complexity_score, 4),
            estimated_cost_usd=round(estimated_cost, 6),
            budget_hint=body.budget_hint,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc