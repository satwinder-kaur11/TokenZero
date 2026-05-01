from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from api.routes.completions import router as completions_router
from core.router import RouteDecision


class FakeContextManager:
    def __init__(self) -> None:
        self.history: list[dict[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

    async def get_context(self) -> list[dict[str, str]]:
        return list(self.history)


@dataclass
class FakeClassifier:
    fixed_score: float = 0.62
    last_query: str | None = None

    def score(self, query: str) -> float:
        self.last_query = query
        return self.fixed_score


class FakeModelRouter:
    def __init__(self) -> None:
        self.last_pick: dict[str, Any] | None = None

    def pick_model(self, score: float, budget: str = "balanced") -> RouteDecision:
        self.last_pick = {"score": score, "budget": budget}
        return RouteDecision(model_id="small-model", tier="small")

    def ab_variant(self, ratio: float | None = None) -> str:
        _ = ratio
        return "control"

    def get_cost(self, tier: str, prompt_tokens: int, completion_tokens: int) -> float:
        _ = (tier, prompt_tokens, completion_tokens)
        return 0.0009


class FakeLLMClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def call(self, model: str, messages: list[dict], **kwargs: object) -> dict[str, Any]:
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        return {
            "id": "cmpl-123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello from mock LLM"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 40, "completion_tokens": 12, "total_tokens": 52},
        }


class FakeMetrics:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    async def record(self, **kwargs: Any) -> None:
        self.rows.append(kwargs)


class FakePromMetrics:
    def __init__(self) -> None:
        self.observations: list[dict[str, Any]] = []

    def observe_completion(self, *, tier: str, cost_usd: float, latency_ms: float) -> None:
        self.observations.append(
            {"tier": tier, "cost_usd": cost_usd, "latency_ms": latency_ms}
        )


@pytest.fixture
def test_app() -> FastAPI:
    app = FastAPI()
    app.include_router(completions_router, prefix="/v1")
    app.state.context_manager = FakeContextManager()
    app.state.classifier = FakeClassifier()
    app.state.model_router = FakeModelRouter()
    app.state.llm_client = FakeLLMClient()
    app.state.metrics = FakeMetrics()
    app.state.prom_metrics = FakePromMetrics()
    return app


@pytest.mark.asyncio
async def test_completions_pipeline_success(test_app: FastAPI) -> None:
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "messages": [{"role": "user", "content": "Explain polymorphism quickly"}],
            "budget_hint": "balanced",
        }
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert response.headers["x-router-model"] == "small-model"
    assert response.headers["x-tier"] == "small"
    assert "x-complexity-score" in response.headers
    body = response.json()
    assert body["id"] == "cmpl-123"
    assert body["choices"][0]["message"]["content"] == "Hello from mock LLM"
    assert body["usage"]["total_tokens"] == 52

    assert len(test_app.state.metrics.rows) == 1
    row = test_app.state.metrics.rows[0]
    assert row["tier"] == "small"
    assert row["prompt_tokens"] == 40
    assert row["completion_tokens"] == 12
    assert len(test_app.state.prom_metrics.observations) == 1


@pytest.mark.asyncio
async def test_completions_rejects_empty_messages(test_app: FastAPI) -> None:
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/chat/completions", json={"messages": []})
    assert response.status_code == 400
