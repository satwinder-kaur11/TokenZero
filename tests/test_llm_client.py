from __future__ import annotations

from dataclasses import dataclass

import httpx
import pytest

from core.llm_client import LLMClient, LLMClientError


@dataclass
class FakeSettings:
    together_api_key: str = "test-key"
    together_base_url: str = "https://api.together.xyz"


@pytest.mark.asyncio
async def test_llm_client_retries_then_succeeds() -> None:
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] < 3:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(
            200,
            json={
                "id": "cmpl-ok",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = LLMClient(
            settings=FakeSettings(),  # type: ignore[arg-type]
            http_client=http_client,
            max_retries=3,
            backoff_base_seconds=0.01,
            sleep_func=fake_sleep,
        )
        result = await client.call(model="mistral", messages=[{"role": "user", "content": "hi"}])

    assert result["id"] == "cmpl-ok"
    assert attempts["count"] == 3
    assert sleeps == [0.01, 0.02]


@pytest.mark.asyncio
async def test_llm_client_raises_on_non_retryable_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        _ = request
        return httpx.Response(400, json={"error": "bad request"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = LLMClient(settings=FakeSettings(), http_client=http_client)  # type: ignore[arg-type]
        with pytest.raises(LLMClientError):
            await client.call(model="mistral", messages=[{"role": "user", "content": "hi"}])
