from __future__ import annotations

from dataclasses import dataclass

import httpx
import pytest

from core.llm_client import LLMClient, LLMClientError


@dataclass
class FakeSettings:
    llm_backend: str = "together"
    together_api_key: str = "test-key"
    together_base_url: str = "https://api.together.xyz"
    ollama_base_url: str = "http://localhost:11434"


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


@pytest.mark.asyncio
async def test_llm_client_ollama_mode_normalizes_response() -> None:
    """Ollama /v1/chat/completions already returns OpenAI-compatible JSON."""

    def handler(request: httpx.Request) -> httpx.Response:
        _ = request
        return httpx.Response(
            200,
            json={
                "id": "ollama-abc",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "llama3.2:1b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hello from ollama"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                },
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        settings = FakeSettings(llm_backend="ollama")
        client = LLMClient(settings=settings, http_client=http_client)  # type: ignore[arg-type]
        result = await client.call(
            model="llama3.2:1b",
            messages=[{"role": "user", "content": "hi"}],
        )

    assert result["choices"][0]["message"]["content"] == "hello from ollama"
    assert result["usage"]["prompt_tokens"] == 10
    assert result["usage"]["completion_tokens"] == 4


def test_llm_client_mock_backend() -> None:
    """Mock backend returns a deterministic response without any HTTP calls."""
    import asyncio

    settings = FakeSettings(llm_backend="mock")
    client = LLMClient(settings=settings)  # type: ignore[arg-type]
    result = asyncio.get_event_loop().run_until_complete(
        client.call(model="mock-model", messages=[{"role": "user", "content": "ping"}])
    )
    assert "Mock response" in result["choices"][0]["message"]["content"]
    assert result["model"] == "mock-model"
