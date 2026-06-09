from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable

import httpx

from core.settings import Settings, get_settings


class LLMClientError(RuntimeError):
    """Raised when the upstream LLM call fails after retries."""


class LLMClient:
    def __init__(
        self,
        settings: Settings | None = None,
        http_client: httpx.AsyncClient | None = None,
        max_retries: int = 3,
        timeout: float = 120.0,
        backoff_base_seconds: float = 0.5,
        sleep_func: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.settings = settings or get_settings()
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self._sleep = sleep_func
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(timeout=timeout)

    async def call(self, model: str, messages: list[dict], **kwargs: object) -> dict[str, Any]:
        backend = self.settings.llm_backend
        if backend == "mock":
            return self._call_mock(model=model, messages=messages)
        if backend == "ollama":
            return await self._call_ollama(model=model, messages=messages, **kwargs)
        # Default: together
        return await self._call_together(model=model, messages=messages, **kwargs)

    # ------------------------------------------------------------------
    # Ollama  (local, OpenAI-compatible /v1/chat/completions endpoint)
    # ------------------------------------------------------------------
    async def _call_ollama(self, model: str, messages: list[dict], **kwargs: object) -> dict[str, Any]:
        base = self.settings.ollama_base_url.rstrip("/")
        url = f"{base}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        # Forward safe optional kwargs (temperature, max_tokens)
        for key in ("temperature", "max_tokens"):
            if key in kwargs:
                payload[key] = kwargs[key]

        response = await self._request_with_retries(url=url, headers=headers, payload=payload)
        data = response.json()
        return self._normalize_ollama_response(data=data, model=model)

    @staticmethod
    def _normalize_ollama_response(data: dict[str, Any], model: str) -> dict[str, Any]:
        """Ensure the response always has the standard OpenAI-like shape."""
        # Ollama's /v1 endpoint already returns OpenAI-compatible JSON,
        # but we guard against unexpected shapes just in case.
        choices = data.get("choices", [])
        usage = data.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        total_tokens = int(
            usage.get("total_tokens", prompt_tokens + completion_tokens) or 0
        )
        return {
            "id": data.get("id", f"ollama-{int(time.time() * 1000)}"),
            "object": data.get("object", "chat.completion"),
            "created": data.get("created", int(time.time())),
            "model": data.get("model", model),
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }

    # ------------------------------------------------------------------
    # Together AI  (cloud fallback)
    # ------------------------------------------------------------------
    async def _call_together(self, model: str, messages: list[dict], **kwargs: object) -> dict[str, Any]:
        url = f'{self.settings.together_base_url.rstrip("/")}/v1/chat/completions'
        headers = {
            "Authorization": f"Bearer {self.settings.together_api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {"model": model, "messages": messages}
        payload.update(kwargs)

        response = await self._request_with_retries(url=url, headers=headers, payload=payload)
        return response.json()

    # ------------------------------------------------------------------
    # HTTP retry logic
    # ------------------------------------------------------------------
    async def _request_with_retries(
        self,
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> httpx.Response:
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.post(url, headers=headers, json=payload)
                if response.status_code < 400:
                    return response

                retryable = response.status_code == 429 or response.status_code >= 500
                if retryable and attempt < self.max_retries:
                    await self._sleep(self.backoff_base_seconds * (2**attempt))
                    continue
                raise LLMClientError(
                    f"LLM provider request failed with status {response.status_code}: {response.text}"
                )
            except httpx.RequestError as exc:
                if attempt < self.max_retries:
                    await self._sleep(self.backoff_base_seconds * (2**attempt))
                    continue
                raise LLMClientError(f"Network error while calling LLM provider: {exc}") from exc

        raise LLMClientError("LLM provider request failed after all retry attempts.")

    # ------------------------------------------------------------------
    # Mock backend  (for unit tests)
    # ------------------------------------------------------------------
    @staticmethod
    def _call_mock(model: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
        latest_user = ""
        for item in reversed(messages):
            if str(item.get("role", "")) == "user":
                latest_user = str(item.get("content", ""))
                break

        reply = f"Mock response: {latest_user[:200]}"
        prompt_tokens = max(1, len(" ".join(str(m.get("content", "")) for m in messages).split()))
        completion_tokens = max(1, len(reply.split()))
        return {
            "id": f"mock-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model or "mock-llm",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()