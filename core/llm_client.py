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
        timeout: float = 30.0,
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
        if backend == "gemini":
            return await self._call_gemini(model=model, messages=messages, **kwargs)
        return await self._call_together(model=model, messages=messages, **kwargs)

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

    async def _call_gemini(self, model: str, messages: list[dict], **kwargs: object) -> dict[str, Any]:
        if not self.settings.gemini_api_key.strip():
            raise LLMClientError(
                "GEMINI_API_KEY is missing. Set GEMINI_API_KEY in your environment."
            )

        target_model = model or self.settings.gemini_model
        url = (
            f'{self.settings.gemini_base_url.rstrip("/")}/v1beta/models/'
            f"{target_model}:generateContent?key={self.settings.gemini_api_key}"
        )
        headers = {"Content-Type": "application/json"}
        contents = self._to_gemini_contents(messages)
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": float(kwargs.get("temperature", 0.2)),
            },
        }

        response = await self._request_with_retries(url=url, headers=headers, payload=payload)
        data = response.json()
        return self._normalize_gemini_response(data=data, model=target_model)

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

    @staticmethod
    def _to_gemini_contents(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        contents: list[dict[str, Any]] = []
        for item in messages:
            role = str(item.get("role", "user"))
            text = str(item.get("content", ""))
            if not text.strip():
                continue
            if role == "assistant":
                gemini_role = "model"
            elif role in {"system", "developer"}:
                gemini_role = "user"
                text = f"[{role}] {text}"
            else:
                gemini_role = "user"
            contents.append({"role": gemini_role, "parts": [{"text": text}]})

        if not contents:
            contents.append({"role": "user", "parts": [{"text": "Hello"}]})
        return contents

    @staticmethod
    def _normalize_gemini_response(data: dict[str, Any], model: str) -> dict[str, Any]:
        candidates = data.get("candidates", []) or []
        first = candidates[0] if candidates else {}
        content = first.get("content", {}) if isinstance(first, dict) else {}
        parts = content.get("parts", []) if isinstance(content, dict) else []
        text = ""
        if parts and isinstance(parts[0], dict):
            text = str(parts[0].get("text", ""))

        usage = data.get("usageMetadata", {}) if isinstance(data, dict) else {}
        prompt_tokens = int(usage.get("promptTokenCount", 0) or 0)
        completion_tokens = int(usage.get("candidatesTokenCount", 0) or 0)
        total_tokens = int(usage.get("totalTokenCount", prompt_tokens + completion_tokens) or 0)

        return {
            "id": f"gemini-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }

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
