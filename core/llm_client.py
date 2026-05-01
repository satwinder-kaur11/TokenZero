from __future__ import annotations

import asyncio
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
        url = f'{self.settings.together_base_url.rstrip("/")}/v1/chat/completions'
        headers = {
            "Authorization": f"Bearer {self.settings.together_api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {"model": model, "messages": messages}
        payload.update(kwargs)

        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.post(url, headers=headers, json=payload)
                if response.status_code < 400:
                    return response.json()

                retryable = response.status_code == 429 or response.status_code >= 500
                if retryable and attempt < self.max_retries:
                    await self._sleep(self.backoff_base_seconds * (2**attempt))
                    continue
                raise LLMClientError(
                    f"Together API request failed with status {response.status_code}: {response.text}"
                )
            except httpx.RequestError as exc:
                if attempt < self.max_retries:
                    await self._sleep(self.backoff_base_seconds * (2**attempt))
                    continue
                raise LLMClientError(f"Network error while calling Together API: {exc}") from exc

        raise LLMClientError("Together API request failed after all retry attempts.")

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
