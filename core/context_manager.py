from __future__ import annotations

from typing import Any, Protocol, TypedDict

import httpx

from core.settings import Settings, get_settings


class TokenEncoder(Protocol):
    def encode(self, text: str) -> list[int]:
        ...

class ChatMessage(TypedDict):
    role: str
    content: str


class ContextManager:
    def __init__(
        self,
        settings: Settings | None = None,
        http_client: httpx.AsyncClient | None = None,
        token_encoder: TokenEncoder | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.history: list[ChatMessage] = []
        self._http_client = http_client
        self._token_encoder = token_encoder or self._load_encoder()

    def add_message(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

    def count_tokens(self, messages: list[ChatMessage]) -> int:
        if not messages:
            return 0

        if self._token_encoder is None:
            # Conservative fallback when tiktoken is unavailable.
            return sum(len(m["content"].split()) + 3 for m in messages)

        total = 0
        for message in messages:
            payload = f'{message["role"]}: {message["content"]}'
            total += len(self._token_encoder.encode(payload))
            total += 3
        return total

    def sliding_window(self, n: int | None = None) -> list[ChatMessage]:
        if n is None:
            n = self.settings.context_window_size
        if n <= 0:
            return []
        return self.history[-n:]

    async def summarize_old(self, messages: list[ChatMessage]) -> str:
        if not messages:
            return ""

        conversation = "\n".join(f'{m["role"]}: {m["content"]}' for m in messages)
        prompt = (
            "Summarize this conversation in 3 sentences, keeping all facts and decisions.\n"
            f"Conversation:\n{conversation}\n"
            "Return the summary string only."
        )

        payload = {
            "model": self.settings.summarizer_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 220,
        }
        headers = {
            "Authorization": f"Bearer {self.settings.together_api_key}",
            "Content-Type": "application/json",
        }
        url = f'{self.settings.together_base_url.rstrip("/")}/v1/chat/completions'

        try:
            if self._http_client is not None:
                response = await self._http_client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data: dict[str, Any] = response.json()
            else:
                async with httpx.AsyncClient(timeout=20.0) as client:
                    response = await client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()

            summary = str(data["choices"][0]["message"]["content"]).strip()
            if summary:
                return summary
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
            pass

        flattened = " ".join(m["content"].strip() for m in messages if m["content"].strip())
        if not flattened:
            return ""
        fallback_summary = flattened[:500].strip()
        return f"{fallback_summary}..."

    async def get_context(self) -> list[ChatMessage]:
        if not self.history:
            return []

        window = self.sliding_window()
        context: list[ChatMessage] = list(window)

        has_older_messages = len(self.history) > len(window)
        if has_older_messages and self.count_tokens(context) > self.settings.context_max_tokens:
            older_messages = self.history[: -len(window)]
            summary = await self.summarize_old(older_messages)
            if summary:
                context = [{"role": "system", "content": f"Previous context: {summary}"}] + context

        return self._trim_to_max_tokens(context)

    def _trim_to_max_tokens(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        trimmed = list(messages)
        while len(trimmed) > 1 and self.count_tokens(trimmed) > self.settings.context_max_tokens:
            if trimmed and trimmed[0]["role"] == "system":
                trimmed.pop(1)
            else:
                trimmed.pop(0)

        if self.count_tokens(trimmed) <= self.settings.context_max_tokens:
            return trimmed

        # Extreme fallback: guarantee bounded context even if one message is oversized.
        last = trimmed[-1]
        content_tokens = last["content"].split()
        if not content_tokens:
            return [{"role": last["role"], "content": ""}]
        truncated = " ".join(content_tokens[-max(1, self.settings.context_max_tokens // 4) :])
        return [{"role": last["role"], "content": truncated}]

    @staticmethod
    def _load_encoder() -> TokenEncoder | None:
        try:
            import tiktoken
        except ImportError:
            return None
        return tiktoken.get_encoding("cl100k_base")
