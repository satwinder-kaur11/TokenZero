from __future__ import annotations

from dataclasses import dataclass

import httpx

from core.context_manager import ContextManager


@dataclass
class FakeSettings:
    context_window_size: int = 3
    context_max_tokens: int = 40
    summarizer_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    together_api_key: str = "test-key"
    together_base_url: str = "https://api.together.xyz"


class FakeEncoder:
    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "bad status",
                request=httpx.Request("POST", "https://api.together.xyz/v1/chat/completions"),
                response=httpx.Response(self.status_code),
            )

    def json(self) -> dict:
        return self._payload


class FakeAsyncClient:
    def __init__(self, response: FakeResponse | None = None, should_fail: bool = False) -> None:
        self.response = response
        self.should_fail = should_fail
        self.calls: list[dict] = []

    async def post(self, url: str, headers: dict, json: dict) -> FakeResponse:
        self.calls.append({"url": url, "headers": headers, "json": json})
        if self.should_fail:
            raise httpx.ConnectError(
                "network failure", request=httpx.Request("POST", url)
            )
        assert self.response is not None
        return self.response


def test_sliding_window_uses_default_setting() -> None:
    manager = ContextManager(settings=FakeSettings(), token_encoder=FakeEncoder())
    manager.add_message("user", "one")
    manager.add_message("assistant", "two")
    manager.add_message("user", "three")
    manager.add_message("assistant", "four")
    window = manager.sliding_window()
    assert len(window) == 3
    assert window[0]["content"] == "two"
    assert window[-1]["content"] == "four"


def test_count_tokens_uses_encoder() -> None:
    manager = ContextManager(settings=FakeSettings(), token_encoder=FakeEncoder())
    tokens = manager.count_tokens([{"role": "user", "content": "alpha beta gamma"}])
    assert tokens == 7


async def test_summarize_old_calls_together_api() -> None:
    response = FakeResponse(
        payload={"choices": [{"message": {"content": "Summary output."}}]},
    )
    client = FakeAsyncClient(response=response)
    manager = ContextManager(
        settings=FakeSettings(),
        http_client=client,  # type: ignore[arg-type]
        token_encoder=FakeEncoder(),
    )
    summary = await manager.summarize_old([{"role": "user", "content": "A long conversation"}])
    assert summary == "Summary output."
    assert len(client.calls) == 1
    assert client.calls[0]["url"] == "https://api.together.xyz/v1/chat/completions"
    assert client.calls[0]["headers"]["Authorization"] == "Bearer test-key"
    assert client.calls[0]["json"]["model"] == "mistralai/Mistral-7B-Instruct-v0.2"


async def test_summarize_old_fallback_on_http_error() -> None:
    client = FakeAsyncClient(should_fail=True)
    manager = ContextManager(
        settings=FakeSettings(),
        http_client=client,  # type: ignore[arg-type]
        token_encoder=FakeEncoder(),
    )
    summary = await manager.summarize_old(
        [{"role": "user", "content": "alpha beta gamma delta epsilon"}]
    )
    assert summary.endswith("...")
    assert "alpha beta gamma" in summary


async def test_get_context_summarizes_when_budget_exceeded() -> None:
    settings = FakeSettings(context_window_size=3, context_max_tokens=35)
    manager = ContextManager(settings=settings, token_encoder=FakeEncoder())
    manager.history = [
        {"role": "user", "content": "one two three four five six seven eight nine ten"},
        {"role": "assistant", "content": "one two three four five six seven eight nine ten"},
        {"role": "user", "content": "one two three four five six seven eight nine ten"},
        {"role": "assistant", "content": "one two three four five six seven eight nine ten"},
        {"role": "user", "content": "one two three four five six seven eight nine ten"},
    ]

    async def fake_summary(messages):
        _ = messages
        return "Short summary."

    manager.summarize_old = fake_summary  # type: ignore[method-assign]
    context = await manager.get_context()
    assert context[0]["role"] == "system"
    assert context[0]["content"].startswith("Previous context: Short summary.")
    assert manager.count_tokens(context) <= 35


async def test_get_context_trims_even_single_oversized_message() -> None:
    settings = FakeSettings(context_window_size=1, context_max_tokens=10)
    manager = ContextManager(settings=settings, token_encoder=FakeEncoder())
    manager.history = [
        {"role": "user", "content": " ".join(["token"] * 30)},
    ]
    context = await manager.get_context()
    assert len(context) == 1
    assert manager.count_tokens(context) <= 10
