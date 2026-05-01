class LLMClient:
    async def call(self, model: str, messages: list[dict], **kwargs: object) -> dict:
        # Section 5 will implement Together API call + retry/backoff.
        _ = (model, messages, kwargs)
        raise NotImplementedError("LLM client is not wired yet.")
