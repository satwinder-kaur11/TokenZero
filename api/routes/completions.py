from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str = Field(..., examples=["user"])
    content: str = Field(..., examples=["Explain recursion in simple terms."])


class ChatCompletionRequest(BaseModel):
    messages: list[Message]
    model: str | None = None
    budget_hint: str = "balanced"
    stream: bool = False


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[dict]
    usage: dict[str, Any]


router = APIRouter(tags=["completions"])


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest, raw_request: Request) -> JSONResponse:
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    context_manager = raw_request.app.state.context_manager
    classifier = raw_request.app.state.classifier
    model_router = raw_request.app.state.model_router
    llm_client = raw_request.app.state.llm_client
    metrics = raw_request.app.state.metrics

    for msg in request.messages:
        context_manager.add_message(msg.role, msg.content)

    context = await context_manager.get_context()
    if not context:
        raise HTTPException(status_code=500, detail="failed to prepare context")

    score = classifier.score(context[-1]["content"])
    decision = model_router.pick_model(score=score, budget=request.budget_hint)
    variant = model_router.ab_variant()

    start = time.monotonic()
    llm_response = await llm_client.call(
        model=request.model or decision.model_id,
        messages=context,
        stream=request.stream,
    )
    latency_ms = (time.monotonic() - start) * 1000.0

    choices = llm_response.get("choices", [])
    usage = llm_response.get("usage", {})
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    cost_usd = model_router.get_cost(
        tier=decision.tier,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )

    if choices:
        maybe_assistant = choices[0].get("message", {})
        assistant_content = maybe_assistant.get("content")
        if isinstance(assistant_content, str) and assistant_content.strip():
            context_manager.add_message("assistant", assistant_content)

    await metrics.record(
        ts=time.time(),
        model=request.model or decision.model_id,
        tier=decision.tier,
        complexity_score=score,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
        ab_variant=variant,
    )

    response_payload = ChatCompletionResponse(
        id=str(llm_response.get("id", f"cmpl-{uuid.uuid4().hex[:12]}")),
        object=str(llm_response.get("object", "chat.completion")),
        created=int(llm_response.get("created", int(time.time()))),
        model=str(llm_response.get("model", request.model or decision.model_id)),
        choices=choices,
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": int(usage.get("total_tokens", prompt_tokens + completion_tokens)),
        },
    )

    return JSONResponse(
        content=response_payload.model_dump(),
        headers={
            "x-router-model": request.model or decision.model_id,
            "x-complexity-score": f"{score:.4f}",
            "x-tier": decision.tier,
            "x-ab-variant": variant,
        },
    )
