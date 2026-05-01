from fastapi import APIRouter
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
    model: str
    choices: list[dict]


router = APIRouter(tags=["completions"])


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    # Stub for Section 1. The full pipeline is wired in later sections.
    return ChatCompletionResponse(
        id="cmpl-bootstrap",
        model=request.model or "router-placeholder",
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Smart Router bootstrap is ready. Core routing is coming in next sections.",
                },
                "finish_reason": "stop",
            }
        ],
    )
