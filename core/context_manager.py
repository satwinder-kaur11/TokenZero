from typing import TypedDict


class ChatMessage(TypedDict):
    role: str
    content: str


class ContextManager:
    def __init__(self) -> None:
        self.history: list[ChatMessage] = []

    def add_message(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

    def sliding_window(self, n: int = 5) -> list[ChatMessage]:
        return self.history[-n:]
