from collections.abc import Sequence
from enum import StrEnum
from typing import Literal, Protocol

import httpx
from pydantic import BaseModel


class Role(StrEnum):
    system = "system"
    user = "user"
    assistant = "assistant"


class ChatMessage(BaseModel):
    role: Role
    content: str


class CompletedMessage(BaseModel):
    content: str


class CompletedChoice(BaseModel):
    message: CompletedMessage
    finish_reason: Literal["stop"]


class ChatCompletion(BaseModel):
    choices: tuple[CompletedChoice]


class ChatModel(Protocol):
    async def complete(self, messages: Sequence[ChatMessage]) -> str: ...


class VllmChatModel:
    def __init__(self, http: httpx.AsyncClient) -> None:
        self.http = http

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        response = await self.http.post(
            "/chat/completions", json={"messages": [message.model_dump(mode="json") for message in messages]}
        )
        response.raise_for_status()
        (choice,) = ChatCompletion.model_validate_json(response.content).choices
        return choice.message.content
