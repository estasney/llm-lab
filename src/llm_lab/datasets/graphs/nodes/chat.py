from collections.abc import AsyncIterator, Sequence
from enum import StrEnum
from typing import Annotated, Literal, Protocol

import httpx
from pregel_redux import Depends, Scope
from pydantic import BaseModel

from llm_lab.settings import Settings, get_settings


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


async def open_chat_model(
    settings: Annotated[Settings, Depends(get_settings, scope=Scope.run)],
) -> AsyncIterator[ChatModel]:
    async with httpx.AsyncClient(
        base_url=f"{settings.vllm_base_url}/v1", timeout=settings.vllm_request_timeout_seconds
    ) as http:
        yield VllmChatModel(http)


type ChatModelDep = Annotated[ChatModel, Depends(open_chat_model, scope=Scope.run)]
