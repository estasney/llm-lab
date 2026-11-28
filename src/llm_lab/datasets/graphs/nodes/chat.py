from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol, assert_never

from openai import AsyncOpenAI
from openai.types import Model
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageParam
from openai.types.chat.chat_completion import Choice
from pydantic import BaseModel


class Role(StrEnum):
    system = "system"
    user = "user"
    assistant = "assistant"


class ChatMessage(BaseModel):
    role: Role
    content: str


class ChatModel(Protocol):
    async def complete(self, messages: Sequence[ChatMessage]) -> str: ...


def to_openai_message(message: ChatMessage) -> ChatCompletionMessageParam:
    match message.role:
        case Role.system:
            return {"role": "system", "content": message.content}
        case Role.user:
            return {"role": "user", "content": message.content}
        case Role.assistant:
            return {"role": "assistant", "content": message.content}
        case _:
            assert_never(message.role)


async def find_served_model(client: AsyncOpenAI) -> str:
    page = await client.models.list()
    match page.data:
        case [Model(id=model)]:
            return model
        case models:
            raise ValueError(f"the server serves {len(models)} models, and a chat model needs exactly one")


class OpenAIChatModel:
    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self.client = client
        self.model = model

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        completion = await self.client.chat.completions.create(
            model=self.model, messages=[to_openai_message(message) for message in messages]
        )
        match completion.choices:
            case [Choice(finish_reason="stop", message=ChatCompletionMessage(content=str() as content))]:
                return content
            case choices:
                raise ValueError(f"expected one reply that stopped by itself, got {choices!r}")
