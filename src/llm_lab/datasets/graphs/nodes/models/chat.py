from collections.abc import Sequence
from typing import Annotated, Literal, Protocol, assert_never

from openai import AsyncOpenAI
from openai.types import CompletionUsage, Model
from openai.types.chat import (
    ChatCompletionFunctionToolParam,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallUnion,
)
from openai.types.chat.chat_completion import Choice
from pydantic import BaseModel, Field, JsonValue, TypeAdapter


class FunctionTool(BaseModel):
    name: str
    description: str
    parameters: dict[str, JsonValue]


class FunctionCall(BaseModel):
    name: str
    arguments: dict[str, JsonValue]


class ToolCall(BaseModel):
    id: str
    function: FunctionCall


class SystemMessage(BaseModel):
    role: Literal["system"] = "system"
    content: str


class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] = []


class ToolMessage(BaseModel):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: str


type ChatMessage = Annotated[SystemMessage | UserMessage | AssistantMessage | ToolMessage, Field(discriminator="role")]


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int


class Completion(BaseModel):
    message: AssistantMessage
    usage: Usage


class ChatModel(Protocol):
    async def complete(self, messages: Sequence[ChatMessage], tools: Sequence[FunctionTool] = ()) -> Completion: ...


def to_openai_tool(tool: FunctionTool) -> ChatCompletionFunctionToolParam:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": dict[str, object](tool.parameters),
        },
    }


def to_openai_tool_call(call: ToolCall) -> ChatCompletionMessageFunctionToolCallParam:
    return {
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.function.name,
            "arguments": TypeAdapter(dict[str, JsonValue]).dump_json(call.function.arguments).decode(),
        },
    }


def to_openai_message(message: ChatMessage) -> ChatCompletionMessageParam:
    match message:
        case SystemMessage(content=content):
            return {"role": "system", "content": content}
        case UserMessage(content=content):
            return {"role": "user", "content": content}
        case AssistantMessage(content=content, tool_calls=[]):
            return {"role": "assistant", "content": content}
        case AssistantMessage(content=content, tool_calls=calls):
            return {
                "role": "assistant",
                "content": content,
                "tool_calls": [to_openai_tool_call(call) for call in calls],
            }
        case ToolMessage(tool_call_id=tool_call_id, content=content):
            return {"role": "tool", "tool_call_id": tool_call_id, "content": content}
        case _:
            assert_never(message)


def from_openai_tool_call(call: ChatCompletionMessageToolCallUnion) -> ToolCall:
    match call:
        case ChatCompletionMessageFunctionToolCall(id=call_id, function=function):
            arguments = TypeAdapter(dict[str, JsonValue]).validate_json(function.arguments)
            return ToolCall(id=call_id, function=FunctionCall(name=function.name, arguments=arguments))
        case _:
            raise ValueError(f"expected a function tool call, got {call!r}")


def from_openai_choices(choices: list[Choice]) -> AssistantMessage:
    match choices:
        case [
            Choice(finish_reason="stop", message=ChatCompletionMessage(content=str() as content, tool_calls=None | []))
        ]:
            return AssistantMessage(content=content)
        case [
            Choice(
                finish_reason="tool_calls", message=ChatCompletionMessage(content=content, tool_calls=[_, *_] as calls)
            )
        ]:
            return AssistantMessage(content=content, tool_calls=[from_openai_tool_call(call) for call in calls])
        case _:
            raise ValueError(f"expected one reply that stopped by itself or called tools, got {choices!r}")


def from_openai_usage(usage: CompletionUsage | None) -> Usage:
    match usage:
        case CompletionUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens):
            return Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        case None:
            raise ValueError("expected the server to report the token usage of the completion")
        case _:
            assert_never(usage)


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

    async def complete(self, messages: Sequence[ChatMessage], tools: Sequence[FunctionTool] = ()) -> Completion:
        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=[to_openai_message(message) for message in messages],
            tools=[to_openai_tool(tool) for tool in tools],
        )
        return Completion(message=from_openai_choices(completion.choices), usage=from_openai_usage(completion.usage))
