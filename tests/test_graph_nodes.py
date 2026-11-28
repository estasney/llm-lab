import asyncio
import json
from collections.abc import Sequence

import httpx
import pytest
from pregel_redux import Endpoint, StateGraph
from pydantic import BaseModel, ValidationError

from llm_lab.datasets.graphs.nodes.chat import ChatMessage, Role, VllmChatModel
from llm_lab.datasets.graphs.nodes.deps import open_chat_model
from llm_lab.datasets.graphs.nodes.user_prompt import (
    PromptKind,
    UserPromptInput,
    UserPromptOutput,
    describe_prompt_kind,
    write_user_prompt,
)


class RecordingChatModel:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.requests: list[Sequence[ChatMessage]] = []

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        self.requests.append(messages)
        return self.reply


class UserPromptState(BaseModel):
    source_text: str
    prompt_kind: PromptKind
    user_prompt: str


def run_user_prompt_graph(chat: RecordingChatModel, source_text: str, kind: PromptKind) -> UserPromptOutput:
    builder = StateGraph(UserPromptState, input_schema=UserPromptInput, output_schema=UserPromptOutput)
    builder.add_node("write_user_prompt", write_user_prompt, input_schema=UserPromptInput)
    builder.add_edge(Endpoint.start, "write_user_prompt")
    graph = builder.compile({open_chat_model: lambda: chat})
    return asyncio.run(graph.ainvoke(UserPromptInput(source_text=source_text, prompt_kind=kind)))


@pytest.mark.parametrize("kind", list(PromptKind))
def test_node_writes_the_reply_of_the_model_as_the_user_prompt(kind: PromptKind) -> None:
    chat = RecordingChatModel("What year did the bridge open?")

    actual = run_user_prompt_graph(chat, "The bridge opened in 1937.", kind)

    assert actual == UserPromptOutput(user_prompt="What year did the bridge open?")


@pytest.mark.parametrize("kind", list(PromptKind))
def test_node_gives_the_model_the_source_text_and_the_prompt_kind(kind: PromptKind) -> None:
    chat = RecordingChatModel("Summarize this.")

    run_user_prompt_graph(chat, "The bridge opened in 1937.", kind)

    [[message]] = chat.requests
    assert message.role is Role.user
    assert "The bridge opened in 1937." in message.content
    assert describe_prompt_kind(kind) in message.content


def build_completion(content: str, finish_reason: str) -> dict[str, object]:
    return {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}]
    }


def complete_with(response: httpx.Response, requests: list[httpx.Request]) -> str:
    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return response

    async def complete() -> str:
        async with httpx.AsyncClient(base_url="http://vllm/v1", transport=httpx.MockTransport(respond)) as http:
            return await VllmChatModel(http).complete([ChatMessage(role=Role.user, content="Hello")])

    return asyncio.run(complete())


def test_chat_model_posts_the_messages_and_returns_the_content() -> None:
    requests: list[httpx.Request] = []

    actual = complete_with(httpx.Response(200, json=build_completion("Hi", "stop")), requests)

    [request] = requests
    assert actual == "Hi"
    assert request.url == "http://vllm/v1/chat/completions"
    assert json.loads(request.content) == {"messages": [{"role": "user", "content": "Hello"}]}


def test_chat_model_rejects_a_reply_that_reached_the_token_limit() -> None:
    with pytest.raises(ValidationError):
        complete_with(httpx.Response(200, json=build_completion("Hi", "length")), [])


def test_chat_model_raises_for_an_error_status() -> None:
    with pytest.raises(httpx.HTTPStatusError):
        complete_with(httpx.Response(500), [])
