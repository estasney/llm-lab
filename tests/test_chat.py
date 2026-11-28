import asyncio
import json
from contextlib import AbstractContextManager

import httpx
import pytest
from pydantic import ValidationError

from llm_lab.datasets.graphs.nodes.chat import ChatMessage, Role, VllmChatModel


def build_choice(content: str, finish_reason: str) -> dict[str, object]:
    return {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}


def complete_with(
    response: httpx.Response, messages: list[ChatMessage], requests: list[httpx.Request] | None = None
) -> str:
    def respond(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(request)
        return response

    async def complete() -> str:
        async with httpx.AsyncClient(base_url="http://vllm/v1", transport=httpx.MockTransport(respond)) as http:
            return await VllmChatModel(http).complete(messages)

    return asyncio.run(complete())


@pytest.mark.parametrize(
    ("messages", "content"),
    [
        pytest.param([ChatMessage(role=Role.user, content="Hello")], "Hi", id="one user message"),
        pytest.param(
            [
                ChatMessage(role=Role.system, content="Answer in one word."),
                ChatMessage(role=Role.user, content="Capital of France?"),
                ChatMessage(role=Role.assistant, content="Paris"),
                ChatMessage(role=Role.user, content="And of Spain?"),
            ],
            "Madrid",
            id="conversation with a system message",
        ),
        pytest.param([ChatMessage(role=Role.user, content="Say nothing.")], "", id="empty reply"),
    ],
)
def test_chat_model_posts_the_messages_and_returns_the_content(messages: list[ChatMessage], content: str) -> None:
    """
    Arrange: a chat model whose HTTP transport replies with one finished choice.
    Act: complete the messages.
    Assert: the request posts the messages to the chat completions path, and the result is the content of the choice.
    """
    requests: list[httpx.Request] = []

    actual = complete_with(httpx.Response(200, json={"choices": [build_choice(content, "stop")]}), messages, requests)

    [request] = requests
    expected_body = {"messages": [message.model_dump(mode="json") for message in messages]}
    assert request.url == "http://vllm/v1/chat/completions", f"Expected the chat completions path, got {request.url}"
    assert json.loads(request.content) == expected_body, f"Expected {expected_body}, got {request.content!r}"
    assert actual == content, f"Expected the content of the only choice, {content!r}, got {actual!r}"


@pytest.mark.parametrize(
    ("response", "expectation"),
    [
        pytest.param(
            httpx.Response(200, json={"choices": [build_choice("Hi", "length")]}),
            pytest.raises(ValidationError),
            id="reply reached the token limit",
        ),
        pytest.param(httpx.Response(200, json={"choices": []}), pytest.raises(ValidationError), id="no choice"),
        pytest.param(
            httpx.Response(200, json={"choices": [build_choice("Hi", "stop"), build_choice("Hello", "stop")]}),
            pytest.raises(ValidationError),
            id="two choices",
        ),
        pytest.param(httpx.Response(400), pytest.raises(httpx.HTTPStatusError), id="client error"),
        pytest.param(httpx.Response(500), pytest.raises(httpx.HTTPStatusError), id="server error"),
    ],
)
def test_chat_model_rejects_a_response_without_one_finished_reply(
    response: httpx.Response, expectation: AbstractContextManager[object]
) -> None:
    """
    Arrange: a chat model whose HTTP transport replies with an error status, a reply that reached the token limit,
        no choice or two choices.
    Act: complete one user message.
    Assert: the chat model raises an error instead of returning a reply.
    """
    with expectation:
        complete_with(response, [ChatMessage(role=Role.user, content="Hello")])
