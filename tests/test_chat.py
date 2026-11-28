import asyncio
import json
from collections.abc import Awaitable, Callable
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import httpx2
import openai
import pytest
from openai import AsyncOpenAI

from llm_lab.datasets.graphs.nodes.chat import ChatMessage, OpenAIChatModel, Role, find_served_model


def build_choice(content: str | None, finish_reason: str) -> dict[str, object]:
    return {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}


def build_completion(choices: list[dict[str, object]]) -> dict[str, object]:
    return {"id": "chat-1", "object": "chat.completion", "created": 0, "model": "served", "choices": choices}


def build_model_list(names: list[str]) -> dict[str, object]:
    return {
        "object": "list",
        "data": [{"id": name, "object": "model", "created": 0, "owned_by": "vllm"} for name in names],
    }


def run_with_client[ResultT](
    response: httpx2.Response,
    use: Callable[[AsyncOpenAI], Awaitable[ResultT]],
    requests: list[httpx2.Request] | None = None,
) -> ResultT:
    def respond(request: httpx2.Request) -> httpx2.Response:
        if requests is not None:
            requests.append(request)
        return response

    async def run() -> ResultT:
        async with (
            httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as http,
            AsyncOpenAI(base_url="http://vllm/v1", api_key="EMPTY", max_retries=0, http_client=http) as client,
        ):
            return await use(client)

    return asyncio.run(run())


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
    Arrange: a chat model whose server replies with one choice that stopped by itself.
    Act: complete the messages.
    Assert: the request posts the messages and the served model to the chat completions path, and the result is
        the content of the choice.
    """
    requests: list[httpx2.Request] = []
    response = httpx2.Response(200, json=build_completion([build_choice(content, "stop")]))

    actual = run_with_client(response, lambda client: OpenAIChatModel(client, "served").complete(messages), requests)

    [request] = requests
    expected_body = {"model": "served", "messages": [message.model_dump(mode="json") for message in messages]}
    assert request.url == "http://vllm/v1/chat/completions", f"Expected the chat completions path, got {request.url}"
    assert json.loads(request.content) == expected_body, f"Expected {expected_body}, got {request.content!r}"
    assert actual == content, f"Expected the content of the only choice, {content!r}, got {actual!r}"


@pytest.mark.parametrize(
    ("response", "expectation"),
    [
        pytest.param(
            httpx2.Response(200, json=build_completion([build_choice("Hi", "length")])),
            pytest.raises(ValueError),
            id="reply reached the token limit",
        ),
        pytest.param(
            httpx2.Response(200, json=build_completion([build_choice(None, "stop")])),
            pytest.raises(ValueError),
            id="reply without content",
        ),
        pytest.param(httpx2.Response(200, json=build_completion([])), pytest.raises(ValueError), id="no choice"),
        pytest.param(
            httpx2.Response(200, json=build_completion([build_choice("Hi", "stop"), build_choice("Hello", "stop")])),
            pytest.raises(ValueError),
            id="two choices",
        ),
        pytest.param(httpx2.Response(400), pytest.raises(openai.BadRequestError), id="client error"),
        pytest.param(httpx2.Response(500), pytest.raises(openai.InternalServerError), id="server error"),
    ],
)
def test_chat_model_rejects_a_response_without_one_finished_reply(
    response: httpx2.Response, expectation: AbstractContextManager[object]
) -> None:
    """
    Arrange: a chat model whose server replies with an error status, a reply that reached the token limit, a reply
        without content, no choice or two choices.
    Act: complete one user message.
    Assert: the chat model raises an error instead of returning a reply.
    """
    messages = [ChatMessage(role=Role.user, content="Hello")]

    with expectation:
        run_with_client(response, lambda client: OpenAIChatModel(client, "served").complete(messages))


@pytest.mark.parametrize(
    ("names", "expectation"),
    [
        pytest.param(["Qwen/Qwen3-8B"], does_not_raise(), id="one model"),
        pytest.param([], pytest.raises(ValueError), id="no model"),
        pytest.param(["Qwen/Qwen3-8B", "Qwen/Qwen3-32B"], pytest.raises(ValueError), id="two models"),
    ],
)
def test_served_model_is_the_only_model_of_the_server(
    names: list[str], expectation: AbstractContextManager[object]
) -> None:
    """
    Arrange: a server that lists no model, one model or two models.
    Act: find the served model.
    Assert: the result is the name of the model when the server lists one model, and otherwise an error.
    """
    with expectation:
        actual = run_with_client(httpx2.Response(200, json=build_model_list(names)), find_served_model)

        assert actual == names[0], f"Expected the only listed model, {names[0]!r}, got {actual!r}"
