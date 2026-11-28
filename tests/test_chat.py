import asyncio
import json
from collections.abc import Awaitable, Callable
from contextlib import AbstractContextManager

import httpx2
import openai
import pytest
from openai import AsyncOpenAI
from pydantic import JsonValue

from llm_lab.datasets.graphs.nodes.models.chat import (
    AssistantMessage,
    ChatMessage,
    Completion,
    FunctionCall,
    FunctionTool,
    OpenAIChatModel,
    SystemMessage,
    ToolCall,
    ToolMessage,
    Usage,
    UserMessage,
)


def build_function_call(call_id: str, name: str, arguments: str) -> dict[str, JsonValue]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def build_choice(
    content: str | None, finish_reason: str, tool_calls: list[JsonValue] | None = None
) -> dict[str, JsonValue]:
    message: dict[str, JsonValue] = {"role": "assistant", "content": content, "tool_calls": tool_calls}
    return {"index": 0, "message": message, "finish_reason": finish_reason}


def build_completion(choices: list[JsonValue]) -> dict[str, JsonValue]:
    return {
        "id": "chat-1",
        "object": "chat.completion",
        "created": 0,
        "model": "served",
        "choices": choices,
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
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


def build_weather_tool() -> FunctionTool:
    return FunctionTool(
        name="get_weather",
        description="Get the weather of a city.",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    )


@pytest.mark.parametrize(
    ("messages", "tools", "expected_body"),
    [
        pytest.param(
            [UserMessage(content="Hello")],
            [],
            {"model": "served", "messages": [{"role": "user", "content": "Hello"}], "tools": []},
            id="one user message",
        ),
        pytest.param(
            [
                SystemMessage(content="Answer in one word."),
                UserMessage(content="Capital of France?"),
                AssistantMessage(content="Paris"),
                UserMessage(content="And of Spain?"),
            ],
            [],
            {
                "model": "served",
                "messages": [
                    {"role": "system", "content": "Answer in one word."},
                    {"role": "user", "content": "Capital of France?"},
                    {"role": "assistant", "content": "Paris"},
                    {"role": "user", "content": "And of Spain?"},
                ],
                "tools": [],
            },
            id="conversation with a system message",
        ),
        pytest.param(
            [
                UserMessage(content="Weather in Oslo?"),
                AssistantMessage(
                    tool_calls=[
                        ToolCall(id="call-1", function=FunctionCall(name="get_weather", arguments={"city": "Oslo"}))
                    ]
                ),
                ToolMessage(tool_call_id="call-1", content="Snow"),
            ],
            [build_weather_tool()],
            {
                "model": "served",
                "messages": [
                    {"role": "user", "content": "Weather in Oslo?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [build_function_call("call-1", "get_weather", '{"city":"Oslo"}')],
                    },
                    {"role": "tool", "tool_call_id": "call-1", "content": "Snow"},
                ],
                "tools": [{"type": "function", "function": build_weather_tool().model_dump(mode="json")}],
            },
            id="tool call and tool result",
        ),
    ],
)
def test_chat_model_posts_the_messages_and_the_tools(
    messages: list[ChatMessage], tools: list[FunctionTool], expected_body: dict[str, JsonValue]
) -> None:
    """
    Arrange: a chat model whose server replies with one choice that stopped by itself.
    Act: complete the messages with the tools.
    Assert: the request posts the served model, the messages and the tools to the chat completions path.
    """
    requests: list[httpx2.Request] = []
    response = httpx2.Response(200, json=build_completion([build_choice("Hi", "stop")]))

    run_with_client(response, lambda client: OpenAIChatModel(client, "served").complete(messages, tools), requests)

    [request] = requests
    actual = json.loads(request.content)
    assert request.url == "http://vllm/v1/chat/completions", f"Expected the chat completions path, got {request.url}"
    assert actual == expected_body, f"Expected {expected_body}, got {actual}"


@pytest.mark.parametrize(
    ("choice", "expected_message"),
    [
        pytest.param(build_choice("Hi", "stop"), AssistantMessage(content="Hi"), id="text reply"),
        pytest.param(build_choice("", "stop", []), AssistantMessage(content=""), id="empty reply"),
        pytest.param(
            build_choice(None, "tool_calls", [build_function_call("call-1", "get_weather", '{"city": "Oslo"}')]),
            AssistantMessage(
                tool_calls=[
                    ToolCall(id="call-1", function=FunctionCall(name="get_weather", arguments={"city": "Oslo"}))
                ]
            ),
            id="tool call without text",
        ),
        pytest.param(
            build_choice(
                "Checking both.",
                "tool_calls",
                [
                    build_function_call("call-1", "get_weather", '{"city": "Oslo"}'),
                    build_function_call("call-2", "get_weather", '{"city": "Lima"}'),
                ],
            ),
            AssistantMessage(
                content="Checking both.",
                tool_calls=[
                    ToolCall(id="call-1", function=FunctionCall(name="get_weather", arguments={"city": "Oslo"})),
                    ToolCall(id="call-2", function=FunctionCall(name="get_weather", arguments={"city": "Lima"})),
                ],
            ),
            id="two tool calls with text",
        ),
    ],
)
def test_chat_model_returns_the_reply_and_the_usage(
    choice: dict[str, JsonValue], expected_message: AssistantMessage
) -> None:
    """
    Arrange: a chat model whose server replies with one choice that stopped by itself or called tools, and reports
        the token usage.
    Act: complete one user message.
    Assert: the result has the reply as an assistant message with parsed tool calls, and the token usage.
    """
    response = httpx2.Response(200, json=build_completion([choice]))
    messages = [UserMessage(content="Hello")]

    actual = run_with_client(response, lambda client: OpenAIChatModel(client, "served").complete(messages))

    expected = Completion(message=expected_message, usage=Usage(prompt_tokens=11, completion_tokens=7))
    assert actual == expected, f"Expected {expected}, got {actual}"


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
        pytest.param(
            httpx2.Response(200, json=build_completion([build_choice(None, "tool_calls", [])])),
            pytest.raises(ValueError),
            id="tool call finish without tool calls",
        ),
        pytest.param(
            httpx2.Response(
                200,
                json=build_completion(
                    [build_choice(None, "tool_calls", [build_function_call("call-1", "get_weather", "{city: Oslo")])]
                ),
            ),
            pytest.raises(ValueError),
            id="tool call arguments that are not JSON",
        ),
        pytest.param(
            httpx2.Response(
                200,
                json=build_completion(
                    [
                        build_choice(
                            None,
                            "tool_calls",
                            [{"id": "call-1", "type": "custom", "custom": {"name": "shell", "input": "ls"}}],
                        )
                    ]
                ),
            ),
            pytest.raises(ValueError),
            id="custom tool call",
        ),
        pytest.param(httpx2.Response(200, json=build_completion([])), pytest.raises(ValueError), id="no choice"),
        pytest.param(
            httpx2.Response(200, json=build_completion([build_choice("Hi", "stop"), build_choice("Hello", "stop")])),
            pytest.raises(ValueError),
            id="two choices",
        ),
        pytest.param(
            httpx2.Response(
                200,
                json={
                    key: value
                    for key, value in build_completion([build_choice("Hi", "stop")]).items()
                    if key != "usage"
                },
            ),
            pytest.raises(ValueError),
            id="no usage",
        ),
        pytest.param(httpx2.Response(400), pytest.raises(openai.BadRequestError), id="client error"),
        pytest.param(httpx2.Response(500), pytest.raises(openai.InternalServerError), id="server error"),
    ],
)
def test_chat_model_rejects_a_response_without_one_finished_reply_and_its_usage(
    response: httpx2.Response, expectation: AbstractContextManager[object]
) -> None:
    """
    Arrange: a chat model whose server replies with an error status, a reply that did not finish, a reply without
        content, a tool call finish without valid function tool calls, no choice, two choices or no usage.
    Act: complete one user message.
    Assert: the chat model raises an error instead of returning a completion.
    """
    messages = [UserMessage(content="Hello")]

    with expectation:
        run_with_client(response, lambda client: OpenAIChatModel(client, "served").complete(messages))
