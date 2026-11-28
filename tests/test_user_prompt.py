import asyncio
from collections.abc import Sequence

import pytest
from pregel_redux import Endpoint, StateGraph
from pydantic import BaseModel

from llm_lab.datasets.graphs.nodes.chat import ChatMessage, Role
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


def run_user_prompt_graph(chat: RecordingChatModel, state: UserPromptInput) -> UserPromptOutput:
    builder = StateGraph(UserPromptState, input_schema=UserPromptInput, output_schema=UserPromptOutput)
    builder.add_node("write_user_prompt", write_user_prompt, input_schema=UserPromptInput)
    builder.add_edge(Endpoint.start, "write_user_prompt")
    graph = builder.compile({open_chat_model: lambda: chat})
    return asyncio.run(graph.ainvoke(state))


@pytest.fixture(params=list(PromptKind))
def prompt_kind(request: pytest.FixtureRequest) -> PromptKind:
    return request.param


@pytest.mark.parametrize(
    ("source_text", "reply"),
    [
        pytest.param("The bridge opened in 1937.", "What year did the bridge open?", id="one sentence"),
        pytest.param("def add(a, b):\n    return a - b", "Fix the bug in this function.", id="code"),
        pytest.param("Line one.\n\nLine two.", "", id="empty reply"),
    ],
)
def test_node_writes_the_reply_of_the_model_as_the_user_prompt(
    prompt_kind: PromptKind, source_text: str, reply: str
) -> None:
    """
    Arrange: a graph whose only node writes the user prompt, with a chat model that gives a fixed reply.
    Act: run the graph on a source text and a prompt kind.
    Assert: the output of the graph has the reply of the chat model as the user prompt.
    """
    chat = RecordingChatModel(reply)

    actual = run_user_prompt_graph(chat, UserPromptInput(source_text=source_text, prompt_kind=prompt_kind))

    expected = UserPromptOutput(user_prompt=reply)
    assert actual == expected, f"Expected the reply of the model as the user prompt, {expected}, got {actual}"


@pytest.mark.parametrize(
    "source_text",
    [
        pytest.param("The bridge opened in 1937.", id="one sentence"),
        pytest.param("def add(a, b):\n    return a - b", id="code"),
        pytest.param("Line one.\n\nLine two.", id="paragraphs"),
    ],
)
def test_node_sends_one_user_message_with_the_source_text_and_the_prompt_kind(
    prompt_kind: PromptKind, source_text: str
) -> None:
    """
    Arrange: a graph whose only node writes the user prompt, with a chat model that records its requests.
    Act: run the graph on a source text and a prompt kind.
    Assert: the chat model gets one user message that contains the source text and the description of the
        prompt kind.
    """
    chat = RecordingChatModel("Summarize this.")

    run_user_prompt_graph(chat, UserPromptInput(source_text=source_text, prompt_kind=prompt_kind))

    [[message]] = chat.requests
    assert message.role is Role.user, f"Expected a user message, got {message.role}"
    assert source_text in message.content, f"The request does not contain the source text: {message.content!r}"
    assert describe_prompt_kind(prompt_kind) in message.content, (
        f"The request does not describe the {prompt_kind} prompt kind: {message.content!r}"
    )


def test_each_prompt_kind_has_its_own_description() -> None:
    """
    Arrange: all prompt kinds.
    Act: describe each prompt kind.
    Assert: no two prompt kinds have the same description.
    """
    descriptions = [describe_prompt_kind(kind) for kind in PromptKind]

    assert len(set(descriptions)) == len(descriptions), f"Two prompt kinds share a description: {descriptions}"
