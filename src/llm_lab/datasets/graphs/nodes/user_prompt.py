from enum import StrEnum
from typing import assert_never

from pydantic import BaseModel

from llm_lab.datasets.graphs.nodes.chat import ChatMessage, ChatModelDep, Role


class PromptKind(StrEnum):
    question = "question"
    task = "task"


class UserPromptInput(BaseModel):
    source_text: str
    prompt_kind: PromptKind


class UserPromptOutput(BaseModel):
    user_prompt: str


def describe_prompt_kind(kind: PromptKind) -> str:
    match kind:
        case PromptKind.question:
            return "The message asks a question that the text answers."
        case PromptKind.task:
            return "The message asks the assistant to do a task with the text."
        case _:
            assert_never(kind)


def build_user_prompt_request(state: UserPromptInput) -> list[ChatMessage]:
    instruction = "\n".join(
        [
            "A user has the text below in front of them.",
            "Write one message that this user sends to an AI assistant about the text.",
            describe_prompt_kind(state.prompt_kind),
            "Write only the message, with no quotes and no label.",
            "",
            "<text>",
            state.source_text,
            "</text>",
        ]
    )
    return [ChatMessage(role=Role.user, content=instruction)]


async def write_user_prompt(state: UserPromptInput, chat: ChatModelDep) -> UserPromptOutput:
    return UserPromptOutput(user_prompt=await chat.complete(build_user_prompt_request(state)))
