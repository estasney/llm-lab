from collections.abc import AsyncIterator
from typing import Annotated

from openai import AsyncOpenAI
from pregel_redux import Depends, Scope

from llm_lab.datasets.graphs.nodes.models.chat import ChatModel, OpenAIChatModel, find_served_model
from llm_lab.settings import Settings, get_settings


async def open_chat_model(
    settings: Annotated[Settings, Depends(get_settings, scope=Scope.run)],
) -> AsyncIterator[ChatModel]:
    async with AsyncOpenAI(
        base_url=f"{settings.vllm_base_url}/v1", api_key="EMPTY", timeout=settings.vllm_request_timeout_seconds
    ) as client:
        yield OpenAIChatModel(client, await find_served_model(client))


type ChatModelDep = Annotated[ChatModel, Depends(open_chat_model, scope=Scope.run)]
