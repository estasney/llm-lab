from collections.abc import AsyncIterator
from typing import Annotated

from openai import AsyncOpenAI
from pregel_redux import Depends, Scope

from llm_lab.datasets.graphs.nodes.models.chat import ChatModelFactory, OpenAIChatModel
from llm_lab.settings import Settings, get_settings


async def open_chat_model_factory(
    settings: Annotated[Settings, Depends(get_settings, scope=Scope.run)],
) -> AsyncIterator[ChatModelFactory]:
    async with AsyncOpenAI(
        base_url=f"{settings.vllm_base_url}/v1", api_key="EMPTY", timeout=settings.vllm_request_timeout_seconds
    ) as client:
        yield lambda model: OpenAIChatModel(client, model)


type ChatModelFactoryDep = Annotated[ChatModelFactory, Depends(open_chat_model_factory, scope=Scope.run)]
