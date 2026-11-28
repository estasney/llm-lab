from collections.abc import AsyncIterator
from typing import Annotated

import httpx
from pregel_redux import Depends, Scope

from llm_lab.datasets.graphs.nodes.chat import ChatModel, VllmChatModel
from llm_lab.settings import Settings, get_settings


async def open_chat_model(
    settings: Annotated[Settings, Depends(get_settings, scope=Scope.run)],
) -> AsyncIterator[ChatModel]:
    async with httpx.AsyncClient(
        base_url=f"{settings.vllm_base_url}/v1", timeout=settings.vllm_request_timeout_seconds
    ) as http:
        yield VllmChatModel(http)


type ChatModelDep = Annotated[ChatModel, Depends(open_chat_model, scope=Scope.run)]
