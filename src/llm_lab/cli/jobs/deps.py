import time
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Annotated

import httpx
from sqlalchemy import Engine, create_engine

from llm_lab.database import build_database_url
from llm_lab.dependencies import Depends
from llm_lab.jobs.orchestrator import InstanceProvider, JobOrchestrator
from llm_lab.jobs.secrets import SettingsSecretSource
from llm_lab.jobs.vast import VastInstanceProvider
from llm_lab.settings import SettingsDep


def get_database_engine(settings: SettingsDep) -> Iterator[Engine]:
    engine = create_engine(build_database_url(settings))
    try:
        yield engine
    finally:
        engine.dispose()


type DatabaseEngineDep = Annotated[Engine, Depends(get_database_engine)]


def get_http_client() -> Iterator[httpx.Client]:
    with httpx.Client(timeout=60) as http:
        yield http


type HttpClientDep = Annotated[httpx.Client, Depends(get_http_client)]


def get_instance_provider(settings: SettingsDep, http: HttpClientDep) -> InstanceProvider:
    if settings.vast_api_key is None:
        raise LookupError("the Vast.ai API key is not set, so set it with `llm-lab secrets set VAST_API_KEY`")
    return VastInstanceProvider(settings, settings.vast_api_key, http, time.sleep)


type InstanceProviderDep = Annotated[InstanceProvider, Depends(get_instance_provider)]


def get_job_orchestrator(
    settings: SettingsDep, provider: InstanceProviderDep, engine: DatabaseEngineDep
) -> JobOrchestrator:
    return JobOrchestrator(provider, engine, SettingsSecretSource(settings), lambda: datetime.now(UTC))


type JobOrchestratorDep = Annotated[JobOrchestrator, Depends(get_job_orchestrator)]
