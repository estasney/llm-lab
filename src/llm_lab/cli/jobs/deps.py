import sqlite3
import time
from collections.abc import Iterator
from contextlib import closing
from datetime import UTC, datetime
from typing import Annotated

import httpx

from llm_lab.dependencies import Depends
from llm_lab.jobs.orchestrator import InstanceProvider, JobOrchestrator
from llm_lab.jobs.secrets import SettingsSecretSource
from llm_lab.jobs.store import SqliteJobRecordStore
from llm_lab.jobs.vast import VastInstanceProvider
from llm_lab.settings import SettingsDep


def get_job_record_store(settings: SettingsDep) -> Iterator[SqliteJobRecordStore]:
    settings.job_database_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(settings.job_database_path)) as connection:
        yield SqliteJobRecordStore(connection)


type JobRecordStoreDep = Annotated[SqliteJobRecordStore, Depends(get_job_record_store)]


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
    settings: SettingsDep, provider: InstanceProviderDep, store: JobRecordStoreDep
) -> JobOrchestrator:
    return JobOrchestrator(provider, store, SettingsSecretSource(settings), lambda: datetime.now(UTC))


type JobOrchestratorDep = Annotated[JobOrchestrator, Depends(get_job_orchestrator)]
