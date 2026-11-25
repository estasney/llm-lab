import time
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Annotated

import httpx

from llm_lab.cli.db.deps import DatabaseSessionDep
from llm_lab.dependencies import Depends
from llm_lab.jobs.access import JobAccessStore, RsyncDownloader
from llm_lab.jobs.orchestrator import ArtifactDownloader, InstanceProvider, JobOrchestrator
from llm_lab.jobs.secrets import SettingsSecretSource
from llm_lab.jobs.vast import VastInstanceProvider
from llm_lab.settings import SettingsDep


def get_http_client(settings: SettingsDep) -> Iterator[httpx.Client]:
    with httpx.Client(timeout=settings.vast_request_timeout_seconds) as http:
        yield http


type HttpClientDep = Annotated[httpx.Client, Depends(get_http_client)]


def get_instance_provider(settings: SettingsDep, http: HttpClientDep) -> InstanceProvider:
    return VastInstanceProvider(settings, http, time.sleep)


type InstanceProviderDep = Annotated[InstanceProvider, Depends(get_instance_provider)]


def get_job_access_store(settings: SettingsDep) -> JobAccessStore:
    return JobAccessStore(settings)


type JobAccessStoreDep = Annotated[JobAccessStore, Depends(get_job_access_store)]


def get_artifact_downloader(settings: SettingsDep, access: JobAccessStoreDep) -> ArtifactDownloader:
    return RsyncDownloader(settings, access)


type ArtifactDownloaderDep = Annotated[ArtifactDownloader, Depends(get_artifact_downloader)]


def get_job_orchestrator(
    settings: SettingsDep,
    provider: InstanceProviderDep,
    session: DatabaseSessionDep,
    access: JobAccessStoreDep,
    downloader: ArtifactDownloaderDep,
) -> JobOrchestrator:
    return JobOrchestrator(
        provider, session, SettingsSecretSource(settings), access, downloader, lambda: datetime.now(UTC)
    )


type JobOrchestratorDep = Annotated[JobOrchestrator, Depends(get_job_orchestrator)]
