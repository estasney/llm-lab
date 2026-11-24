import time
from collections.abc import Callable, Generator
from pathlib import PurePosixPath
from typing import Annotated, Literal, assert_never, override

import httpx
from pydantic import BaseModel, Field, RootModel, SecretStr

from llm_lab.dependencies import Depends
from llm_lab.settings import SettingsDep
from llm_lab.studio.recipe import JobRequest
from llm_lab.studio.ssh import SshSessionDep


class Token(BaseModel):
    access_token: str


class RecipeError(BaseModel):
    message: str
    path: str | None
    code: str | None


class RecipeValidation(BaseModel):
    valid: bool
    errors: list[RecipeError]
    raw_detail: str | None


class StartedJob(BaseModel):
    job_id: str


class RunningJob(BaseModel):
    status: Literal["created", "pending", "active", "cancelling"]
    stage: str | None
    current_column: str | None


class CompletedJob(BaseModel):
    status: Literal["completed"]
    artifact_path: PurePosixPath


class FailedJob(BaseModel):
    status: Literal["error"]
    error: str


class CancelledJob(BaseModel):
    status: Literal["cancelled"]


class JobSnapshot(RootModel[RunningJob | CompletedJob | FailedJob | CancelledJob]):
    root: RunningJob | CompletedJob | FailedJob | CancelledJob = Field(discriminator="status")


def read_body(response: httpx.Response) -> bytes:
    if response.is_error:
        raise RuntimeError(f"Studio answered {response.status_code} for {response.request.url.path}: {response.text}")
    return response.content


class StudioLogin(httpx.Auth):
    requires_response_body = True

    def __init__(self, username: str, password: SecretStr) -> None:
        self.username = username
        self.password = password
        self.access_token: str | None = None

    def log_in(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, str]:
        response = yield httpx.Request(
            "POST",
            request.url.join("/api/auth/login"),
            json={"username": self.username, "password": self.password.get_secret_value()},
        )
        return Token.model_validate_json(read_body(response)).access_token

    @override
    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response]:
        if self.access_token is None:
            self.access_token = yield from self.log_in(request)
        request.headers["Authorization"] = f"Bearer {self.access_token}"
        response = yield request
        if response.status_code == httpx.codes.UNAUTHORIZED:
            self.access_token = yield from self.log_in(request)
            request.headers["Authorization"] = f"Bearer {self.access_token}"
            yield request


class StudioClient:
    def __init__(self, http: httpx.Client, poll_seconds: float) -> None:
        self.http = http
        self.poll_seconds = poll_seconds

    def run_job(self, request: JobRequest, report: Callable[[RunningJob], None]) -> CompletedJob:
        payload = request.model_dump(mode="json")
        validation = RecipeValidation.model_validate_json(
            read_body(self.http.post("/api/data-recipe/validate", json=payload))
        )
        if not validation.valid:
            raise ValueError(f"Studio rejected the recipe: {validation.model_dump_json()}")
        job_id = StartedJob.model_validate_json(read_body(self.http.post("/api/data-recipe/jobs", json=payload))).job_id
        reported: RunningJob | None = None
        while True:
            job = JobSnapshot.model_validate_json(
                read_body(self.http.get(f"/api/data-recipe/jobs/{job_id}/status"))
            ).root
            match job:
                case CompletedJob():
                    return job
                case FailedJob():
                    raise RuntimeError(f"Studio job {job_id} failed: {job.error}")
                case CancelledJob():
                    raise RuntimeError(f"Studio job {job_id} was cancelled")
                case RunningJob():
                    if job != reported:
                        report(job)
                        reported = job
                    time.sleep(self.poll_seconds)
                case _:
                    assert_never(job)


def open_studio_client(settings: SettingsDep, ssh: SshSessionDep) -> Generator[StudioClient]:
    with httpx.Client(
        transport=httpx.HTTPTransport(uds=str(ssh.studio_socket)),
        base_url=f"http://127.0.0.1:{settings.studio_port}",
        auth=StudioLogin(settings.studio_username, settings.studio_password),
        timeout=settings.studio_request_timeout_seconds,
    ) as http:
        yield StudioClient(http, settings.studio_poll_seconds)


type StudioClientDep = Annotated[StudioClient, Depends(open_studio_client)]
