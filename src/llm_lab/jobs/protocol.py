from collections.abc import Mapping
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

type JobName = Annotated[str, Field(pattern=r"^[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?$")]


class JobSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: JobName
    image: Annotated[str, Field(min_length=1)]
    command: tuple[str, ...] = ()
    environment: Mapping[str, str] = Field(default_factory=dict)


class Running(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    condition: Literal["running"] = "running"
    started_at: datetime


class Succeeded(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    condition: Literal["succeeded"] = "succeeded"
    started_at: datetime
    finished_at: datetime


class Failed(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    condition: Literal["failed"] = "failed"
    started_at: datetime
    finished_at: datetime
    exit_code: int


class JobRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    spec: JobSpec
    status: Annotated[Running | Succeeded | Failed, Field(discriminator="condition")]


def conclude(running: Running, finished_at: datetime, exit_code: int) -> Succeeded | Failed:
    if exit_code == 0:
        return Succeeded(started_at=running.started_at, finished_at=finished_at)
    return Failed(started_at=running.started_at, finished_at=finished_at, exit_code=exit_code)
