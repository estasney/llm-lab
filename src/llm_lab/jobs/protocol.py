from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Literal, assert_never

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt

type JobName = Annotated[str, Field(pattern=r"^[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?$")]


class ResourceRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["resources"]
    gpu_name: Annotated[str, Field(min_length=1)]
    gpu_count: PositiveInt
    max_dollars_per_hour: PositiveFloat


class ExplicitOffer(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["offer"]
    offer_id: int


class JobSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: JobName
    image: Annotated[str, Field(min_length=1)]
    command: Annotated[tuple[str, ...], Field(min_length=1)]
    environment: Mapping[str, str] = Field(default_factory=dict)
    disk_gb: PositiveFloat
    ttl_seconds_after_finished: PositiveInt
    placement: Annotated[ResourceRequest | ExplicitOffer, Field(discriminator="kind")]


class Pending(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    condition: Literal["pending"] = "pending"
    submitted_at: datetime


class Running(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    condition: Literal["running"] = "running"
    submitted_at: datetime
    instance_id: int


class Succeeded(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    condition: Literal["succeeded"] = "succeeded"
    submitted_at: datetime
    finished_at: datetime
    instance_id: int


class Failed(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    condition: Literal["failed"] = "failed"
    submitted_at: datetime
    finished_at: datetime
    instance_id: int
    exit_code: int


class Aborted(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    condition: Literal["aborted"] = "aborted"
    submitted_at: datetime
    finished_at: datetime
    reason: str


class JobRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    spec: JobSpec
    status: Annotated[Pending | Running | Succeeded | Failed | Aborted, Field(discriminator="condition")]


@dataclass(frozen=True)
class Active:
    pass


@dataclass(frozen=True)
class Completed:
    exit_code: int


@dataclass(frozen=True)
class Halted:
    instance_state: str


def conclude(running: Running, finished_at: datetime, outcome: Completed | Halted) -> Succeeded | Failed | Aborted:
    match outcome:
        case Completed(exit_code=0):
            return Succeeded(
                submitted_at=running.submitted_at, finished_at=finished_at, instance_id=running.instance_id
            )
        case Completed(exit_code=exit_code):
            return Failed(
                submitted_at=running.submitted_at,
                finished_at=finished_at,
                instance_id=running.instance_id,
                exit_code=exit_code,
            )
        case Halted(instance_state=instance_state):
            return Aborted(
                submitted_at=running.submitted_at,
                finished_at=finished_at,
                reason=f"instance {running.instance_id} became {instance_state} before the command reported an exit code",
            )
        case _:
            assert_never(outcome)
