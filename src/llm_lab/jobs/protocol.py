from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Self, assert_never

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator

from llm_lab.keyring_settings import SecretName


class PlacementKind(StrEnum):
    resources = "resources"
    offer = "offer"


class JobCondition(StrEnum):
    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    aborted = "aborted"


type JobName = Annotated[str, Field(pattern=r"^[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?$")]
type EnvironmentName = Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]
type CountryCode = Annotated[str, Field(pattern=r"^[A-Z]{2}$")]
type CountryCodes = Annotated[tuple[CountryCode, ...], Field(min_length=1)]


class IncludedCountries(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    include: CountryCodes


class ExcludedCountries(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    exclude: CountryCodes


class ResourceRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[PlacementKind.resources]
    gpu_name: Annotated[str, Field(min_length=1)]
    gpu_count: PositiveInt
    max_dollars_per_hour: PositiveFloat
    max_compute_dollars_per_hour: PositiveFloat | None = None
    max_storage_dollars_per_gb_month: PositiveFloat | None = None
    max_download_dollars_per_gb: PositiveFloat | None = None
    max_upload_dollars_per_gb: PositiveFloat | None = None
    countries: IncludedCountries | ExcludedCountries | None = None
    datacenter_only: bool = False
    min_cuda: PositiveFloat = 13.0


class ExplicitOffer(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[PlacementKind.offer]
    offer_id: int


class JobSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: JobName
    image: Annotated[str, Field(min_length=1)]
    command: Annotated[tuple[str, ...], Field(min_length=1)]
    environment: Mapping[EnvironmentName, str] = Field(default_factory=dict)
    secret_environment: tuple[SecretName, ...] = ()
    disk_gb: PositiveFloat
    ttl_seconds_after_finished: PositiveInt
    placement: Annotated[ResourceRequest | ExplicitOffer, Field(discriminator="kind")]
    artifacts_directory: Annotated[str, Field(pattern=r"^/")] | None = None

    @model_validator(mode="after")
    def reject_names_in_both_environments(self) -> Self:
        repeated = sorted(set(self.environment) & set(self.secret_environment))
        if repeated:
            raise ValueError(f"names {repeated!r} are in environment and in secret_environment")
        return self


class Pending(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    condition: Literal[JobCondition.pending] = JobCondition.pending
    submitted_at: datetime


class Running(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    condition: Literal[JobCondition.running] = JobCondition.running
    submitted_at: datetime
    instance_id: int


class Succeeded(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    condition: Literal[JobCondition.succeeded] = JobCondition.succeeded
    submitted_at: datetime
    finished_at: datetime
    instance_id: int


class Failed(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    condition: Literal[JobCondition.failed] = JobCondition.failed
    submitted_at: datetime
    finished_at: datetime
    instance_id: int
    exit_code: int


class Aborted(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    condition: Literal[JobCondition.aborted] = JobCondition.aborted
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
