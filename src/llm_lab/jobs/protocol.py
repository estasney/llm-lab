from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Self, assert_never

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator
from pydantic_extra_types.country import CountryAlpha2

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
type CountryCodes = Annotated[tuple[CountryAlpha2, ...], Field(min_length=1)]


class IncludedCountries(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    include: CountryCodes = Field(description="Countries where the host must be, as ISO 3166-1 alpha-2 codes.")


class ExcludedCountries(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    exclude: CountryCodes = Field(description="Countries where the host must not be, as ISO 3166-1 alpha-2 codes.")


class ResourceRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[PlacementKind.resources]
    gpu_name: Annotated[str, Field(min_length=1)]
    gpu_count: PositiveInt
    max_dollars_per_hour: PositiveFloat = Field(
        description="Ceiling on the total hourly price, which is the compute price plus the storage price for disk_gb."
    )
    max_compute_dollars_per_hour: PositiveFloat | None = Field(
        default=None, description="Ceiling on the hourly price of the machine without storage."
    )
    max_storage_dollars_per_gb_month: PositiveFloat | None = Field(
        default=None, description="Ceiling on the storage price in dollars per GB per month."
    )
    max_download_dollars_per_gb: PositiveFloat | None = Field(
        default=None,
        description="Ceiling on the price of internet download traffic, which Vast.ai bills on top of the hourly price.",
    )
    max_upload_dollars_per_gb: PositiveFloat | None = Field(
        default=None,
        description="Ceiling on the price of internet upload traffic, which Vast.ai bills on top of the hourly price.",
    )
    countries: IncludedCountries | ExcludedCountries | None = Field(
        default=None, description="Countries to include or to exclude. Without it, a host can be in any country."
    )
    datacenter_only: bool = Field(default=False, description="Rent only hosts that Vast.ai lists as datacenters.")
    min_duration_hours: PositiveFloat | None = Field(
        default=None, description="Minimum time in hours that the host must stay available for rent."
    )
    min_cuda: PositiveFloat = Field(
        default=13.0, description="Minimum CUDA version that the driver of the host must support."
    )


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
