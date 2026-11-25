from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol, assert_never

from pydantic import SecretStr
from sqlalchemy import select
from sqlalchemy.orm import Session

from llm_lab.errors import CommandError
from llm_lab.jobs.access import InstanceAccess, SshEndpoint
from llm_lab.jobs.protocol import (
    Aborted,
    Active,
    Completed,
    Failed,
    Halted,
    JobRecord,
    JobSpec,
    Pending,
    Running,
    Succeeded,
    conclude,
)
from llm_lab.jobs.tables import JobTransitionRow, select_latest_transitions
from llm_lab.keyring_settings import SecretName


class InstanceProvider(Protocol):
    def find_offers(self, spec: JobSpec) -> Sequence[int]: ...

    def rent_offer(
        self, offer_id: int, spec: JobSpec, secrets: Mapping[SecretName, SecretStr], access: InstanceAccess
    ) -> int | None: ...

    def observe(self, instance_id: int) -> Active | Completed | Halted: ...

    def find_ssh_endpoint(self, instance_id: int) -> SshEndpoint: ...

    def find_instances(self, job_name: str) -> Sequence[int]: ...

    def destroy(self, instance_id: int) -> None: ...


class SecretSource(Protocol):
    def read(self, name: SecretName) -> SecretStr: ...


class AccessKeyStore(Protocol):
    def create(self, job_name: str) -> InstanceAccess: ...

    def remove(self, job_name: str) -> None: ...


class ArtifactDownloader(Protocol):
    def download(self, job_name: str, endpoint: SshEndpoint, remote_directory: str) -> Path: ...


@dataclass(frozen=True)
class Abandonment:
    record: JobRecord
    downloaded_to: Path | None


class JobOrchestrator:
    def __init__(
        self,
        provider: InstanceProvider,
        session: Session,
        secret_source: SecretSource,
        access_keys: AccessKeyStore,
        downloader: ArtifactDownloader,
        read_clock: Callable[[], datetime],
    ) -> None:
        self.provider = provider
        self.access_keys = access_keys
        self.downloader = downloader
        self.secret_source = secret_source
        self.session = session
        self.read_clock = read_clock

    def submit(self, spec: JobSpec) -> JobRecord:
        previous = self.read_latest(spec.name)
        if previous is not None:
            match previous.status:
                case Pending() | Running():
                    raise CommandError(f"Job {spec.name} is not finished. Run `llm-lab jobs status {spec.name}`.")
                case Succeeded() | Failed() | Aborted():
                    pass
                case _:
                    assert_never(previous.status)
        secrets = {name: self.secret_source.read(name) for name in spec.secret_environment}
        pending = Pending(submitted_at=self.read_clock())
        self.record_transition(JobRecord(spec=spec, status=pending))
        access = self.access_keys.create(spec.name)
        rented: list[int] = []
        recorded = False
        try:
            status = self.rent_instance(spec, secrets, access, pending, rented)
            record = JobRecord(spec=spec, status=status)
            self.record_transition(record)
            recorded = True
        finally:
            if not recorded:
                self.destroy_instances(spec.name, rented)
                self.access_keys.remove(spec.name)
        match status:
            case Running():
                pass
            case Aborted():
                self.access_keys.remove(spec.name)
            case _:
                assert_never(status)
        return record

    def sync(self, name: str) -> JobRecord:
        record = self.read(name)
        match record.status:
            case Pending(submitted_at=submitted_at):
                return self.finish(
                    record.spec,
                    [],
                    Aborted(
                        submitted_at=submitted_at,
                        finished_at=self.read_clock(),
                        reason="the submission stopped before an instance was recorded",
                    ),
                )
            case Running(instance_id=instance_id) as running:
                observation = self.provider.observe(instance_id)
                match observation:
                    case Active():
                        return record
                    case Completed() | Halted():
                        return self.finish(
                            record.spec, [instance_id], conclude(running, self.read_clock(), observation)
                        )
                    case _:
                        assert_never(observation)
            case Succeeded() | Failed() | Aborted():
                return record
            case _:
                assert_never(record.status)

    def sync_all(self) -> Sequence[JobRecord]:
        names = [row.name for row in self.session.scalars(select_latest_transitions())]
        return [self.sync(name) for name in names]

    def abandon(self, name: str, *, download: bool) -> Abandonment:
        record = self.read(name)
        match record.status:
            case Pending(submitted_at=submitted_at):
                known_instances: list[int] = []
                downloaded_to = None
            case Running(submitted_at=submitted_at, instance_id=instance_id):
                known_instances = [instance_id]
                downloaded_to = self.download_artifacts(record.spec, instance_id) if download else None
            case Succeeded() | Failed() | Aborted():
                self.destroy_instances(name, [])
                return Abandonment(record, None)
            case _:
                assert_never(record.status)
        finished = self.finish(
            record.spec,
            known_instances,
            Aborted(submitted_at=submitted_at, finished_at=self.read_clock(), reason="abandoned by the operator"),
        )
        return Abandonment(finished, downloaded_to)

    def download_artifacts(self, spec: JobSpec, instance_id: int) -> Path | None:
        match spec.artifacts_directory:
            case None:
                return None
            case str() as remote_directory:
                endpoint = self.provider.find_ssh_endpoint(instance_id)
                return self.downloader.download(spec.name, endpoint, remote_directory)
            case _:
                assert_never(spec.artifacts_directory)

    def find_ssh_endpoint(self, name: str) -> SshEndpoint:
        record = self.read(name)
        match record.status:
            case Running(instance_id=instance_id):
                return self.provider.find_ssh_endpoint(instance_id)
            case Pending() | Succeeded() | Failed() | Aborted():
                raise CommandError(f"Job {name} is {record.status.condition}, so it has no instance to connect to.")
            case _:
                assert_never(record.status)

    def read(self, name: str) -> JobRecord:
        record = self.read_latest(name)
        if record is None:
            raise CommandError(f"No job has the name {name}.")
        return record

    def read_latest(self, name: str) -> JobRecord | None:
        record = self.session.scalar(
            select(JobTransitionRow.record)
            .where(JobTransitionRow.name == name)
            .order_by(JobTransitionRow.sequence.desc())
            .limit(1)
        )
        return None if record is None else JobRecord.model_validate_json(record)

    def record_transition(self, record: JobRecord) -> None:
        self.session.add(
            JobTransitionRow(
                name=record.spec.name, condition=record.status.condition.value, record=record.model_dump_json()
            )
        )
        self.session.commit()

    def rent_instance(
        self,
        spec: JobSpec,
        secrets: Mapping[SecretName, SecretStr],
        access: InstanceAccess,
        pending: Pending,
        rented: list[int],
    ) -> Running | Aborted:
        for offer_id in self.provider.find_offers(spec):
            instance_id = self.provider.rent_offer(offer_id, spec, secrets, access)
            if instance_id is not None:
                rented.append(instance_id)
                return Running(submitted_at=pending.submitted_at, instance_id=instance_id)
        return Aborted(
            submitted_at=pending.submitted_at, finished_at=self.read_clock(), reason="no offer was available to rent"
        )

    def finish(self, spec: JobSpec, known_instances: Sequence[int], status: Succeeded | Failed | Aborted) -> JobRecord:
        self.destroy_instances(spec.name, known_instances)
        self.access_keys.remove(spec.name)
        record = JobRecord(spec=spec, status=status)
        self.record_transition(record)
        return record

    def destroy_instances(self, job_name: str, known_instances: Sequence[int]) -> None:
        for instance_id in known_instances:
            self.provider.destroy(instance_id)
        for instance_id in self.provider.find_instances(job_name):
            if instance_id not in known_instances:
                self.provider.destroy(instance_id)
