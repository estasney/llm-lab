from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol, assert_never

from pydantic import SecretStr
from sqlalchemy import select
from sqlalchemy.orm import Session

from llm_lab.errors import CommandError
from llm_lab.jobs.access import CodeDelivery, InstanceAccess, SshEndpoint
from llm_lab.jobs.protocol import (
    Aborted,
    Active,
    Completed,
    Failed,
    Halted,
    JobCode,
    JobRecord,
    JobSpec,
    Pending,
    Running,
    Succeeded,
    conclude,
)
from llm_lab.jobs.tables import JobTransitionRow, select_latest_transitions
from llm_lab.keyring_settings import SecretName
from llm_lab.settings import Settings


class InstanceProvider(Protocol):
    def find_offers(self, spec: JobSpec) -> Sequence[int]: ...

    def rent_offer(
        self, offer_id: int, spec: JobSpec, secrets: Mapping[SecretName, SecretStr], access: InstanceAccess
    ) -> int | None: ...

    def observe(self, instance_id: int) -> Active | Completed | Halted: ...

    def find_ssh_endpoint(self, instance_id: int) -> SshEndpoint | None: ...

    def find_instances(self, job_name: str) -> Sequence[int]: ...

    def destroy(self, instance_id: int) -> None: ...


class SecretSource(Protocol):
    def read(self, name: SecretName) -> SecretStr: ...


class AccessKeyStore(Protocol):
    def create(self, job_name: str) -> InstanceAccess: ...

    def remove(self, job_name: str) -> None: ...


class ArtifactDownloader(Protocol):
    def download(self, job_name: str, endpoint: SshEndpoint, remote_directory: str) -> Path: ...


class JobCodeSource(Protocol):
    def read(self, package: str) -> JobCode: ...


class CodeUploader(Protocol):
    def upload(self, job_name: str, endpoint: SshEndpoint, code: JobCode) -> CodeDelivery: ...


@dataclass(frozen=True)
class Abandonment:
    record: JobRecord
    downloaded_to: Path | None


class JobOrchestrator:
    def __init__(
        self,
        settings: Settings,
        provider: InstanceProvider,
        session: Session,
        secret_source: SecretSource,
        access_keys: AccessKeyStore,
        code_source: JobCodeSource,
        uploader: CodeUploader,
        downloader: ArtifactDownloader,
        read_clock: Callable[[], datetime],
        sleep: Callable[[float], None],
    ) -> None:
        self.settings = settings
        self.provider = provider
        self.access_keys = access_keys
        self.code_source = code_source
        self.uploader = uploader
        self.downloader = downloader
        self.secret_source = secret_source
        self.session = session
        self.read_clock = read_clock
        self.sleep = sleep

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
        match spec.package:
            case None:
                code = None
            case str() as package:
                code = self.code_source.read(package)
            case _:
                assert_never(spec.package)
        pending = Pending(submitted_at=self.read_clock())
        submitted = JobRecord(spec=spec, status=pending, code_digest=None if code is None else code.digest)
        self.record_transition(submitted)
        access = self.access_keys.create(spec.name)
        rented: list[int] = []
        recorded = False
        try:
            status = self.rent_instance(spec, secrets, access, pending, rented)
            record = submitted.model_copy(update={"status": status})
            self.record_transition(record)
            recorded = True
        finally:
            if not recorded:
                self.destroy_instances(spec.name, rented)
                self.access_keys.remove(spec.name)
        match status:
            case Running() as running:
                match code:
                    case None:
                        return record
                    case JobCode():
                        return self.deliver_code(record, running, code)
                    case _:
                        assert_never(code)
            case Aborted():
                self.access_keys.remove(spec.name)
                return record
            case _:
                assert_never(status)

    def deliver_code(self, record: JobRecord, running: Running, code: JobCode) -> JobRecord:
        delivered = False
        try:
            delivered = self.upload_code(record.spec.name, running.instance_id, code)
        finally:
            if not delivered:
                record = self.finish(
                    record,
                    [running.instance_id],
                    Aborted(
                        submitted_at=running.submitted_at,
                        finished_at=self.read_clock(),
                        reason=f"the code did not reach instance {running.instance_id}",
                    ),
                )
        return record

    def upload_code(self, job_name: str, instance_id: int, code: JobCode) -> bool:
        deadline = self.read_clock() + timedelta(seconds=self.settings.job_code_timeout_seconds)
        while True:
            match self.provider.find_ssh_endpoint(instance_id):
                case SshEndpoint() as endpoint:
                    delivery = self.uploader.upload(job_name, endpoint, code)
                case None:
                    delivery = CodeDelivery.unreachable
                case _ as endpoint:
                    assert_never(endpoint)
            match delivery:
                case CodeDelivery.delivered:
                    return True
                case CodeDelivery.unreachable:
                    if self.read_clock() >= deadline:
                        return False
                    self.sleep(self.settings.job_code_poll_seconds)
                case _:
                    assert_never(delivery)

    def sync(self, name: str) -> JobRecord:
        record = self.read(name)
        match record.status:
            case Pending(submitted_at=submitted_at):
                return self.finish(
                    record,
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
                        return self.finish(record, [instance_id], conclude(running, self.read_clock(), observation))
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
            record,
            known_instances,
            Aborted(submitted_at=submitted_at, finished_at=self.read_clock(), reason="abandoned by the operator"),
        )
        return Abandonment(finished, downloaded_to)

    def download_artifacts(self, spec: JobSpec, instance_id: int) -> Path | None:
        match spec.artifacts_directory:
            case None:
                return None
            case str() as remote_directory:
                endpoint = self.require_ssh_endpoint(instance_id)
                return self.downloader.download(spec.name, endpoint, remote_directory)
            case _:
                assert_never(spec.artifacts_directory)

    def find_ssh_endpoint(self, name: str) -> SshEndpoint:
        record = self.read(name)
        match record.status:
            case Running(instance_id=instance_id):
                return self.require_ssh_endpoint(instance_id)
            case Pending() | Succeeded() | Failed() | Aborted():
                raise CommandError(f"Job {name} is {record.status.condition}, so it has no instance to connect to.")
            case _:
                assert_never(record.status)

    def require_ssh_endpoint(self, instance_id: int) -> SshEndpoint:
        match self.provider.find_ssh_endpoint(instance_id):
            case SshEndpoint() as endpoint:
                return endpoint
            case None:
                raise CommandError(
                    f"Instance {instance_id} does not map the SSH port {self.settings.vast_ssh_port} to a public port."
                )
            case _ as endpoint:
                assert_never(endpoint)

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

    def finish(
        self, record: JobRecord, known_instances: Sequence[int], status: Succeeded | Failed | Aborted
    ) -> JobRecord:
        self.destroy_instances(record.spec.name, known_instances)
        self.access_keys.remove(record.spec.name)
        finished = record.model_copy(update={"status": status})
        self.record_transition(finished)
        return finished

    def destroy_instances(self, job_name: str, known_instances: Sequence[int]) -> None:
        for instance_id in known_instances:
            self.provider.destroy(instance_id)
        for instance_id in self.provider.find_instances(job_name):
            if instance_id not in known_instances:
                self.provider.destroy(instance_id)
