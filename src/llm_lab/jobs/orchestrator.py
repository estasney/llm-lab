from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Protocol, assert_never

from pydantic import SecretStr
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

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


class InstanceProvider(Protocol):
    def find_offers(self, spec: JobSpec) -> Sequence[int]: ...

    def rent_offer(self, offer_id: int, spec: JobSpec, secrets: Mapping[str, SecretStr]) -> int | None: ...

    def observe(self, instance_id: int) -> Active | Completed | Halted: ...

    def find_instances(self, job_name: str) -> Sequence[int]: ...

    def destroy(self, instance_id: int) -> None: ...


class SecretSource(Protocol):
    def read(self, name: str) -> SecretStr: ...


class JobOrchestrator:
    def __init__(
        self,
        provider: InstanceProvider,
        engine: Engine,
        secret_source: SecretSource,
        read_clock: Callable[[], datetime],
    ) -> None:
        self.provider = provider
        self.secret_source = secret_source
        self.engine = engine
        self.read_clock = read_clock

    def submit(self, spec: JobSpec) -> JobRecord:
        previous = self.read_latest(spec.name)
        if previous is not None:
            match previous.status:
                case Pending() | Running():
                    raise ValueError(f"job {spec.name!r} is not finished, so it cannot be submitted again")
                case Succeeded() | Failed() | Aborted():
                    pass
                case _:
                    assert_never(previous.status)
        secrets = {name: self.secret_source.read(name) for name in spec.secret_environment}
        pending = Pending(submitted_at=self.read_clock())
        self.record_transition(JobRecord(spec=spec, status=pending))
        rented: list[int] = []
        recorded = False
        try:
            record = JobRecord(spec=spec, status=self.rent_instance(spec, secrets, pending, rented))
            self.record_transition(record)
            recorded = True
        finally:
            if not recorded:
                self.destroy_instances(spec.name, rented)
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
        with Session(self.engine) as session:
            names = [row.name for row in session.scalars(select_latest_transitions())]
        return [self.sync(name) for name in names]

    def abandon(self, name: str) -> JobRecord:
        record = self.read(name)
        match record.status:
            case Pending(submitted_at=submitted_at):
                known_instances: list[int] = []
            case Running(submitted_at=submitted_at, instance_id=instance_id):
                known_instances = [instance_id]
            case Succeeded() | Failed() | Aborted():
                self.destroy_instances(name, [])
                return record
            case _:
                assert_never(record.status)
        return self.finish(
            record.spec,
            known_instances,
            Aborted(submitted_at=submitted_at, finished_at=self.read_clock(), reason="abandoned by the operator"),
        )

    def read(self, name: str) -> JobRecord:
        record = self.read_latest(name)
        if record is None:
            raise LookupError(f"no job has the name {name!r}")
        return record

    def read_latest(self, name: str) -> JobRecord | None:
        with Session(self.engine) as session:
            record = session.scalar(
                select(JobTransitionRow.record)
                .where(JobTransitionRow.name == name)
                .order_by(JobTransitionRow.sequence.desc())
                .limit(1)
            )
        return None if record is None else JobRecord.model_validate_json(record)

    def record_transition(self, record: JobRecord) -> None:
        with Session(self.engine) as session, session.begin():
            session.add(
                JobTransitionRow(
                    name=record.spec.name, condition=record.status.condition, record=record.model_dump_json()
                )
            )

    def rent_instance(
        self, spec: JobSpec, secrets: Mapping[str, SecretStr], pending: Pending, rented: list[int]
    ) -> Running | Aborted:
        for offer_id in self.provider.find_offers(spec):
            instance_id = self.provider.rent_offer(offer_id, spec, secrets)
            if instance_id is not None:
                rented.append(instance_id)
                return Running(submitted_at=pending.submitted_at, instance_id=instance_id)
        return Aborted(
            submitted_at=pending.submitted_at, finished_at=self.read_clock(), reason="no offer was available to rent"
        )

    def finish(self, spec: JobSpec, known_instances: Sequence[int], status: Succeeded | Failed | Aborted) -> JobRecord:
        self.destroy_instances(spec.name, known_instances)
        record = JobRecord(spec=spec, status=status)
        self.record_transition(record)
        return record

    def destroy_instances(self, job_name: str, known_instances: Sequence[int]) -> None:
        for instance_id in known_instances:
            self.provider.destroy(instance_id)
        for instance_id in self.provider.find_instances(job_name):
            if instance_id not in known_instances:
                self.provider.destroy(instance_id)
