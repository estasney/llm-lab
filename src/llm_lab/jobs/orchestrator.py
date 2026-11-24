from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Protocol, assert_never

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
from llm_lab.settings import Settings


class InstanceProvider(Protocol):
    def find_offers(self, spec: JobSpec) -> Sequence[int]: ...

    def rent_offer(self, offer_id: int, spec: JobSpec) -> int | None: ...

    def observe(self, instance_id: int) -> Active | Completed | Halted: ...

    def find_instances(self, job_name: str) -> Sequence[int]: ...

    def destroy(self, instance_id: int) -> None: ...


class JobRecordStore(Protocol):
    def write(self, record: JobRecord) -> None: ...

    def read(self, name: str) -> JobRecord: ...


class FileJobRecordStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def write(self, record: JobRecord) -> None:
        self.settings.job_records_directory.mkdir(parents=True, exist_ok=True)
        (self.settings.job_records_directory / f"{record.spec.name}.json").write_text(record.model_dump_json(indent=2))

    def read(self, name: str) -> JobRecord:
        return JobRecord.model_validate_json((self.settings.job_records_directory / f"{name}.json").read_text())


class JobOrchestrator:
    def __init__(
        self,
        settings: Settings,
        provider: InstanceProvider,
        store: JobRecordStore,
        read_clock: Callable[[], datetime],
        sleep: Callable[[float], None],
    ) -> None:
        self.settings = settings
        self.provider = provider
        self.store = store
        self.read_clock = read_clock
        self.sleep = sleep

    def execute(self, spec: JobSpec) -> JobRecord:
        pending = Pending(submitted_at=self.read_clock())
        self.store.write(JobRecord(spec=spec, status=pending))
        rented: list[int] = []
        try:
            status = self.run_on_rented_instance(spec, pending, rented)
        finally:
            self.destroy_instances(spec.name, rented)
        record = JobRecord(spec=spec, status=status)
        self.store.write(record)
        return record

    def abandon(self, name: str) -> JobRecord:
        record = self.store.read(name)
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
        self.destroy_instances(name, known_instances)
        abandoned = JobRecord(
            spec=record.spec,
            status=Aborted(
                submitted_at=submitted_at, finished_at=self.read_clock(), reason="abandoned by the operator"
            ),
        )
        self.store.write(abandoned)
        return abandoned

    def run_on_rented_instance(
        self, spec: JobSpec, pending: Pending, rented: list[int]
    ) -> Succeeded | Failed | Aborted:
        for offer_id in self.provider.find_offers(spec):
            instance_id = self.provider.rent_offer(offer_id, spec)
            if instance_id is not None:
                rented.append(instance_id)
                running = Running(submitted_at=pending.submitted_at, instance_id=instance_id)
                self.store.write(JobRecord(spec=spec, status=running))
                return conclude(running, self.read_clock(), self.wait_for_outcome(instance_id))
        return Aborted(
            submitted_at=pending.submitted_at, finished_at=self.read_clock(), reason="no offer was available to rent"
        )

    def wait_for_outcome(self, instance_id: int) -> Completed | Halted:
        while True:
            observation = self.provider.observe(instance_id)
            match observation:
                case Active():
                    self.sleep(self.settings.job_poll_interval_seconds)
                case Completed() | Halted():
                    return observation
                case _:
                    assert_never(observation)

    def destroy_instances(self, job_name: str, known_instances: Sequence[int]) -> None:
        for instance_id in known_instances:
            self.provider.destroy(instance_id)
        for instance_id in self.provider.find_instances(job_name):
            if instance_id not in known_instances:
                self.provider.destroy(instance_id)
