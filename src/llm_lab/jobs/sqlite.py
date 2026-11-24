from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import Engine, Index, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from llm_lab.jobs.protocol import JobRecord


class Base(DeclarativeBase):
    pass


class JobTransitionRow(Base):
    __tablename__ = "job_transitions"
    __table_args__ = (Index("job_transitions_by_name", "name", "sequence"), {"sqlite_autoincrement": True})

    sequence: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column()
    condition: Mapped[str] = mapped_column()
    recorded_at: Mapped[datetime] = mapped_column(server_default=func.now())
    record: Mapped[str] = mapped_column()


@dataclass(frozen=True)
class JobTransition:
    recorded_at: datetime
    record: JobRecord


@dataclass(frozen=True)
class SqliteJobRecordStore:
    engine: Engine

    def append(self, record: JobRecord) -> None:
        with Session(self.engine) as session, session.begin():
            session.add(
                JobTransitionRow(
                    name=record.spec.name, condition=record.status.condition, record=record.model_dump_json()
                )
            )

    def read_latest(self, name: str) -> JobRecord | None:
        with Session(self.engine) as session:
            record = session.scalar(
                select(JobTransitionRow.record)
                .where(JobTransitionRow.name == name)
                .order_by(JobTransitionRow.sequence.desc())
                .limit(1)
            )
            return None if record is None else JobRecord.model_validate_json(record)

    def list_latest(self) -> Sequence[JobRecord]:
        latest_sequences = select(func.max(JobTransitionRow.sequence)).group_by(JobTransitionRow.name)
        with Session(self.engine) as session:
            records = session.scalars(
                select(JobTransitionRow.record)
                .where(JobTransitionRow.sequence.in_(latest_sequences))
                .order_by(JobTransitionRow.name)
            )
            return [JobRecord.model_validate_json(record) for record in records]

    def read_history(self, name: str) -> Sequence[JobTransition]:
        with Session(self.engine) as session:
            rows = session.scalars(
                select(JobTransitionRow).where(JobTransitionRow.name == name).order_by(JobTransitionRow.sequence)
            )
            return [
                JobTransition(row.recorded_at.replace(tzinfo=UTC), JobRecord.model_validate_json(row.record))
                for row in rows
            ]
