from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import Dialect, Index, Text, TypeDecorator, func, select
from sqlalchemy.orm import Mapped, Session, mapped_column, sessionmaker

from llm_lab.database import Base
from llm_lab.jobs.protocol import JobRecord


class JobRecordJson(TypeDecorator[JobRecord]):
    impl = Text
    cache_ok = True

    def process_bind_param(self, value: JobRecord | None, dialect: Dialect) -> str | None:
        return None if value is None else value.model_dump_json()

    def process_result_value(self, value: str | None, dialect: Dialect) -> JobRecord | None:
        return None if value is None else JobRecord.model_validate_json(value)


class JobTransitionRow(Base):
    __tablename__ = "job_transitions"
    __table_args__ = (Index("job_transitions_by_name", "name", "sequence"), {"sqlite_autoincrement": True})

    sequence: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column()
    condition: Mapped[str] = mapped_column()
    recorded_at: Mapped[datetime] = mapped_column(server_default=func.now())
    record: Mapped[JobRecord] = mapped_column(JobRecordJson)


@dataclass(frozen=True)
class JobTransition:
    recorded_at: datetime
    record: JobRecord


class SqlAlchemyJobRecordStore:
    def __init__(self, open_session: sessionmaker[Session]) -> None:
        self.open_session = open_session

    def append(self, record: JobRecord) -> None:
        with self.open_session.begin() as session:
            session.add(JobTransitionRow(name=record.spec.name, condition=record.status.condition, record=record))

    def read_latest(self, name: str) -> JobRecord | None:
        with self.open_session() as session:
            return session.scalars(
                select(JobTransitionRow.record)
                .where(JobTransitionRow.name == name)
                .order_by(JobTransitionRow.sequence.desc())
                .limit(1)
            ).one_or_none()

    def list_latest(self) -> Sequence[JobRecord]:
        latest_sequences = select(func.max(JobTransitionRow.sequence)).group_by(JobTransitionRow.name)
        with self.open_session() as session:
            return session.scalars(
                select(JobTransitionRow.record)
                .where(JobTransitionRow.sequence.in_(latest_sequences))
                .order_by(JobTransitionRow.name)
            ).all()

    def read_history(self, name: str) -> Sequence[JobTransition]:
        with self.open_session() as session:
            rows = session.scalars(
                select(JobTransitionRow).where(JobTransitionRow.name == name).order_by(JobTransitionRow.sequence)
            ).all()
            return [JobTransition(row.recorded_at.replace(tzinfo=UTC), row.record) for row in rows]
