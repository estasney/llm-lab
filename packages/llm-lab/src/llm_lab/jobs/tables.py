from datetime import datetime

from sqlalchemy import Index, Select, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


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


def select_latest_transitions() -> Select[JobTransitionRow]:
    latest_sequences = select(func.max(JobTransitionRow.sequence)).group_by(JobTransitionRow.name)
    return (
        select(JobTransitionRow).where(JobTransitionRow.sequence.in_(latest_sequences)).order_by(JobTransitionRow.name)
    )
