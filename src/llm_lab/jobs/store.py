import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from llm_lab.jobs.protocol import JobRecord


@dataclass(frozen=True)
class JobTransition:
    recorded_at: datetime
    record: JobRecord


class SqliteJobRecordStore:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        with self.connection:
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS job_transitions ("
                "sequence INTEGER PRIMARY KEY AUTOINCREMENT, "
                "name TEXT NOT NULL, "
                "recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                "record TEXT NOT NULL)"
            )
            self.connection.execute(
                "CREATE INDEX IF NOT EXISTS job_transitions_by_name ON job_transitions (name, sequence)"
            )

    def append(self, record: JobRecord) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO job_transitions (name, record) VALUES (?, ?)", (record.spec.name, record.model_dump_json())
            )

    def read_latest(self, name: str) -> JobRecord | None:
        row: tuple[str] | None = self.connection.execute(
            "SELECT record FROM job_transitions WHERE name = ? ORDER BY sequence DESC LIMIT 1", (name,)
        ).fetchone()
        return None if row is None else JobRecord.model_validate_json(row[0])

    def list_latest(self) -> Sequence[JobRecord]:
        rows: list[tuple[str]] = self.connection.execute(
            "SELECT record FROM job_transitions "
            "WHERE sequence IN (SELECT MAX(sequence) FROM job_transitions GROUP BY name) ORDER BY name"
        ).fetchall()
        return [JobRecord.model_validate_json(record) for (record,) in rows]

    def read_history(self, name: str) -> Sequence[JobTransition]:
        rows: list[tuple[str, str]] = self.connection.execute(
            "SELECT recorded_at, record FROM job_transitions WHERE name = ? ORDER BY sequence", (name,)
        ).fetchall()
        return [
            JobTransition(
                datetime.fromisoformat(recorded_at).replace(tzinfo=UTC), JobRecord.model_validate_json(record)
            )
            for recorded_at, record in rows
        ]
