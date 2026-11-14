import sqlite3
from contextlib import closing
from os import PathLike
from types import TracebackType
from typing import Self

from rrm.exceptions import StorageConflict
from rrm.protocol.models import RunEvent
from rrm.storage.base import StorageBackend


class SQLiteStorage(StorageBackend):
    def __init__(self, path: str | PathLike[str]) -> None:
        self._connection = sqlite3.connect(path, isolation_level="IMMEDIATE")
        self._connection.execute("""
            CREATE TABLE IF NOT EXISTS rrm_events (
                run_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY (run_id, sequence)
            ) WITHOUT ROWID
        """)

    def append(self, run_id: str, expected_sequence: int, event: RunEvent) -> None:
        payload = event.model_dump_json()
        with self._connection:
            self._connection.execute("BEGIN IMMEDIATE")
            with closing(
                self._connection.execute(
                    "SELECT COALESCE(MAX(sequence), 0) FROM rrm_events WHERE run_id = ?",
                    (run_id,),
                )
            ) as cursor:
                sequence = cursor.fetchone()[0]
            if sequence != expected_sequence:
                raise StorageConflict(f"Run {run_id} was updated concurrently")
            self._connection.execute(
                "INSERT INTO rrm_events VALUES (?, ?, ?)", (run_id, sequence + 1, payload)
            )

    def events(self, run_id: str) -> tuple[RunEvent, ...]:
        with closing(
            self._connection.execute(
                "SELECT payload FROM rrm_events WHERE run_id = ? ORDER BY sequence", (run_id,)
            )
        ) as cursor:
            return tuple(RunEvent.model_validate_json(payload) for (payload,) in cursor)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
