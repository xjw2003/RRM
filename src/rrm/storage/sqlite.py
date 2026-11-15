import sqlite3
from contextlib import closing
from os import PathLike
from types import TracebackType
from typing import Self

from rrm.exceptions import StorageConflict
from rrm.protocol.models import (
    Conversation,
    ConversationTurn,
    CrossRunReference,
    RunEvent,
    SourceNode,
)
from rrm.storage.base import StorageBackend
from rrm.storage.conversation import check_import_event, check_reference


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

        self._connection.executescript("""
            CREATE TABLE IF NOT EXISTS rrm_conversations (
                conversation_id TEXT PRIMARY KEY, payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS rrm_conversation_turns (
                run_id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL,
                turn_index INTEGER NOT NULL, payload TEXT NOT NULL,
                UNIQUE (conversation_id, turn_index)
            );
            CREATE TABLE IF NOT EXISTS rrm_cross_run_refs (
                source_run_id TEXT NOT NULL, source_node_id TEXT NOT NULL,
                target_run_id TEXT NOT NULL, sequence INTEGER NOT NULL,
                PRIMARY KEY (source_run_id, source_node_id, target_run_id)
            ) WITHOUT ROWID;
        """)

    def _sequence(self, run_id: str) -> int:
        with closing(
            self._connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) FROM rrm_events WHERE run_id = ?", (run_id,)
            )
        ) as cursor:
            return cursor.fetchone()[0]

    def append(self, run_id: str, expected_sequence: int, event: RunEvent) -> None:
        payload = event.model_dump_json()
        with self._connection:
            self._connection.execute("BEGIN IMMEDIATE")
            sequence = self._sequence(run_id)
            if sequence != expected_sequence:
                raise StorageConflict(f"Run {run_id} was updated concurrently")
            self._connection.execute(
                "INSERT INTO rrm_events VALUES (?, ?, ?)", (run_id, sequence + 1, payload)
            )

            if event.turn is not None:
                self._insert_turn(event.turn, run_id, expected_sequence)
            if event.cross_run_reference is not None:
                ref = event.cross_run_reference
                check_reference(
                    ref,
                    self.get_turn_by_run(run_id),
                    self.get_turn_by_run(ref.source_run_id),
                    self._sequence(ref.source_run_id),
                )
                self._connection.execute(
                    "INSERT INTO rrm_cross_run_refs VALUES (?, ?, ?, ?)",
                    (ref.source_run_id, ref.source_node_id, run_id, sequence + 1),
                )

            check_import_event(run_id, event)

    def _insert_turn(self, turn: ConversationTurn, run_id: str, expected_sequence: int) -> None:
        conversation = self.get_conversation(turn.conversation_id)
        with closing(
            self._connection.execute(
                "SELECT COALESCE(MAX(turn_index), 0) FROM rrm_conversation_turns WHERE conversation_id = ?",
                (turn.conversation_id,),
            )
        ) as cursor:
            next_index = cursor.fetchone()[0] + 1
        if turn.run_id != run_id or expected_sequence != 0 or turn.turn_index != next_index:
            raise StorageConflict("Conversation changed while creating the turn")
        self._connection.execute(
            "INSERT INTO rrm_conversation_turns VALUES (?, ?, ?, ?)",
            (
                run_id,
                turn.conversation_id,
                turn.turn_index,
                turn.model_dump_json(),
            ),
        )
        updated = conversation.model_copy(
            update={"updated_at": max(conversation.updated_at, turn.created_at)}
        )
        self._connection.execute(
            "UPDATE rrm_conversations SET payload = ? WHERE conversation_id = ?",
            (
                updated.model_dump_json(),
                turn.conversation_id,
            ),
        )

    def create_conversation(self, conversation: Conversation) -> None:
        with self._connection:
            self._connection.execute(
                "INSERT OR IGNORE INTO rrm_conversations VALUES (?, ?)",
                (
                    conversation.conversation_id,
                    conversation.model_dump_json(),
                ),
            )

    def get_conversation(self, conversation_id: str) -> Conversation:
        with closing(
            self._connection.execute(
                "SELECT payload FROM rrm_conversations WHERE conversation_id = ?",
                (conversation_id,),
            )
        ) as cursor:
            row = cursor.fetchone()
        if row is None:
            raise KeyError(conversation_id)
        return Conversation.model_validate_json(row[0])

    def conversation_turns(self, conversation_id: str) -> tuple[ConversationTurn, ...]:
        with closing(
            self._connection.execute(
                "SELECT payload FROM rrm_conversation_turns WHERE conversation_id = ? ORDER BY turn_index",
                (conversation_id,),
            )
        ) as cursor:
            return tuple(ConversationTurn.model_validate_json(payload) for (payload,) in cursor)

    def get_turn_by_run(self, run_id: str) -> ConversationTurn | None:
        with closing(
            self._connection.execute(
                "SELECT payload FROM rrm_conversation_turns WHERE run_id = ?",
                (run_id,),
            )
        ) as cursor:
            row = cursor.fetchone()
        return ConversationTurn.model_validate_json(row[0]) if row else None

    def references_to(self, source: SourceNode) -> tuple[CrossRunReference, ...]:
        with closing(
            self._connection.execute(
                """SELECT e.payload FROM rrm_cross_run_refs AS r JOIN rrm_events AS e
               ON e.run_id = r.target_run_id AND e.sequence = r.sequence
               WHERE r.source_run_id = ? AND r.source_node_id = ? ORDER BY r.target_run_id""",
                (source.source_run_id, source.source_node_id),
            )
        ) as cursor:
            return tuple(
                ref
                for (payload,) in cursor
                if (ref := RunEvent.model_validate_json(payload).cross_run_reference) is not None
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
