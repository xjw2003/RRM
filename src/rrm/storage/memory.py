from threading import Lock

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


class MemoryStorage(StorageBackend):
    def __init__(self) -> None:
        self._streams: dict[str, list[str]] = {}
        self._lock = Lock()
        self._conversations: dict[str, Conversation] = {}
        self._turns: dict[str, ConversationTurn] = {}
        self._references: dict[tuple[str, str], list[CrossRunReference]] = {}

    def append(self, run_id: str, expected_sequence: int, event: RunEvent) -> None:
        payload = event.model_dump_json()
        with self._lock:
            stream = self._streams.get(run_id, [])
            if len(stream) != expected_sequence:
                raise StorageConflict(f"Run {run_id} was updated concurrently")
            turn = event.turn
            reference = event.cross_run_reference
            if turn is not None:
                conversation = self._conversations.get(turn.conversation_id)
                next_index = 1 + sum(
                    t.conversation_id == turn.conversation_id for t in self._turns.values()
                )
                if (
                    conversation is None
                    or turn.run_id != run_id
                    or expected_sequence != 0
                    or turn.turn_index != next_index
                    or run_id in self._turns
                ):
                    raise StorageConflict("Conversation changed while creating the turn")
            if reference is not None:
                check_reference(
                    reference,
                    self._turns.get(run_id),
                    self._turns.get(reference.source_run_id),
                    len(self._streams.get(reference.source_run_id, ())),
                )
                key = (reference.source_run_id, reference.source_node_id)
                if any(ref.target_run_id == run_id for ref in self._references.get(key, ())):
                    raise StorageConflict("This node was already imported into the run")
            check_import_event(run_id, event)
            self._streams.setdefault(run_id, []).append(payload)
            if turn is not None:
                self._turns[run_id] = turn
                self._conversations[turn.conversation_id] = conversation.model_copy(
                    update={"updated_at": max(conversation.updated_at, turn.created_at)},
                )
            if reference is not None:
                self._references.setdefault(key, []).append(reference)

    def events(self, run_id: str) -> tuple[RunEvent, ...]:
        with self._lock:
            payloads = tuple(self._streams.get(run_id, ()))
        return tuple(RunEvent.model_validate_json(payload) for payload in payloads)

    def create_conversation(self, conversation: Conversation) -> None:
        with self._lock:
            self._conversations.setdefault(conversation.conversation_id, conversation)

    def get_conversation(self, conversation_id: str) -> Conversation:
        with self._lock:
            return self._conversations[conversation_id]

    def conversation_turns(self, conversation_id: str) -> tuple[ConversationTurn, ...]:
        with self._lock:
            return tuple(t for t in self._turns.values() if t.conversation_id == conversation_id)

    def get_turn_by_run(self, run_id: str) -> ConversationTurn | None:
        with self._lock:
            return self._turns.get(run_id)

    def references_to(self, source: SourceNode) -> tuple[CrossRunReference, ...]:
        with self._lock:
            return tuple(self._references.get((source.source_run_id, source.source_node_id), ()))
