import json
from abc import ABC, abstractmethod

from rrm.exceptions import GraphError
from rrm.graph.dag import ReasoningGraph
from rrm.protocol.models import (
    Conversation,
    ConversationTurn,
    CrossRunReference,
    RunEvent,
    SourceNode,
)


class StorageBackend(ABC):
    @abstractmethod
    def append(self, run_id: str, expected_sequence: int, event: RunEvent) -> None:
        """Atomically append one event if the current sequence matches."""

    @abstractmethod
    def events(self, run_id: str) -> tuple[RunEvent, ...]:
        """Read a consistent, ordered snapshot of one run's events."""

    def create_conversation(self, conversation: Conversation) -> None:
        """Create a conversation if its ID is absent."""
        raise NotImplementedError("This backend does not support conversations")

    def get_conversation(self, conversation_id: str) -> Conversation:
        raise NotImplementedError("This backend does not support conversations")

    def conversation_turns(self, conversation_id: str) -> tuple[ConversationTurn, ...]:
        raise NotImplementedError("This backend does not support conversations")

    def get_turn_by_run(self, run_id: str) -> ConversationTurn | None:
        raise NotImplementedError("This backend does not support conversations")

    def references(self, run_id: str) -> tuple[CrossRunReference, ...]:
        return tuple(
            event.cross_run_reference
            for event in self.events(run_id)
            if event.cross_run_reference is not None
        )

    def references_to(self, source: SourceNode) -> tuple[CrossRunReference, ...]:
        raise NotImplementedError("This backend does not support conversations")

    def get_graph(
        self, run_id: str, *, through_sequence: int | None = None, audit: bool = False
    ) -> ReasoningGraph:
        events = self.events(run_id)
        if not events:
            raise KeyError(run_id)
        if events[0].run is None or events[0].run.run_id != run_id:
            raise GraphError(f"Invalid creation event for run {run_id}")
        if through_sequence is not None:
            if not 1 <= through_sequence <= len(events):
                raise ValueError("through_sequence must identify an existing event")
            events = events[:through_sequence]
        return ReasoningGraph.from_events(events, audit=audit)

    def export_json(self, run_id: str) -> str:
        return json.dumps(self.get_graph(run_id).to_dict(), ensure_ascii=False, indent=2)
