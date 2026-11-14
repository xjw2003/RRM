import json
from abc import ABC, abstractmethod

from rrm.exceptions import GraphError
from rrm.graph.dag import ReasoningGraph
from rrm.protocol.models import RunEvent


class StorageBackend(ABC):
    @abstractmethod
    def append(self, run_id: str, expected_sequence: int, event: RunEvent) -> None:
        """Atomically append one event if the current sequence matches."""

    @abstractmethod
    def events(self, run_id: str) -> tuple[RunEvent, ...]:
        """Read a consistent, ordered snapshot of one run's events."""

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
