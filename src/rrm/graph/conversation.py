from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from rrm.exceptions import GraphError
from rrm.graph.dag import ReasoningGraph
from rrm.protocol.models import Conversation, CrossRunReference

Vertex = tuple[str, str]


class ConversationGraph:
    """Historical DAG with run-qualified vertices and explicit IMPORT edges."""

    def __init__(self, conversation: Conversation, graphs: Iterable[ReasoningGraph]) -> None:
        self.conversation = conversation
        self.runs: dict[str, ReasoningGraph] = {}
        self.references: list[CrossRunReference] = []
        self._parents: dict[Vertex, set[Vertex]] = defaultdict(set)
        self._children: dict[Vertex, set[Vertex]] = defaultdict(set)
        last_index = 0
        for graph in graphs:
            turn = graph.turn
            if (
                turn is None
                or turn.conversation_id != conversation.conversation_id
                or (turn.turn_index <= last_index or turn.run_id in self.runs)
            ):
                raise GraphError("Invalid conversation turn order")
            last_index = turn.turn_index
            for ref in graph.references.values():
                source = self.runs.get(ref.source_run_id)
                node = source.nodes.get(ref.source_node_id) if source else None
                if source is None or node is None or ref.target_run_id != turn.run_id:
                    raise GraphError(
                        "Cross-run edge must reference an existing node in an earlier turn"
                    )
                self.references.append(ref)
                self._connect(
                    (ref.source_run_id, ref.source_node_id),
                    (ref.target_run_id, ref.import_operation_id),
                )
            for edge in graph.dependencies:
                start, end = edge.endpoints
                self._connect((turn.run_id, start), (turn.run_id, end))
            self.runs[turn.run_id] = graph

    def _connect(self, start: Vertex, end: Vertex) -> None:
        self._parents[end].add(start)
        self._children[start].add(end)

    def parents(self, vertex: Vertex) -> frozenset[Vertex]:
        return frozenset(self._parents.get(vertex, ()))

    def children(self, vertex: Vertex) -> frozenset[Vertex]:
        return frozenset(self._children.get(vertex, ()))

    def ancestors(self, vertex: Vertex) -> set[Vertex]:
        visited: set[Vertex] = set()
        pending = list(self.parents(vertex))
        while pending:
            item = pending.pop()
            if item not in visited:
                visited.add(item)
                pending.extend(self.parents(item))
        return visited

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation": self.conversation.model_dump(mode="json"),
            "runs": {run_id: graph.to_dict() for run_id, graph in self.runs.items()},
            "references": [ref.model_dump(mode="json") for ref in self.references],
        }
