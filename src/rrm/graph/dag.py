from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Self

from rrm.exceptions import GraphError
from rrm.protocol.models import (
    NodeInvalidation,
    NodeState,
    OperationStatus,
    PlanningRecord,
    ReasoningDependency,
    ReasoningNode,
    ReasoningOperation,
    ReasoningRun,
    RunEvent,
    VerificationRecord,
)


class ReasoningGraph:
    def __init__(self, run: ReasoningRun) -> None:
        self.run = run
        self.nodes: dict[str, ReasoningNode] = {}
        self.operations: dict[str, ReasoningOperation] = {}
        self.planning: list[PlanningRecord] = []
        self.dependencies: list[ReasoningDependency] = []
        self.invalidations: dict[str, NodeInvalidation] = {}
        self.current_versions: dict[str, str] = {}
        self.verifications: list[VerificationRecord] = []
        self.audit_issues: list[tuple[str, str]] = []
        self.sequence = 0
        self._parents: dict[str, set[str]] = defaultdict(set)
        self._children: dict[str, set[str]] = defaultdict(set)

    def add_node(self, node: ReasoningNode) -> None:
        if node.run_id != self.run.run_id or (
            node.node_id in self.nodes or node.node_id in self.operations
        ):
            raise GraphError(f"Foreign or duplicate node {node.node_id}")
        logical_id = node.logical_id or node.node_id
        previous_id = self.current_versions.get(logical_id)
        if previous_id != node.supersedes_node_id or node.revision != (
            self.nodes[previous_id].revision + 1 if previous_id is not None else 1
        ):
            raise GraphError(f"Invalid revision chain for node {node.node_id}")
        self.nodes[node.node_id] = node
        self.current_versions[logical_id] = node.node_id

    def add_operation(self, operation: ReasoningOperation) -> None:
        if (
            operation.run_id != self.run.run_id
            or operation.operation_id in self.nodes
            or operation.operation_id in self.operations
        ):
            raise GraphError(f"Foreign or duplicate operation {operation.operation_id}")
        self.operations[operation.operation_id] = operation

    def add_dependency(self, edge: ReasoningDependency) -> None:
        if (
            edge.run_id != self.run.run_id
            or edge.node_id not in self.nodes
            or edge.operation_id not in self.operations
        ):
            raise GraphError("Dependency endpoints must exist in the same run")
        start, end = edge.endpoints
        if end in self._children.get(start, ()):
            raise GraphError(f"Duplicate dependency {start} -> {end}")
        if start in self.descendants(end):
            raise GraphError(f"Cyclic dependency {start} -> {end}")
        self.dependencies.append(edge)
        self._children[start].add(end)
        self._parents[end].add(start)

    def parents(self, vertex_id: str) -> frozenset[str]:
        return frozenset(self._parents.get(vertex_id, ()))

    def children(self, vertex_id: str) -> frozenset[str]:
        return frozenset(self._children.get(vertex_id, ()))

    @staticmethod
    def _walk(vertex_id: str, links: dict[str, set[str]]) -> set[str]:
        visited: set[str] = set()
        pending = list(links.get(vertex_id, ()))
        while pending:
            item = pending.pop()
            if item not in visited:
                visited.add(item)
                pending.extend(links.get(item, ()))
        return visited

    def ancestors(self, vertex_id: str) -> set[str]:
        return self._walk(vertex_id, self._parents)

    def descendants(self, vertex_id: str) -> set[str]:
        return self._walk(vertex_id, self._children)

    def trace(self, node_id: str) -> tuple[ReasoningDependency, ...]:
        self.nodes[node_id]
        path = self.ancestors(node_id) | {node_id}
        return tuple(edge for edge in self.dependencies if set(edge.endpoints) <= path)

    def node_state(self, node_id: str) -> NodeState:
        node = self.nodes[node_id]
        invalidation = self.invalidations.get(node_id)
        if invalidation is not None:
            return invalidation.state
        if self.current_versions[node.logical_id or node_id] != node_id:
            return NodeState.SUPERSEDED
        return NodeState.CURRENT

    def apply(self, event: RunEvent, *, audit: bool = False) -> None:
        def reject(target_id: str, problem: GraphError) -> None:
            if not audit:
                raise problem
            self.audit_issues.append((target_id, str(problem)))

        if event.run is not None:
            if event.run.run_id != self.run.run_id:
                reject(self.run.run_id, GraphError("Foreign run event"))
            else:
                self.run = event.run
        for node in event.nodes:
            try:
                self.add_node(node)
            except GraphError as exc:
                reject(node.node_id, exc)
        for operation in event.operations:
            operation_id = operation.operation_id
            previous = self.operations.get(operation_id)
            try:
                if previous is None:
                    self.add_operation(operation)
                elif previous.status != OperationStatus.RUNNING:
                    raise GraphError(f"Operation {operation_id} was already finalized")
                elif operation.run_id != self.run.run_id:
                    raise GraphError(f"Foreign operation {operation_id}")
                else:
                    self.operations[operation_id] = operation
                    final_fields = {
                        "status",
                        "completed_at",
                        "output_node_ids",
                        "output_snapshot_hash",
                        "error",
                    }
                    if operation.status == OperationStatus.RUNNING or (
                        previous.model_dump(exclude=final_fields)
                        != operation.model_dump(exclude=final_fields)
                    ):
                        raise GraphError(f"Operation {operation_id} changed its execution record")
            except GraphError as exc:
                if not audit and previous is not None:
                    self.operations[operation_id] = previous
                reject(operation_id, exc)
        for edge in event.dependencies:
            try:
                self.add_dependency(edge)
            except GraphError as exc:
                reject(edge.operation_id, exc)
        for invalidation in event.invalidations:
            if (
                invalidation.node_id not in self.nodes
                or invalidation.cause_node_id not in self.nodes
            ):
                reject(invalidation.node_id, GraphError("Invalidation references a missing node"))
            else:
                self.invalidations[invalidation.node_id] = invalidation
        self.verifications.extend(event.verifications)
        if event.planning is not None:
            self.planning.append(event.planning)
        self.sequence += 1

    @classmethod
    def from_events(cls, events: Iterable[RunEvent], *, audit: bool = False) -> Self:
        stream = iter(events)
        first = next(stream, None)
        if first is None or first.run is None:
            raise KeyError("Run not found")
        graph = cls(first.run)
        graph.apply(first, audit=audit)
        for event in stream:
            graph.apply(event, audit=audit)
        return graph

    def to_dict(self) -> dict[str, Any]:
        return {
            "run": self.run.model_dump(mode="json"),
            "nodes": [node.model_dump(mode="json") for node in self.nodes.values()],
            "operations": [op.model_dump(mode="json") for op in self.operations.values()],
            "dependencies": [edge.model_dump(mode="json") for edge in self.dependencies],
            "planning": [plan.model_dump(mode="json") for plan in self.planning],
            "current_versions": dict(self.current_versions),
            "node_states": {node_id: self.node_state(node_id) for node_id in self.nodes},
            "invalidations": [item.model_dump(mode="json") for item in self.invalidations.values()],
            "verifications": [item.model_dump(mode="json") for item in self.verifications],
        }
