from collections import defaultdict
from collections.abc import Iterable
from copy import copy
from typing import Any, Self

from rrm.exceptions import GraphError
from rrm.protocol.models import (
    NodeBlock,
    NodeInvalidation,
    NodeState,
    NodeType,
    NodeUsability,
    OperationStatus,
    OperationType,
    PlanningRecord,
    ReasoningDependency,
    ReasoningNode,
    ReasoningOperation,
    ReasoningRun,
    RecoveryRecord,
    RunEvent,
    ToolConfirmation,
    UsabilityStatus,
    VerificationRecord,
    VerificationStatus,
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
        self.node_blocks: dict[str, NodeBlock] = {}
        self.verified_operations: set[str] = set()
        self._verified_snapshots: set[str] = set()
        self._verified_computations: set[str] = set()
        self.recoveries: list[RecoveryRecord] = []
        self.tool_confirmations: dict[str, ToolConfirmation] = {}
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

    def node_usability(self, node_id: str) -> NodeUsability:
        state = self.node_state(node_id)
        block = self.node_blocks.get(node_id)
        if state != NodeState.CURRENT:
            status = UsabilityStatus(state.value)
        elif block is not None:
            status = UsabilityStatus.BLOCKED
        else:
            node = self.nodes[node_id]
            op = self.operations.get(node.produced_by or "")
            if op and op.verification_required and op.operation_id not in self.verified_operations:
                status = UsabilityStatus.PENDING
            elif node.node_type == NodeType.QUESTION:
                status = UsabilityStatus.ELIGIBLE
            elif op is None or op.executor_type == "model" or op.operation_type == "REVISE":
                status = UsabilityStatus.UNVERIFIED
            elif op.operation_type == OperationType.COMPUTE:
                status = (
                    UsabilityStatus.ELIGIBLE
                    if op.operation_id in self._verified_computations
                    else UsabilityStatus.UNVERIFIED
                )
            elif node_id in self._verified_snapshots:
                status = UsabilityStatus.ELIGIBLE
            else:
                status = UsabilityStatus.UNVERIFIED
        return NodeUsability(node_id=node_id, node_state=state, status=status, block=block)

    def pending_tool_operations(self) -> tuple[str, ...]:
        return tuple(
            op.operation_id
            for op in self.operations.values()
            if op.executor_type == "tool"
            and not op.retry_safe
            and op.status == OperationStatus.INTERRUPTED
            and op.operation_id not in self.tool_confirmations
        )

    def prepare(self, event: RunEvent) -> Self:
        """Validate changes on detached collections before publishing a committed event."""
        draft = copy(self)
        if event.nodes:
            draft.nodes = self.nodes.copy()
            draft.current_versions = self.current_versions.copy()
        if event.operations:
            draft.operations = self.operations.copy()
        if event.dependencies:
            draft.dependencies = self.dependencies.copy()
            draft._parents = self._parents.copy()
            draft._children = self._children.copy()
            for edge in event.dependencies:
                start, end = edge.endpoints
                draft._children[start] = set(self._children.get(start, ()))
                draft._parents[end] = set(self._parents.get(end, ()))
        if event.invalidations:
            draft.invalidations = self.invalidations.copy()
        if event.node_blocks:
            draft.node_blocks = self.node_blocks.copy()
        if event.verified_operation_ids:
            draft.verified_operations = self.verified_operations.copy()
        if event.verifications:
            draft.verifications = self.verifications.copy()
            draft._verified_snapshots = self._verified_snapshots.copy()
            draft._verified_computations = self._verified_computations.copy()
        if event.planning is not None:
            draft.planning = self.planning.copy()
        if event.recovery is not None:
            draft.recoveries = self.recoveries.copy()
        if event.tool_confirmation is not None:
            draft.tool_confirmations = self.tool_confirmations.copy()
        draft.apply(event)
        return draft

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
                        "invocation_stop",
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
        for block in event.node_blocks:
            if block.node_id not in self.nodes or block.cause_node_id not in self.nodes:
                reject(block.node_id, GraphError("Block references a missing node"))
            else:
                self.node_blocks.setdefault(block.node_id, block)
        for op_id in event.verified_operation_ids:
            if (
                op_id not in self.operations
                or self.operations[op_id].status != OperationStatus.COMPLETED
            ):
                reject(op_id, GraphError("Verification settlement requires a completed operation"))
            else:
                self.verified_operations.add(op_id)
        self.verifications.extend(event.verifications)
        for record in event.verifications:
            if record.status != VerificationStatus.PASSED:
                continue
            if record.verification_type == "provenance_snapshot":
                self._verified_snapshots.add(record.target_id)
            elif record.verification_type == "deterministic":
                op = self.operations.get(record.target_id)
                if (
                    op
                    and op.operation_type == OperationType.COMPUTE
                    and all(
                        node_id in self.nodes
                        and self.node_usability(node_id).status == UsabilityStatus.ELIGIBLE
                        for node_id in op.provided_node_ids
                    )
                ):
                    self._verified_computations.add(record.target_id)
        if event.planning is not None:
            self.planning.append(event.planning)
        if event.recovery is not None:
            self.recoveries.append(event.recovery)
        if event.tool_confirmation is not None:
            op_id = event.tool_confirmation.operation_id
            if op_id not in self.pending_tool_operations():
                reject(op_id, GraphError("Tool confirmation requires a pending operation"))
            else:
                self.tool_confirmations[op_id] = event.tool_confirmation
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
            "node_blocks": [block.model_dump(mode="json") for block in self.node_blocks.values()],
            "verified_operation_ids": sorted(self.verified_operations),
            "verifications": [item.model_dump(mode="json") for item in self.verifications],
            "recoveries": [item.model_dump(mode="json") for item in self.recoveries],
            "pending_tool_operation_ids": self.pending_tool_operations(),
            "tool_confirmations": [
                item.model_dump(mode="json") for item in self.tool_confirmations.values()
            ],
        }
