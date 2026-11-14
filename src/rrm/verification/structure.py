from collections import defaultdict
from collections.abc import Iterator

from rrm.graph.dag import ReasoningGraph
from rrm.protocol.models import (
    NodeState,
    NodeType,
    OperationStatus,
    ReasoningOperation,
    RunStatus,
    VerificationRecord,
    VerificationStatus,
)
from rrm.protocol.validation import content_hash, node_input, snapshot_hash
from rrm.verification.base import TargetType, Verifier


class StructuralVerifier(Verifier):
    verifier_id = "rrm.structure"

    def _check(self, target: str, kind: TargetType, problems: list[str]) -> VerificationRecord:
        status = VerificationStatus.FAILED if problems else VerificationStatus.PASSED
        return self.record(target, kind, "structure", status, problems=problems)

    @staticmethod
    def _operation(graph: ReasoningGraph, op: ReasoningOperation) -> list[str]:
        problems = []
        if op.run_id != graph.run.run_id:
            problems.append("Operation belongs to a different run")
        if op.status != OperationStatus.COMPLETED or op.completed_at is None or op.error:
            problems.append(f"Operation did not complete successfully ({op.status})")
        elif op.completed_at < op.started_at:
            problems.append("Operation timestamps are out of order")
        if op.input_node_ids != op.provided_node_ids:
            problems.append("Declared inputs differ from the provided inputs")
        for label, ids, adjacent, saved_hash in (
            ("input", op.provided_node_ids, graph.parents(op.operation_id), op.input_snapshot_hash),
            (
                "output",
                op.output_node_ids,
                graph.children(op.operation_id),
                op.output_snapshot_hash,
            ),
        ):
            if len(ids) != len(set(ids)) or set(ids) != adjacent:
                problems.append(f"{label.capitalize()} IDs do not match dependency edges")
            missing = [node_id for node_id in ids if node_id not in graph.nodes]
            if missing:
                problems.append(f"Missing {label} nodes: {', '.join(missing)}")
                continue
            nodes = [graph.nodes[node_id] for node_id in ids]
            if any(node.run_id != op.run_id for node in nodes):
                problems.append(f"{label.capitalize()} belongs to a different run")
            if (label == "input" or op.status == OperationStatus.COMPLETED) and (
                snapshot_hash([node_input(node) for node in nodes]) != saved_hash
            ):
                problems.append(f"{label.capitalize()} snapshot hash mismatch")
        missing_declared = set(op.input_node_ids) - graph.nodes.keys()
        if missing_declared:
            problems.append(f"Missing declared inputs: {', '.join(sorted(missing_declared))}")
        if any(
            graph.nodes[node_id].produced_by != op.operation_id
            for node_id in op.output_node_ids
            if node_id in graph.nodes
        ):
            problems.append("Output names a different producing operation")
        return problems

    def verify(self, graph: ReasoningGraph) -> Iterator[VerificationRecord]:
        issues: dict[str, list[str]] = defaultdict(list)
        for target_id, error in graph.audit_issues:
            issues[target_id].append(error)

        # Kahn's algorithm also checks graphs supplied directly to the verifier.
        incoming = {
            vertex: len(graph.parents(vertex)) for vertex in (*graph.nodes, *graph.operations)
        }
        roots = [vertex for vertex, degree in incoming.items() if degree == 0]
        count = 0
        while roots:
            count += 1
            for child in graph.children(roots.pop()):
                if child not in incoming:
                    continue
                incoming[child] -= 1
                if incoming[child] == 0:
                    roots.append(child)
        run_problems = list(issues[graph.run.run_id])
        if count != len(incoming) or any("Cyclic" in error for _, error in graph.audit_issues):
            run_problems.append("Execution graph contains a cycle")
        for target_id, error in graph.audit_issues:
            if target_id not in graph.nodes and target_id not in graph.operations:
                if target_id != graph.run.run_id:
                    run_problems.append(f"{target_id}: {error}")
        if graph.audit_issues:
            run_problems.append("Event stream contains inconsistent records")
        if graph.run.status == RunStatus.COMPLETED and not graph.run.final_node_ids:
            run_problems.append("Completed run has no final node")
        for node_id in graph.run.final_node_ids:
            node = graph.nodes.get(node_id)
            if node is None or node.node_type != NodeType.RESULT:
                run_problems.append(f"Invalid final node {node_id}")
            elif graph.node_state(node_id) != NodeState.CURRENT:
                run_problems.append(f"Final node {node_id} is not current")
        yield self._check(graph.run.run_id, "run", run_problems)

        for op_id, op in graph.operations.items():
            yield self._check(op_id, "operation", self._operation(graph, op) + issues[op_id])
            unusable = [
                node_id
                for node_id in (*op.provided_node_ids, *op.output_node_ids)
                if node_id in graph.nodes and graph.node_state(node_id) != NodeState.CURRENT
            ]
            yield self.record(
                op_id,
                "operation",
                "current_validity",
                VerificationStatus.FAILED if unusable else VerificationStatus.PASSED,
                unavailable_node_ids=unusable,
            )

        for node_id, node in graph.nodes.items():
            problems = list(issues[node_id])
            if node.run_id != graph.run.run_id:
                problems.append("Node belongs to a different run")
            if content_hash(node.content) != node.content_hash:
                problems.append("Node content hash mismatch")
            producer = graph.operations.get(node.produced_by or "")
            parents = graph.parents(node_id)
            if node.node_type == NodeType.QUESTION and node.produced_by is None:
                if parents or node.content != graph.run.goal:
                    problems.append("Root question does not match the run")
            elif (
                producer is None
                or parents != {node.produced_by}
                or producer.status != OperationStatus.COMPLETED
                or node_id not in producer.output_node_ids
            ):
                problems.append("Node lacks one consistent, completed producing operation")
            yield self._check(node_id, "node", problems)
            state = graph.node_state(node_id)
            invalidation = graph.invalidations.get(node_id)
            yield self.record(
                node_id,
                "node",
                "current_validity",
                VerificationStatus.PASSED
                if state == NodeState.CURRENT
                else VerificationStatus.FAILED,
                state=state,
                reason=invalidation.reason if invalidation else None,
                cause_node_id=invalidation.cause_node_id if invalidation else None,
            )
