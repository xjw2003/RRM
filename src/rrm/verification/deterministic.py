from collections.abc import Iterator
from decimal import Decimal

from rrm.graph.dag import ReasoningGraph
from rrm.protocol.models import (
    OperationStatus,
    OperationType,
    VerificationRecord,
    VerificationStatus,
)
from rrm.protocol.validation import node_input, snapshot_hash
from rrm.runtime.executor import arithmetic
from rrm.verification.base import Verifier


class DeterministicVerifier(Verifier):
    verifier_id = "rrm.deterministic"

    def verify(self, graph: ReasoningGraph) -> Iterator[VerificationRecord]:
        for op in graph.operations.values():
            if op.status != OperationStatus.COMPLETED:
                continue
            if op.operation_type != OperationType.COMPUTE:
                is_semantic = op.executor_type == "model" or op.operation_type == "REVISE"
                yield self.record(
                    op.operation_id,
                    "operation",
                    "deterministic",
                    VerificationStatus.NOT_VERIFIED
                    if is_semantic
                    else VerificationStatus.NOT_APPLICABLE,
                    reason="No deterministic rule for this result",
                )
                continue
            try:
                inputs = [node_input(graph.nodes[node_id]) for node_id in op.provided_node_ids]
                if snapshot_hash(inputs) != op.input_snapshot_hash:
                    raise ValueError("Recorded input snapshot does not match")
                expected = arithmetic(inputs, op.parameters)
                (output_id,) = op.output_node_ids
                actual = graph.nodes[output_id].content
                matches = type(actual) in (str, int, float) and Decimal(str(actual)) == Decimal(
                    expected
                )
            except (ArithmeticError, KeyError, ValueError) as exc:
                yield self.record(
                    op.operation_id,
                    "operation",
                    "deterministic",
                    VerificationStatus.FAILED,
                    reason=f"Cannot validate recorded computation ({type(exc).__name__})",
                )
                continue
            yield self.record(
                op.operation_id,
                "operation",
                "deterministic",
                VerificationStatus.PASSED if matches else VerificationStatus.FAILED,
                expected=expected,
                actual=actual,
                output_node_id=output_id,
            )
