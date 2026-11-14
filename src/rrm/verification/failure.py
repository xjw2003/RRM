from rrm.graph.dag import ReasoningGraph
from rrm.protocol.models import (
    NodeInvalidation,
    NodeState,
    ReasoningNode,
    ReasoningOperation,
    Record,
    VerificationRecord,
    VerificationStatus,
)
from rrm.verification.deterministic import DeterministicVerifier
from rrm.verification.provenance import ProvenanceVerifier
from rrm.verification.structure import StructuralVerifier


class FailureReport(Record):
    node: ReasoningNode
    node_state: NodeState
    invalidation: NodeInvalidation | None
    operation: ReasoningOperation | None
    actual_inputs: tuple[ReasoningNode, ...]
    evidence: tuple[ReasoningNode, ...]
    verification_failures: tuple[VerificationRecord, ...]
    direct_affected_node_ids: tuple[str, ...]
    downstream_node_ids: tuple[str, ...]
    located_operation_ids: tuple[str, ...]
    candidate_operation_ids: tuple[str, ...]
    root_cause: str


def inspect_failure(graph: ReasoningGraph, node_id: str) -> FailureReport:
    node = graph.nodes[node_id]
    producer = graph.operations.get(node.produced_by or "")
    ancestors = graph.ancestors(node_id)
    downstream = graph.descendants(node_id)
    candidates = tuple(op_id for op_id in graph.operations if op_id in ancestors)
    targets = ancestors | {node_id}
    if producer is not None:
        targets.add(producer.operation_id)
    # Recheck local data even if verify_run has never been called. Preserve the latest
    # recorded result for checks that require a separately configured source adapter.
    latest = {}
    for record in graph.verifications:
        key = (record.target_id, record.verifier_id, record.verification_type)
        latest[key] = record
    for verifier in (StructuralVerifier(), ProvenanceVerifier(), DeterministicVerifier()):
        for record in verifier.verify(graph):
            if record.verification_type != "source_version":
                key = (record.target_id, record.verifier_id, record.verification_type)
                prior = latest.get(key)
                if prior is None or prior.status != VerificationStatus.FAILED:
                    latest[key] = record
    block_ids = {
        verification_id
        for target in targets
        if target in graph.node_blocks
        for verification_id in graph.node_blocks[target].verification_ids
    }
    known_failures = {
        record.verification_id: record
        for record in graph.verifications
        if record.verification_id in block_ids
    }
    known_failures.update(
        (record.verification_id, record)
        for record in latest.values()
        if record.target_id in targets and record.status == VerificationStatus.FAILED
    )
    failures = tuple(known_failures.values())
    located = set()
    for record in failures:
        if record.verification_type == "current_validity":
            continue
        if record.target_type == "operation":
            located.add(record.target_id)
        elif record.target_type == "node":
            operation_id = graph.nodes[record.target_id].produced_by
            if operation_id in graph.operations:
                located.add(operation_id)
    direct = {
        descendant
        for operation_id in graph.children(node_id)
        for descendant in graph.children(operation_id)
    }
    return FailureReport(
        node=node,
        node_state=graph.node_state(node_id),
        invalidation=graph.invalidations.get(node_id),
        operation=producer,
        actual_inputs=tuple(
            graph.nodes[input_id]
            for input_id in producer.provided_node_ids
            if input_id in graph.nodes
        )
        if producer
        else (),
        evidence=tuple(
            graph.nodes[item]
            for item in graph.nodes
            if item in targets and graph.nodes[item].provenance is not None
        ),
        verification_failures=failures,
        direct_affected_node_ids=tuple(item for item in graph.nodes if item in direct),
        downstream_node_ids=tuple(item for item in graph.nodes if item in downstream),
        located_operation_ids=tuple(op_id for op_id in graph.operations if op_id in located),
        candidate_operation_ids=candidates,
        root_cause="verification_failures_located" if located else "unconfirmed",
    )
