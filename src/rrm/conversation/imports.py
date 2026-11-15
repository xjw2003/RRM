from uuid import uuid4

from rrm.exceptions import InvalidReference
from rrm.graph.dag import ReasoningGraph
from rrm.protocol.models import (
    ContinuityType,
    CrossRunReference,
    DependencyType,
    OperationStatus,
    ReasoningDependency,
    ReasoningOperation,
    ReuseDecision,
    RunEvent,
    SourceNode,
    UsabilityStatus,
    now,
)
from rrm.protocol.validation import node_input, snapshot_hash
from rrm.runtime.engine import ReasoningRuntime
from rrm.verification.reference import ReferenceVerifier


def import_node(
    core: ReasoningRuntime,
    graph: ReasoningGraph,
    source: SourceNode,
    reason: str,
    *,
    allow_unverified: bool,
) -> str:
    turn, continuity = graph.turn, graph.continuity
    if (
        turn is None
        or continuity is None
        or continuity.continuity_type
        not in {
            ContinuityType.CONTINUE,
            ContinuityType.COUNTERFACTUAL,
        }
    ):
        raise InvalidReference("This turn does not allow historical reuse")
    verifier = ReferenceVerifier(core.storage)
    source_graph, node, status = verifier.source(turn, source)
    if status == UsabilityStatus.UNVERIFIED and not allow_unverified:
        raise InvalidReference("Unverified sources are disabled for this conversation runtime")
    if continuity.continuity_type == ContinuityType.COUNTERFACTUAL:
        if not continuity.changed_premises:
            raise InvalidReference("Changed historical premises must be identified before reuse")
        changed = set()
        for premise in continuity.changed_premises:
            premise_graph, premise_node, _ = verifier.source(turn, premise)
            changed.add((premise_graph.run.run_id, premise_node.node_id))
            if premise_node.node_id in premise_graph.references:
                changed.add(verifier.origin(premise_graph, premise_node.node_id))
        if changed & verifier.lineage(source_graph, node.node_id):
            raise InvalidReference("Source depends on a premise changed by this question")
    for ref in graph.references.values():
        if (ref.source_run_id, ref.source_node_id) == (source.source_run_id, source.source_node_id):
            verifier.check_reference(graph, ref)
            return ref.target_node_id

    operation = ReasoningOperation(
        operation_id=uuid4().hex,
        run_id=turn.run_id,
        operation_type="IMPORT",
        input_node_ids=(),
        executor_type="runtime",
        executor_id="conversation_import",
        objective="Import a specific historical node version",
        verification_required=True,
        retry_safe=True,
        input_snapshot_hash=snapshot_hash((node_input(node),)),
    )
    imported = core._new_node(
        turn.run_id,
        node.node_type,
        node.content,
        produced_by=operation.operation_id,
        provenance=node.provenance,
    )
    ref = CrossRunReference(
        **source.model_dump(),
        source_content_hash=node.content_hash,
        source_revision=node.revision,
        source_sequence=source_graph.sequence,
        source_usability=status,
        target_run_id=turn.run_id,
        target_node_id=imported.node_id,
        import_operation_id=operation.operation_id,
    )
    core._append(graph, RunEvent(operations=(operation,)))
    completed = operation.model_copy(
        update={
            "status": OperationStatus.COMPLETED,
            "completed_at": now(),
            "output_node_ids": (imported.node_id,),
            "output_snapshot_hash": snapshot_hash((node_input(imported),)),
        }
    )
    core._append(
        graph,
        RunEvent(
            operations=(completed,),
            nodes=(imported,),
            cross_run_reference=ref,
            dependencies=(
                ReasoningDependency(
                    run_id=turn.run_id,
                    edge_type=DependencyType.PRODUCES,
                    node_id=imported.node_id,
                    operation_id=operation.operation_id,
                ),
            ),
            reuse_decision=ReuseDecision(
                **source.model_dump(), reason=reason, target_node_id=imported.node_id
            ),
        ),
    )
    core._verify_step(graph, completed)
    return imported.node_id
