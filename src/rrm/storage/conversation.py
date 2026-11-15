from rrm.exceptions import StorageConflict
from rrm.protocol.models import ConversationTurn, CrossRunReference, OperationStatus, RunEvent


def check_reference(
    reference: CrossRunReference,
    target: ConversationTurn | None,
    source: ConversationTurn | None,
    source_sequence: int,
) -> None:
    if (
        target is None
        or source is None
        or reference.target_run_id != target.run_id
        or reference.source_run_id != source.run_id
        or source.conversation_id != target.conversation_id
        or source.turn_index >= target.turn_index
    ):
        raise StorageConflict("IMPORT requires an earlier Run in the same conversation")
    if source_sequence != reference.source_sequence:
        raise StorageConflict("IMPORT source changed while the reference was being committed")


def check_import_event(run_id: str, event: RunEvent) -> None:
    imports = tuple(
        op
        for op in event.operations
        if op.operation_type == "IMPORT" and op.status == OperationStatus.COMPLETED
    )
    reference = event.cross_run_reference
    if not imports and reference is None:
        return
    if (
        len(imports) != 1
        or reference is None
        or reference.target_run_id != run_id
        or imports[0].operation_id != reference.import_operation_id
        or imports[0].output_node_ids != (reference.target_node_id,)
        or not any(
            node.node_id == reference.target_node_id
            and node.run_id == run_id
            and node.produced_by == imports[0].operation_id
            for node in event.nodes
        )
        or not any(
            edge.endpoints == (reference.import_operation_id, reference.target_node_id)
            and edge.run_id == run_id
            for edge in event.dependencies
        )
    ):
        raise StorageConflict("IMPORT output, operation, edge and reference must commit together")
