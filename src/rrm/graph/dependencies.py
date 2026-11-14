from rrm.graph.dag import ReasoningGraph
from rrm.protocol.models import InvalidationReport, NodeInvalidation, NodeState


def propagate_invalidation(
    graph: ReasoningGraph,
    node_id: str,
    state: NodeState,
    reason: str,
    replacement_id: str | None = None,
) -> tuple[InvalidationReport, tuple[NodeInvalidation, ...]]:
    downstream = graph.descendants(node_id)
    affected = tuple(op_id for op_id in graph.operations if op_id in downstream)
    stale = tuple(
        current_id
        for current_id in graph.nodes
        if current_id in downstream and graph.node_state(current_id) == NodeState.CURRENT
    )
    records = (
        NodeInvalidation(node_id=node_id, state=state, cause_node_id=node_id, reason=reason),
        *(
            NodeInvalidation(
                node_id=current_id,
                state=NodeState.STALE,
                cause_node_id=node_id,
                reason=reason,
            )
            for current_id in stale
        ),
    )
    report = InvalidationReport(
        node_id=node_id,
        replacement_node_id=replacement_id,
        affected_operation_ids=affected,
        stale_node_ids=stale,
        reason=reason,
    )
    return report, records
