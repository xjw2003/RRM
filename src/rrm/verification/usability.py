from collections import defaultdict
from collections.abc import Iterable

from rrm.graph.dag import ReasoningGraph
from rrm.protocol.models import NodeBlock, VerificationRecord, VerificationStatus


def blocked_nodes(
    graph: ReasoningGraph,
    records: Iterable[VerificationRecord],
) -> tuple[NodeBlock, ...]:
    causes: dict[str, list[VerificationRecord]] = defaultdict(list)
    for record in records:
        if (
            record.status != VerificationStatus.FAILED
            or record.verification_type == "current_validity"
        ):
            continue
        if record.target_type == "node" and record.target_id in graph.nodes:
            causes[record.target_id].append(record)
        elif record.target_type == "operation" and record.target_id in graph.operations:
            operation = graph.operations[record.target_id]
            for node_id in dict.fromkeys(
                (*operation.output_node_ids, *graph.children(record.target_id))
            ):
                if node_id in graph.nodes:
                    causes[node_id].append(record)

    blocks = {}
    for cause_id, failures in causes.items():
        targets = (cause_id, *sorted(graph.descendants(cause_id) & graph.nodes.keys()))
        for node_id in targets:
            if node_id in graph.node_blocks or (node_id in causes and node_id != cause_id):
                continue
            blocks.setdefault(
                node_id,
                NodeBlock(
                    node_id=node_id,
                    cause_node_id=cause_id,
                    failed_checks=tuple(
                        dict.fromkeys(record.verification_type for record in failures)
                    ),
                    verification_ids=tuple(record.verification_id for record in failures),
                    reasons=tuple(
                        dict.fromkeys(
                            str(
                                record.details.get("reason")
                                or record.details.get("problems")
                                or f"{record.verification_type} check failed"
                            )
                            for record in failures
                        )
                    ),
                ),
            )
    return tuple(blocks.values())
