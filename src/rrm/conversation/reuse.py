from collections.abc import Iterable
from itertools import islice

from rrm.conversation.continuity import HistoricalCandidate, relevance
from rrm.exceptions import GraphError
from rrm.graph.dag import ReasoningGraph
from rrm.protocol.models import ConversationTurn, NodeType, RunStatus
from rrm.storage.base import StorageBackend


class NodeReuseResolver:
    def __init__(self, storage: StorageBackend) -> None:
        self.storage = storage

    def candidates(
        self,
        question: str,
        earlier_turns: Iterable[ConversationTurn],
    ) -> tuple[HistoricalCandidate, ...]:
        turns = tuple(earlier_turns)
        scored = []
        for turn in islice(reversed(turns), 12):
            try:
                graph: ReasoningGraph = self.storage.get_graph(turn.run_id)
            except (KeyError, GraphError):
                continue
            if graph.run.status != RunStatus.COMPLETED:
                continue
            final = set(graph.run.final_node_ids)
            ancestors = set().union(*(graph.ancestors(node_id) for node_id in final))
            for node in graph.nodes.values():
                if node.node_type == NodeType.QUESTION:
                    continue
                label = node.provenance.document_id if node.provenance else None
                text = node.model_dump_json(include={"content"})
                match = relevance(question, f"{label or ''} {text}")
                recent = turn == turns[-1]
                priority = (
                    match,
                    recent and node.node_id in final,
                    recent and node.node_id in ancestors,
                    node.node_type == NodeType.EVIDENCE,
                    turn.turn_index,
                )
                candidate = HistoricalCandidate(
                    source_run_id=turn.run_id,
                    source_node_id=node.node_id,
                    revision=node.revision,
                    node_type=node.node_type,
                    label=label,
                    content=node.content if len(text) <= 512 else text[:512],
                    usability=graph.node_usability(node.node_id).status,
                )
                scored.append((priority, candidate))
        scored.sort(key=lambda item: item[0], reverse=True)
        return tuple(candidate for _, candidate in scored[:24])
