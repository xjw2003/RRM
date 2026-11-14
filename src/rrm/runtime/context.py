import re
from collections.abc import Iterable
from itertools import islice, zip_longest

from rrm.adapters.model_base import PlanningContext, ToolDescription
from rrm.graph.dag import ReasoningGraph
from rrm.protocol.models import NodeState, NodeType, UsabilityStatus
from rrm.protocol.validation import node_input


class ContextSelector:
    """Bound planning summaries while keeping evidence, branches and ancestry visible."""

    def __init__(self, *, max_nodes: int = 21, max_summary_chars: int = 512) -> None:
        if any(type(limit) is not int or limit < 1 for limit in (max_nodes, max_summary_chars)):
            raise ValueError("Context limits must be positive integers")
        self.max_nodes = max_nodes
        self.max_summary_chars = max_summary_chars

    def select(self, graph: ReasoningGraph, tools: Iterable[ToolDescription]) -> PlanningContext:
        usability = {node_id: graph.node_usability(node_id) for node_id in graph.nodes}
        available = {
            node_id: graph.nodes[node_id] for node_id, status in usability.items() if status.usable
        }
        if len(available) <= self.max_nodes:
            selected = set(available)
        else:
            recent = list(reversed(available))
            latest = next(
                (
                    node_id
                    for node_id in recent
                    if available[node_id].node_type != NodeType.QUESTION
                ),
                recent[0],
            )
            ancestors = graph.ancestors(latest)
            terms = set(re.findall(r"\w+", graph.run.goal.casefold()))

            def relevance(node_id: str) -> int:
                text = str(available[node_id].content).casefold()
                return sum(term in text for term in terms)

            evidence = [
                node_id
                for node_id in available
                if available[node_id].node_type in (NodeType.EVIDENCE, NodeType.FACT)
            ]
            evidence.sort(
                key=lambda node_id: (
                    node_id in ancestors,
                    relevance(node_id),
                    len(graph.children(node_id)),
                ),
                reverse=True,
            )
            frontier = [
                node_id
                for node_id in available
                if not any(
                    child in available
                    for op_id in graph.children(node_id)
                    for child in graph.children(op_id)
                )
            ]
            roots = [
                node_id
                for node_id in available
                if available[node_id].node_type == NodeType.QUESTION
            ]
            selected = set(roots[:1])
            for row in zip_longest(
                recent, evidence, frontier, (node_id for node_id in recent if node_id in ancestors)
            ):
                for node_id in row:
                    if len(selected) < self.max_nodes and node_id is not None:
                        selected.add(node_id)
                if len(selected) == self.max_nodes:
                    break
        summaries = []
        for node_id, node in available.items():
            if node_id not in selected:
                continue
            summary = node_input(node)
            text = node.model_dump_json(include={"content"})
            if len(text) > self.max_summary_chars:
                summary = summary.model_copy(update={"content": text[: self.max_summary_chars]})
            summaries.append(summary)
        blocks = (
            block
            for block in reversed(graph.node_blocks.values())
            if graph.node_state(block.node_id) == NodeState.CURRENT
        )
        return PlanningContext(
            goal=graph.run.goal,
            nodes=tuple(summaries),
            tools=tuple(tools),
            unverified_node_ids=tuple(
                summary.node_id
                for summary in summaries
                if usability[summary.node_id].status == UsabilityStatus.UNVERIFIED
            ),
            blocked_nodes=tuple(islice(blocks, self.max_nodes)),
        )
