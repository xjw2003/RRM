from collections.abc import Iterable

from rrm.adapters.model_base import PlanningContext, ToolDescription
from rrm.graph.dag import ReasoningGraph
from rrm.protocol.models import NodeInput, NodeState, NodeType
from rrm.protocol.validation import node_input


def planning_context(graph: ReasoningGraph, tools: Iterable[ToolDescription]) -> PlanningContext:
    nodes = tuple(
        node for node in graph.nodes.values() if graph.node_state(node.node_id) == NodeState.CURRENT
    )
    questions = [node for node in nodes if node.node_type == NodeType.QUESTION]
    recent = [node for node in nodes if node.node_type != NodeType.QUESTION][-20:]
    summaries = []
    for node in questions[-1:] + recent:
        summary = node_input(node)
        text = node.model_dump_json(include={"content"})
        if len(text) > 512:
            summary = NodeInput(node_id=node.node_id, node_type=node.node_type, content=text[:512])
        summaries.append(summary)
    return PlanningContext(goal=graph.run.goal, nodes=tuple(summaries), tools=tuple(tools))
