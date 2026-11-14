import pytest

from rrm import ReasoningRuntime
from rrm.adapters import MockModelAdapter
from rrm.exceptions import GraphError
from rrm.protocol.models import DependencyType, ReasoningDependency


def test_reject_cycle():
    runtime = ReasoningRuntime(
        MockModelAdapter(
            lambda ctx: {
                "operation_type": "FINISH",
                "objective": "Answer",
                "input_node_ids": [ctx.nodes[0].node_id],
            },
            None,
            lambda _: {"answer": "ok"},
        )
    )
    result = runtime.run("Question")
    graph = runtime.get_graph(result.run_id)
    (op,) = graph.operations.values()
    original = graph.to_dict()
    with pytest.raises(GraphError, match="Cyclic"):
        graph.add_dependency(
            ReasoningDependency(
                run_id=result.run_id,
                node_id=op.output_node_ids[0],
                operation_id=op.operation_id,
                edge_type=DependencyType.CONSUMES,
            )
        )
    assert graph.to_dict() == original
