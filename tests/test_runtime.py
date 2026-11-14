from threading import Event

import pytest

from rrm import Limits, ReasoningRuntime
from rrm.adapters import MockModelAdapter, MockRetrievalTool, ModelReply
from rrm.protocol.models import NodeType, OperationStatus, OperationType, RunStatus


def proposal(kind, nodes, objective="Evaluate", **options):
    return {
        "operation_type": kind,
        "input_node_ids": [n.node_id for n in nodes],
        "objective": objective,
        **options,
    }


def test_dynamic_dag_branches_and_actual_executor_inputs():
    received = []
    tool = MockRetrievalTool({"a": "value alpha", "b": "value beta"})

    def plan(ctx):
        nodes = ctx.nodes
        evidence = [n for n in nodes if n.node_type == NodeType.EVIDENCE]
        derived = [n for n in nodes if n.node_type == NodeType.INFERENCE]
        if not evidence:
            return proposal("RETRIEVE", nodes[:1], tool=tool.name, parameters={"query": "value"})
        if len(derived) < len(evidence):
            return proposal("INFER", evidence[len(derived) : len(derived) + 1])
        if len(derived) == len(evidence):
            return proposal("COMPARE", derived)
        return proposal("FINISH", derived[-1:])

    def infer(ctx):
        received.append(ctx)
        return {"node_type": "inference", "content": "|".join(n.content for n in ctx.inputs)}

    runtime = ReasoningRuntime(
        MockModelAdapter(plan, infer, lambda ctx: {"answer": ctx.inputs[0].content}), tools=[tool]
    )
    result = runtime.run("Combine facts discovered at runtime")
    assert result.status == RunStatus.COMPLETED
    assert result.answer == "value alpha|value beta"
    graph = runtime.get_graph(result.run_id)
    retrieve, a, b, merge, finish = graph.operations.values()
    assert len(graph.planning) == 5
    assert [op.operation_type for op in graph.operations.values()] == [
        OperationType.RETRIEVE,
        OperationType.INFER,
        OperationType.INFER,
        OperationType.COMPARE,
        OperationType.FINISH,
    ]
    assert set(retrieve.output_node_ids) == set(a.input_node_ids + b.input_node_ids)
    assert graph.parents(merge.operation_id) == set(a.output_node_ids + b.output_node_ids)
    assert graph.parents(finish.operation_id) == set(merge.output_node_ids)
    assert len(graph.trace(result.final_node_ids[0])) == len(graph.dependencies)
    assert [len(ctx.inputs) for ctx in received] == [1, 1, 2]
    assert [node.node_id for node in received[2].inputs] == list(merge.provided_node_ids)
    assert [node.content for node in received[2].inputs] == [
        graph.nodes[a.output_node_ids[0]].content,
        graph.nodes[b.output_node_ids[0]].content,
    ]


@pytest.mark.parametrize("cross_run", [False, True])
def test_unknown_and_cross_run_references(cross_run):
    model = MockModelAdapter(
        lambda ctx: proposal("FINISH", ctx.nodes), None, lambda _: {"answer": "ok"}
    )
    runtime = ReasoningRuntime(model)
    first = runtime.run("First run")
    node_id = first.final_node_ids[0] if cross_run else "missing"
    runtime = ReasoningRuntime(
        MockModelAdapter(
            lambda _: {
                "operation_type": "INFER",
                "input_node_ids": [node_id],
                "objective": "Try",
            },
            None,
            None,
        ),
        storage=runtime.storage,
    )
    result = runtime.run("Second run")
    assert result.status == RunStatus.FAILED and node_id in result.error
    assert not runtime.get_graph(result.run_id).operations


def test_invalid_plan_retries_then_returns_locatable_error():
    runtime = ReasoningRuntime(MockModelAdapter(lambda _: {"unexpected": True}, None, None))
    result = runtime.run("Test")
    assert result.status == RunStatus.FAILED
    assert "planning: invalid OperationProposal at operation_type" in result.error
    assert len(runtime.get_graph(result.run_id).planning) == 2


def test_invalid_execution_leaves_no_outputs_and_was_recorded_before_adapter():
    def infer(ctx):
        nonlocal run_id
        run_id = next(iter(runtime.storage._streams))
        graph = runtime.get_graph(run_id)
        assert [op.status for op in graph.operations.values()] == [OperationStatus.RUNNING]
        return {"node_type": "inference", "content": "bad", "node_id": "forged"}

    run_id = None
    runtime = ReasoningRuntime(
        MockModelAdapter(lambda ctx: proposal("INFER", ctx.nodes[:1]), infer, None)
    )
    result = runtime.run("Test")
    assert result.run_id == run_id
    graph = runtime.get_graph(result.run_id)
    (op,) = graph.operations.values()
    assert result.status == RunStatus.FAILED
    assert op.status == OperationStatus.FAILED
    assert "node_id" in op.error
    assert len(graph.nodes) == 1 and not op.output_node_ids


@pytest.mark.parametrize(
    ("limits", "reason", "steps"),
    [
        (Limits(max_steps=2), "max_steps", 2),
        (Limits(max_model_calls=3), "max_model_calls", 1),
        (Limits(max_tokens=4), "max_tokens", 1),
    ],
)
def test_budget_limits(limits, reason, steps):
    runtime = ReasoningRuntime(
        MockModelAdapter(
            lambda ctx: ModelReply(proposal("INFER", ctx.nodes[:1]), 2),
            lambda _: ModelReply({"node_type": "inference", "content": "continue"}, 2),
            None,
        )
    )
    result = runtime.run("Continue indefinitely", limits=limits)
    assert result.status == RunStatus.PARTIAL
    assert reason in result.error and result.steps_executed == steps


def test_timeout_does_not_commit_late_result():
    release = Event()
    exited = Event()

    def infer(_):
        release.wait(2)
        exited.set()
        return {"node_type": "inference", "content": "too late"}

    runtime = ReasoningRuntime(
        MockModelAdapter(lambda ctx: proposal("INFER", ctx.nodes), infer, None)
    )
    try:
        result = runtime.run("Slow", limits=Limits(timeout_seconds=0.1))
        assert result.status == RunStatus.PARTIAL
    finally:
        release.set()
    assert exited.wait(1)
    assert len(runtime.get_graph(result.run_id).nodes) == 1
