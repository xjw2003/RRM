import pytest
from test_verification import make_calculation, result_record

from rrm import ReasoningRuntime
from rrm.adapters import MockModelAdapter
from rrm.exceptions import InvalidReference
from rrm.protocol.models import NodeState, NodeType, RunStatus


def test_revision_keeps_history_and_only_invalidates_execution_descendants():
    runtime, result, (retrieve, compute, finish) = make_calculation()
    run_id = result.run_id
    revenue, cost = retrieve.output_node_ids
    old = runtime.get_node(run_id, cost)
    records = runtime.verify_run(run_id)
    report = runtime.revise_node(run_id, cost, 100, reason="Corrected cost figure")
    graph = runtime.get_graph(run_id)
    new = graph.nodes[report.replacement_node_id]
    assert graph.nodes[cost] == old
    assert (new.revision, new.logical_id, new.supersedes_node_id) == (2, old.logical_id, cost)
    assert new.content == 100
    assert graph.node_state(cost) == NodeState.SUPERSEDED
    assert graph.node_state(new.node_id) == graph.node_state(revenue) == NodeState.CURRENT
    assert report.stale_node_ids == compute.output_node_ids + finish.output_node_ids
    assert report.affected_operation_ids == (compute.operation_id, finish.operation_id)
    assert all(graph.node_state(n) == NodeState.STALE for n in report.stale_node_ids)
    assert all(graph.invalidations[n].cause_node_id == cost for n in report.stale_node_ids)
    assert graph.operations[compute.operation_id] == compute
    assert graph.run.status == RunStatus.PARTIAL and not graph.run.final_node_ids
    assert all(record in graph.verifications for record in records)
    latest = runtime.verify_run(run_id)
    assert result_record(latest, compute.operation_id, "current_validity").status == "failed"
    assert result_record(latest, compute.operation_id, "deterministic").status == "passed"
    assert result_record(latest, new.node_id, "provenance_snapshot").status == "not_verified"
    failure = runtime.inspect_failure(run_id, cost)
    assert failure.direct_affected_node_ids == compute.output_node_ids
    assert failure.downstream_node_ids == report.stale_node_ids
    with pytest.raises(InvalidReference):
        runtime.revise_node(run_id, cost, 101, reason="Outdated edit")


def test_revoke_filters_planner_context_and_rejects_stale_reference():
    runtime, result, (retrieve, compute, _) = make_calculation()
    revenue, cost = retrieve.output_node_ids
    report = runtime.revoke_node(result.run_id, cost, reason="Withdrawn")
    stale = compute.output_node_ids[0]
    assert stale in report.stale_node_ids

    def planner(ctx):
        visible = {node.node_id for node in ctx.nodes}
        assert visible.issuperset({revenue})
        assert not visible.intersection({cost, *report.stale_node_ids})
        return {"operation_type": "FINISH", "objective": "Answer", "input_node_ids": [stale]}

    other = ReasoningRuntime(MockModelAdapter(planner, None, None), runtime.storage)
    resumed = other.resume(result.run_id)
    assert resumed.status == RunStatus.FAILED and "is stale" in resumed.error
    graph = other.get_graph(result.run_id)
    assert graph.node_state(cost) == NodeState.REVOKED


def test_resume_can_recompute_using_corrected_version():
    runtime, result, (retrieve, compute, _) = make_calculation()
    revenue, cost = retrieve.output_node_ids
    revision = runtime.revise_node(result.run_id, cost, 100, reason="Correct cost")

    def planner(ctx):
        derived = [node.node_id for node in ctx.nodes if node.node_type == NodeType.INFERENCE]
        if not derived:
            return {
                "operation_type": "COMPUTE",
                "objective": "Update profit",
                "input_node_ids": [revenue, revision.replacement_node_id],
                "parameters": {"operator": "subtract"},
            }
        return {"operation_type": "FINISH", "objective": "Answer", "input_node_ids": derived}

    other = ReasoningRuntime(
        MockModelAdapter(planner, None, lambda ctx: {"answer": ctx.inputs[0].content}),
        runtime.storage,
    )
    resumed = other.resume(result.run_id)
    graph = other.get_graph(result.run_id)
    assert resumed.answer == "20" and resumed.status == RunStatus.COMPLETED
    assert graph.node_state(compute.output_node_ids[0]) == NodeState.STALE
    assert cost not in graph.ancestors(resumed.final_node_ids[0])


def test_revoke_keeps_the_unrelated_branch_current():
    from test_runtime import proposal

    from rrm.adapters import MockRetrievalTool

    tool = MockRetrievalTool({"a": "source A", "b": "source B"})

    def planner(ctx):
        evidence = [node for node in ctx.nodes if node.node_type == NodeType.EVIDENCE]
        derived = [node for node in ctx.nodes if node.node_type == NodeType.INFERENCE]
        if not evidence:
            return proposal(
                "RETRIEVE", ctx.nodes[:1], tool=tool.name, parameters={"query": "source"}
            )
        if len(derived) < 2:
            return proposal("INFER", evidence[len(derived) : len(derived) + 1])
        return proposal("FINISH", derived)

    runtime = ReasoningRuntime(
        MockModelAdapter(
            planner,
            lambda ctx: {"node_type": "inference", "content": ctx.inputs[0].content},
            lambda ctx: {"answer": " / ".join(node.content for node in ctx.inputs)},
        ),
        tools=[tool],
    )
    result = runtime.run("Compare independent sources")
    graph = runtime.get_graph(result.run_id)
    retrieval, a, b, finish = graph.operations.values()
    report = runtime.revoke_node(
        result.run_id, retrieval.output_node_ids[0], reason="Source A withdrawn"
    )
    assert report.affected_operation_ids == (a.operation_id, finish.operation_id)
    assert report.stale_node_ids == a.output_node_ids + finish.output_node_ids
    assert runtime.get_graph(result.run_id).node_state(b.output_node_ids[0]) == NodeState.CURRENT


def test_resume_completed_run_with_stale_side_branch():
    from test_runtime import proposal

    from rrm.adapters import MockRetrievalTool

    tool = MockRetrievalTool({"a": "source A", "b": "source B"})

    def planner(ctx):
        evidence = [node for node in ctx.nodes if node.node_type == NodeType.EVIDENCE]
        derived = [node for node in ctx.nodes if node.node_type == NodeType.INFERENCE]
        if not evidence:
            return proposal(
                "RETRIEVE", ctx.nodes[:1], tool=tool.name, parameters={"query": "source"}
            )
        if len(derived) < len(evidence):
            return proposal("INFER", evidence[len(derived) : len(derived) + 1])
        return proposal("FINISH", derived[:1])

    runtime = ReasoningRuntime(
        MockModelAdapter(
            planner,
            lambda ctx: {"node_type": "inference", "content": ctx.inputs[0].content},
            lambda ctx: {"answer": ctx.inputs[0].content},
        ),
        tools=[tool],
    )
    result = runtime.run("Answer with the first source and inspect the second separately")
    graph = runtime.get_graph(result.run_id)
    retrieval, _, branch, _ = graph.operations.values()
    report = runtime.revise_node(
        result.run_id, retrieval.output_node_ids[1], "corrected", reason="Fix"
    )
    assert report.stale_node_ids == branch.output_node_ids
    assert runtime.get_graph(result.run_id).run.status == RunStatus.COMPLETED

    other = ReasoningRuntime(
        MockModelAdapter(
            lambda ctx: proposal(
                "FINISH", [n for n in ctx.nodes if n.node_id == report.replacement_node_id]
            ),
            lambda _: None,
            lambda ctx: {"answer": ctx.inputs[0].content},
        ),
        runtime.storage,
    )
    assert other.resume(result.run_id).answer == "corrected"
