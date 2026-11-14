from test_verification import make_calculation

from rrm import Limits, ReasoningRuntime
from rrm.adapters import MockModelAdapter
from rrm.protocol.models import NodeState, NodeType, OperationStatus, RunStatus
from rrm.storage import SQLiteStorage


def test_replay_rebuilds_history_without_model_or_tool_calls(tmp_path):
    db = tmp_path / "replay.db"
    with SQLiteStorage(db) as storage:
        runtime, result, (retrieve, compute, _) = make_calculation(storage)
        run_id = result.run_id
        before = runtime.get_graph(run_id)
        runtime.revoke_node(run_id, retrieve.output_node_ids[1], reason="Source withdrawn")
        after = runtime.get_graph(run_id)

    def forbidden(*_):
        raise AssertionError("Replay called an adapter")

    with SQLiteStorage(db) as storage:
        runtime = ReasoningRuntime(MockModelAdapter(forbidden, forbidden, forbidden), storage)
        history = storage.events(run_id)
        start_sequence = next(
            i
            for i, event in enumerate(history, start=1)
            if any(op.operation_id == compute.operation_id for op in event.operations)
        )
        start = runtime.replay(run_id, through_sequence=start_sequence)
        assert start.operations[compute.operation_id].status == OperationStatus.RUNNING
        assert (
            runtime.replay(run_id, through_sequence=before.sequence).to_dict() == before.to_dict()
        )
        restored = runtime.replay(run_id)
        assert restored.to_dict() == after.to_dict()
        assert restored.node_state(compute.output_node_ids[0]) == NodeState.STALE
        assert storage.events(run_id) == history


def test_cross_model_resume_uses_persisted_node_payload(tmp_path):
    db = tmp_path / "resume.db"

    def plan_a(ctx):
        return {
            "operation_type": "INFER",
            "objective": "Derive",
            "input_node_ids": [ctx.nodes[0].node_id],
        }

    with SQLiteStorage(db) as storage:
        a = ReasoningRuntime(
            MockModelAdapter(
                plan_a,
                lambda _: {"node_type": "hypothesis", "content": {"value": 7}},
                None,
                model_id="model-a",
            ),
            storage,
        )
        partial = a.run("Create and interpret a value", limits=Limits(max_steps=1))
        assert partial.status == RunStatus.PARTIAL
        graph_a = a.get_graph(partial.run_id)
        (op_a,) = graph_a.operations.values()

    def plan_b(ctx):
        previous = next(node for node in ctx.nodes if node.node_type == NodeType.HYPOTHESIS)
        return {
            "operation_type": "FINISH",
            "objective": "Explain",
            "input_node_ids": [previous.node_id],
        }

    def finalize_b(ctx):
        (node,) = ctx.inputs
        assert (node.node_id, node.content) == (op_a.output_node_ids[0], {"value": 7})
        return {"answer": f"Value {node.content['value']}"}

    with SQLiteStorage(db) as storage:
        b = ReasoningRuntime(
            MockModelAdapter(plan_b, None, finalize_b, model_id="model-b"), storage
        )
        final = b.resume(partial.run_id)
        assert final.status == RunStatus.COMPLETED and final.answer == "Value 7"
        graph_b = b.get_graph(final.run_id)
        assert [op.executor_id for op in graph_b.operations.values()] == ["model-a", "model-b"]
        assert graph_b.run.model_id == "model-b" and final.run_id == partial.run_id
