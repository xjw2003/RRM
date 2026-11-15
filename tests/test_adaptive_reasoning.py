from test_runtime import proposal

from rrm import Limits, ReasoningRuntime
from rrm.adapters import MockModelAdapter
from rrm.protocol.models import OperationType


def direct_run(limits=None):
    calls = []

    def plan(ctx):
        calls.append("plan")
        return proposal("FINISH", ctx.nodes, "Answer")

    def infer(_):
        raise AssertionError("Direct execution called the reasoning adapter")

    def finish(ctx):
        calls.append("finish")
        return {"answer": ctx.inputs[0].content.upper()}

    runtime = ReasoningRuntime(MockModelAdapter(plan, infer, finish))
    return runtime, runtime.run("hello", limits=limits), calls


def test_simple_question_finishes_without_extra_inference():
    runtime, result, calls = direct_run()
    assert calls == ["plan", "finish"]
    assert result.reasoning_mode == "DIRECT" and result.model_calls == 2
    assert result.answer == "HELLO"
    assert len(runtime.replay(result.run_id).nodes) == 2


def test_no_fixed_minimum_reasoning_nodes():
    _, result, _ = direct_run(Limits(max_steps=1, max_model_calls=2))
    assert result.status == "completed" and result.steps_executed == 1


def test_direct_result_has_real_execution_record():
    runtime, result, _ = direct_run()
    graph = runtime.replay(result.run_id)
    (operation,) = graph.operations.values()
    question, answer = graph.nodes.values()
    assert operation.operation_type == OperationType.FINISH
    assert operation.provided_node_ids == (question.node_id,)
    assert operation.output_node_ids == (answer.node_id,)
    assert graph.ancestors(answer.node_id) == {operation.operation_id, question.node_id}
    assert graph.node_usability(answer.node_id).status == "unverified"


def test_complex_question_creates_dynamic_dag():
    observed = []

    def plan(ctx):
        last = ctx.nodes[-1]
        return proposal("FINISH" if last.content == "second" else "INFER", [last])

    def infer(ctx):
        observed.append(ctx.inputs[0])
        return {
            "node_type": "inference",
            "content": "second" if ctx.inputs[0].content == "first" else "first",
        }

    runtime = ReasoningRuntime(
        MockModelAdapter(plan, infer, lambda ctx: {"answer": ctx.inputs[0].content})
    )
    result = runtime.run("Develop an explanation")
    graph = runtime.get_graph(result.run_id)
    first, second, finish = graph.operations.values()
    assert result.reasoning_mode == "EXTENDED" and result.model_calls == 6
    assert observed[1].node_id == first.output_node_ids[0]
    assert second.provided_node_ids == first.output_node_ids
    assert finish.provided_node_ids == second.output_node_ids
    assert result.answer == "second"


def test_extended_reasoning_still_runs_step_verification():
    def plan(ctx):
        if len(ctx.nodes) == 1:
            return proposal("INFER", ctx.nodes)
        assert ctx.unverified_node_ids == (ctx.nodes[-1].node_id,)
        return proposal("FINISH", ctx.nodes[-1:])

    runtime = ReasoningRuntime(
        MockModelAdapter(
            plan,
            lambda _: {"node_type": "hypothesis", "content": "candidate"},
            lambda ctx: {"answer": ctx.inputs[0].content},
        )
    )
    result = runtime.run("Test a candidate")
    graph = runtime.get_graph(result.run_id)
    assert result.reasoning_mode == "EXTENDED"
    assert graph.verified_operations == graph.operations.keys()
