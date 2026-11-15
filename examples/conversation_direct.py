"""Run a simple question with one planning call and one finalization call."""

from rrm import Limits, ReasoningRuntime
from rrm.adapters import MockModelAdapter
from rrm.adapters.model_base import ExecutionContext, PlanningContext


def plan(ctx: PlanningContext) -> dict[str, object]:
    return {
        "operation_type": "FINISH",
        "input_node_ids": [ctx.nodes[0].node_id],
        "objective": "Reply briefly",
    }


def finish(ctx: ExecutionContext) -> dict[str, str]:
    return {"answer": f"{ctx.inputs[0].content}！"}


def main() -> None:
    runtime = ReasoningRuntime(MockModelAdapter(plan, lambda _: None, finish))
    result = runtime.run("你好", limits=Limits(max_steps=1, max_model_calls=2))
    print(result.model_dump_json(indent=2))
    print(runtime.export_json(result.run_id))


if __name__ == "__main__":
    main()
