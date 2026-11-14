"""Persist model A's partial run, then finish it with model B on a reopened database."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from mock_reasoning import propose

from rrm import Limits, ReasoningRuntime
from rrm.adapters import MockModelAdapter
from rrm.adapters.model_base import ExecutionContext, PlanningContext
from rrm.protocol.models import NodeType
from rrm.storage import SQLiteStorage


def plan_a(ctx: PlanningContext) -> dict[str, Any]:
    return propose("INFER", list(ctx.nodes), "从问题列出两个数")


def plan_b(ctx: PlanningContext) -> dict[str, Any]:
    previous = [node for node in ctx.nodes if node.node_type == NodeType.HYPOTHESIS]
    return propose("FINISH", previous, "用已保存的候选值回答")


def finalize_b(ctx: ExecutionContext) -> dict[str, str]:
    candidate = ctx.inputs[0].content
    assert isinstance(candidate, list) and all(isinstance(value, int) for value in candidate)
    return {"answer": f"两个数的和是 {sum(candidate)}。"}


def main() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "reasoning.db"
        with SQLiteStorage(path) as storage:
            a = ReasoningRuntime(
                MockModelAdapter(
                    plan_a,
                    lambda _: {"node_type": "hypothesis", "content": [7, 11]},
                    lambda _: None,
                    model_id="mock-a",
                ),
                storage,
            )
            first = a.run("7 与 11 的和是多少？", limits=Limits(max_steps=1))
            print(first.model_dump_json(indent=2))
        with SQLiteStorage(path) as storage:
            b = ReasoningRuntime(
                MockModelAdapter(plan_b, lambda _: None, finalize_b, model_id="mock-b"),
                storage,
            )
            result = b.resume(first.run_id)
            assert result.answer == "两个数的和是 18。", result.error
            print(result.model_dump_json(indent=2))
            for op in b.get_graph(result.run_id).operations.values():
                print(op.executor_id, op.operation_type, op.input_node_ids, op.output_node_ids)


if __name__ == "__main__":
    main()
