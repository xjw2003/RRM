"""Build a branching reasoning graph from a synthetic document corpus."""

from typing import Any

from rrm import ReasoningRuntime
from rrm.adapters import MockModelAdapter, MockRetrievalTool
from rrm.adapters.model_base import ExecutionContext, PlanningContext
from rrm.protocol.models import NodeInput, NodeType
from rrm.storage import StorageBackend

GOAL = "为什么销售额增加，但利润却下降？"
DOCUMENTS = {
    "sales": "本季销售额同比增长 20%，由折扣促销带动。",
    "costs": "本季成本同比增长 35%，主要来自履约支出。",
}


def propose(kind: str, inputs: list[NodeInput], objective: str, **options: Any) -> dict[str, Any]:
    return {
        "operation_type": kind,
        "input_node_ids": [item.node_id for item in inputs],
        "objective": objective,
        **options,
    }


def plan(ctx: PlanningContext) -> dict[str, Any]:
    evidence = [node for node in ctx.nodes if node.node_type == NodeType.EVIDENCE]
    derived = [node for node in ctx.nodes if node.node_type == NodeType.INFERENCE]
    if not evidence:
        return propose(
            "RETRIEVE",
            list(ctx.nodes[:1]),
            "查找销售额与成本证据",
            tool="mock_search",
            parameters={"query": "本季"},
        )
    if len(derived) < len(evidence):
        return propose("INFER", evidence[len(derived) : len(derived) + 1], "解释单项变化")
    if len(derived) == len(evidence):
        return propose("COMPARE", derived, "比较收入与成本变化，说明可能机制")
    return propose("FINISH", derived[-1:], "给出结论并说明证据缺口")


def infer(ctx: ExecutionContext) -> dict[str, str]:
    combined = " ".join(str(node.content) for node in ctx.inputs)
    if ctx.operation_type == "COMPARE":
        content = (
            f"{combined} 促销带动收入增长，同时履约支出推高成本。"
            "若成本增加额超过收入增加额，利润就会下降。同比增速本身不足以判定增加额。"
        )
    else:
        content = f"单项证据支持以下变化。{combined}"
    return {"node_type": "inference", "content": content}


def answer(ctx: ExecutionContext) -> dict[str, str]:
    return {"answer": str(ctx.inputs[0].content)}


def make_runtime(storage: StorageBackend | None = None) -> ReasoningRuntime:
    model = MockModelAdapter(plan, infer, answer)
    return ReasoningRuntime(model, storage, [MockRetrievalTool(DOCUMENTS)])


def main() -> None:
    runtime = make_runtime()
    result = runtime.run(GOAL)
    print(result.model_dump_json(indent=2))
    print(runtime.export_json(result.run_id))


if __name__ == "__main__":
    main()
