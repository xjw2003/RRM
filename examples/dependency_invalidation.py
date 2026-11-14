"""Correct a source figure and show which actual descendants need recomputing."""

from typing import Any

from mock_reasoning import answer, propose

from rrm import ReasoningRuntime
from rrm.adapters import MockModelAdapter, MockRetrievalTool
from rrm.adapters.model_base import PlanningContext
from rrm.protocol.models import NodeType


def plan(ctx: PlanningContext) -> dict[str, Any]:
    evidence = [node for node in ctx.nodes if node.node_type == NodeType.EVIDENCE]
    derived = [node for node in ctx.nodes if node.node_type == NodeType.INFERENCE]
    if not evidence:
        return propose(
            "RETRIEVE",
            list(ctx.nodes[:1]),
            "取收入和成本",
            tool="mock_search",
            parameters={"query": "120 95"},
        )
    if not derived:
        return propose("COMPUTE", evidence, "计算利润", parameters={"operator": "subtract"})
    return propose("FINISH", derived, "给出利润")


def main() -> None:
    runtime = ReasoningRuntime(
        MockModelAdapter(plan, lambda _: None, answer),
        tools=[MockRetrievalTool({"revenue": 120, "cost": 95})],
    )
    result = runtime.run("收入减去成本后，利润是多少？")
    assert result.answer == "25", result.error
    before = runtime.get_graph(result.run_id)
    retrieval, calculation, finish = before.operations.values()
    revenue, cost = retrieval.output_node_ids
    for record in runtime.verify_run(result.run_id):
        if (
            record.target_id == calculation.operation_id
            and record.verification_type == "deterministic"
        ):
            print(record.model_dump_json(indent=2))

    report = runtime.revise_node(result.run_id, cost, 100, reason="将履约成本补录为 100")
    print(report.model_dump_json(indent=2))
    graph = runtime.get_graph(result.run_id)
    for label, node_id in (
        ("收入", revenue),
        ("旧成本", cost),
        ("新成本", report.replacement_node_id),
        ("旧利润", calculation.output_node_ids[0]),
        ("旧结论", finish.output_node_ids[0]),
    ):
        assert node_id is not None
        print(label, graph.nodes[node_id].content, graph.node_state(node_id), node_id)

    inspection = runtime.inspect_failure(result.run_id, finish.output_node_ids[0])
    print("旧结论的生成操作", inspection.operation.operation_id)
    print("实际输入", [node.content for node in inspection.actual_inputs])
    print("失效原因", inspection.invalidation.reason)

    # Use explicit input order after revision. Graph order is version creation order.
    def recompute(ctx: PlanningContext) -> dict[str, Any]:
        available = {node.node_id: node for node in ctx.nodes}
        derived = [node for node in ctx.nodes if node.node_type == NodeType.INFERENCE]
        if derived:
            return propose("FINISH", derived, "给出修正后的利润")
        assert report.replacement_node_id is not None
        inputs = [available[revenue], available[report.replacement_node_id]]
        return propose("COMPUTE", inputs, "重新计算", parameters={"operator": "subtract"})

    second = ReasoningRuntime(MockModelAdapter(recompute, lambda _: None, answer), runtime.storage)
    final = second.resume(result.run_id)
    assert final.answer == "20", final.error
    print(final.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
