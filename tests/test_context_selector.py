from test_runtime import proposal

from rrm import Limits, ReasoningRuntime
from rrm.adapters import MockModelAdapter, MockRetrievalTool
from rrm.protocol.models import NodeType
from rrm.runtime.context import ContextSelector


def test_long_dag_keeps_early_evidence_and_open_branch_discoverable():
    count = 0
    early_text = "calibration source " + "x" * 700
    early_id = None
    branch_id = None
    observed = []

    def plan(ctx):
        nonlocal early_id, branch_id
        assert len(ctx.nodes) <= 12
        root = ctx.nodes[0]
        assert root.node_type == NodeType.QUESTION
        evidence = [node for node in ctx.nodes if node.node_type == NodeType.EVIDENCE]
        derived = [node for node in ctx.nodes if node.node_type == NodeType.INFERENCE]
        if not evidence:
            return proposal(
                "RETRIEVE", (root,), tool="mock_search", parameters={"query": "calibration"}
            )
        early = next(
            node for node in evidence if node.content.startswith('{"content":"calibration')
        )
        early_id = early.node_id
        assert len(early.content) == 80
        if count < 52:
            if count == 2:
                branch_id = derived[0].node_id
            source = (early,) if count == 0 else (root,) if count == 1 else derived[-1:]
            return proposal("INFER", source)
        if count == 52:
            branch = next(node for node in derived if node.node_id == branch_id)
            return proposal("COMPARE", (early, branch, derived[-1]))
        return proposal("FINISH", derived[-1:])

    def infer(ctx):
        nonlocal count
        count += 1
        observed.append(ctx.inputs)
        return {"node_type": "inference", "content": f"Step {count}"}

    runtime = ReasoningRuntime(
        MockModelAdapter(plan, infer, lambda ctx: {"answer": ctx.inputs[0].content}),
        tools=[MockRetrievalTool({"early": early_text})],
        context_selector=ContextSelector(max_nodes=12, max_summary_chars=80),
    )
    result = runtime.run("Find the calibration", limits=Limits(max_steps=60, max_model_calls=120))
    assert result.answer == "Step 53", result.error
    graph = runtime.get_graph(result.run_id)
    assert len(graph.nodes) == 56 and early_id not in list(graph.nodes)[-20:]
    assert [node.node_id for node in observed[-1]][:2] == [early_id, branch_id]
    assert observed[-1][0].content == early_text and len(observed[-1]) == 3
    assert {early_id, branch_id} <= graph.ancestors(result.final_node_ids[0])
    runtime.revoke_node(result.run_id, early_id, reason="Calibration withdrawn")
    graph = runtime.replay(result.run_id)
    selected = ContextSelector(max_nodes=4).select(graph, [])
    assert len(selected.nodes) <= 4
    assert early_id not in {node.node_id for node in selected.nodes}
    assert all(graph.node_usability(node.node_id).usable for node in selected.nodes)
