import pytest
from test_runtime import proposal

from rrm import Limits, ReasoningRuntime
from rrm.adapters import MockModelAdapter, MockRetrievalTool
from rrm.protocol.models import NodeType, UsabilityStatus
from rrm.runtime import engine


def calculator(monkeypatch, *, retry=False, bypass=False, always_wrong=False, storage=None):
    real_arithmetic = engine.arithmetic
    observed = []

    def arithmetic(inputs, parameters):
        observed.append(inputs)
        if len(observed) == 1 or always_wrong:
            return "35"
        return real_arithmetic(inputs, parameters)

    monkeypatch.setattr(engine, "arithmetic", arithmetic)
    finalized = []

    def plan(ctx):
        evidence = [node for node in ctx.nodes if node.node_type == NodeType.EVIDENCE]
        derived = [node for node in ctx.nodes if node.node_type == NodeType.INFERENCE]
        if not evidence:
            return proposal(
                "RETRIEVE", ctx.nodes[:1], tool="mock_search", parameters={"query": "120 95"}
            )
        if ctx.blocked_nodes and not derived:
            block = ctx.blocked_nodes[-1]
            assert block.node_id not in {node.node_id for node in ctx.nodes}
            assert "deterministic" in block.failed_checks
            if bypass:
                return {
                    "operation_type": "FINISH",
                    "input_node_ids": [block.node_id],
                    "objective": "Attempt to use failed output",
                }
            if not retry:
                return proposal("FINISH", evidence, "Use original evidence")
        if not derived:
            return proposal("COMPUTE", evidence, parameters={"operator": "subtract"})
        return proposal("FINISH", derived)

    def finalize(ctx):
        finalized.append(ctx.inputs)
        if len(ctx.inputs) == 2:
            return {"answer": "Only the original figures are available"}
        return {"answer": str(ctx.inputs[0].content)}

    runtime = ReasoningRuntime(
        MockModelAdapter(plan, lambda _: None, finalize),
        storage,
        tools=[MockRetrievalTool({"revenue": 120, "cost": 95})],
    )
    return runtime, observed, finalized


def test_failed_compute_is_blocked(monkeypatch):
    runtime, observed, finalized = calculator(monkeypatch, retry=True)
    result = runtime.run("Find profit")
    graph = runtime.replay(result.run_id)
    _, wrong, corrected, finish = graph.operations.values()
    bad, good = wrong.output_node_ids[0], corrected.output_node_ids[0]
    assert graph.nodes[bad].content == "35"
    assert graph.node_state(bad) == "current"
    assert graph.node_usability(bad).status == "blocked"
    assert graph.node_usability(good).status == "eligible"
    assert graph.nodes[good].content == result.answer == "25"
    assert finish.provided_node_ids == (good,)
    assert len(observed) == 2 and finalized[0][0].node_id == good
    check = next(
        record
        for record in graph.verifications
        if record.target_id == wrong.operation_id and record.verification_type == "deterministic"
    )
    assert check.status == "failed"
    assert (check.details["expected"], check.details["actual"]) == ("25", "35")
    assert check.verification_id in graph.node_blocks[bad].verification_ids
    events = runtime.storage.events(result.run_id)
    committed = next(i for i, event in enumerate(events, 1) if graph.nodes[bad] in event.nodes)
    assert runtime.replay(result.run_id, through_sequence=committed).node_usability(bad).status == (
        "pending_verification"
    )
    assert events[committed].node_blocks


def test_verification_failure_cannot_be_bypassed(monkeypatch):
    runtime, _, finalized = calculator(monkeypatch, bypass=True)
    result = runtime.run("Find profit")
    graph = runtime.get_graph(result.run_id)
    assert result.status == "failed" and "is blocked" in result.error
    assert not finalized and len(graph.operations) == 2 and not result.final_node_ids
    runtime.verify_run(result.run_id)
    bad = tuple(graph.operations.values())[-1].output_node_ids[0]
    assert runtime.get_node_usability(result.run_id, bad).status == "blocked"


def test_failed_step_does_not_loop_forever(monkeypatch):
    runtime, calls, finalized = calculator(monkeypatch, retry=True, always_wrong=True)
    result = runtime.run("Find profit", limits=Limits(max_steps=4))
    assert result.status == "partial" and result.error == "max_steps reached"
    assert len(calls) == 3 and not finalized
    assert len(runtime.replay(result.run_id).node_blocks) == 3


@pytest.mark.parametrize("claimed_type", ["inference", "fact"])
def test_unverified_inference_can_continue(claimed_type):
    def plan(ctx):
        if len(ctx.nodes) == 1:
            return proposal("INFER", ctx.nodes)
        assert ctx.unverified_node_ids == (ctx.nodes[-1].node_id,)
        assert ctx.nodes[-1].node_type == ("hypothesis" if claimed_type == "fact" else "inference")
        return proposal("FINISH", ctx.nodes[-1:])

    runtime = ReasoningRuntime(
        MockModelAdapter(
            plan,
            lambda _: {"node_type": claimed_type, "content": "A possibility"},
            lambda ctx: {"answer": ctx.inputs[0].content},
        )
    )
    result = runtime.run("Suggest a possibility")
    graph = runtime.get_graph(result.run_id)
    assert result.answer == "A possibility"
    assert (
        len(
            [
                r
                for r in graph.verifications
                if r.verification_type == "deterministic" and r.status == "not_verified"
            ]
        )
        == 2
    )
    assert graph.node_usability(result.final_node_ids[0]).status == UsabilityStatus.UNVERIFIED


@pytest.mark.parametrize("damage", ["missing", "wrong"])
def test_failed_evidence_is_not_trusted(monkeypatch, damage):
    original = engine.ReasoningRuntime._new_node

    def new_node(*args, **kwargs):
        node = original(*args, **kwargs)
        if node.node_type == NodeType.EVIDENCE:
            provenance = (
                None
                if damage == "missing"
                else node.provenance.model_copy(update={"content": "corrupted"})
            )
            return node.model_copy(update={"provenance": provenance})
        return node

    monkeypatch.setattr(engine.ReasoningRuntime, "_new_node", staticmethod(new_node))

    def plan(ctx):
        if ctx.blocked_nodes:
            node_id = ctx.blocked_nodes[0].node_id
            assert node_id not in {node.node_id for node in ctx.nodes}
            return {"operation_type": "INFER", "input_node_ids": [node_id], "objective": "Test"}
        return proposal("RETRIEVE", ctx.nodes, tool="mock_search", parameters={"query": "source"})

    runtime = ReasoningRuntime(
        MockModelAdapter(plan, None, None), tools=[MockRetrievalTool({"doc": "source data"})]
    )
    result = runtime.run("Use data")
    graph = runtime.replay(result.run_id)
    assert result.status == "failed" and "is blocked" in result.error
    assert len(graph.operations) == 1
    assert next(iter(graph.node_blocks.values())).failed_checks == ("provenance_snapshot",)


def test_step_verification_does_not_scan_full_graph(monkeypatch):
    def full_scan(*_):
        pytest.fail("The runtime reverified the entire graph")

    for verifier in (
        engine.StructuralVerifier,
        engine.DeterministicVerifier,
        engine.ProvenanceVerifier,
    ):
        monkeypatch.setattr(verifier, "verify", full_scan)
    runtime, _, _ = calculator(monkeypatch, retry=True)
    assert runtime.run("Find profit").answer == "25"


def test_verified_arithmetic_keeps_unverified_assumptions_provisional():
    def plan(ctx):
        if len(ctx.nodes) < 3:
            return proposal("INFER", ctx.nodes[:1])
        if len(ctx.nodes) == 3:
            return proposal("COMPUTE", ctx.nodes[1:], parameters={"operator": "subtract"})
        assert ctx.nodes[-1].node_id in ctx.unverified_node_ids
        return proposal("FINISH", ctx.nodes[-1:])

    numbers = iter((120, 95))
    runtime = ReasoningRuntime(
        MockModelAdapter(
            plan,
            lambda _: {"node_type": "hypothesis", "content": next(numbers)},
            lambda ctx: {"answer": ctx.inputs[0].content},
        )
    )
    result = runtime.run("Calculate a conditional result")
    graph = runtime.replay(result.run_id)
    assert result.answer == "25"
    computation = next(op for op in graph.operations.values() if op.operation_type == "COMPUTE")
    assert graph.node_usability(computation.output_node_ids[0]).status == "unverified"
    assert any(
        record.target_id == computation.operation_id
        and record.status == "passed"
        and record.verification_type == "deterministic"
        for record in graph.verifications
    )
