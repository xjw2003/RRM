import json
import sqlite3

import pytest
from test_runtime import proposal
from test_verification import AlteredStorage

from rrm import Limits, ReasoningRuntime
from rrm.adapters import MockModelAdapter, MockRetrievalTool
from rrm.conversation import (
    ContinuityDecision,
    ConversationRuntime,
    MockContinuityResolver,
    SourceNode,
)
from rrm.exceptions import GraphError, StorageConflict, StorageError
from rrm.graph.conversation import ConversationGraph
from rrm.protocol.models import NodeType, RunEvent
from rrm.protocol.validation import node_input, protocol_schema, snapshot_hash
from rrm.storage import SQLiteStorage

FIRST = "收入和成本变化后，利润变化是多少？"
FOLLOWUP = "刚才成本增加了多少？"
HYPOTHETICAL = "如果成本没有增加呢？"


class CountingTool(MockRetrievalTool):
    def __init__(self):
        super().__init__({"收入": 100, "成本": 150})
        self.calls = 0

    def execute(self, parameters, inputs):
        self.calls += 1
        return super().execute(parameters, inputs)


def make_chat(storage=None, *, resolver=None, allow_unverified=True):
    tool = CountingTool()
    delivered = []

    def plan(ctx):
        question = ctx.nodes[:1]
        evidence = [node for node in ctx.nodes if node.node_type == NodeType.EVIDENCE]
        hypotheses = [node for node in ctx.nodes if node.node_type == NodeType.HYPOTHESIS]
        results = [node for node in ctx.nodes if node.node_type == NodeType.INFERENCE]
        if ctx.goal == FIRST and not evidence:
            return proposal("RETRIEVE", question, tool=tool.name, parameters={"query": "100 150"})
        if ctx.goal == HYPOTHETICAL and evidence and not hypotheses:
            return proposal("INFER", question, "Represent the hypothetical change as a number")
        if ctx.goal in {FIRST, HYPOTHETICAL} and evidence and not results:
            return proposal("COMPUTE", evidence + hypotheses, parameters={"operator": "subtract"})
        return proposal("FINISH", results or evidence or question)

    def finish(ctx):
        delivered.append(ctx.inputs)
        return {"answer": str(ctx.inputs[0].content)}

    model = MockModelAdapter(plan, lambda _: {"node_type": "hypothesis", "content": 0}, finish)
    core = ReasoningRuntime(model, storage, tools=[tool])
    return (
        ConversationRuntime(core, resolver=resolver, allow_unverified=allow_unverified),
        tool,
        delivered,
    )


@pytest.fixture
def followup():
    chat, tool, delivered = make_chat()
    first = chat.ask(FIRST, conversation_id="C")
    second = chat.ask(FOLLOWUP, conversation_id="C")
    return chat, tool, delivered, first, second


def test_conversation_creates_independent_runs(followup):
    chat, _, _, first, second = followup
    turns = chat.list_turns("C")
    assert [turn.run_id for turn in turns] == [first.run_id, second.run_id]
    assert [turn.turn_index for turn in turns] == [1, 2]
    assert first.turn_id != second.turn_id and first.run_id != second.run_id
    a, b = (chat.get_run("C", turn.run_id) for turn in turns)
    assert a.run.protocol_version == b.run.protocol_version == "0.2"
    assert not a.nodes.keys() & b.nodes.keys()
    assert all(edge.run_id == b.run.run_id for edge in b.dependencies)


def test_followup_imports_existing_node(followup):
    chat, _, _, first, second = followup
    (ref,) = chat.get_references("C", second.run_id)
    assert ref.source_run_id == first.run_id
    assert ref.target_node_id in second.imported_node_ids
    assert second.continuity_type == "CONTINUE" and second.answer == "150"
    assert any(
        r.target_id == ref.target_node_id
        and r.verification_type == "cross_run_reference"
        and r.status == "passed"
        for r in chat.get_run("C", second.run_id).verifications
    )


def test_followup_does_not_repeat_retrieval(followup):
    _, tool, _, first, second = followup
    assert (first.model_calls, second.model_calls) == (4, 2)
    assert tool.calls == 1


def test_followup_direct_answer_without_extra_infer(followup):
    chat, _, _, _, second = followup
    graph = chat.get_run("C", second.run_id)
    assert [op.operation_type for op in graph.operations.values()] == ["IMPORT", "FINISH"]
    assert second.reasoning_mode == "DIRECT" and second.steps_executed == 2


def test_cross_run_import_preserves_source(followup):
    chat, _, _, first, second = followup
    (ref,) = chat.get_references("C", second.run_id)
    source = chat.core.get_node(first.run_id, ref.source_node_id)
    imported = chat.core.get_node(second.run_id, ref.target_node_id)
    assert source.run_id != imported.run_id and source.node_id != imported.node_id
    assert imported.content == source.content and imported.provenance == source.provenance
    assert ref.source_revision == source.revision
    assert imported.content_hash == ref.source_content_hash == source.content_hash


def test_imported_unverified_node_remains_unverified():
    def select(ctx):
        final = next(c for c in ctx.candidates if c.node_type == NodeType.RESULT)
        source = SourceNode(source_run_id=final.source_run_id, source_node_id=final.source_node_id)
        return ContinuityDecision(
            continuity_type="CONTINUE", reason="Reuse text", selected=(source,)
        )

    chat, _, _ = make_chat(resolver=MockContinuityResolver(select))
    chat.ask(FIRST, conversation_id="C")
    second = chat.ask("Use the previous conclusion", conversation_id="C")
    (node_id,) = second.imported_node_ids
    assert chat.core.get_node_usability(second.run_id, node_id).status == "unverified"


@pytest.mark.parametrize(
    "damage",
    [
        pytest.param("blocked", id="blocked_source_cannot_be_imported"),
        pytest.param("revoked", id="revoked_source_cannot_be_imported"),
        pytest.param("hash", id="source_hash_mismatch_rejects_import"),
    ],
)
def test_bad_source_is_rejected(damage, monkeypatch):
    storage = AlteredStorage()
    source_key = None

    def select(_):
        return ContinuityDecision(
            continuity_type="CONTINUE",
            reason="Try damaged cost evidence",
            selected=(source_key,),
        )

    chat, _, _ = make_chat(storage, resolver=MockContinuityResolver(select))
    first = chat.ask(FIRST, conversation_id="C")
    graph = chat.get_run("C", first.run_id)
    cost = next(node for node in graph.nodes.values() if node.content == 150)
    source_key = SourceNode(source_run_id=first.run_id, source_node_id=cost.node_id)
    if damage == "hash":

        def corrupt(event):
            return event.model_copy(
                update={
                    "nodes": tuple(
                        node.model_copy(update={"content_hash": "broken"}) if node == cost else node
                        for node in event.nodes
                    )
                }
            )

        storage.alter = corrupt
    elif damage == "revoked":
        # Keep the source run completed so the node-specific guard is exercised.
        monkeypatch.setattr(chat.core, "_after_invalidation", lambda current, _: current.run)
        chat.core.revoke_node(first.run_id, cost.node_id, reason="Withdraw source")
    else:
        from rrm.protocol.models import NodeBlock

        storage.append(
            first.run_id,
            graph.sequence,
            RunEvent(
                node_blocks=(
                    NodeBlock(
                        node_id=cost.node_id,
                        cause_node_id=cost.node_id,
                        failed_checks=("provenance_snapshot",),
                        verification_ids=(),
                    ),
                )
            ),
        )
    second = chat.ask(FOLLOWUP, conversation_id="C")
    assert not second.imported_node_ids
    (refusal,) = chat.get_run("C", second.run_id).reuse_decisions
    assert {"blocked": "blocked", "revoked": "revoked", "hash": "hash mismatch"}[
        damage
    ] in refusal.reason


def test_cross_run_reference_is_persisted(tmp_path):
    path = tmp_path / "conversation.db"
    with SQLiteStorage(path) as storage:
        chat, _, _ = make_chat(storage)
        first = chat.ask(FIRST, conversation_id="C")
        second = chat.ask(FOLLOWUP, conversation_id="C")
        (ref,) = chat.get_references("C", second.run_id)
    with SQLiteStorage(path) as storage:
        assert storage.references(second.run_id) == (ref,)
        source = SourceNode(source_run_id=first.run_id, source_node_id=ref.source_node_id)
        assert storage.references_to(source) == (ref,)
        (event,) = [event for event in storage.events(second.run_id) if event.cross_run_reference]
        assert event.nodes[0].node_id == ref.target_node_id
        assert event.operations[0].operation_id == ref.import_operation_id


def test_conversation_graph_replays_imports(followup):
    chat, tool, delivered, first, second = followup
    before = len(delivered)
    combined = chat.get_conversation_graph("C")
    (ref,) = combined.references
    assert combined.parents((second.run_id, ref.import_operation_id)) == {
        (first.run_id, ref.source_node_id),
    }
    assert (first.run_id, ref.source_node_id) in combined.ancestors(
        (second.run_id, second.final_node_ids[0])
    )
    assert len(delivered) == before and tool.calls == 1


def test_counterfactual_creates_new_branch(followup):
    chat, tool, _, first, _ = followup
    third = chat.ask(HYPOTHETICAL, conversation_id="C")
    graph = chat.get_run("C", third.run_id)
    assert third.continuity_type == "COUNTERFACTUAL" and third.answer == "100"
    assert third.reasoning_mode == "EXTENDED" and tool.calls == 1
    (ref,) = graph.references.values()
    assert ref.source_run_id == first.run_id
    assumption = next(
        node for node in graph.nodes.values() if node.node_type == NodeType.HYPOTHESIS
    )
    computation = next(op for op in graph.operations.values() if op.operation_type == "COMPUTE")
    assert computation.provided_node_ids == (ref.target_node_id, assumption.node_id)
    assert graph.nodes[assumption.node_id].content == 0
    assert graph.node_usability(computation.output_node_ids[0]).status == "unverified"
    assert any(
        r.target_id == computation.operation_id
        and r.verification_type == "deterministic"
        and r.status == "passed"
        for r in graph.verifications
    )


def test_counterfactual_does_not_modify_old_facts(followup):
    chat, _, _, first, _ = followup
    before = chat.core.replay(first.run_id).to_dict()
    chat.ask(HYPOTHETICAL, conversation_id="C")
    assert chat.core.replay(first.run_id).to_dict() == before


def test_unrelated_question_does_not_import_old_nodes(followup):
    chat, _, _, _, _ = followup
    result = chat.ask("换个话题，太阳是什么？", conversation_id="C")
    assert result.continuity_type == "NEW_TOPIC" and not result.imported_node_ids


def test_correction_is_recorded_for_explicit_review(followup):
    chat, _, _, first, _ = followup
    before = chat.core.replay(first.run_id).to_dict()
    result = chat.ask("成本数据错了", conversation_id="C")
    assert result.continuity_type == "CORRECTION" and result.correction_pending
    assert result.status == "partial" and result.steps_executed == 0
    assert chat.core.replay(first.run_id).to_dict() == before


def test_cross_run_cycle_is_rejected(followup):
    chat, _, _, first, second = followup
    a = chat.get_run("C", first.run_id)
    b = chat.get_run("C", second.run_id)
    (ref,) = b.references.values()
    forged = ref.model_copy(
        update={
            "source_run_id": b.run.run_id,
            "source_node_id": b.run.final_node_ids[0],
            "target_run_id": a.run.run_id,
            "source_sequence": b.sequence,
        }
    )
    with pytest.raises(StorageConflict, match="earlier Run"):
        chat.storage.append(a.run.run_id, a.sequence, RunEvent(cross_run_reference=forged))
    a.references[forged.target_node_id] = forged
    with pytest.raises(GraphError, match="earlier turn"):
        ConversationGraph(chat.storage.get_conversation("C"), [a, b])


def test_real_cross_turn_dependency_and_independent_replay(followup):
    chat, tool, delivered, first, second = followup
    (ref,) = chat.get_references("C", second.run_id)
    source = chat.core.get_node(first.run_id, ref.source_node_id)
    target = chat.core.get_node(second.run_id, ref.target_node_id)
    (actual,) = delivered[-1]
    assert actual == node_input(target) and actual.content == source.content == 150
    a, b = (chat.core.replay(run.run_id) for run in (first, second))
    finish = list(b.operations.values())[-1]
    assert finish.input_snapshot_hash == snapshot_hash(delivered[-1])
    assert b.operations[ref.import_operation_id].input_snapshot_hash == snapshot_hash(
        (node_input(source),)
    )
    assert a.run.final_node_ids and b.run.final_node_ids and tool.calls == 1


def test_source_revocation_is_checked_on_later_use(followup):
    chat, tool, _, first, second = followup
    (ref,) = chat.get_references("C", second.run_id)
    historical = chat.core.replay(second.run_id).to_dict()
    chat.core.revoke_node(first.run_id, ref.source_node_id, reason="Source withdrawn")
    assert chat.core.get_node_usability(second.run_id, ref.target_node_id).status == "blocked"
    assert chat.core.get_node_usability(second.run_id, second.final_node_ids[0]).status == "blocked"
    assert chat.core.replay(second.run_id).to_dict() == historical
    records = chat.core.verify_run(second.run_id)
    assert any(
        r.verification_type == "cross_run_reference" and r.status == "failed" for r in records
    )
    assert chat.core.get_graph(second.run_id).run.status == "partial"
    assert tool.calls == 1


def test_model_cannot_forge_import_or_foreign_candidate(followup):
    from pydantic import ValidationError

    from rrm.protocol.models import OperationProposal

    with pytest.raises(ValidationError):
        OperationProposal(operation_type="IMPORT", objective="Forge", input_node_ids=("source",))
    chat, _, _, first, _ = followup
    foreign = SourceNode(source_run_id=first.run_id, source_node_id=first.final_node_ids[0])
    chat.resolver = MockContinuityResolver(
        lambda _: ContinuityDecision(
            continuity_type="CONTINUE",
            reason="Forged ID",
            selected=(foreign,),
        )
    )
    chat.ask("Other conversation", conversation_id="OTHER")
    result = chat.ask(FOLLOWUP, conversation_id="OTHER")
    assert not result.imported_node_ids
    assert "offered candidate" in chat.get_run("OTHER", result.run_id).reuse_decisions[0].reason


def test_import_event_and_index_roll_back_together(tmp_path):
    with SQLiteStorage(tmp_path / "atomic.db") as storage:
        chat, _, _ = make_chat(storage)
        first = chat.ask(FIRST, conversation_id="C")
        storage._connection.execute("""
            CREATE TEMP TRIGGER reject_import BEFORE INSERT ON rrm_cross_run_refs
            BEGIN SELECT RAISE(ABORT, 'index unavailable'); END
        """)
        with pytest.raises(StorageError):
            chat.ask(FOLLOWUP, conversation_id="C")
        target = chat.list_turns("C")[-1]
        graph = chat.core.replay(target.run_id)
        (op,) = graph.operations.values()
        assert op.status == "running" and not op.output_node_ids
        assert len(graph.nodes) == 1 and not storage.references(target.run_id)
        original = next(
            node for node in chat.get_run("C", first.run_id).nodes.values() if node.content == 150
        )
        assert not storage.references_to(
            SourceNode(source_run_id=first.run_id, source_node_id=original.node_id)
        )


def test_old_rrp_records_remain_readable(tmp_path):
    from pathlib import Path

    from jsonschema import Draft202012Validator

    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE rrm_events(run_id TEXT, sequence INTEGER, payload TEXT, PRIMARY KEY(run_id, sequence))"
        )
        record = {
            "run_id": "old",
            "goal": "Original goal",
            "model_id": "legacy",
            "protocol_version": "0.1",
        }
        conn.execute(
            "INSERT INTO rrm_events VALUES (?, 1, ?)", ("old", json.dumps({"run": record}))
        )
    with SQLiteStorage(path) as storage:
        assert storage.get_graph("old").run.protocol_version == "0.1"
        assert storage.get_turn_by_run("old") is None
        assert storage.references("old") == ()
        chat, _, _ = make_chat(storage)
        result = chat.ask(FIRST, conversation_id="new")
        event = storage.events(result.run_id)[0]
        schema = json.loads(
            (Path(__file__).parents[1] / "schemas/rrp-v0.2.schema.json").read_text()
        )
        assert schema == protocol_schema("0.2")
        Draft202012Validator({**schema, "$ref": "#/$defs/RunEvent"}).validate(
            event.model_dump(mode="json")
        )
        assert storage.get_graph("old").run.goal == "Original goal"


def test_model_resolver_uses_one_call_and_shares_run_budget():
    from rrm.adapters import ModelReply
    from rrm.conversation import ContinuityResolver

    calls = []

    class Resolver(ContinuityResolver):
        uses_model = True

        def resolve(self, context):
            calls.append(context)
            return ModelReply({"continuity_type": "NEW_TOPIC", "reason": "No reuse"})

    chat, _, _ = make_chat(resolver=Resolver())
    first = chat.ask("First", conversation_id="C")
    second = chat.ask("Second", conversation_id="C", limits=Limits(max_model_calls=2))
    assert first.model_calls == 2 and len(calls) == 1
    assert second.model_calls == 2 and second.status == "partial"


def test_counterfactual_rejects_old_conclusion_that_depends_on_changed_cost(followup):
    chat, _, _, first, _ = followup
    original = chat.get_run("C", first.run_id)
    cost = next(node for node in original.nodes.values() if node.content == 150)
    chat.resolver = MockContinuityResolver(
        lambda _: ContinuityDecision(
            continuity_type="COUNTERFACTUAL",
            reason="Try old conclusion",
            selected=(
                SourceNode(source_run_id=first.run_id, source_node_id=first.final_node_ids[0]),
            ),
            changed_premises=(SourceNode(source_run_id=first.run_id, source_node_id=cost.node_id),),
        )
    )
    result = chat.ask(HYPOTHETICAL, conversation_id="C")
    assert not result.imported_node_ids
    assert "changed" in chat.get_run("C", result.run_id).reuse_decisions[0].reason


def test_same_source_is_imported_once(followup):
    chat, _, _, first, _ = followup
    original = chat.get_run("C", first.run_id)
    revenue = next(node for node in original.nodes.values() if node.content == 100)
    key = SourceNode(source_run_id=first.run_id, source_node_id=revenue.node_id)
    chat.resolver = MockContinuityResolver(
        lambda _: ContinuityDecision(
            continuity_type="CONTINUE",
            reason="Repeated selection",
            selected=(key, key),
        )
    )
    result = chat.ask("Earlier revenue", conversation_id="C")
    assert len(result.imported_node_ids) == 1
    assert len(chat.storage.references(result.run_id)) == 1


def test_revocation_between_planning_and_execution_blocks_local_id(followup):
    from rrm.exceptions import InvalidReference

    chat, _, _, first, second = followup
    target = chat.get_run("C", second.run_id)
    (ref,) = target.references.values()
    local_id = ref.target_node_id
    assert chat.core._inputs(target, (local_id,))[0].content == 150
    chat.core.revoke_node(first.run_id, ref.source_node_id, reason="Withdrawn during planning")
    with pytest.raises(InvalidReference):
        chat.core._inputs(target, (local_id,))


def test_import_feeds_actual_followup_inference_without_retrieval(followup):
    chat, tool, _, first, _ = followup
    contexts = []

    def plan(ctx):
        evidence = [node for node in ctx.nodes if node.node_type == NodeType.EVIDENCE]
        derived = [node for node in ctx.nodes if node.node_type == NodeType.INFERENCE]
        return proposal("FINISH", derived) if derived else proposal("INFER", evidence)

    def infer(ctx):
        contexts.append(ctx)
        return {"node_type": "inference", "content": f"成本增加 {ctx.inputs[0].content} 万元"}

    core = ReasoningRuntime(
        MockModelAdapter(plan, infer, lambda ctx: {"answer": ctx.inputs[0].content}),
        chat.storage,
        tools=[tool],
    )
    followup_runtime = ConversationRuntime(core)
    result = followup_runtime.ask(FOLLOWUP, conversation_id="C")
    graph = core.replay(result.run_id)
    (ref,) = graph.references.values()
    # Follow the reference chain to the actual RETRIEVE in the first turn.
    combined = followup_runtime.get_conversation_graph("C")
    origin = core.replay(first.run_id)
    cost = next(node for node in origin.nodes.values() if node.content == 150)
    infer_op = next(op for op in graph.operations.values() if op.operation_type == "INFER")
    assert contexts[0].inputs == (node_input(graph.nodes[ref.target_node_id]),)
    assert infer_op.input_snapshot_hash == snapshot_hash(contexts[0].inputs)
    assert (first.run_id, cost.node_id) in combined.ancestors(
        (result.run_id, infer_op.operation_id)
    )
    assert result.answer == "成本增加 150 万元" and tool.calls == 1


def test_revision_of_source_version_blocks_reuse(followup):
    chat, _, _, first, second = followup
    (ref,) = chat.get_references("C", second.run_id)
    change = chat.core.revise_node(first.run_id, ref.source_node_id, 155, reason="Revised figure")
    assert change.replacement_node_id is not None
    assert chat.core.get_node_usability(second.run_id, ref.target_node_id).status == "blocked"
    later = chat.ask(FOLLOWUP, conversation_id="C")
    assert not later.imported_node_ids
