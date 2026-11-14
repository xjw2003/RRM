import json
import sqlite3
import subprocess
import sys

import pytest
from test_runtime import proposal

from rrm import ReasoningRuntime
from rrm.adapters import MockModelAdapter, MockRetrievalTool
from rrm.exceptions import RRMError, StorageConflict, StorageError
from rrm.protocol.models import OperationStatus
from rrm.storage import SQLiteStorage


def forbidden(*_):
    pytest.fail("Unexpected adapter call")


def test_recovery_after_process_interruption(tmp_path):
    db = tmp_path / "interrupted.db"
    child = """
import os
import sys
from rrm import ReasoningRuntime
from rrm.adapters import MockModelAdapter
from rrm.storage import SQLiteStorage

def plan(ctx):
    return dict(operation_type="INFER", input_node_ids=[ctx.nodes[0].node_id], objective="Test")

runtime = ReasoningRuntime(MockModelAdapter(plan, lambda _: os._exit(17), lambda _: None),
                           SQLiteStorage(sys.argv[1]))
runtime.run("Recover me")
"""
    proc = subprocess.run([sys.executable, "-c", child, str(db)], check=False)
    assert proc.returncode == 17
    with sqlite3.connect(db) as con:
        (run_id,) = con.execute("SELECT run_id FROM rrm_events WHERE sequence=1").fetchone()
    with SQLiteStorage(db) as storage:
        runtime = ReasoningRuntime(MockModelAdapter(forbidden, forbidden, forbidden), storage)
        graph = runtime.replay(run_id)
        (op,) = graph.operations.values()
        assert op.status == OperationStatus.RUNNING
        report = runtime.recover_run(run_id)
        assert report.status == "partial"
        assert report.recovery.interrupted_operation_ids == (op.operation_id,)
        recovered = runtime.replay(run_id)
        assert recovered.operations[op.operation_id].status == OperationStatus.INTERRUPTED
        assert recovered.recoveries == [report.recovery]
        assert runtime.recover_run(run_id).recovery is None
        assert runtime.replay(run_id).sequence == recovered.sequence
        next_runtime = ReasoningRuntime(
            MockModelAdapter(
                lambda ctx: proposal("FINISH", ctx.nodes),
                forbidden,
                lambda ctx: {"answer": ctx.inputs[0].content},
            ),
            storage,
        )
        assert next_runtime.resume(run_id).answer == "Recover me"


class FailedWrite(SQLiteStorage):
    def __init__(self, path, fail_on):
        super().__init__(path)
        self.fail_on = fail_on
        self.run_id = None

    def append(self, run_id, expected_sequence, event):
        self.run_id = run_id
        if self.fail_on(event):
            self._connection.execute("""
                CREATE TEMP TRIGGER IF NOT EXISTS fail_event BEFORE INSERT ON rrm_events
                BEGIN SELECT RAISE(ABORT, 'injected SQLite write failure'); END
            """)
        super().append(run_id, expected_sequence, event)


class ObservedRuntime(ReasoningRuntime):
    def _append(self, graph, event):
        self.graph_under_test = graph
        return super()._append(graph, event)


def test_failed_storage_commit_keeps_consistency(tmp_path):
    db = tmp_path / "commit.db"
    model = MockModelAdapter(
        lambda ctx: proposal("FINISH", ctx.nodes), forbidden, lambda _: {"answer": "answer"}
    )
    with FailedWrite(db, lambda event: bool(event.nodes and event.operations)) as storage:
        runtime = ObservedRuntime(model, storage)
        with pytest.raises(StorageError, match="Cannot commit"):
            runtime.run("Question")
        run_id = storage.run_id
        persisted = runtime.replay(run_id)
        assert runtime.graph_under_test.to_dict() == persisted.to_dict()
        assert runtime.graph_under_test.sequence == persisted.sequence
        assert len(persisted.nodes) == len(persisted.dependencies) == 1
    with SQLiteStorage(db) as storage:
        assert storage.get_graph(run_id).to_dict() == persisted.to_dict()


def test_recovery_of_running_operation(tmp_path):
    db = tmp_path / "verify-pending.db"
    model = MockModelAdapter(
        lambda ctx: proposal("FINISH", ctx.nodes), forbidden, lambda _: {"answer": "committed"}
    )
    with FailedWrite(db, lambda event: bool(event.verified_operation_ids)) as storage:
        with pytest.raises(StorageError):
            ReasoningRuntime(model, storage).run("Question")
        run_id = storage.run_id
    with SQLiteStorage(db) as storage:
        runtime = ReasoningRuntime(MockModelAdapter(forbidden, forbidden, forbidden), storage)
        original = runtime.replay(run_id)
        (op,) = original.operations.values()
        (node_id,) = op.output_node_ids
        assert original.node_usability(node_id).status == "pending_verification"
        result = runtime.recover_run(run_id)
        assert result.status == "completed"
        assert result.recovery.settled_operation_ids == (op.operation_id,)
        recovered = runtime.replay(run_id)
        assert recovered.nodes == original.nodes
        assert recovered.run.final_node_ids == (node_id,)
        assert recovered.node_usability(node_id).status == "unverified"
        assert runtime.recover_run(run_id).recovery is None


def test_no_duplicate_side_effect_on_recovery(tmp_path):
    calls = []

    class SideEffect(MockRetrievalTool):
        retry_safe = False

        def execute(self, parameters, inputs):
            calls.append(parameters)
            return super().execute(parameters, inputs)

    tool = SideEffect({"doc": "fixture"})
    db = tmp_path / "effect.db"
    model = MockModelAdapter(
        lambda ctx: proposal(
            "RETRIEVE", ctx.nodes, tool=tool.name, parameters={"query": "fixture"}
        ),
        forbidden,
        forbidden,
    )
    with FailedWrite(db, lambda event: bool(event.nodes and event.operations)) as storage:
        with pytest.raises(StorageError):
            ReasoningRuntime(model, storage, tools=[tool]).run("Question")
        run_id = storage.run_id
    with SQLiteStorage(db) as storage:
        runtime = ReasoningRuntime(model, storage, tools=[tool])
        first = runtime.recover_run(run_id)
        second = runtime.recover_run(run_id)
        (op_id,) = first.pending_tool_operation_ids
        assert second.pending_tool_operation_ids == (op_id,) and second.recovery is None
        with pytest.raises(RRMError, match="Confirm the outcome"):
            runtime.resume(run_id)
        assert calls == [{"query": "fixture"}]
        runtime.confirm_tool_outcome(run_id, op_id, reason="External result reconciled")
        assert runtime.recover_run(run_id).pending_tool_operation_ids == ()
        assert len(runtime.replay(run_id).tool_confirmations) == 1


def test_replay_remains_read_only(tmp_path):
    with FailedWrite(
        tmp_path / "run.db", lambda event: bool(event.verified_operation_ids)
    ) as store:
        model = MockModelAdapter(
            lambda ctx: proposal("FINISH", ctx.nodes), forbidden, lambda _: {"answer": "committed"}
        )
        with pytest.raises(StorageError):
            ReasoningRuntime(model, store).run("Question")
        run_id = store.run_id
        runtime = ReasoningRuntime(MockModelAdapter(forbidden, forbidden, forbidden), store)
        events = store.events(run_id)
        replayed = runtime.replay(run_id)
        assert replayed.run.status == "running"
        assert store.events(run_id) == events


def test_legacy_sqlite_events_load_and_audit_without_rewriting(tmp_path):
    db = tmp_path / "legacy.db"
    model = MockModelAdapter(
        lambda ctx: proposal("FINISH", ctx.nodes), forbidden, lambda _: {"answer": "legacy"}
    )
    with SQLiteStorage(db) as storage:
        result = ReasoningRuntime(model, storage).run("Question")
    fields = {
        "verification_required",
        "retry_safe",
        "verified_operation_ids",
        "node_blocks",
        "recovery",
        "tool_confirmation",
    }
    with sqlite3.connect(db) as con:
        payloads = []
        for seq, payload in con.execute(
            "SELECT sequence, payload FROM rrm_events ORDER BY sequence"
        ):
            event = json.loads(payload)
            if event["verified_operation_ids"]:
                continue
            event = {k: v for k, v in event.items() if k not in fields}
            for operation in event["operations"]:
                for field in fields:
                    operation.pop(field, None)
            if event["nodes"] and event["operations"]:
                event["run"]["status"] = "completed"
                event["run"]["final_node_ids"] = [event["nodes"][0]["node_id"]]
            payloads.append((result.run_id, seq, json.dumps(event)))
        con.execute("DELETE FROM rrm_events")
        con.executemany("INSERT INTO rrm_events VALUES (?, ?, ?)", payloads)
    with SQLiteStorage(db) as storage:
        runtime = ReasoningRuntime(MockModelAdapter(forbidden, forbidden, forbidden), storage)
        legacy = runtime.replay(result.run_id)
        assert legacy.run.status == "completed"
        assert legacy.node_usability(result.final_node_ids[0]).status == "unverified"
        before = storage.events(result.run_id)
        assert runtime.verify_run(result.run_id)
        assert storage.events(result.run_id)[:-1] == before


def test_stale_writer_cannot_publish_after_recovery(tmp_path):
    db = tmp_path / "race.db"

    class OtherWriter(ReasoningRuntime):
        def _verify_step(self, graph, operation, **kwargs):
            self.before = graph.to_dict()
            self.graph = graph
            with SQLiteStorage(db) as other:
                recovered = ReasoningRuntime(
                    MockModelAdapter(forbidden, forbidden, forbidden),
                    other,
                ).recover_run(graph.run.run_id)
                assert recovered.status == "completed"
            return super()._verify_step(graph, operation, **kwargs)

    with SQLiteStorage(db) as storage:
        runtime = OtherWriter(
            MockModelAdapter(
                lambda ctx: proposal("FINISH", ctx.nodes),
                forbidden,
                lambda _: {"answer": "one commit"},
            ),
            storage,
        )
        with pytest.raises(StorageConflict):
            runtime.run("Race")
        assert runtime.graph.to_dict() == runtime.before
        persisted = runtime.replay(runtime.graph.run.run_id)
        assert persisted.run.status == "completed"
        assert len(persisted.nodes) == 2 and len(persisted.recoveries) == 1
