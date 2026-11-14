import json
import os
import subprocess
import sys

import pytest

from rrm import ReasoningRuntime
from rrm.adapters import MockModelAdapter
from rrm.exceptions import StorageConflict, StorageError
from rrm.protocol.models import OperationStatus, ReasoningRun, RunEvent
from rrm.storage import MemoryStorage, SQLiteStorage


def test_sqlite_graph_is_recovered_by_new_process(tmp_path):
    db = tmp_path / "graph.db"
    with SQLiteStorage(db) as storage:
        runtime = ReasoningRuntime(
            MockModelAdapter(
                lambda ctx: {
                    "operation_type": "FINISH",
                    "objective": "Answer",
                    "input_node_ids": [ctx.nodes[0].node_id],
                },
                None,
                lambda _: {"answer": "ok"},
            ),
            storage,
        )
        result = runtime.run("Question")
        before = json.loads(runtime.export_json(result.run_id))
    code = """
import sys
from rrm.storage import SQLiteStorage
with SQLiteStorage(sys.argv[1]) as storage:
    print(storage.export_json(sys.argv[2]))
"""
    output = subprocess.check_output([sys.executable, "-c", code, os.fspath(db), result.run_id])
    assert json.loads(output) == before


@pytest.mark.parametrize("sqlite", [False, True])
def test_conflicting_appends_are_atomic(tmp_path, sqlite):
    store = SQLiteStorage(tmp_path / "graph.db") if sqlite else MemoryStorage()
    event = RunEvent(run=ReasoningRun(run_id="r", goal="g", model_id="mock"))
    store.append("r", 0, event)
    with pytest.raises(StorageConflict):
        store.append("r", 0, event)
    assert store.events("r") == (event,)
    if sqlite:
        store.close()


def test_failed_step_commit_preserves_the_start_event_only():
    class FailingStorage(MemoryStorage):
        def append(self, run_id, expected_sequence, event):
            if any(op.output_node_ids for op in event.operations):
                raise OSError("simulated disk write failure")
            super().append(run_id, expected_sequence, event)

    storage = FailingStorage()
    runtime = ReasoningRuntime(
        MockModelAdapter(
            lambda ctx: {
                "operation_type": "FINISH",
                "objective": "Answer",
                "input_node_ids": [ctx.nodes[0].node_id],
            },
            lambda _: None,
            lambda _: {"answer": "ok"},
        ),
        storage,
    )
    with pytest.raises(StorageError, match="Cannot commit run event"):
        runtime.run("Question")
    (run_id,) = storage._streams
    graph = storage.get_graph(run_id)
    (op,) = graph.operations.values()
    assert op.status == OperationStatus.RUNNING
    assert len(graph.nodes) == len(graph.dependencies) == 1
    assert not graph.run.final_node_ids
