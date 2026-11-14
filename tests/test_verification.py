from collections.abc import Callable

import pytest

from rrm import ReasoningRuntime
from rrm.adapters import MockModelAdapter, MockRetrievalTool
from rrm.protocol.models import (
    DependencyType,
    NodeType,
    ReasoningDependency,
    RunEvent,
    RunStatus,
    VerificationStatus,
)
from rrm.protocol.validation import content_hash, node_input, snapshot_hash
from rrm.storage import MemoryStorage
from rrm.verification import SourceAdapter


def result_record(records, target_id, kind):
    return next(
        item for item in records if item.target_id == target_id and item.verification_type == kind
    )


class AlteredStorage(MemoryStorage):
    """Simulate persisted corruption without using runtime mutation APIs."""

    alter: Callable[[RunEvent], RunEvent] = staticmethod(lambda event: event)

    def events(self, run_id: str) -> tuple[RunEvent, ...]:
        return tuple(self.alter(event) for event in super().events(run_id))


def make_calculation(storage=None):
    tool = MockRetrievalTool({"revenue": 120, "cost": 95})

    def plan(ctx):
        evidence = [n.node_id for n in ctx.nodes if n.node_type == NodeType.EVIDENCE]
        derived = [n.node_id for n in ctx.nodes if n.node_type == NodeType.INFERENCE]
        if not evidence:
            return {
                "operation_type": "RETRIEVE",
                "objective": "Fetch figures",
                "input_node_ids": [ctx.nodes[0].node_id],
                "tool": "mock_search",
                "parameters": {"query": "120 95"},
            }
        if not derived:
            return {
                "operation_type": "COMPUTE",
                "objective": "Profit",
                "input_node_ids": evidence,
                "parameters": {"operator": "subtract"},
            }
        return {"operation_type": "FINISH", "objective": "Answer", "input_node_ids": derived}

    runtime = ReasoningRuntime(
        MockModelAdapter(plan, None, lambda ctx: {"answer": ctx.inputs[0].content}),
        storage,
        [tool],
    )
    result = runtime.run("Find profit")
    assert result.status == RunStatus.COMPLETED and result.answer == "25"
    return runtime, result, tuple(runtime.get_graph(result.run_id).operations.values())


def test_structure_detects_missing_reference_snapshot_and_rejected_cycle():
    storage = AlteredStorage()
    runtime, result, (_, compute, _) = make_calculation(storage)

    def corrupt(event):
        if event.operations and event.operations[0] == compute:
            op = compute.model_copy(
                update={
                    "input_node_ids": ("missing",),
                    "input_snapshot_hash": "wrong",
                }
            )
            edge = ReasoningDependency(
                run_id=result.run_id,
                edge_type=DependencyType.CONSUMES,
                node_id=compute.output_node_ids[0],
                operation_id=compute.operation_id,
            )
            return event.model_copy(
                update={
                    "operations": (op,),
                    "dependencies": (*event.dependencies, edge),
                }
            )
        return event

    storage.alter = corrupt
    records = runtime.verify_run(result.run_id)
    op = result_record(records, compute.operation_id, "structure")
    assert op.status == VerificationStatus.FAILED
    assert "Missing declared inputs: missing" in op.details["problems"]
    assert "Input snapshot hash mismatch" in op.details["problems"]
    assert (
        "Execution graph contains a cycle"
        in result_record(records, result.run_id, "structure").details["problems"]
    )


def test_provenance_detects_altered_snapshot():
    storage = AlteredStorage()
    runtime, result, (retrieve, _, _) = make_calculation(storage)
    node_id = retrieve.output_node_ids[0]

    def corrupt(event):
        nodes = tuple(
            node.model_copy(
                update={
                    "provenance": node.provenance.model_copy(update={"content": 999}),
                }
            )
            if node.node_id == node_id
            else node
            for node in event.nodes
        )
        return event.model_copy(update={"nodes": nodes})

    storage.alter = corrupt
    record = result_record(runtime.verify_run(result.run_id), node_id, "provenance_snapshot")
    assert record.status == VerificationStatus.FAILED
    assert "hash mismatch" in record.details["reason"]
    report = runtime.inspect_failure(result.run_id, result.final_node_ids[0])
    assert retrieve.operation_id in report.located_operation_ids


@pytest.mark.parametrize("fetched, expected", [(120, "passed"), (35, "failed")])
def test_source_adapter_checks_pinned_version(fetched, expected):
    runtime, result, (retrieve, _, _) = make_calculation()

    class Source(SourceAdapter):
        source_type = "mock_corpus"
        source_id = "local_fixture"

        def fetch_version(self, document_id: str, document_version: str) -> int:
            assert document_version == "1"
            return fetched if document_id == "revenue" else 95

    records = runtime.verify_run(result.run_id, sources=[Source()])
    assert result_record(records, retrieve.output_node_ids[0], "source_version").status == expected


def test_recompute_wrong_result_even_if_its_output_hash_is_consistent():
    storage = AlteredStorage()
    runtime, result, (retrieve, compute, finish) = make_calculation(storage)
    output = runtime.get_node(result.run_id, compute.output_node_ids[0])
    wrong = output.model_copy(update={"content": "35", "content_hash": content_hash("35")})

    def corrupt(event):
        if event.nodes == (output,):
            op = compute.model_copy(
                update={"output_snapshot_hash": snapshot_hash([node_input(wrong)])}
            )
            return event.model_copy(update={"nodes": (wrong,), "operations": (op,)})
        return event

    storage.alter = corrupt
    records = runtime.verify_run(result.run_id)
    assert result_record(records, compute.operation_id, "structure").status == "passed"
    wrong_result = result_record(records, compute.operation_id, "deterministic")
    assert wrong_result.status == "failed"
    assert (wrong_result.details["expected"], wrong_result.details["actual"]) == ("25", "35")
    assert result_record(records, finish.operation_id, "deterministic").status == "not_verified"
    report = runtime.inspect_failure(result.run_id, result.final_node_ids[0])
    assert report.operation == finish and report.actual_inputs == (wrong,)
    assert [item.node_id for item in report.evidence] == list(retrieve.output_node_ids)
    assert compute.operation_id in report.located_operation_ids
    assert report.root_cause == "verification_failures_located"


def test_semantic_root_cause_is_unconfirmed():
    runtime, result, (_, compute, finish) = make_calculation()
    records = runtime.verify_run(result.run_id)
    assert result_record(records, compute.operation_id, "deterministic").status == "passed"
    assert result_record(records, finish.operation_id, "deterministic").status == "not_verified"
    assert (
        runtime.inspect_failure(result.run_id, result.final_node_ids[0]).root_cause == "unconfirmed"
    )
