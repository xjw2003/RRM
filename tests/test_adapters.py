import io
import json

import pytest

from rrm import ReasoningRuntime
from rrm.adapters import MockModelAdapter, MockRetrievalTool, OpenAICompatibleAdapter
from rrm.protocol.models import OperationStatus, RunStatus


def test_openai_compatible_http_request_is_phase_specific(monkeypatch):
    adapter = OpenAICompatibleAdapter(
        base_url="http://provider.invalid/v1/",
        api_key="test-only",
        model="example",
    )
    requests = []

    def fake_open(request, *, timeout):
        body = json.loads(request.data)
        requests.append(body)
        assert request.full_url == "http://provider.invalid/v1/chat/completions"
        assert request.get_header("Authorization") == "Bearer test-only"
        assert timeout == 30 and body["model"] == "example"
        assert body["response_format"] == {"type": "json_object"}
        assert [entry["role"] for entry in body["messages"]] == ["system", "user"]
        ctx = json.loads(body["messages"][1]["content"])
        if "nodes" in ctx:
            payload = {
                "operation_type": "FINISH",
                "objective": "Summarize",
                "input_node_ids": [ctx["nodes"][0]["node_id"]],
            }
        else:
            assert set(ctx) == {"operation_type", "objective", "inputs"}
            assert [node["content"] for node in ctx["inputs"]] == ["Question"]
            payload = {"answer": "Answer"}
        raw = {
            "choices": [{"message": {"content": json.dumps(payload)}}],
            "usage": {"total_tokens": 42},
        }
        return io.BytesIO(json.dumps(raw).encode())

    monkeypatch.setattr(adapter._opener, "open", fake_open)
    result = ReasoningRuntime(adapter).run("Question")
    assert result.status == RunStatus.COMPLETED and result.answer == "Answer"
    assert len(requests) == 2


@pytest.mark.parametrize(
    "payload", ["not JSON", {"content": float("nan"), "node_type": "inference"}]
)
def test_invalid_model_values_retry(payload):
    calls = []

    def infer(_):
        calls.append(1)
        return payload

    runtime = ReasoningRuntime(
        MockModelAdapter(
            lambda ctx: {
                "operation_type": "INFER",
                "objective": "Produce JSON",
                "input_node_ids": [ctx.nodes[0].node_id],
            },
            infer,
            lambda _: None,
        )
    )
    result = runtime.run("Question")
    assert result.status == RunStatus.FAILED and len(calls) == 2
    graph = runtime.get_graph(result.run_id)
    assert len(graph.nodes) == 1
    assert next(iter(graph.operations.values())).status == OperationStatus.FAILED


def test_tool_output_schema_failure_leaves_no_outputs():
    class BrokenTool(MockRetrievalTool):
        def execute(self, parameters, inputs):
            return [{"wrong": 1}]

    tool = BrokenTool({})
    runtime = ReasoningRuntime(
        MockModelAdapter(
            lambda ctx: {
                "operation_type": "RETRIEVE",
                "objective": "Search",
                "input_node_ids": [ctx.nodes[0].node_id],
                "tool": tool.name,
                "parameters": {"query": "x"},
            },
            lambda _: None,
            lambda _: None,
        ),
        tools=[tool],
    )
    result = runtime.run("Question")
    assert result.status == RunStatus.FAILED and "output schema at 0" in result.error
    graph = runtime.get_graph(result.run_id)
    (operation,) = graph.operations.values()
    assert operation.status == OperationStatus.FAILED
    assert not operation.output_node_ids and len(graph.nodes) == 1
