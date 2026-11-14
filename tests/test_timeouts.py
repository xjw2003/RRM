from threading import Event
from urllib.error import URLError

import pytest
from test_runtime import proposal

from rrm import Limits, ReasoningRuntime
from rrm.adapters import (
    MockModelAdapter,
    MockRetrievalTool,
    OpenAICompatibleAdapter,
    current_invocation,
)
from rrm.exceptions import RRMError


def test_model_wait_timeout_records_cooperative_cancellation():
    started, cancelled, released = Event(), Event(), Event()
    execution_ids = []

    def slow(_):
        control = current_invocation()
        assert control is not None
        execution_ids.append(control.execution_id)
        started.set()
        if control.wait_for_cancellation(2):
            cancelled.set()
        released.wait(2)
        return {"node_type": "inference", "content": "late"}

    model = MockModelAdapter(lambda ctx: proposal("INFER", ctx.nodes), slow, lambda _: None)
    model.timeout = 0.1
    runtime = ReasoningRuntime(model)
    try:
        result = runtime.run("Slow model")
        assert started.is_set() and cancelled.wait(1)
        graph = runtime.replay(result.run_id)
        (op,) = graph.operations.values()
        assert result.status == "failed" and not op.output_node_ids
        assert op.invocation_stop.reason == "wait_timeout"
        assert op.invocation_stop.worker_state == "unknown"
        assert op.invocation_stop.cancellation_requested
        assert op.invocation_stop.execution_id == execution_ids[0]
        assert current_invocation() is None
    finally:
        released.set()


def test_model_transport_timeout_differs_from_cancellation(monkeypatch):
    model = OpenAICompatibleAdapter(base_url="https://model.invalid/v1", api_key=None, model="test")

    def request(*args, **kwargs):
        raise URLError(TimeoutError("transport timeout"))

    monkeypatch.setattr(model._opener, "open", request)
    runtime = ReasoningRuntime(model)
    result = runtime.run("Transport")
    graph = runtime.replay(result.run_id)
    (attempt,) = graph.planning
    stop = attempt.invocation_stop
    assert result.status == "failed" and not graph.operations
    assert stop.reason == "request_timeout" and stop.worker_state == "exited"
    assert not stop.cancellation_requested


def test_interrupted_adapter_call_is_recorded_as_cancelled():
    def interrupt(_):
        raise KeyboardInterrupt

    model = MockModelAdapter(lambda ctx: proposal("INFER", ctx.nodes), interrupt, lambda _: None)
    runtime = ReasoningRuntime(model)
    result = runtime.run("Cancellation")
    (op,) = runtime.replay(result.run_id).operations.values()
    assert result.status == "cancelled"
    assert op.invocation_stop.reason == "cancelled"
    assert op.invocation_stop.worker_state == "exited"


def test_tool_timeout_late_result_cannot_enter_a_new_run():
    started, released, returned = Event(), Event(), Event()
    calls = []

    class ExternalTool(MockRetrievalTool):
        retry_safe = False
        timeout = 0.1

        def execute(self, parameters, inputs):
            control = current_invocation()
            calls.append(control.execution_id)
            started.set()
            released.wait(2)
            returned.set()
            return super().execute(parameters, inputs)

    tool = ExternalTool({"doc": "late data"})

    def plan(ctx):
        if ctx.goal == "New run":
            return proposal("FINISH", ctx.nodes)
        return proposal("RETRIEVE", ctx.nodes, tool=tool.name, parameters={"query": "late"})

    runtime = ReasoningRuntime(
        MockModelAdapter(plan, None, lambda _: {"answer": "fresh"}), tools=[tool]
    )
    try:
        old = runtime.run("Old run")
        assert old.status == "failed" and started.is_set()
        graph = runtime.replay(old.run_id)
        (op,) = graph.operations.values()
        assert op.invocation_stop.worker_state == "unknown"
        assert op.invocation_stop.reason == "wait_timeout"
        assert op.invocation_stop.execution_id == calls[0]
        assert op.status == "interrupted"
        assert runtime.recover_run(old.run_id).pending_tool_operation_ids == (op.operation_id,)
        assert runtime.recover_run(old.run_id).pending_tool_operation_ids == (op.operation_id,)
        with pytest.raises(RRMError, match="Confirm"):
            runtime.resume(old.run_id)
        new = runtime.run("New run", limits=Limits(timeout_seconds=1))
        assert new.answer == "fresh"
    finally:
        released.set()
    assert returned.wait(1)
    assert len(calls) == 1
    assert runtime.replay(old.run_id).to_dict() == graph.to_dict()
    assert all(node.content != "late data" for node in runtime.replay(new.run_id).nodes.values())
