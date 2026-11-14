from collections.abc import Iterable
from typing import Any
from uuid import uuid4

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as SchemaError
from pydantic import JsonValue, TypeAdapter

from rrm.adapters.model_base import ExecutionContext, ModelAdapter, ToolDescription
from rrm.adapters.tool_base import EVIDENCE_ITEMS, ToolAdapter
from rrm.exceptions import InvalidReference, LimitReached, RRMError, StorageError
from rrm.graph.dag import ReasoningGraph
from rrm.graph.dependencies import propagate_invalidation
from rrm.protocol.models import (
    DependencyType,
    EvidenceSnapshot,
    FinalOutput,
    InvalidationReport,
    Limits,
    ModelOutput,
    NodeInput,
    NodeState,
    NodeType,
    OperationProposal,
    OperationStatus,
    OperationType,
    PlanningRecord,
    PositiveSeconds,
    ReasoningDependency,
    ReasoningNode,
    ReasoningOperation,
    ReasoningRun,
    RunEvent,
    RunResult,
    RunStatus,
    VerificationRecord,
    now,
)
from rrm.protocol.validation import content_hash, node_input, snapshot_hash
from rrm.runtime.context import planning_context
from rrm.runtime.executor import arithmetic
from rrm.runtime.planner import ExecutionBudget
from rrm.storage import MemoryStorage, StorageBackend
from rrm.verification import (
    DeterministicVerifier,
    ProvenanceVerifier,
    SourceAdapter,
    StructuralVerifier,
)
from rrm.verification.failure import FailureReport, inspect_failure


class ReasoningRuntime:
    def __init__(
        self,
        model: ModelAdapter,
        storage: StorageBackend | None = None,
        tools: Iterable[ToolAdapter] = (),
        *,
        model_retries: int = 1,
    ) -> None:
        if type(model_retries) is not int or model_retries < 0:
            raise ValueError("model_retries must be a nonnegative integer")
        self.model = model
        self.storage = storage if storage is not None else MemoryStorage()
        self.model_retries = model_retries
        seconds = TypeAdapter(PositiveSeconds)
        self._model_timeout = seconds.validate_python(model.timeout)
        self.tools: dict[str, ToolAdapter] = {}
        for tool in tools:
            if not tool.name or tool.name in self.tools:
                raise ValueError(f"Missing or duplicate tool name {tool.name!r}")
            seconds.validate_python(tool.timeout)
            if tool.max_calls is not None and (
                type(tool.max_calls) is not int or tool.max_calls < 0
            ):
                raise ValueError("tool max_calls must be a nonnegative integer")
            Draft202012Validator.check_schema(tool.input_schema)
            Draft202012Validator.check_schema(tool.output_schema)
            self.tools[tool.name] = tool

    def get_graph(self, run_id: str) -> ReasoningGraph:
        return self.storage.get_graph(run_id)

    def replay(self, run_id: str, *, through_sequence: int | None = None) -> ReasoningGraph:
        return self.storage.get_graph(run_id, through_sequence=through_sequence)

    def verify_run(
        self, run_id: str, *, sources: Iterable[SourceAdapter] = ()
    ) -> tuple[VerificationRecord, ...]:
        graph = self.storage.get_graph(run_id, audit=True)
        verifiers = (StructuralVerifier(), ProvenanceVerifier(sources), DeterministicVerifier())
        records = tuple(record for verifier in verifiers for record in verifier.verify(graph))
        if graph.run.status != RunStatus.RUNNING:
            self.storage.append(run_id, graph.sequence, RunEvent(verifications=records))
        return records

    def inspect_failure(self, run_id: str, node_id: str) -> FailureReport:
        graph = self.storage.get_graph(run_id, audit=True)
        return inspect_failure(graph, node_id)

    def get_node(self, run_id: str, node_id: str) -> ReasoningNode:
        return self.get_graph(run_id).nodes[node_id]

    def get_dependencies(self, run_id: str, vertex_id: str) -> tuple[ReasoningDependency, ...]:
        return tuple(
            edge for edge in self.get_graph(run_id).dependencies if vertex_id in edge.endpoints
        )

    def export_json(self, run_id: str) -> str:
        return self.storage.export_json(run_id)

    def _append(self, graph: ReasoningGraph, event: RunEvent) -> None:
        sequence = graph.sequence
        graph.apply(event)
        try:
            self.storage.append(graph.run.run_id, sequence, event)
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(f"Cannot commit run event ({type(exc).__name__})") from exc

    @staticmethod
    def _new_node(
        run_id: str,
        node_type: NodeType,
        content: JsonValue,
        *,
        produced_by: str | None = None,
        provenance: EvidenceSnapshot | None = None,
        supersedes: ReasoningNode | None = None,
        node_id: str | None = None,
    ) -> ReasoningNode:
        node_id = node_id or uuid4().hex
        return ReasoningNode(
            node_id=node_id,
            logical_id=(supersedes.logical_id or supersedes.node_id) if supersedes else node_id,
            revision=supersedes.revision + 1 if supersedes else 1,
            supersedes_node_id=supersedes.node_id if supersedes else None,
            run_id=run_id,
            node_type=node_type,
            content=content,
            produced_by=produced_by,
            content_hash=content_hash(content),
            provenance=provenance,
        )

    def _set_status(
        self,
        graph: ReasoningGraph,
        status: RunStatus,
        *,
        error: str | None = None,
        final_node_ids: tuple[str, ...] = (),
        limits: Limits | None = None,
    ) -> None:
        run = graph.run.model_copy(
            update={
                "status": status,
                "error": error,
                "updated_at": now(),
                "final_node_ids": final_node_ids,
                "model_id": self.model.model_id,
                "limits": limits if limits is not None else graph.run.limits,
            }
        )
        self._append(graph, RunEvent(run=run))

    def run(self, goal: str, *, limits: Limits | None = None) -> RunResult:
        if not goal.strip():
            raise ValueError("goal must not be blank")
        run = ReasoningRun(
            run_id=uuid4().hex,
            goal=goal,
            model_id=self.model.model_id,
            limits=limits if limits is not None else Limits(),
        )
        graph = ReasoningGraph(run)
        question = self._new_node(run.run_id, NodeType.QUESTION, goal)
        self._append(graph, RunEvent(run=run, nodes=(question,)))
        self._set_status(graph, RunStatus.RUNNING)
        return self._drive(graph, run.limits)

    def resume(self, run_id: str, *, limits: Limits | None = None) -> RunResult:
        graph = self.get_graph(run_id)
        if graph.run.status == RunStatus.RUNNING:
            raise RRMError("Cannot resume a running run")
        if graph.run.status == RunStatus.COMPLETED and not any(
            item.state == NodeState.STALE for item in graph.invalidations.values()
        ):
            raise RRMError("Cannot resume a completed run without stale results")
        self._set_status(graph, RunStatus.RUNNING, limits=limits)
        return self._drive(graph, graph.run.limits)

    def revise_node(
        self, run_id: str, node_id: str, content: JsonValue, *, reason: str
    ) -> InvalidationReport:
        graph = self.get_graph(run_id)
        old = self._editable_node(graph, node_id, reason)
        replacement_id = uuid4().hex
        report, invalidations = propagate_invalidation(
            graph,
            node_id,
            NodeState.SUPERSEDED,
            reason,
            replacement_id,
        )
        run = self._after_invalidation(graph, report)
        operation = ReasoningOperation(
            operation_id=uuid4().hex,
            run_id=run_id,
            operation_type="REVISE",
            input_node_ids=(),
            executor_type="runtime",
            executor_id="manual_revision",
            objective=reason,
            input_snapshot_hash=snapshot_hash(()),
        )
        # Validate caller data before recording the operation. The old node's content
        # is never used to produce the replacement, so lineage uses supersedes_node_id.
        content = TypeAdapter(JsonValue).validate_python(content)
        content_hash(content)
        self._append(
            graph,
            RunEvent(
                run=graph.run.model_copy(update={"status": RunStatus.RUNNING, "updated_at": now()}),
                operations=(operation,),
            ),
        )
        replacement = self._new_node(
            run_id,
            old.node_type,
            content,
            produced_by=operation.operation_id,
            supersedes=old,
            node_id=replacement_id,
        )
        completed = operation.model_copy(
            update={
                "status": OperationStatus.COMPLETED,
                "completed_at": now(),
                "output_node_ids": (replacement_id,),
                "output_snapshot_hash": snapshot_hash((node_input(replacement),)),
            }
        )
        edge = ReasoningDependency(
            run_id=run_id,
            edge_type=DependencyType.PRODUCES,
            node_id=replacement_id,
            operation_id=operation.operation_id,
        )
        self._append(
            graph,
            RunEvent(
                run=run.model_copy(update={"updated_at": now()}),
                nodes=(replacement,),
                operations=(completed,),
                dependencies=(edge,),
                invalidations=invalidations,
            ),
        )
        return report

    def revoke_node(self, run_id: str, node_id: str, *, reason: str) -> InvalidationReport:
        graph = self.get_graph(run_id)
        self._editable_node(graph, node_id, reason)
        report, invalidations = propagate_invalidation(graph, node_id, NodeState.REVOKED, reason)
        self._append(
            graph,
            RunEvent(
                run=self._after_invalidation(graph, report),
                invalidations=invalidations,
            ),
        )
        return report

    @staticmethod
    def _editable_node(graph: ReasoningGraph, node_id: str, reason: str) -> ReasoningNode:
        if not reason.strip():
            raise ValueError("Provide a reason for the change")
        if graph.run.status == RunStatus.RUNNING:
            raise RRMError("Cannot change nodes during execution")
        node = graph.nodes[node_id]
        if node.node_type == NodeType.QUESTION:
            raise RRMError("Changing the root question requires a new run")
        if graph.current_versions[node.logical_id or node_id] != node_id:
            raise InvalidReference(f"Node {node_id!r} is not the latest version")
        return node

    @staticmethod
    def _after_invalidation(graph: ReasoningGraph, report: InvalidationReport) -> ReasoningRun:
        update: dict[str, Any] = {"updated_at": now()}
        if set(graph.run.final_node_ids).intersection((report.node_id, *report.stale_node_ids)):
            update.update(
                status=RunStatus.PARTIAL,
                final_node_ids=(),
                error=f"Final answer invalidated: {report.reason}",
            )
        return graph.run.model_copy(update=update)

    def _plan(self, graph: ReasoningGraph, budget: ExecutionBudget) -> OperationProposal:
        context = planning_context(
            graph,
            (
                ToolDescription(
                    name=tool.name, description=tool.description, input_schema=tool.input_schema
                )
                for tool in self.tools.values()
            ),
        )
        digest = content_hash(context.model_dump(mode="json"))

        def log(proposal: OperationProposal | None, error: str | None) -> None:
            self._append(
                graph,
                RunEvent(
                    planning=PlanningRecord(
                        model_id=self.model.model_id,
                        context_hash=digest,
                        proposed=proposal,
                        error=error,
                    )
                ),
            )

        return budget.model_response(
            lambda: self.model.propose_next_operation(context.model_copy(deep=True)),
            OperationProposal,
            timeout=self._model_timeout,
            retries=self.model_retries,
            label="planning",
            on_attempt=log,
        )

    def _drive(self, graph: ReasoningGraph, limits: Limits) -> RunResult:
        budget = ExecutionBudget(limits)
        try:
            for _ in range(limits.max_steps):
                proposal = self._plan(graph, budget)
                finished = self._execute(graph, proposal, budget)
                if finished:
                    break
            else:
                raise LimitReached("max_steps reached")
        except LimitReached as exc:
            self._set_status(graph, RunStatus.PARTIAL, error=str(exc))
        except KeyboardInterrupt:
            self._set_status(graph, RunStatus.CANCELLED, error="Interrupted")
        except StorageError:
            raise
        except RRMError as exc:
            self._set_status(graph, RunStatus.FAILED, error=str(exc))
        except Exception as exc:
            self._set_status(graph, RunStatus.FAILED, error=f"Runtime error ({type(exc).__name__})")
        return self._result(graph)

    @staticmethod
    def _result(graph: ReasoningGraph) -> RunResult:
        run = graph.run
        answer = graph.nodes[run.final_node_ids[0]].content if run.final_node_ids else None
        return RunResult(
            run_id=run.run_id,
            status=run.status,
            answer=answer if isinstance(answer, str) else None,
            final_node_ids=run.final_node_ids,
            steps_executed=sum(op.operation_type != "REVISE" for op in graph.operations.values()),
            error=run.error,
        )

    @staticmethod
    def _inputs(graph: ReasoningGraph, ids: tuple[str, ...]) -> tuple[NodeInput, ...]:
        inputs = []
        for node_id in ids:
            node = graph.nodes.get(node_id)
            if node is None:
                raise InvalidReference(f"Input node {node_id!r} does not exist in this run")
            state = graph.node_state(node_id)
            if state != NodeState.CURRENT:
                raise InvalidReference(f"Input node {node_id!r} is {state}")
            if node.content_hash != content_hash(node.content):
                raise InvalidReference(f"Input node {node_id!r} has an invalid content hash")
            inputs.append(node_input(node))
        return tuple(inputs)

    def _execute(
        self, graph: ReasoningGraph, proposal: OperationProposal, budget: ExecutionBudget
    ) -> bool:
        inputs = self._inputs(graph, proposal.input_node_ids)
        kind = proposal.operation_type
        tool = self.tools.get(proposal.tool) if proposal.tool else None
        if kind == OperationType.RETRIEVE and tool is None:
            raise RRMError(f"Tool {proposal.tool!r} is not registered")
        if tool is not None:
            self._check_tool(graph, tool, proposal.parameters)
        elif kind != OperationType.COMPUTE:
            budget.check_model()
        run_id = graph.run.run_id
        operation = ReasoningOperation(
            operation_id=uuid4().hex,
            run_id=run_id,
            operation_type=kind,
            input_node_ids=proposal.input_node_ids,
            provided_node_ids=tuple(node.node_id for node in inputs),
            executor_type="tool"
            if tool
            else "runtime"
            if kind == OperationType.COMPUTE
            else "model",
            executor_id=tool.name
            if tool
            else "decimal"
            if kind == OperationType.COMPUTE
            else self.model.model_id,
            parameters=proposal.parameters,
            objective=proposal.objective,
            input_snapshot_hash=snapshot_hash(inputs),
        )

        def edges(ids: Iterable[str], direction: DependencyType) -> tuple[ReasoningDependency, ...]:
            return tuple(
                ReasoningDependency(
                    run_id=run_id,
                    edge_type=direction,
                    node_id=node_id,
                    operation_id=operation.operation_id,
                )
                for node_id in ids
            )

        self._append(
            graph,
            RunEvent(
                operations=(operation,),
                dependencies=edges(operation.provided_node_ids, DependencyType.CONSUMES),
            ),
        )
        try:
            nodes = self._invoke_operation(operation, inputs, tool, budget)
            budget.check_time()
            output_ids = tuple(node.node_id for node in nodes)
            completed = operation.model_copy(
                update={
                    "status": OperationStatus.COMPLETED,
                    "completed_at": now(),
                    "output_node_ids": output_ids,
                    "output_snapshot_hash": snapshot_hash(
                        tuple(node_input(node) for node in nodes)
                    ),
                }
            )
        except StorageError:
            raise
        except (Exception, KeyboardInterrupt) as exc:
            error = str(exc) if isinstance(exc, RRMError) else type(exc).__name__
            failed = operation.model_copy(
                update={
                    "status": OperationStatus.FAILED,
                    "completed_at": now(),
                    "error": error,
                }
            )
            self._append(graph, RunEvent(operations=(failed,)))
            if isinstance(exc, (RRMError, KeyboardInterrupt)):
                raise
            raise RRMError(f"{kind} execution failed ({error})") from exc
        finished = kind == OperationType.FINISH
        run = graph.run.model_copy(
            update={
                "updated_at": now(),
                "status": RunStatus.COMPLETED if finished else RunStatus.RUNNING,
                "final_node_ids": output_ids if finished else (),
            }
        )
        self._append(
            graph,
            RunEvent(
                run=run,
                operations=(completed,),
                nodes=nodes,
                dependencies=edges(output_ids, DependencyType.PRODUCES),
            ),
        )
        return finished

    def _invoke_operation(
        self,
        op: ReasoningOperation,
        inputs: tuple[NodeInput, ...],
        tool: ToolAdapter | None,
        budget: ExecutionBudget,
    ) -> tuple[ReasoningNode, ...]:
        if tool is not None:
            return self._retrieve(op, inputs, tool, budget)
        if op.operation_type == OperationType.COMPUTE:
            value = budget.invoke(
                lambda: arithmetic(inputs, op.parameters), float("inf"), "COMPUTE"
            )
            node_type = NodeType.INFERENCE
        else:
            context = ExecutionContext(
                operation_type=op.operation_type,
                objective=op.objective,
                inputs=inputs,
            )
            label = f"{op.operation_type} {op.operation_id}"
            if op.operation_type == OperationType.FINISH:
                final = budget.model_response(
                    lambda: self.model.generate_final_answer(context.model_copy(deep=True)),
                    FinalOutput,
                    timeout=self._model_timeout,
                    retries=self.model_retries,
                    label=label,
                )
                node_type, value = NodeType.RESULT, final.answer
            else:
                result = budget.model_response(
                    lambda: self.model.execute_reasoning(context.model_copy(deep=True)),
                    ModelOutput,
                    timeout=self._model_timeout,
                    retries=self.model_retries,
                    label=label,
                )
                node_type, value = result.node_type, result.content
        return (self._new_node(op.run_id, node_type, value, produced_by=op.operation_id),)

    @staticmethod
    def _validate_tool_schema(
        tool: ToolAdapter,
        value: object,
        schema: dict[str, Any],
        direction: str,
    ) -> None:
        try:
            Draft202012Validator(schema).validate(value)
        except SchemaError as exc:
            location = ".".join(map(str, exc.absolute_path)) or "<root>"
            raise RRMError(
                f"Tool {tool.name!r} violated its {direction} schema at {location}"
            ) from exc

    def _check_tool(
        self, graph: ReasoningGraph, tool: ToolAdapter, parameters: dict[str, JsonValue]
    ) -> None:
        calls = sum(
            op.executor_type == "tool" and op.executor_id == tool.name
            for op in graph.operations.values()
        )
        if tool.max_calls is not None and calls >= tool.max_calls:
            raise RRMError(f"Tool {tool.name!r} max_calls reached")
        self._validate_tool_schema(tool, parameters, tool.input_schema, "input")

    def _retrieve(
        self,
        op: ReasoningOperation,
        inputs: tuple[NodeInput, ...],
        tool: ToolAdapter,
        budget: ExecutionBudget,
    ) -> tuple[ReasoningNode, ...]:
        request = op.model_copy(deep=True).parameters
        provided = tuple(item.model_copy(deep=True) for item in inputs)
        raw = budget.invoke(lambda: tool.execute(request, provided), tool.timeout, tool.name)
        self._validate_tool_schema(tool, raw, tool.output_schema, "output")
        items = EVIDENCE_ITEMS.validate_python(raw)
        return tuple(
            self._new_node(
                op.run_id,
                NodeType.EVIDENCE,
                item.content,
                produced_by=op.operation_id,
                provenance=EvidenceSnapshot(
                    source_type=tool.source_type,
                    source_id=tool.source_id,
                    document_id=item.document_id,
                    document_version=item.document_version,
                    content=item.content,
                    content_hash=content_hash(item.content),
                ),
            )
            for item in items
        )
