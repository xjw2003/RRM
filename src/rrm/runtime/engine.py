from collections.abc import Iterable
from typing import Any
from uuid import uuid4

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as SchemaError
from pydantic import JsonValue, TypeAdapter

from rrm.adapters.model_base import ExecutionContext, ModelAdapter, ToolDescription
from rrm.adapters.tool_base import EVIDENCE_ITEMS, ToolAdapter
from rrm.exceptions import (
    InvalidReference,
    InvocationAborted,
    LimitReached,
    RRMError,
    StorageError,
)
from rrm.graph.dag import ReasoningGraph
from rrm.graph.dependencies import propagate_invalidation
from rrm.protocol.models import (
    ContinuityType,
    ConversationTurn,
    DependencyType,
    EvidenceSnapshot,
    FinalOutput,
    InvalidationReport,
    InvocationStop,
    Limits,
    ModelOutput,
    NodeInput,
    NodeState,
    NodeType,
    NodeUsability,
    OperationProposal,
    OperationStatus,
    OperationType,
    PlanningRecord,
    PositiveSeconds,
    ReasoningDependency,
    ReasoningNode,
    ReasoningOperation,
    ReasoningRun,
    RecoveryRecord,
    RecoveryResult,
    RunEvent,
    RunResult,
    RunStatus,
    ToolConfirmation,
    VerificationRecord,
    VerificationStatus,
    now,
)
from rrm.protocol.validation import content_hash, node_input, snapshot_hash
from rrm.runtime.context import ContextSelector
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
from rrm.verification.reference import ReferenceVerifier
from rrm.verification.usability import blocked_nodes


class ReasoningRuntime:
    def __init__(
        self,
        model: ModelAdapter,
        storage: StorageBackend | None = None,
        tools: Iterable[ToolAdapter] = (),
        *,
        model_retries: int = 1,
        context_selector: ContextSelector | None = None,
    ) -> None:
        if type(model_retries) is not int or model_retries < 0:
            raise ValueError("model_retries must be a nonnegative integer")
        self.model = model
        self.storage = storage if storage is not None else MemoryStorage()
        self.model_retries = model_retries
        self.context_selector = (
            context_selector if context_selector is not None else ContextSelector()
        )
        seconds = TypeAdapter(PositiveSeconds)
        self._model_timeout = seconds.validate_python(model.timeout)
        self.tools: dict[str, ToolAdapter] = {}
        for tool in tools:
            if not tool.name or tool.name in self.tools:
                raise ValueError(f"Missing or duplicate tool name {tool.name!r}")
            seconds.validate_python(tool.timeout)
            if type(tool.retry_safe) is not bool:
                raise ValueError("tool retry_safe must be a boolean")
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
        verifiers = (
            StructuralVerifier(),
            ProvenanceVerifier(sources),
            DeterministicVerifier(),
            ReferenceVerifier(self.storage),
        )
        records = tuple(record for verifier in verifiers for record in verifier.verify(graph))
        if graph.run.status != RunStatus.RUNNING:
            blocks = blocked_nodes(graph, records)
            run = graph.run
            if set(run.final_node_ids).intersection(block.node_id for block in blocks):
                run = run.model_copy(
                    update={
                        "status": RunStatus.PARTIAL,
                        "updated_at": now(),
                        "final_node_ids": (),
                        "error": "Final answer blocked by verification",
                    }
                )
            self._append(
                graph,
                RunEvent(
                    run=run if run != graph.run else None,
                    verifications=records,
                    node_blocks=blocks,
                    verified_operation_ids=tuple(
                        op_id
                        for op_id, operation in graph.operations.items()
                        if operation.status == OperationStatus.COMPLETED
                        and op_id not in graph.verified_operations
                    ),
                ),
            )
        return records

    def inspect_failure(self, run_id: str, node_id: str) -> FailureReport:
        graph = self.storage.get_graph(run_id, audit=True)
        return inspect_failure(graph, node_id)

    def get_node_usability(self, run_id: str, node_id: str) -> NodeUsability:
        graph = self.get_graph(run_id)
        self._refresh_imports(graph)
        return graph.node_usability(node_id)

    def get_node(self, run_id: str, node_id: str) -> ReasoningNode:
        return self.get_graph(run_id).nodes[node_id]

    def get_dependencies(self, run_id: str, vertex_id: str) -> tuple[ReasoningDependency, ...]:
        return tuple(
            edge for edge in self.get_graph(run_id).dependencies if vertex_id in edge.endpoints
        )

    def export_json(self, run_id: str) -> str:
        return self.storage.export_json(run_id)

    def _append(self, graph: ReasoningGraph, event: RunEvent) -> None:
        draft = graph.prepare(event)
        try:
            self.storage.append(graph.run.run_id, graph.sequence, event)
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(f"Cannot commit run event ({type(exc).__name__})") from exc
        graph.__dict__.update(draft.__dict__)

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
        graph = self._create_run(goal, limits=limits)
        self._set_status(graph, RunStatus.RUNNING)
        return self._drive(graph, graph.run.limits)

    def _create_run(
        self, goal: str, *, limits: Limits | None = None, turn: ConversationTurn | None = None
    ) -> ReasoningGraph:
        if not goal.strip():
            raise ValueError("goal must not be blank")
        run = ReasoningRun(
            run_id=turn.run_id if turn else uuid4().hex,
            goal=goal,
            protocol_version="0.2" if turn else "0.1",
            model_id=self.model.model_id,
            limits=limits if limits is not None else Limits(),
        )
        graph = ReasoningGraph(run)
        question = self._new_node(run.run_id, NodeType.QUESTION, goal)
        self._append(graph, RunEvent(run=run, nodes=(question,), turn=turn))
        return graph

    def recover_run(
        self, run_id: str, *, reason: str = "Execution owner stopped before completion"
    ) -> RecoveryResult:
        """Reconcile a stopped execution without calling any model or tool."""
        graph = self.get_graph(run_id)
        recovery = None
        if graph.run.status == RunStatus.RUNNING:
            if not reason.strip():
                raise ValueError("Provide a recovery reason")
            interrupted = tuple(
                op for op in graph.operations.values() if op.status == OperationStatus.RUNNING
            )
            pending = tuple(
                op
                for op in graph.operations.values()
                if op.status == OperationStatus.COMPLETED
                and op.verification_required
                and op.operation_id not in graph.verified_operations
            )
            recovery = RecoveryRecord(
                recovery_id=uuid4().hex,
                reason=reason,
                interrupted_operation_ids=tuple(op.operation_id for op in interrupted),
                settled_operation_ids=tuple(op.operation_id for op in pending),
            )
            finalized = tuple(
                op.model_copy(
                    update={
                        "status": OperationStatus.INTERRUPTED,
                        "completed_at": now(),
                        "error": reason,
                    }
                )
                for op in interrupted
            )
            run = graph.run
            records: list[VerificationRecord] = []
            for operation in pending:
                records.extend(self._step_records(graph, operation))
            blocks = blocked_nodes(graph, records)
            bad = {block.node_id for block in blocks} | graph.node_blocks.keys()
            final_ids = (
                pending[-1].output_node_ids
                if pending
                and not interrupted
                and pending[-1].operation_type == OperationType.FINISH
                and not bad.intersection(pending[-1].output_node_ids)
                and not any(
                    record.target_id == pending[-1].operation_id
                    and record.verification_type == "structure"
                    and record.status == VerificationStatus.FAILED
                    for record in records
                )
                else ()
            )
            run = graph.run.model_copy(
                update={
                    "status": RunStatus.COMPLETED if final_ids else RunStatus.PARTIAL,
                    "updated_at": now(),
                    "error": None if final_ids else reason,
                    "final_node_ids": final_ids,
                }
            )
            self._append(
                graph,
                RunEvent(
                    run=run,
                    operations=finalized,
                    recovery=recovery,
                    node_blocks=blocks,
                    verifications=tuple(records),
                    verified_operation_ids=recovery.settled_operation_ids,
                ),
            )
        return RecoveryResult(
            run_id=run_id,
            status=graph.run.status,
            recovery=recovery,
            pending_tool_operation_ids=graph.pending_tool_operations(),
        )

    def confirm_tool_outcome(self, run_id: str, operation_id: str, *, reason: str) -> None:
        if not reason.strip():
            raise ValueError("Provide a confirmation reason")
        graph = self.get_graph(run_id)
        if operation_id not in graph.pending_tool_operations():
            raise RRMError(f"Tool operation {operation_id!r} is not awaiting confirmation")
        self._append(
            graph,
            RunEvent(
                tool_confirmation=ToolConfirmation(
                    operation_id=operation_id,
                    reason=reason,
                )
            ),
        )

    def resume(self, run_id: str, *, limits: Limits | None = None) -> RunResult:
        graph = self.get_graph(run_id)
        if graph.run.status == RunStatus.RUNNING:
            raise RRMError("Cannot resume a running run; recover it after its execution stops")
        if graph.turn is not None and graph.continuity is None:
            raise RRMError("Turn stopped before continuity was recorded; submit the question again")
        if graph.continuity and graph.continuity.continuity_type == ContinuityType.CORRECTION:
            raise RRMError("Correction requires explicit review before starting a new turn")
        if graph.pending_tool_operations():
            raise RRMError("Confirm the outcome of pending tool operations before resuming")
        if any(
            op.status == OperationStatus.COMPLETED
            and op.operation_id not in graph.verified_operations
            for op in graph.operations.values()
        ):
            self.verify_run(run_id)
            graph = self.get_graph(run_id)
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
            verification_required=True,
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
                run=run.model_copy(update={"updated_at": now(), "status": RunStatus.RUNNING}),
                nodes=(replacement,),
                operations=(completed,),
                dependencies=(edge,),
                invalidations=invalidations,
            ),
        )
        self._verify_step(graph, completed, settled_run=run)
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
        self._refresh_imports(graph)
        context = self.context_selector.select(
            graph,
            (
                ToolDescription(
                    name=tool.name, description=tool.description, input_schema=tool.input_schema
                )
                for tool in self.tools.values()
            ),
        )
        digest = content_hash(context.model_dump(mode="json"))

        def log(
            proposal: OperationProposal | None,
            error: str | None,
            stop: InvocationStop | None,
        ) -> None:
            self._append(
                graph,
                RunEvent(
                    planning=PlanningRecord(
                        model_id=self.model.model_id,
                        context_hash=digest,
                        proposed=proposal,
                        invocation_stop=stop,
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

    def _budget(self, graph: ReasoningGraph, limits: Limits) -> ExecutionBudget:
        return ExecutionBudget(
            limits,
            record_model_call=lambda: self._append(graph, RunEvent(model_call_started=True)),
        )

    def _drive(
        self,
        graph: ReasoningGraph,
        limits: Limits,
        *,
        budget: ExecutionBudget | None = None,
    ) -> RunResult:
        budget = budget if budget is not None else self._budget(graph, limits)
        try:
            for _ in range(limits.max_steps):
                proposal = self._plan(graph, budget)
                finished = self._execute(graph, proposal, budget)
                if finished:
                    break
            else:
                raise LimitReached("max_steps reached")
        except InvocationAborted as exc:
            status = {
                "run_timeout": RunStatus.PARTIAL,
                "cancelled": RunStatus.CANCELLED,
            }.get(exc.stop.reason, RunStatus.FAILED)
            self._set_status(graph, status, error=str(exc))
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
            reasoning_mode=graph.reasoning_mode,
            model_calls=graph.model_calls,
            error=run.error,
        )

    def _refresh_imports(self, graph: ReasoningGraph) -> None:
        graph.unavailable_imports.clear()
        verifier = ReferenceVerifier(self.storage)
        for node_id, reference in graph.references.items():
            try:
                verifier.check_reference(graph, reference)
            except InvalidReference:
                graph.unavailable_imports.add(node_id)
                graph.unavailable_imports.update(graph.descendants(node_id) & graph.nodes.keys())

    def _inputs(self, graph: ReasoningGraph, ids: tuple[str, ...]) -> tuple[NodeInput, ...]:
        ReferenceVerifier(self.storage).check_inputs(graph, ids)
        inputs = []
        for node_id in ids:
            node = graph.nodes.get(node_id)
            if node is None:
                raise InvalidReference(f"Input node {node_id!r} does not exist in this run")
            usability = graph.node_usability(node_id)
            if not usability.usable:
                raise InvalidReference(f"Input node {node_id!r} is {usability.status}")
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
            verification_required=True,
            retry_safe=tool is None or tool.retry_safe,
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
            ReferenceVerifier(self.storage).check_inputs(graph, operation.provided_node_ids)
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
            stop = exc.stop if isinstance(exc, InvocationAborted) else None
            uncertain_effect = tool is not None and not operation.retry_safe
            failed = operation.model_copy(
                update={
                    "status": OperationStatus.INTERRUPTED
                    if uncertain_effect
                    else OperationStatus.FAILED,
                    "completed_at": now(),
                    "error": error,
                    "invocation_stop": stop,
                }
            )
            self._append(graph, RunEvent(operations=(failed,)))
            if isinstance(exc, (RRMError, KeyboardInterrupt)):
                raise
            raise RRMError(f"{kind} execution failed ({error})") from exc
        self._append(
            graph,
            RunEvent(
                run=graph.run.model_copy(update={"updated_at": now()}),
                operations=(completed,),
                nodes=nodes,
                dependencies=edges(output_ids, DependencyType.PRODUCES),
            ),
        )
        return self._verify_step(graph, completed)

    def _verify_step(
        self,
        graph: ReasoningGraph,
        operation: ReasoningOperation,
        *,
        settled_run: ReasoningRun | None = None,
    ) -> bool:
        records = self._step_records(graph, operation)
        blocks = blocked_nodes(graph, records)
        finished = (
            operation.operation_type == OperationType.FINISH
            and bool(operation.output_node_ids)
            and not blocks
        )
        run = settled_run or graph.run
        if finished:
            run = run.model_copy(
                update={
                    "status": RunStatus.COMPLETED,
                    "final_node_ids": operation.output_node_ids,
                    "error": None,
                }
            )
        self._append(
            graph,
            RunEvent(
                run=run.model_copy(update={"updated_at": now()}),
                verifications=tuple(records),
                node_blocks=blocks,
                verified_operation_ids=(operation.operation_id,),
            ),
        )
        return finished

    def _step_records(
        self, graph: ReasoningGraph, operation: ReasoningOperation
    ) -> list[VerificationRecord]:
        structure = StructuralVerifier()
        provenance = ProvenanceVerifier()
        records = [
            *structure.verify_operation(graph, operation),
            *DeterministicVerifier().verify_operation(graph, operation),
        ]
        for node_id in operation.output_node_ids:
            records.extend(structure.verify_node(graph, node_id))
            records.extend(provenance.verify_node(graph, graph.nodes[node_id]))
            if node_id in graph.references:
                records.append(
                    ReferenceVerifier(self.storage).verify_node(graph, graph.references[node_id])
                )
        try:
            ReferenceVerifier(self.storage).check_inputs(graph, operation.provided_node_ids)
        except InvalidReference as exc:
            records.append(
                ReferenceVerifier(self.storage).record(
                    operation.operation_id,
                    "operation",
                    "cross_run_reference",
                    VerificationStatus.FAILED,
                    reason=str(exc),
                )
            )
        return records

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
                node_type = (
                    NodeType.HYPOTHESIS if result.node_type == NodeType.FACT else result.node_type
                )
                value = result.content
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
        raw = budget.invoke(
            lambda: tool.execute(request, provided),
            tool.timeout,
            tool.name,
            execution_id=op.operation_id,
        )
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
