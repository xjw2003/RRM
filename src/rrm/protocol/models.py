from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

PROTOCOL_VERSION = "0.1"
NonEmpty = Annotated[str, Field(min_length=1)]
PositiveInt = Annotated[int, Field(gt=0, strict=True)]
PositiveSeconds = Annotated[float, Field(gt=0, allow_inf_nan=False)]


def now() -> datetime:
    return datetime.now(UTC)


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class RunStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NodeType(StrEnum):
    QUESTION = "question"
    FACT = "fact"
    EVIDENCE = "evidence"
    HYPOTHESIS = "hypothesis"
    INFERENCE = "inference"
    RESULT = "result"


class NodeState(StrEnum):
    CURRENT = "current"
    STALE = "stale"
    SUPERSEDED = "superseded"
    REVOKED = "revoked"


class UsabilityStatus(StrEnum):
    ELIGIBLE = "eligible"
    UNVERIFIED = "unverified"
    BLOCKED = "blocked"
    PENDING = "pending_verification"
    STALE = "stale"
    SUPERSEDED = "superseded"
    REVOKED = "revoked"


class NodeBlock(Record):
    node_id: NonEmpty
    cause_node_id: NonEmpty
    failed_checks: tuple[str, ...]
    verification_ids: tuple[str, ...]
    reasons: tuple[str, ...] = ()
    recorded_at: datetime = Field(default_factory=now)


class NodeUsability(Record):
    node_id: NonEmpty
    node_state: NodeState
    status: UsabilityStatus
    block: NodeBlock | None = None

    @property
    def usable(self) -> bool:
        return self.status in (UsabilityStatus.ELIGIBLE, UsabilityStatus.UNVERIFIED)


class OperationType(StrEnum):
    RETRIEVE = "RETRIEVE"
    INFER = "INFER"
    COMPARE = "COMPARE"
    COMPUTE = "COMPUTE"
    FINISH = "FINISH"


class OperationStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class DependencyType(StrEnum):
    CONSUMES = "operation_consumes_node"
    PRODUCES = "operation_produces_node"


class VerificationStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_VERIFIED = "not_verified"
    NOT_APPLICABLE = "not_applicable"


class Limits(Record):
    max_steps: PositiveInt = 12
    max_model_calls: PositiveInt = 24
    timeout_seconds: PositiveSeconds = 120
    max_tokens: PositiveInt | None = None


class ReasoningRun(Record):
    run_id: NonEmpty
    goal: NonEmpty
    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(default_factory=now)
    status: RunStatus = RunStatus.CREATED
    model_id: NonEmpty
    protocol_version: Literal["0.1"] = PROTOCOL_VERSION
    limits: Limits = Field(default_factory=Limits)
    final_node_ids: tuple[str, ...] = ()
    error: str | None = None


class EvidenceSnapshot(Record):
    source_type: NonEmpty
    source_id: NonEmpty
    document_id: str | None = None
    document_version: str | None = None
    content: JsonValue
    content_hash: NonEmpty
    retrieved_at: datetime = Field(default_factory=now)


class ReasoningNode(Record):
    node_id: NonEmpty
    run_id: NonEmpty
    node_type: NodeType
    content: JsonValue
    produced_by: str | None = None
    created_at: datetime = Field(default_factory=now)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    content_hash: NonEmpty
    provenance: EvidenceSnapshot | None = None
    logical_id: str | None = None
    revision: PositiveInt = 1
    supersedes_node_id: str | None = None


class NodeInvalidation(Record):
    node_id: NonEmpty
    state: Literal[NodeState.STALE, NodeState.SUPERSEDED, NodeState.REVOKED]
    cause_node_id: NonEmpty
    reason: NonEmpty
    recorded_at: datetime = Field(default_factory=now)


class InvalidationReport(Record):
    node_id: NonEmpty
    replacement_node_id: str | None = None
    affected_operation_ids: tuple[str, ...]
    stale_node_ids: tuple[str, ...]
    reason: NonEmpty


class NodeInput(Record):
    node_id: NonEmpty
    node_type: NodeType
    content: JsonValue


class InvocationStop(Record):
    execution_id: NonEmpty
    reason: Literal["request_timeout", "wait_timeout", "run_timeout", "cancelled"]
    worker_state: Literal["exited", "unknown"]
    cancellation_requested: bool


class ReasoningOperation(Record):
    operation_id: NonEmpty
    run_id: NonEmpty
    operation_type: OperationType | Literal["REVISE"]
    input_node_ids: tuple[str, ...]
    output_node_ids: tuple[str, ...] = ()
    provided_node_ids: tuple[str, ...] = ()
    executor_type: Literal["model", "tool", "runtime"]
    executor_id: NonEmpty
    objective: NonEmpty
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    status: OperationStatus = OperationStatus.RUNNING
    started_at: datetime = Field(default_factory=now)
    completed_at: datetime | None = None
    input_snapshot_hash: NonEmpty
    output_snapshot_hash: str | None = None
    verification_required: bool = False
    retry_safe: bool = False
    invocation_stop: InvocationStop | None = None
    error: str | None = None


class ReasoningDependency(Record):
    run_id: NonEmpty
    edge_type: DependencyType
    node_id: NonEmpty
    operation_id: NonEmpty

    @property
    def endpoints(self) -> tuple[str, str]:
        if self.edge_type == DependencyType.CONSUMES:
            return self.node_id, self.operation_id
        return self.operation_id, self.node_id


class VerificationRecord(Record):
    verification_id: NonEmpty
    target_id: NonEmpty
    target_type: Literal["run", "node", "operation"]
    verifier_id: NonEmpty
    verification_type: NonEmpty
    status: VerificationStatus
    details: dict[str, JsonValue] = Field(default_factory=dict)
    checked_at: datetime = Field(default_factory=now)


class ComputeParameters(Record):
    operator: Literal["add", "subtract", "multiply", "divide"]


class OperationProposal(Record):
    operation_type: OperationType
    input_node_ids: Annotated[tuple[NonEmpty, ...], Field(min_length=1)]
    objective: NonEmpty
    tool: NonEmpty | None = None
    parameters: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_operation(self) -> Self:
        if len(set(self.input_node_ids)) != len(self.input_node_ids):
            raise ValueError("input node IDs must be unique")
        if self.operation_type == OperationType.RETRIEVE:
            if self.tool is None:
                raise ValueError("RETRIEVE requires a registered tool")
        elif self.operation_type == OperationType.COMPUTE:
            if self.tool is not None:
                raise ValueError("COMPUTE uses the built-in arithmetic executor")
            ComputeParameters.model_validate(self.parameters)
            if len(self.input_node_ids) != 2:
                raise ValueError("COMPUTE requires exactly two input nodes")
        elif self.tool is not None or self.parameters:
            raise ValueError("model execution accepts only an objective and input IDs")
        return self


class ModelOutput(Record):
    node_type: Literal[NodeType.FACT, NodeType.HYPOTHESIS, NodeType.INFERENCE]
    content: JsonValue


class FinalOutput(Record):
    answer: NonEmpty


class EvidenceItem(Record):
    content: JsonValue
    document_id: str | None = None
    document_version: str | None = None


class PlanningRecord(Record):
    invocation_stop: InvocationStop | None = None
    model_id: NonEmpty
    context_hash: NonEmpty
    proposed: OperationProposal | None = None
    error: str | None = None
    checked_at: datetime = Field(default_factory=now)


class RunResult(Record):
    run_id: NonEmpty
    status: RunStatus
    answer: str | None = None
    final_node_ids: tuple[str, ...] = ()
    steps_executed: int
    error: str | None = None


class RecoveryRecord(Record):
    recovery_id: NonEmpty
    reason: NonEmpty
    interrupted_operation_ids: tuple[str, ...] = ()
    settled_operation_ids: tuple[str, ...] = ()
    recorded_at: datetime = Field(default_factory=now)


class ToolConfirmation(Record):
    operation_id: NonEmpty
    reason: NonEmpty
    recorded_at: datetime = Field(default_factory=now)


class RecoveryResult(Record):
    run_id: NonEmpty
    status: RunStatus
    recovery: RecoveryRecord | None = None
    pending_tool_operation_ids: tuple[str, ...] = ()


class RunEvent(Record):
    """One atomic change in an ordered, per-run event stream."""

    run: ReasoningRun | None = None
    nodes: tuple[ReasoningNode, ...] = ()
    operations: tuple[ReasoningOperation, ...] = ()
    dependencies: tuple[ReasoningDependency, ...] = ()
    planning: PlanningRecord | None = None
    invalidations: tuple[NodeInvalidation, ...] = ()
    verifications: tuple[VerificationRecord, ...] = ()
    verified_operation_ids: tuple[str, ...] = ()
    node_blocks: tuple[NodeBlock, ...] = ()
    recovery: RecoveryRecord | None = None
    tool_confirmation: ToolConfirmation | None = None
