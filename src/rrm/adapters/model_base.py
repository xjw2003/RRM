from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rrm.protocol.models import NodeInput, OperationType, Record


class ToolDescription(Record):
    name: str
    description: str
    input_schema: dict[str, Any]


class PlanningContext(Record):
    goal: str
    nodes: tuple[NodeInput, ...]
    tools: tuple[ToolDescription, ...]


class ExecutionContext(Record):
    operation_type: OperationType
    objective: str
    inputs: tuple[NodeInput, ...]


@dataclass(frozen=True)
class ModelReply:
    payload: Any
    total_tokens: int | None = None


class ModelAdapter(ABC):
    model_id: str
    timeout: float = 30

    @abstractmethod
    def propose_next_operation(self, context: PlanningContext) -> ModelReply:
        """Propose one operation without creating result nodes."""

    @abstractmethod
    def execute_reasoning(self, context: ExecutionContext) -> ModelReply:
        """Execute INFER or COMPARE using the supplied input snapshot."""

    @abstractmethod
    def generate_final_answer(self, context: ExecutionContext) -> ModelReply:
        """Synthesize a final answer from the supplied input snapshot."""


class MockModelAdapter(ModelAdapter):
    def __init__(
        self,
        planner: Callable[[PlanningContext], Any],
        reasoner: Callable[[ExecutionContext], Any],
        finalizer: Callable[[ExecutionContext], Any],
        *,
        model_id: str = "mock",
    ) -> None:
        self.model_id = model_id
        self._planner = planner
        self._reasoner = reasoner
        self._finalizer = finalizer

    @staticmethod
    def _reply(payload: Any) -> ModelReply:
        return payload if isinstance(payload, ModelReply) else ModelReply(payload)

    def propose_next_operation(self, context: PlanningContext) -> ModelReply:
        return self._reply(self._planner(context))

    def execute_reasoning(self, context: ExecutionContext) -> ModelReply:
        return self._reply(self._reasoner(context))

    def generate_final_answer(self, context: ExecutionContext) -> ModelReply:
        return self._reply(self._finalizer(context))
