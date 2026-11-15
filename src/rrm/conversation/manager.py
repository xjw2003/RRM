from uuid import uuid4

from pydantic import TypeAdapter

from rrm.adapters.openai_compatible import OpenAICompatibleAdapter
from rrm.conversation.continuity import (
    ContinuityContext,
    ContinuityDecision,
    ContinuityResolver,
    OpenAIContinuityResolver,
    RuleBasedContinuityResolver,
)
from rrm.conversation.imports import import_node
from rrm.conversation.reuse import NodeReuseResolver
from rrm.exceptions import InvalidReference, InvocationAborted, LimitReached, StorageError
from rrm.graph.conversation import ConversationGraph
from rrm.graph.dag import ReasoningGraph
from rrm.protocol.models import (
    ContinuityRecord,
    ContinuityType,
    Conversation,
    ConversationTurn,
    CrossRunReference,
    Limits,
    PositiveSeconds,
    ReuseDecision,
    RunEvent,
    RunResult,
    RunStatus,
    SourceNode,
)
from rrm.runtime.engine import ReasoningRuntime
from rrm.storage.base import StorageBackend


class ConversationResult(RunResult):
    conversation_id: str
    turn_id: str
    continuity_type: ContinuityType
    imported_node_ids: tuple[str, ...]
    correction_pending: bool = False

    @property
    def run_status(self) -> RunStatus:
        return self.status


class ConversationRuntime:
    def __init__(
        self,
        core: ReasoningRuntime,
        *,
        storage: StorageBackend | None = None,
        resolver: ContinuityResolver | None = None,
        allow_unverified: bool = True,
    ) -> None:
        if storage is not None and storage is not core.storage:
            raise ValueError("Conversation and core must use the same storage instance")
        self.core = core
        self.storage = core.storage
        if resolver is None:
            resolver = (
                OpenAIContinuityResolver(core.model)
                if isinstance(core.model, OpenAICompatibleAdapter)
                else RuleBasedContinuityResolver()
            )
        self.resolver = resolver
        self._resolver_timeout = TypeAdapter(PositiveSeconds).validate_python(resolver.timeout)
        self.reuse = NodeReuseResolver(self.storage)
        self.allow_unverified = allow_unverified

    def create_conversation(self, conversation_id: str | None = None) -> Conversation:
        if conversation_id is not None and not conversation_id.strip():
            raise ValueError("Provide a nonblank conversation ID")
        conversation = Conversation(conversation_id=conversation_id or uuid4().hex)
        self.storage.create_conversation(conversation)
        return self.storage.get_conversation(conversation.conversation_id)

    def list_turns(self, conversation_id: str) -> tuple[ConversationTurn, ...]:
        self.storage.get_conversation(conversation_id)
        return self.storage.conversation_turns(conversation_id)

    def get_run(self, conversation_id: str, run_id: str) -> ReasoningGraph:
        turn = self.storage.get_turn_by_run(run_id)
        if turn is None or turn.conversation_id != conversation_id:
            raise KeyError(run_id)
        return self.core.get_graph(run_id)

    def get_references(self, conversation_id: str, run_id: str) -> tuple[CrossRunReference, ...]:
        return tuple(self.get_run(conversation_id, run_id).references.values())

    def get_conversation_graph(self, conversation_id: str) -> ConversationGraph:
        return ConversationGraph(
            self.storage.get_conversation(conversation_id),
            (self.core.replay(turn.run_id) for turn in self.list_turns(conversation_id)),
        )

    def ask(
        self,
        question: str,
        *,
        conversation_id: str | None = None,
        limits: Limits | None = None,
    ) -> ConversationResult:
        if not question.strip():
            raise ValueError("Provide a nonblank question")
        conversation = self.create_conversation(conversation_id)
        earlier = self.list_turns(conversation.conversation_id)
        turn = ConversationTurn(
            conversation_id=conversation.conversation_id,
            turn_id=uuid4().hex,
            run_id=uuid4().hex,
            question=question,
            turn_index=len(earlier) + 1,
        )
        graph = self.core._create_run(question, limits=limits, turn=turn)
        self.core._set_status(graph, RunStatus.RUNNING)
        budget = self.core._budget(graph, graph.run.limits)
        decision = ContinuityDecision(continuity_type=ContinuityType.NEW_TOPIC, reason="First turn")
        candidates: set[tuple[str, str]] = set()
        if earlier:
            context = ContinuityContext(
                question=question,
                previous_question=earlier[-1].question,
                candidates=self.reuse.candidates(question, earlier),
            )
            candidates = {(c.source_run_id, c.source_node_id) for c in context.candidates}
            try:
                if self.resolver.uses_model:
                    decision = budget.model_response(
                        lambda: self.resolver.resolve(context),
                        ContinuityDecision,
                        timeout=self._resolver_timeout,
                        retries=0,
                        label="continuity",
                    )
                else:
                    reply = budget.invoke(
                        lambda: self.resolver.resolve(context), self._resolver_timeout, "continuity"
                    )
                    decision = ContinuityDecision.model_validate(reply)
            except StorageError:
                raise
            except InvocationAborted as exc:
                if exc.stop.reason == "cancelled":
                    self.core._append(
                        graph,
                        RunEvent(
                            continuity=ContinuityRecord(
                                continuity_type=ContinuityType.NEW_TOPIC,
                                reason="Continuity cancelled",
                            )
                        ),
                    )
                    self.core._set_status(graph, RunStatus.CANCELLED, error=str(exc))
                    return self._result(graph)
                decision = ContinuityDecision(
                    continuity_type=ContinuityType.NEW_TOPIC,
                    reason=f"Continuity interrupted ({exc.stop.reason}); reuse disabled",
                )
            except Exception as exc:
                decision = ContinuityDecision(
                    continuity_type=ContinuityType.NEW_TOPIC,
                    reason=f"Continuity could not be resolved ({type(exc).__name__}); reuse disabled",
                )
        self.core._append(
            graph,
            RunEvent(
                continuity=ContinuityRecord.model_validate(
                    decision.model_dump(exclude={"selected"}),
                )
            ),
        )
        for selection in dict.fromkeys(
            (source.source_run_id, source.source_node_id) for source in decision.selected
        ):
            source = SourceNode(source_run_id=selection[0], source_node_id=selection[1])
            try:
                if selection not in candidates:
                    raise InvalidReference("Source is outside the offered candidate set")
                if any(
                    (p.source_run_id, p.source_node_id) not in candidates
                    for p in decision.changed_premises
                ):
                    raise InvalidReference("Changed premise is outside the offered candidate set")
                budget.check_time()
                import_node(
                    self.core,
                    graph,
                    source,
                    decision.reason,
                    allow_unverified=self.allow_unverified,
                )
            except InvalidReference as exc:
                self.core._append(
                    graph,
                    RunEvent(
                        reuse_decision=ReuseDecision(**source.model_dump(), reason=str(exc)),
                    ),
                )
            except LimitReached:
                break

        if decision.continuity_type == ContinuityType.CORRECTION:
            self.core._set_status(
                graph,
                RunStatus.PARTIAL,
                error="Correction pending explicit review",
            )
        else:
            self.core._drive(graph, graph.run.limits, budget=budget)
        return self._result(graph)

    def _result(self, graph: ReasoningGraph) -> ConversationResult:
        turn, continuity = graph.turn, graph.continuity
        assert turn is not None and continuity is not None
        return ConversationResult(
            **self.core._result(graph).model_dump(),
            conversation_id=turn.conversation_id,
            turn_id=turn.turn_id,
            continuity_type=continuity.continuity_type,
            imported_node_ids=tuple(graph.references),
            correction_pending=continuity.continuity_type == ContinuityType.CORRECTION,
        )
