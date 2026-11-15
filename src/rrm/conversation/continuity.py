import re
from abc import ABC, abstractmethod
from collections.abc import Callable

from pydantic import Field, JsonValue

from rrm.adapters.model_base import ModelReply
from rrm.adapters.openai_compatible import OpenAICompatibleAdapter
from rrm.protocol.models import (
    ContinuityRecord,
    ContinuityType,
    NodeType,
    Record,
    SourceNode,
    UsabilityStatus,
)


class HistoricalCandidate(SourceNode):
    revision: int
    node_type: NodeType
    label: str | None = None
    content: JsonValue
    usability: UsabilityStatus


class ContinuityContext(Record):
    question: str
    previous_question: str
    candidates: tuple[HistoricalCandidate, ...]


class ContinuityDecision(ContinuityRecord):
    selected: tuple[SourceNode, ...] = Field(default=(), max_length=8)


class ContinuityResolver(ABC):
    uses_model: bool = False
    timeout: float = 30

    @abstractmethod
    def resolve(self, context: ContinuityContext) -> ContinuityDecision | ModelReply:
        """Suggest one relation and a bounded set of historical node versions."""


class MockContinuityResolver(ContinuityResolver):
    def __init__(self, resolve: Callable[[ContinuityContext], ContinuityDecision]) -> None:
        self._resolve = resolve

    def resolve(self, context: ContinuityContext) -> ContinuityDecision:
        return self._resolve(context)


class OpenAIContinuityResolver(ContinuityResolver):
    uses_model = True

    def __init__(self, model: OpenAICompatibleAdapter) -> None:
        self.model = model
        self.timeout = model.timeout

    def resolve(self, context: ContinuityContext) -> ModelReply:
        return self.model._request(
            context,
            ContinuityDecision,
            "Decide whether the user continues, changes a premise hypothetically, corrects a "
            "premise, or starts a new topic. Select the minimum necessary IDs from candidates. "
            "For COUNTERFACTUAL, list the changed historical premises in changed_premises. "
            "For CORRECTION and NEW_TOPIC, leave selected empty. Explain the relationship briefly.",
        )


def relevance(question: str, text: str) -> int:
    def tokens(value: str) -> set[str]:
        words = set(re.findall(r"[a-z0-9_]+", value.casefold()))
        for phrase in re.findall(r"[\u3400-\u9fff]+", value):
            words.update(phrase[i : i + 2] for i in range(len(phrase) - 1))
        return words

    return len(tokens(question) & tokens(text))


class RuleBasedContinuityResolver(ContinuityResolver):
    """Conservative offline resolver for explicit follow-up wording."""

    def resolve(self, context: ContinuityContext) -> ContinuityDecision:
        question = context.question.casefold().strip()
        if re.search(r"换个|换一个|新话题|new topic|unrelated", question):
            kind = ContinuityType.NEW_TOPIC
        elif re.search(r"^\s*(如果|假如|假设|what if|suppose|assuming)", question):
            kind = ContinuityType.COUNTERFACTUAL
        elif re.search(r"错了|更正|纠正|incorrect|correction|was wrong|is wrong", question):
            kind = ContinuityType.CORRECTION
        elif re.search(r"刚才|上一轮|前面|此前|继续|earlier|previous|above|continue", question):
            kind = ContinuityType.CONTINUE
        else:
            kind = ContinuityType.NEW_TOPIC
        evidence = [
            candidate
            for candidate in context.candidates
            if candidate.node_type == NodeType.EVIDENCE
        ]

        def score(candidate: HistoricalCandidate) -> int:
            return relevance(question, f"{candidate.label or ''} {candidate.content}")

        def key(candidate: HistoricalCandidate) -> SourceNode:
            return SourceNode(
                source_run_id=candidate.source_run_id, source_node_id=candidate.source_node_id
            )

        matches = sorted((c for c in evidence if score(c)), key=score, reverse=True)
        selected: tuple[SourceNode, ...] = ()
        changed: tuple[SourceNode, ...] = ()
        if kind == ContinuityType.CONTINUE:
            related = matches or [c for c in context.candidates if score(c)]
            selected = tuple(key(c) for c in related[:1])
        elif kind == ContinuityType.COUNTERFACTUAL and matches:
            changed = tuple(key(c) for c in matches[:8])
            matching_runs = {c.source_run_id for c in matches}
            independent = [c for c in evidence if not score(c) and c.source_run_id in matching_runs]
            if len(independent) == 1:
                selected = (key(independent[0]),)
        return ContinuityDecision(
            continuity_type=kind,
            reason=f"Offline wording rule selected {kind.value}",
            selected=selected,
            changed_premises=changed,
        )
