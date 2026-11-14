from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator

from pydantic import JsonValue, TypeAdapter

from rrm.graph.dag import ReasoningGraph
from rrm.protocol.models import (
    Limits,
    NodeType,
    OperationStatus,
    OperationType,
    VerificationRecord,
    VerificationStatus,
)
from rrm.protocol.validation import content_hash
from rrm.runtime.planner import ExecutionBudget
from rrm.verification.base import Verifier

_SCOPE = "Checks snapshot integrity or correspondence to a source version, not factual truth."


class SourceAdapter(ABC):
    source_type: str
    source_id: str
    timeout: float = 10

    @abstractmethod
    def fetch_version(self, document_id: str, document_version: str) -> JsonValue:
        """Return the same evidence representation from the pinned source version."""


class ProvenanceVerifier(Verifier):
    verifier_id = "rrm.provenance"

    def __init__(self, sources: Iterable[SourceAdapter] = ()) -> None:
        self.sources: dict[tuple[str, str], SourceAdapter] = {}
        for source in sources:
            key = (source.source_type, source.source_id)
            if key in self.sources:
                raise ValueError(f"Duplicate source adapter {key}")
            Limits(timeout_seconds=source.timeout)
            self.sources[key] = source

    def verify(self, graph: ReasoningGraph) -> Iterator[VerificationRecord]:
        for node in graph.nodes.values():
            if node.node_type != NodeType.EVIDENCE:
                continue
            provenance = node.provenance
            producer = graph.operations.get(node.produced_by or "")
            registered_origin = (
                producer is not None
                and producer.operation_type == OperationType.RETRIEVE
                and producer.executor_type == "tool"
                and producer.status == OperationStatus.COMPLETED
                and node.node_id in producer.output_node_ids
                and graph.parents(node.node_id) == {producer.operation_id}
            )
            if not registered_origin or provenance is None:
                yield self.record(
                    node.node_id,
                    "node",
                    "provenance_snapshot",
                    VerificationStatus.NOT_VERIFIED,
                    reason="No recorded retrieval origin and source snapshot",
                    scope=_SCOPE,
                )
                continue
            digest = content_hash(provenance.content)
            if (
                digest != provenance.content_hash
                or digest != node.content_hash
                or (digest != content_hash(node.content))
            ):
                yield self.record(
                    node.node_id,
                    "node",
                    "provenance_snapshot",
                    VerificationStatus.FAILED,
                    reason="Evidence snapshot hash mismatch",
                    scope=_SCOPE,
                )
                continue
            yield self.record(
                node.node_id,
                "node",
                "provenance_snapshot",
                VerificationStatus.PASSED,
                source_id=provenance.source_id,
                document_version=provenance.document_version,
                scope=_SCOPE,
            )
            source = self.sources.get((provenance.source_type, provenance.source_id))
            doc, version = provenance.document_id, provenance.document_version
            status = VerificationStatus.NOT_VERIFIED
            reason = "No source adapter or pinned document version"
            if source is not None and doc and version:
                try:
                    budget = ExecutionBudget(Limits(timeout_seconds=source.timeout))
                    fetched = budget.invoke(
                        lambda: source.fetch_version(doc, version), source.timeout, "Source lookup"
                    )
                    fetched_hash = content_hash(TypeAdapter(JsonValue).validate_python(fetched))
                except Exception as exc:
                    reason = f"Source lookup failed ({type(exc).__name__})"
                else:
                    matches = fetched_hash == digest
                    status = VerificationStatus.PASSED if matches else VerificationStatus.FAILED
                    reason = "Source version matches" if matches else "Source version differs"
            yield self.record(
                node.node_id,
                "node",
                "source_version",
                status,
                reason=reason,
                scope=_SCOPE,
            )
