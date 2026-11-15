from collections.abc import Iterable, Iterator

from rrm.exceptions import GraphError, InvalidReference
from rrm.graph.dag import ReasoningGraph
from rrm.protocol.models import (
    ConversationTurn,
    CrossRunReference,
    NodeType,
    OperationStatus,
    ReasoningNode,
    RunStatus,
    SourceNode,
    UsabilityStatus,
    VerificationRecord,
    VerificationStatus,
)
from rrm.protocol.validation import content_hash, node_input, snapshot_hash
from rrm.storage.base import StorageBackend
from rrm.verification.base import Verifier
from rrm.verification.provenance import ProvenanceVerifier
from rrm.verification.structure import StructuralVerifier


class ReferenceVerifier(Verifier):
    verifier_id = "rrm.reference"

    def __init__(self, storage: StorageBackend) -> None:
        self.storage = storage
        self._graphs: dict[str, ReasoningGraph] = {}

    def _load(self, run_id: str) -> ReasoningGraph:
        if run_id not in self._graphs:
            try:
                self._graphs[run_id] = self.storage.get_graph(run_id)
            except (KeyError, GraphError) as exc:
                raise InvalidReference("Source run is missing or structurally invalid") from exc
        return self._graphs[run_id]

    def _source(
        self, target: ConversationTurn, source: SourceNode
    ) -> tuple[ReasoningGraph, ReasoningNode, UsabilityStatus]:
        turn = self.storage.get_turn_by_run(source.source_run_id)
        if turn is None or turn.conversation_id != target.conversation_id:
            raise InvalidReference("Source is outside the target conversation")
        if turn.turn_index >= target.turn_index or turn.run_id == target.run_id:
            raise InvalidReference("Source must belong to an earlier turn")
        graph = self._load(turn.run_id)
        if graph.turn != turn or graph.run.status != RunStatus.COMPLETED:
            raise InvalidReference("Source Run must be a completed turn")
        node = graph.nodes.get(source.source_node_id)
        if node is None or node.node_type == NodeType.QUESTION:
            raise InvalidReference("Source must be an existing reasoning output")
        producer = graph.operations.get(node.produced_by or "")
        if (
            producer is None
            or producer.status != OperationStatus.COMPLETED
            or (
                graph.parents(node.node_id) != {producer.operation_id}
                or node.node_id not in producer.output_node_ids
            )
        ):
            raise InvalidReference("Source lacks a consistent completed producer")
        status = graph.node_usability(node.node_id)
        if not status.usable:
            raise InvalidReference(f"Source node is {status.status}")
        if content_hash(node.content) != node.content_hash:
            raise InvalidReference("Source content hash mismatch")
        problems = StructuralVerifier._operation(graph, producer)
        if problems:
            raise InvalidReference("Source producer failed structure validation: " + problems[0])
        if any(
            record.status == VerificationStatus.FAILED
            for record in ProvenanceVerifier().verify_node(graph, node)
        ):
            raise InvalidReference("Source evidence snapshot failed validation")
        return graph, node, status.status

    def source(
        self, target: ConversationTurn, source: SourceNode
    ) -> tuple[ReasoningGraph, ReasoningNode, UsabilityStatus]:
        graph, node, status = self._source(target, source)
        self.check_inputs(graph, (node.node_id,))
        return graph, node, status

    def lineage(self, graph: ReasoningGraph, node_id: str) -> set[tuple[str, str]]:
        """Collect data ancestry across recorded IMPORT edges."""
        pending = [(graph, node_id)]
        result: set[tuple[str, str]] = set()
        while pending:
            current, target_id = pending.pop()
            for local_id in ({target_id} | current.ancestors(target_id)) & current.nodes.keys():
                key = (current.run.run_id, local_id)
                if key in result:
                    continue
                result.add(key)
                ref = current.references.get(local_id)
                if ref is not None:
                    pending.append((self._load(ref.source_run_id), ref.source_node_id))
        return result

    def origin(self, graph: ReasoningGraph, node_id: str) -> tuple[str, str]:
        while ref := graph.references.get(node_id):
            source = self._load(ref.source_run_id)
            if (
                graph.turn is None
                or source.turn is None
                or (
                    source.turn.conversation_id != graph.turn.conversation_id
                    or source.turn.turn_index >= graph.turn.turn_index
                )
            ):
                raise InvalidReference("IMPORT lineage must point to an earlier turn")
            graph, node_id = source, ref.source_node_id
        return graph.run.run_id, node_id

    def _check_reference(self, graph: ReasoningGraph, ref: CrossRunReference) -> ReasoningGraph:
        if graph.turn is None or ref.target_run_id != graph.run.run_id:
            raise InvalidReference("Reference has an invalid target turn")
        source_graph, source, status = self._source(graph.turn, ref)
        target = graph.nodes.get(ref.target_node_id)
        op = graph.operations.get(ref.import_operation_id)
        if (
            target is None
            or op is None
            or op.operation_type != "IMPORT"
            or op.executor_type != "runtime"
            or op.executor_id != "conversation_import"
            or op.status != OperationStatus.COMPLETED
            or op.output_node_ids != (target.node_id,)
            or graph.parents(target.node_id) != {op.operation_id}
            or target.produced_by != op.operation_id
            or source.revision != ref.source_revision
            or source.content_hash != ref.source_content_hash
            or target.content_hash != source.content_hash
            or content_hash(target.content) != source.content_hash
            or target.node_type != source.node_type
            or target.provenance != source.provenance
            or op.input_snapshot_hash != snapshot_hash((node_input(source),))
            or ref.source_sequence > source_graph.sequence
            or (ref.source_usability == UsabilityStatus.ELIGIBLE and status != ref.source_usability)
        ):
            raise InvalidReference(
                "IMPORT record, snapshot, version or source usability does not match"
            )

        return source_graph

    def check_reference(self, graph: ReasoningGraph, ref: CrossRunReference) -> None:
        source = self._check_reference(graph, ref)
        self.check_inputs(source, (ref.source_node_id,))

    def check_inputs(self, graph: ReasoningGraph, node_ids: Iterable[str]) -> None:
        if not graph.references:
            return
        pending = [(graph, node_id) for node_id in node_ids]
        checked: set[tuple[str, str]] = set()
        while pending:
            current, node_id = pending.pop()
            ancestors = {node_id} | current.ancestors(node_id)
            for imported_id in ancestors & current.references.keys():
                key = (current.run.run_id, imported_id)
                if key in checked:
                    continue
                checked.add(key)
                if not current.node_usability(imported_id).usable:
                    raise InvalidReference("An imported ancestor is not usable")
                ref = current.references[imported_id]
                source = self._check_reference(current, ref)
                pending.append((source, ref.source_node_id))

    def verify_node(self, graph: ReasoningGraph, ref: CrossRunReference) -> VerificationRecord:
        try:
            self.check_reference(graph, ref)
        except InvalidReference as exc:
            status, reason = VerificationStatus.FAILED, str(exc)
        else:
            status, reason = VerificationStatus.PASSED, "Reference matches its source version"
        return self.record(
            ref.target_node_id,
            "node",
            "cross_run_reference",
            status,
            reason=reason,
            source_run_id=ref.source_run_id,
            source_node_id=ref.source_node_id,
            scope="Reference integrity and current eligibility",
        )

    def verify(self, graph: ReasoningGraph) -> Iterator[VerificationRecord]:
        for ref in graph.references.values():
            yield self.verify_node(graph, ref)
