from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Literal
from uuid import uuid4

from pydantic import JsonValue

from rrm.graph.dag import ReasoningGraph
from rrm.protocol.models import VerificationRecord, VerificationStatus

TargetType = Literal["run", "node", "operation"]


class Verifier(ABC):
    verifier_id: str

    @abstractmethod
    def verify(self, graph: ReasoningGraph) -> Iterable[VerificationRecord]:
        """Check a snapshot independently of the model that produced it."""

    def record(
        self,
        target_id: str,
        target_type: TargetType,
        kind: str,
        status: VerificationStatus,
        **details: JsonValue,
    ) -> VerificationRecord:
        return VerificationRecord(
            verification_id=uuid4().hex,
            target_id=target_id,
            target_type=target_type,
            verifier_id=self.verifier_id,
            verification_type=kind,
            status=status,
            details=details,
        )
