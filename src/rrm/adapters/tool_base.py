from abc import ABC, abstractmethod
from typing import Any

from pydantic import JsonValue, TypeAdapter

from rrm.protocol.models import EvidenceItem, NodeInput

EVIDENCE_ITEMS = TypeAdapter(list[EvidenceItem])


class ToolAdapter(ABC):
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] = EVIDENCE_ITEMS.json_schema()
    timeout: float = 10
    max_calls: int | None = None
    retry_safe: bool = False
    source_type: str
    source_id: str

    @abstractmethod
    def execute(self, parameters: dict[str, JsonValue], inputs: tuple[NodeInput, ...]) -> Any:
        """Return evidence items. Both parameters and inputs are untrusted data."""


class MockRetrievalTool(ToolAdapter):
    name = "mock_search"
    retry_safe = True
    description = "Search the synthetic local documents for query words."
    source_type = "mock_corpus"
    source_id = "local_fixture"
    input_schema = {
        "type": "object",
        "properties": {"query": {"type": "string", "minLength": 1}},
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, documents: dict[str, JsonValue]) -> None:
        self._documents = documents.copy()

    def execute(
        self, parameters: dict[str, JsonValue], inputs: tuple[NodeInput, ...]
    ) -> list[dict[str, JsonValue]]:
        words = str(parameters["query"]).casefold().split()
        return [
            {"document_id": doc_id, "document_version": "1", "content": content}
            for doc_id, content in self._documents.items()
            if any(word in str(content).casefold() for word in words)
        ]
