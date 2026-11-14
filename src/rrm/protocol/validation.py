import hashlib
import json
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel
from pydantic.json_schema import models_json_schema

from rrm.protocol.models import (
    EvidenceItem,
    FinalOutput,
    ModelOutput,
    NodeInput,
    OperationProposal,
    ReasoningDependency,
    ReasoningNode,
    ReasoningOperation,
    ReasoningRun,
    RunEvent,
    RunResult,
    VerificationRecord,
)


def content_hash(value: Any) -> str:
    data = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def node_input(node: ReasoningNode) -> NodeInput:
    return NodeInput(node_id=node.node_id, node_type=node.node_type, content=node.content)


def snapshot_hash(nodes: Sequence[NodeInput]) -> str:
    return content_hash([node.model_dump(mode="json") for node in nodes])


def protocol_schema() -> dict[str, Any]:
    models: list[type[BaseModel]] = [
        ReasoningRun,
        ReasoningNode,
        ReasoningOperation,
        ReasoningDependency,
        VerificationRecord,
        OperationProposal,
        ModelOutput,
        FinalOutput,
        EvidenceItem,
        RunResult,
        RunEvent,
    ]
    _, schema = models_json_schema([(model, "validation") for model in models], title="RRP v0.1")
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", **schema}
