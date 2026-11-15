import hashlib
import json
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel
from pydantic.json_schema import models_json_schema

from rrm.protocol.models import (
    EvidenceItem,
    FinalOutput,
    ModelOutput,
    NodeInput,
    NodeUsability,
    OperationProposal,
    ReasoningDependency,
    ReasoningNode,
    ReasoningOperation,
    ReasoningRun,
    RecoveryResult,
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


def protocol_schema(version: Literal["0.1", "0.2"] = "0.1") -> dict[str, Any]:
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
        RecoveryResult,
        NodeUsability,
        RunEvent,
    ]
    _, schema = models_json_schema(
        [(model, "validation") for model in models],
        title=f"RRP v{version}",
    )
    if version == "0.1":
        definitions = schema["$defs"]
        definitions["ReasoningRun"]["properties"]["protocol_version"] = {
            "const": "0.1",
            "type": "string",
            "default": "0.1",
        }
        kinds = definitions["ReasoningOperation"]["properties"]["operation_type"]["anyOf"]
        kinds[-1] = {"const": "REVISE", "type": "string"}
        for field in ("turn", "continuity", "reuse_decision", "cross_run_reference"):
            definitions["RunEvent"]["properties"][field] = {"type": "null", "default": None}
        for definition in (
            "ContinuityRecord",
            "ContinuityType",
            "ConversationTurn",
            "CrossRunReference",
            "ReuseDecision",
            "SourceNode",
        ):
            del definitions[definition]
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", **schema}
