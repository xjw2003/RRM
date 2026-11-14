import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from rrm.protocol.models import OperationProposal, ReasoningRun
from rrm.protocol.validation import content_hash, protocol_schema


def test_schema_matches_protocol():
    schema = json.loads((Path(__file__).parents[1] / "schemas/rrp-v0.1.schema.json").read_text())
    assert schema == protocol_schema()
    Draft202012Validator.check_schema(schema)
    example = ReasoningRun(run_id="r", goal="example", model_id="mock").model_dump(mode="json")
    Draft202012Validator({**schema, "$ref": "#/$defs/ReasoningRun"}).validate(example)


def test_content_hash_is_order_independent_and_rejects_nonfinite_values():
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})
    with pytest.raises(ValueError):
        content_hash(float("nan"))


@pytest.mark.parametrize(
    "invalid",
    [
        {"output_node_ids": ["invented"]},
        {"operation_type": "VERIFY"},
        {"input_node_ids": ["a", "a"]},
        {"input_node_ids": []},
        {"operation_type": "RETRIEVE"},
        {"parameters": {"invented": True}},
    ],
)
def test_reject_invalid_proposals(invalid):
    proposal = {"operation_type": "INFER", "input_node_ids": ["a"], "objective": "Infer"}
    with pytest.raises(ValidationError):
        OperationProposal.model_validate(proposal | invalid)
