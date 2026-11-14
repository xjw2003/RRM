from collections.abc import Sequence
from decimal import (
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DivisionByZero,
    InvalidOperation,
    Overflow,
    localcontext,
)
from operator import add, mul, sub, truediv

from pydantic import JsonValue

from rrm.protocol.models import ComputeParameters, NodeInput

_OPERATORS = {"add": add, "subtract": sub, "multiply": mul, "divide": truediv}


def arithmetic(inputs: Sequence[NodeInput], parameters: dict[str, JsonValue]) -> str:
    operator = ComputeParameters.model_validate(parameters).operator

    def number(value: JsonValue) -> Decimal:
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ValueError("COMPUTE inputs must be finite decimal values")
        result = Decimal(str(value))
        if not result.is_finite():
            raise ValueError("COMPUTE inputs must be finite decimal values")
        return result

    left, right = (number(item.content) for item in inputs)
    context = Context(
        prec=34,
        rounding=ROUND_HALF_EVEN,
        Emin=-6143,
        Emax=6144,
        clamp=1,
        traps=[InvalidOperation, DivisionByZero, Overflow],
    )
    with localcontext(context):
        return str(_OPERATORS[operator](left, right))
