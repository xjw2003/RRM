from collections.abc import Callable
from concurrent.futures import Future, wait
from threading import Thread
from time import monotonic
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from rrm.adapters.model_base import ModelReply
from rrm.exceptions import InvalidOutput, LimitReached, RRMError
from rrm.protocol.models import Limits
from rrm.protocol.validation import content_hash

T = TypeVar("T")
Response = TypeVar("Response", bound=BaseModel)


class ExecutionBudget:
    def __init__(self, limits: Limits) -> None:
        self.limits = limits
        self.deadline = monotonic() + limits.timeout_seconds
        self.model_calls = 0
        self.tokens = 0

    def check_time(self) -> float:
        remaining = self.deadline - monotonic()
        if remaining <= 0:
            raise LimitReached("timeout_seconds reached")
        return remaining

    def check_model(self) -> None:
        self.check_time()
        if self.model_calls >= self.limits.max_model_calls:
            raise LimitReached("max_model_calls reached")
        if self.limits.max_tokens is not None and self.tokens >= self.limits.max_tokens:
            raise LimitReached("max_tokens reached")

    def invoke(self, function: Callable[[], T], timeout: float, label: str) -> T:
        remaining = self.check_time()
        result: Future[T] = Future()

        def worker() -> None:
            try:
                result.set_result(function())
            except BaseException as exc:
                result.set_exception(exc)

        # A late return has no path to the graph or storage.
        Thread(target=worker, daemon=True).start()
        done, _ = wait((result,), timeout=min(remaining, timeout))
        if not done:
            if remaining <= timeout:
                raise LimitReached("timeout_seconds reached")
            raise RRMError(f"{label} timed out")
        self.check_time()
        return result.result()

    def model_response(
        self,
        function: Callable[[], ModelReply],
        response: type[Response],
        *,
        timeout: float,
        retries: int,
        label: str,
        on_attempt: Callable[[Response | None, str | None], None] | None = None,
    ) -> Response:
        last_error: str | None = None
        for _ in range(retries + 1):
            self.check_model()
            self.model_calls += 1
            try:
                reply = self.invoke(function, timeout, label)
                if not isinstance(reply, ModelReply):
                    raise InvalidOutput("adapter must return ModelReply")
                if self.limits.max_tokens is not None:
                    if type(reply.total_tokens) is not int or reply.total_tokens < 0:
                        raise RRMError("max_tokens requires model usage for every reply")
                    self.tokens += reply.total_tokens
                    if self.tokens > self.limits.max_tokens:
                        raise LimitReached("max_tokens exceeded")
                payload = reply.payload
                if isinstance(payload, BaseModel):
                    payload = payload.model_dump()
                output = response.model_validate(payload)
                try:
                    content_hash(output.model_dump(mode="json"))
                except ValueError as exc:
                    raise InvalidOutput("response contains values outside canonical JSON") from exc
            except ValidationError as exc:
                first = exc.errors(include_input=False, include_url=False)[0]
                field = ".".join(map(str, first["loc"])) or "<root>"
                last_error = f"{label}: invalid {response.__name__} at {field} ({first['type']})"
            except InvalidOutput as exc:
                last_error = f"{label}: {exc}"
            except Exception as exc:
                if on_attempt is not None:
                    on_attempt(None, str(exc) if isinstance(exc, RRMError) else type(exc).__name__)
                raise
            else:
                if on_attempt is not None:
                    on_attempt(output, None)
                return output
            if on_attempt is not None:
                on_attempt(None, last_error)
        raise InvalidOutput(last_error or f"{label}: invalid response")
