from contextvars import ContextVar
from dataclasses import dataclass
from threading import Event


@dataclass(frozen=True)
class InvocationControl:
    """Per-call control available to an adapter in its execution thread."""

    execution_id: str
    _cancelled: Event

    @property
    def cancellation_requested(self) -> bool:
        return self._cancelled.is_set()

    def wait_for_cancellation(self, timeout: float | None = None) -> bool:
        return self._cancelled.wait(timeout)


_invocation: ContextVar[InvocationControl | None] = ContextVar("rrm_invocation", default=None)


def current_invocation() -> InvocationControl | None:
    """Return cooperative cancellation control during a runtime adapter call."""
    return _invocation.get()
