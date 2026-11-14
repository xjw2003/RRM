from rrm.protocol.models import InvocationStop


class RRMError(Exception):
    """A runtime error suitable for exposing in the run record."""


class InvalidOutput(RRMError):
    """An adapter response violated the expected contract."""


class InvalidReference(RRMError):
    """An input node is missing or unusable in this run."""


class GraphError(RRMError):
    """An edge violates the graph's invariants."""


class StorageError(RRMError):
    """The requested storage change could not be committed."""


class StorageConflict(StorageError):
    """The event stream changed since it was read."""


class LimitReached(RRMError):
    """The runtime reached a configured execution limit."""


class AdapterRequestTimeout(RRMError):
    """The adapter reported a transport timeout."""


class InvocationAborted(RRMError):
    def __init__(self, message: str, stop: InvocationStop) -> None:
        super().__init__(message)
        self.stop = stop
