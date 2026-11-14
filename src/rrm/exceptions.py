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
