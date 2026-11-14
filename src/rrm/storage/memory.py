from threading import Lock

from rrm.exceptions import StorageConflict
from rrm.protocol.models import RunEvent
from rrm.storage.base import StorageBackend


class MemoryStorage(StorageBackend):
    def __init__(self) -> None:
        self._streams: dict[str, list[str]] = {}
        self._lock = Lock()

    def append(self, run_id: str, expected_sequence: int, event: RunEvent) -> None:
        payload = event.model_dump_json()
        with self._lock:
            stream = self._streams.setdefault(run_id, [])
            if len(stream) != expected_sequence:
                raise StorageConflict(f"Run {run_id} was updated concurrently")
            stream.append(payload)

    def events(self, run_id: str) -> tuple[RunEvent, ...]:
        with self._lock:
            payloads = tuple(self._streams.get(run_id, ()))
        return tuple(RunEvent.model_validate_json(payload) for payload in payloads)
