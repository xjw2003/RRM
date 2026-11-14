"""Register a synthetic RAG tool and persist the graph in a local SQLite database."""

import argparse
from pathlib import Path

from mock_reasoning import GOAL, make_runtime

from rrm.storage import SQLiteStorage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("reasoning.db"))
    args = parser.parse_args()
    with SQLiteStorage(args.db) as storage:
        runtime = make_runtime(storage)
        result = runtime.run(GOAL)
        records = runtime.verify_run(result.run_id)
        print(result.model_dump_json(indent=2))
        for record in records:
            if record.verification_type in {
                "provenance_snapshot",
                "source_version",
                "deterministic",
            }:
                print(record.model_dump_json())


if __name__ == "__main__":
    main()
