"""Rebuild a persisted graph as of a specific event without configuring a model or tool."""

import argparse
from pathlib import Path

from rrm.storage import SQLiteStorage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db", type=Path)
    parser.add_argument("run_id")
    parser.add_argument("--through-sequence", type=int)
    args = parser.parse_args()
    if not args.db.is_file():
        parser.error("Database does not exist")
    with SQLiteStorage(args.db) as storage:
        try:
            graph = storage.get_graph(args.run_id, through_sequence=args.through_sequence)
        except KeyError:
            parser.error("Run does not exist")
        except ValueError as exc:
            parser.error(str(exc))
        print(f"Replayed {graph.sequence} events for {graph.run.run_id}")
        for op in graph.operations.values():
            print(op.operation_type, op.status, op.input_node_ids, op.output_node_ids)


if __name__ == "__main__":
    main()
