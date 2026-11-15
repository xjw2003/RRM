"""Read a saved conversation graph directly from SQLite."""

import argparse
import json
from pathlib import Path

from rrm.graph.conversation import ConversationGraph
from rrm.storage import SQLiteStorage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db", type=Path)
    parser.add_argument("conversation_id")
    args = parser.parse_args()
    if not args.db.is_file():
        parser.error("Database does not exist")
    with SQLiteStorage(args.db) as storage:
        graph = ConversationGraph(
            storage.get_conversation(args.conversation_id),
            (
                storage.get_graph(turn.run_id)
                for turn in storage.conversation_turns(args.conversation_id)
            ),
        )
        print(json.dumps(graph.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
