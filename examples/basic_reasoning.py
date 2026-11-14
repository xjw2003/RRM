"""Use a provider's OpenAI-compatible Chat Completions endpoint."""

import argparse
import os

from rrm import ReasoningRuntime
from rrm.adapters import OpenAICompatibleAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("goal")
    args = parser.parse_args()
    model = OpenAICompatibleAdapter(
        base_url=os.environ["RRM_BASE_URL"],
        api_key=os.environ.get("RRM_API_KEY"),
        model=os.environ["RRM_MODEL"],
        timeout=float(os.environ.get("RRM_HTTP_TIMEOUT", "30")),
    )
    runtime = ReasoningRuntime(model)
    result = runtime.run(args.goal)
    print(result.model_dump_json(indent=2))
    print(runtime.export_json(result.run_id))
    if result.status != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
