"""Demonstrate retrieval, direct follow-up and optional counterfactual on one conversation."""

import argparse
from pathlib import Path
from typing import Any

from rrm import ReasoningRuntime
from rrm.adapters import MockModelAdapter, MockRetrievalTool
from rrm.adapters.model_base import ExecutionContext, PlanningContext
from rrm.conversation import ConversationRuntime
from rrm.protocol.models import NodeInput, NodeType
from rrm.storage import SQLiteStorage, StorageBackend

QUESTIONS = (
    "为什么收入增长，利润却下降？",
    "刚才成本增加了多少？",
    "如果成本没有增加呢？",
)


class CountingRetrievalTool(MockRetrievalTool):
    def __init__(self) -> None:
        super().__init__({"收入": 100, "成本": 150})
        self.calls = 0

    def execute(self, parameters, inputs):
        self.calls += 1
        return super().execute(parameters, inputs)


def plan(ctx: PlanningContext) -> dict[str, Any]:
    def propose(
        kind: str, inputs: tuple[NodeInput, ...], objective: str, **kwargs: Any
    ) -> dict[str, Any]:
        return {
            "operation_type": kind,
            "input_node_ids": [node.node_id for node in inputs],
            "objective": objective,
            **kwargs,
        }

    question = ctx.nodes[:1]
    evidence = tuple(node for node in ctx.nodes if node.node_type == NodeType.EVIDENCE)
    hypothesis = tuple(node for node in ctx.nodes if node.node_type == NodeType.HYPOTHESIS)
    computed = tuple(node for node in ctx.nodes if node.node_type == NodeType.INFERENCE)
    if ctx.goal == QUESTIONS[0] and not evidence:
        return propose(
            "RETRIEVE",
            question,
            "查询收入和成本增量",
            tool="mock_search",
            parameters={"query": "100 150"},
        )
    if ctx.goal == QUESTIONS[2] and evidence and not hypothesis:
        return propose("INFER", question, "把假设中的成本增量表示为数字")
    if ctx.goal in (QUESTIONS[0], QUESTIONS[2]) and evidence and not computed:
        return propose(
            "COMPUTE",
            evidence + hypothesis,
            "收入增量减去成本增量",
            parameters={"operator": "subtract"},
        )
    return propose("FINISH", computed or evidence or question, "回答本轮问题")


def infer(ctx: ExecutionContext) -> dict[str, Any]:
    assert ctx.inputs[0].content == QUESTIONS[2]
    return {"node_type": "hypothesis", "content": 0}


def finalize(ctx: ExecutionContext) -> dict[str, str]:
    node = ctx.inputs[0]
    if node.node_type == NodeType.EVIDENCE:
        answer = f"成本增加 {node.content} 万元。"
    else:
        value = int(str(node.content))
        answer = f"利润{'减少' if value < 0 else '增加'} {abs(value)} 万元。"
        if value >= 0:
            answer = "在成本增量为 0 的假设下，" + answer
    return {"answer": answer}


def demo(storage: StorageBackend | None, count: int) -> None:
    tool = CountingRetrievalTool()
    core = ReasoningRuntime(MockModelAdapter(plan, infer, finalize), storage, tools=[tool])
    chat = ConversationRuntime(core)
    conversation = chat.create_conversation()
    print(f"conversation_id {conversation.conversation_id}")
    for question in QUESTIONS[:count]:
        before = tool.calls
        result = chat.ask(question, conversation_id=conversation.conversation_id)
        print(question)
        print(result.model_dump_json(indent=2))
        print(f"retrieval_calls {tool.calls - before}")
        graph = chat.get_run(conversation.conversation_id, result.run_id)
        for node in graph.nodes.values():
            print(
                "node",
                node.node_id,
                node.node_type,
                graph.node_usability(node.node_id).status,
                node.content,
            )
        for op in graph.operations.values():
            print(
                "operation",
                op.operation_id,
                op.operation_type,
                op.provided_node_ids,
                op.output_node_ids,
            )
        for ref in graph.references.values():
            print("reference", ref.model_dump_json())
        for record in graph.verifications:
            if record.verification_type in {"cross_run_reference", "deterministic"}:
                print("verification", record.target_id, record.verification_type, record.status)
    assert tool.calls == 1


def main(count: int = 2) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path)
    args = parser.parse_args()
    if args.db is None:
        demo(None, count)
    else:
        with SQLiteStorage(args.db) as storage:
            demo(storage, count)


if __name__ == "__main__":
    main()
