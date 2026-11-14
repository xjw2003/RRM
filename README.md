# RRM · Reasoning Runtime Middleware

RRM 是一个独立的 Python 推理运行时。模型每轮提出一项操作，运行时读取该操作声明的节点版本并执行。结果成为新节点。运行时在下一轮规划前执行适用的独立检查，符合使用条件的节点可以成为真实输入。节点、操作和数据依赖能独立保存、验证和回放。

RRP（Reasoning Runtime Protocol）是本项目自定义的协议。包版本为 0.2.0，沿用 RRP v0.1 的操作协议，并通过可选字段保存版本、失效、步骤验证、恢复与超时记录。早期 SQLite 事件可直接读取。

| 方式 | 执行与记录 |
| --- | --- |
| 普通 RAG | 通常先检索证据，再把证据交给模型生成回答。RRM 可将检索注册为工具，并让后续步骤选择已持久化的证据版本。 |
| Agentic RAG | 通常让 Agent 动态检索和使用工具。RRM 可以作为其执行组件，维护实际输入快照、产生的节点以及可重放的依赖。 |
| LangGraph 工作流状态 | 由应用定义工作流图并维护运行状态。RRM 的二部 DAG 由运行时发生的数据消费和产出构成，可作为工作流中的独立记录。 [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api) |

## 安装和示例

需要 Python 3.11 或更新版本。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python examples/mock_reasoning.py
python examples/rag_reasoning.py --db reasoning.db
python examples/dependency_invalidation.py
python examples/cross_model_resume.py
python -m pytest -q
ruff check .
```

历史回放示例使用 `rag_reasoning.py` 输出的 `run_id`。

```bash
python examples/replay.py reasoning.db RUN_ID
```

## 运行、检查与回放

```python
from rrm import Limits, ReasoningRuntime
from rrm.adapters import MockModelAdapter
from rrm.storage import SQLiteStorage


def plan(ctx):
    return {
        "operation_type": "FINISH",
        "input_node_ids": [ctx.nodes[0].node_id],
        "objective": "简洁回答",
    }


model = MockModelAdapter(
    plan,
    lambda _: None,
    lambda ctx: {
        "answer": f"已收到问题：{ctx.inputs[0].content}",
    },
)
with SQLiteStorage("reasoning.db") as storage:
    runtime = ReasoningRuntime(model, storage)
    result = runtime.run("这个示例如何保存结果？", limits=Limits(max_steps=4))
    print(result.answer)
    print(runtime.export_json(result.run_id))
    for check in runtime.verify_run(result.run_id):
        print(check.target_id, check.verification_type, check.status)
    restored = runtime.replay(result.run_id)
    print(restored.trace(result.final_node_ids[0]))
```

`get_graph(run_id)` 返回当前图，`get_node(run_id, node_id)` 返回指定历史版本。图支持 `parents()`、`children()`、`ancestors()`、`descendants()` 和 `trace()`。也可用 `get_dependencies(run_id, vertex_id)` 查询直接相连的边。默认存储为 `MemoryStorage`。SQLite 的事件序号可用于 `replay(run_id, through_sequence=n)`，读历史时可查询各版本当时的状态。只读取存储也可通过 `storage.get_graph(run_id)` 重建图。

每步自动验证操作和产出的节点。
`get_node_usability(run_id, node_id)` 分别返回版本状态和使用资格。`eligible` 表示已满足适用的局部检查，`unverified` 表示仍可作为待核实输入，`blocked` 和 `pending_verification` 节点会被排除。模型输出的 `fact` 会保存为 `hypothesis`，旧 `fact` 仍可读取。

`verify_run()` 保留全图检查。结构、快照、来源版本、确定性计算与当前有效性分别报告。终止状态的检查记录会追加到事件流，明确失败会阻断相关节点和下游结果。用 `inspect_failure(run_id, node_id)` 查看生成操作、实际输入、上游证据、已定位的验证问题和受影响节点。

## 接入模型与 RAG 工具

实现 `ModelAdapter` 的 `propose_next_operation()`、`execute_reasoning()` 和 `generate_final_answer()`。每个方法返回 `ModelReply(payload, total_tokens=None)`。规划输入包含问题、筛选后的节点摘要、未验证节点 ID、独立的失败摘要和工具目录。`ContextSelector(max_nodes=21, max_summary_chars=512)` 优先保留任务根节点，并从最近节点、相关证据、分支末端和上游节点选取摘要。模型执行输入包含操作类型、目标及所选节点的完整内容，适配器应保持执行调用无会话历史。协议在 [RRP 文档](docs/protocol.md) 中定义。

真实模型可使用 `OpenAICompatibleAdapter`。提供兼容 Chat Completions 与 JSON object 模式的 API 基址，例如以 `/v1` 结尾的地址。按 [.env.example](.env.example) 将 `RRM_BASE_URL`、`RRM_MODEL` 和适用的 `RRM_API_KEY` 注入环境，`RRM_HTTP_TIMEOUT` 可选。示例从进程环境读取配置。 [Chat Completions API](https://developers.openai.com/api/reference/resources/chat)

```bash
python examples/basic_reasoning.py "比较增收与降本对利润的影响"
```

实现 `ToolAdapter` 并传入 `ReasoningRuntime(model, tools=[tool])`。工具提供唯一名称、描述、JSON Schema、来源标识、超时与可选的调用上限。检索工具接收参数和声明的节点快照，返回由 `content` 及可选 `document_id`、`document_version` 组成的列表。运行时创建证据节点和来源快照。工具示例见 [MockRetrievalTool](src/rrm/adapters/tool_base.py)。

## 修订与换模型接续

`revise_node(run_id, node_id, content, reason=...)` 创建新版本。历史操作仍引用原节点。递归下游结果被标记为 `stale`，无关分支可继续使用。`revoke_node()` 只撤销旧版本。若当前结论受影响，运行变为 `partial`。新建一个使用同一存储的 Runtime，传入同一或另一模型并调用 `resume(run_id)`，即可基于当前可用节点重算。结论仍有效而旁支有失效结果时，也可接续。每次执行阶段有独立预算，工具调用上限按整个 Run 计算。

## 中断恢复与超时

确认原执行已停止后，用 `recover_run(run_id)` 协调持久化状态，再用 `resume(run_id)` 接续可运行的任务。恢复会验证已提交但待检查的输出，并为未完成操作记录 `interrupted`。可以通过报告里的 `pending_tool_operation_ids` 找到结果待确认的外部操作。核对外部系统后，以 `confirm_tool_outcome(run_id, operation_id, reason=...)` 记录确认。工具默认 `retry_safe=False`，只读或具有幂等保障的工具可明确设为 `True`。恢复本身只处理事件和验证。

模型和工具适配器可在调用内使用 `rrm.adapters.current_invocation()` 获取执行标识与取消信号。中止记录区分传输请求超时、运行时等待超时、Run 截止和取消。未确认退出的调用标记为 `unknown`。详见 [当前边界](docs/limitations.md)。

## 验证范围

RRM 记录模型显式产出的外部结果。隐藏思维链无法还原，日志也无法证明模型内部因果性地使用了某个输入。哈希检查说明保存的数据与快照是否一致，来源版本检查说明内容是否匹配指定来源。事实真实性与自然语言推断正确性需要独立证据或审查。受支持的算术可以重算，其余语义结果保留 `not_verified`。

- [架构与扩展点](docs/architecture.md)
- [协议与数据格式](docs/protocol.md)
- [验证与修订](docs/verification.md)
- [当前边界](docs/limitations.md)

[MIT License](LICENSE)
