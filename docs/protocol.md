# RRP v0.1

Reasoning Runtime Protocol 是 RRM 的自定义协议。Pydantic 定义位于 [models.py](../src/rrm/protocol/models.py)，生成的 [JSON Schema](../schemas/rrp-v0.1.schema.json) 使用 Draft 2020-12，模型定义位于 `$defs`。跨字段校验在 Pydantic 和 Runtime 内执行。工具 I/O 的结构约束由 Draft 2020-12 校验器处理。 [jsonschema 验证文档](https://python-jsonschema.readthedocs.io/en/stable/validate/)

| 记录 | 含义 |
| --- | --- |
| `ReasoningRun` | 任务、状态、协议、模型、预算及当前结论。 |
| `ReasoningNode` | 运行时生成 ID 的具体数据版本，记录内容哈希、生产者和可选证据快照。 |
| `ReasoningOperation` | 执行目标、执行器、参数、声明输入、运行时交付的输入快照、输出与执行状态。 |
| `ReasoningDependency` | 真实消费或产出关系，通过 `edge_type` 确定方向。 |
| `PlanningRecord` | 每次规划的模型、上下文哈希及校验后的提案或错误。 |
| `VerificationRecord` | 对特定对象的一项验证，附状态、时间和细节。 |
| `RunEvent` | 每个运行中一次原子变更，可携带验证结算、节点阻断、恢复或工具确认。 |
| `NodeBlock` | 不可用节点、失败来源节点，以及关联验证 ID 和原因。 |
| `RecoveryRecord` | 恢复原因、被标为中断的操作、已结算的待验证操作。 |

模型提案示例。

```json
{
  "operation_type": "INFER",
  "input_node_ids": ["runtime-generated-node-id"],
  "objective": "分析证据对当前问题的含义"
}
```

`INFER` 和 `COMPARE` 返回一个 `fact`、`hypothesis` 或 `inference` 类型节点的内容。`FINISH` 返回 `{"answer":"..."}`。`RETRIEVE` 需指定注册的 `tool` 及符合输入 Schema 的 `parameters`。`COMPUTE` 接收两个有序节点和 `operator` 参数，支持 `add`、`subtract`、`multiply`、`divide`。运行时修订使用的 `REVISE` 只存在于执行记录，验证活动使用独立记录。

内容哈希是确定性 JSON 编码的 SHA-256。编码使用排序的对象键、无额外空白、UTF-8，并拒绝非有限浮点数。快照哈希使用有序数组，每项包含节点 ID、类型与内容。旧版本仍通过原 ID 读取，逻辑 ID 标识版本族。缺少逻辑 ID 的早期记录使用初始节点 ID。

`max_steps` 按执行阶段限制操作数量，`max_model_calls` 包括规划、推断、回答及重试。`timeout_seconds` 限制运行时等待。可选的 `max_tokens` 使用模型报告的累计消耗，超限回复不会提交。若提供方未报告 token 用量，启用预算的运行会返回错误。

## 兼容扩展

`ReasoningOperation.verification_required` 在新操作中启用。输出通过 `RunEvent.verified_operation_ids` 完成验证结算。验证记录与 `node_blocks` 同时写入。旧操作默认 `verification_required=False`，历史图可直接回放。接续含旧操作的 Run 时，运行时先补做适用检查，旧事件保留原样。

`retry_safe` 在操作开始时保存。旧记录默认 `False`。恢复发现结果未知的工具操作时，使用此持久化设置判断是否等待确认。`invocation_stop` 可出现在操作和规划记录，包含 `execution_id`、停止原因、是否发出取消信号及本地调用是否已退出。工具调用的 `execution_id` 与其操作 ID 相同，便于对照外部日志。

## RRP v0.2 对话扩展

独立 Run 继续使用 v0.1，Conversation Turn 使用 v0.2。参见 [RRP v0.2 JSON Schema](../schemas/rrp-v0.2.schema.json)。现有模型提案仍使用原有操作集合和至少一个本地输入 ID 的约束。

内部 `IMPORT` 操作使用空的本地输入列表。其输入哈希记录实际读取的来源节点快照。输出节点与 `CrossRunReference` 同事件保存，引用记录包含来源和目标 Run/节点 ID、来源修订号和哈希、读取时的来源事件序号、来源使用资格以及导入操作 ID。单轮 DAG 记录本地生产边，跨 Run 消费边由联合图从引用事件重建。

`RunResult.reasoning_mode` 由实际操作推导。模型调用计数来自新事件中的调用开始标记，较早的记录没有精确调用计数。失败及重试的已启动调用计入统计。
