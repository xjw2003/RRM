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
| `RunEvent` | 每个运行中一次原子变更。 |

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
