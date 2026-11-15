# 对话与跨 Run 复用

`ConversationRuntime(core, storage=..., resolver=..., allow_unverified=True)` 使用与 Core 相同的存储实例。示例可用内存运行，使用 SQLite 可在进程重启后重建联合图。

| API | 作用 |
| --- | --- |
| `create_conversation(conversation_id=None)` | 创建或打开对话。 |
| `ask(question, conversation_id=..., limits=...)` | 创建一轮新 Run，返回关系、模式、调用数和导入节点 ID。 |
| `list_turns(conversation_id)` | 按顺序列出 Turn。 |
| `get_run(conversation_id, run_id)` | 读取单轮执行图。 |
| `get_references(conversation_id, run_id)` | 查询本轮导入来源。 |
| `get_conversation_graph(conversation_id)` | 重建联合 DAG。联合顶点由 `(run_id, vertex_id)` 标识。 |
| `storage.references_to(SourceNode(...))` | 反向查询使用指定历史版本的所有导入。 |

## 追问关系

- `CONTINUE` 可复用经核对的历史节点。后续 Planner 仍可选择直接回答或展开推理。
- `COUNTERFACTUAL` 要求在复用前标识改变的历史前提。运行时拒绝依赖这些前提的候选节点。可从本轮问题创建条件性假设，再和独立的导入节点共同计算。
- `CORRECTION` 保存待审状态。应用可显式调用已有的单 Run 修订或撤销 API，再提交新问题。
- `NEW_TOPIC` 在新 Run 中独立执行。

可替换的连续性解析器只返回候选关系和节点标识。开启 `allow_unverified` 后可以引用未验证假设，引用仍保留其暂定资格。每个尝试导入的候选都有接受或拒绝原因。默认离线规则适合带明确追问标记的输入。OpenAI-compatible 适配器使用一次结构化解析，首轮跳过该调用。

## 持久化与历史

单轮执行图只含本轮节点和边。对话图根据真实引用事件连接历史来源节点与目标轮的 `IMPORT` 操作。跨轮边必须指向更晚的同对话 Turn。同一来源版本在一个 Run 中只导入一次。

SQLite 保留原有 `rrm_events`。新增表存储对话、Turn 索引及反向引用索引。引用的权威记录来自事件流。IMPORT 开始事件可单独保留，输出、产出边和引用索引要么共同提交，要么共同回滚。旧数据无需迁移为对话。

`replay()` 与联合图读取历史记录。当前使用资格通过 `get_node_usability()` 查询，导入后继续执行时也会核对引用及其祖先。如果来源后来被撤销、替代或阻断，下一次使用会发现。跨轮主动失效传播尚未实现。
