# VABench Agent 输入、工具与轨迹：可复核的数据流

本文以 `v4-001` 的已运行条件说明 Apollo Chips 当前实际传给 Agent 什么、工具如何被发现、
哪些结果进入下一轮思考，以及最终评分为何不能被 Agent 调用。这里的“System”只指
Apollo 构造的 `AgentTask.system`；Agent CLI 或模型服务另有自己的提示与上下文，
当前证据不能还原服务商每次请求的完整消息体。

## 一次回合中的四类输入

| 类别 | 来源与当前内容 | Agent 何时看到 | 是否允许改动 |
| --- | --- | --- | --- |
| Apollo 启动指令 | [`chips_vabench_agent.py`](../../alphaapollo/workflows/chips_vabench_agent.py) 的 `AgentTask.system` 和 `prompt` | 回合开始 | 由 Harness 固定；Pi 与 Codex 的文本目前略有差异 |
| 工具契约 | [`vabench_session.py`](../../alphaapollo/common/execution/chips/vabench_session.py) 的 `TOOLS` 名称、英文描述、JSON 参数 schema | MCP `tools/list` 后，由 Agent 运行时注册/发现 | 回合内不可由候选修改 |
| 公开任务 | 服务器会话的 `task/instruction.md`、`task/public_contract.json`、`task/visible_test.scs`、`task/evas_runtime.json` | Agent 主动调用 `vabench_read` 后 | 只读；`submission/bbpd_ref.va` 是声明的可写候选 |
| 反馈与历史 | `vabench_read/write/simulate/submit` 的返回值和当前 Agent 对话 | 每次调用返回后；本回合后续步骤可使用 | 不等于跨回合 Memory |

以 `v4-001` 为例，`instruction.md` 要求实现 Bang-Bang Phase Detector 的 Verilog-A 模块
`bbpd_ref`；五个 `electrical` 端口依次为 `data`、`clk`、`retimed_data`、`up`、`down`，
目标文件为 `bbpd_ref.va`。`public_contract.json` 进一步给出参数、端口、允许的 artifact、
波形信号和公开性质。`visible_test.scs` 是可读的测试激励；`evas_runtime.json` 描述公开
EVAS 运行环境。**初始提示并未直接附上这些文件内容**，而是要求 Agent 先列文件，再读任务。

Apollo 给 Pi 的 `AgentTask.system` 原文是：

> Use only the four VABench public tools. Read task/instruction.md and task/public_contract.json. Implement the requested candidate, use public simulation feedback, revise if needed, then call vabench_submit. Public simulation success is not a final correctness score. No hidden evaluator is available.

给原生 Codex 的版本将首句改成 “Use the four VABench public MCP tools for this task.”，
其余含义相同。两者的 `AgentTask.prompt` 均为：

> Complete the VABench task. Start by listing public files.

但它们进入 Agent 的**角色不同**：[`PiSession`](../../alphaapollo/reasoning/runtime/external/agents/pi.py)
把 `system` 作为 `--system-prompt` 参数，把 `prompt` 作为 CLI stdin 输入；Pi 同时使用
`--no-builtin-tools` 并关闭自动扩展、Skills 与提示模板。这里可以确认 Apollo 显式交给 Pi
一个系统提示，但 Pi 自身/服务商是否还包裹其他文字，不能仅凭本仓库断言。
[`CodexSession`](../../alphaapollo/reasoning/runtime/external/agents/codex.py) 没有相应的
`--system-prompt` 参数，因此在第一次 `codex exec` 的 stdin 中发送
`system + "\n\n" + prompt`，并记录 `system_prompt_inlined=true`。这段文本是**首条任务输入的
一部分**，不能声称它在模型协议中具有 system 角色。原生 Codex 的内建工具仍可用；
`read-only` 本机沙箱和“请使用四个工具”的文字都不能把它变成 Pi 的仅四工具条件。

## 工具说明怎样到达模型

`TOOLS` 是当前四个工具说明的唯一代码定义；`tool_schemas()` 将它们转换成函数 schema：

```json
{
  "name": "vabench_read",
  "description": "Read a task/... or submission/... file; empty path lists public files.",
  "inputSchema": {
    "type": "object",
    "properties": {"path": {"type": "string"}},
    "required": ["path"],
    "additionalProperties": false
  }
}
```

另三个工具是 `vabench_write(path, content)`（写声明的候选文件）、
`vabench_simulate()`（固定公开 EVAS 仿真，只返回诊断）、`vabench_submit()`（冻结候选，
不返回终评分数）。这些英文描述加上参数 schema 是 Agent 获得的工具接口；
Markdown 文档是给人审核的，不是每次自动拼入提示的工具说明。调用时服务端还会验证
工具名、参数、路径、文件大小、动作/仿真预算和冻结状态；自然语言约束之外有可执行边界。

传递链如下。服务器上的 Pi 条件省去 SSH 一跳；本机 Codex 条件仍在服务器执行 EVAS。

```text
仓库 TOOLS → tool_schemas() → Apollo stdio MCP serve → tools/list
                                         ├─ Pi 扩展：registerTool(name, description, parameters)
                                         └─ Codex：MCP 客户端发现 vabench 服务的工具
模型选择工具/参数 → tools/call → Apollo MCP serve → 本地调用或 SSH → 服务器公开会话
            ← JSON 工具结果 ← MCP 文本结果 ← 服务器动作/EVAS 反馈
```

Pi 侧实现见 [`pi_extension.ts`](../../alphaapollo/reasoning/runtime/external/bridge/pi_extension.ts)：
启动 MCP 子进程、执行 `initialize` 和 `tools/list`，逐个 `registerTool`，再将 Agent 的
工具调用转发到 `tools/call`。Codex 侧通过
[`mcp_config.py`](../../alphaapollo/reasoning/runtime/external/bridge/mcp_config.py)
生成启动配置，连接同一个 MCP 服务。MCP 子进程属于**控制面适配器**，并非另一个推理模型；
它经清空环境启动，不接收模型 Key。

## 从公开任务到最终判定

一条典型回合为：

1. Agent 收到上述启动指令和四个工具 schema，调用 `vabench_read({"path":""})`
   得到公开文件列表，再读取 `instruction.md`、`public_contract.json` 和需要的测试/运行信息。
2. Agent 自己决定候选实现，调用 `vabench_write({"path":"bbpd_ref.va","content":"..."})`。
   服务端只允许写公开契约声明的文件，返回候选 SHA-256。
3. Agent 调用 `vabench_simulate({})`。Tool 使用固定的公开 EVAS 流程，返回退出码、日志摘要、
   波形摘要，以及 `authority=public_diagnostic`、`task_correctness=not_evaluated`。
   Agent 看见此结果后可以修改候选并再次仿真，直到预算用完或自行停止。
4. Agent 调用 `vabench_submit({})` 后，服务器冻结候选并返回身份和
   `final_score=not_exposed`。Apollo **只在确认提交成功后**由操作者控制流程调用
   `vabench-finalize`，以冻结文件运行原评分器，归档公开 episode 与终评 job，下载后
   校验候选摘要一致，最后写 `report.json`。

这里的“独立终评”指评分发生在冻结后、通过不对模型开放的控制面。它不保证每题都有
一份不同的隐藏激励：`v4-001` 的公开契约明确写有 `visible_and_final_test=identical`；
最终判定仍含独立属性检查和评分规则，公开仿真成功不能代替 `verdict=pass`。
`pin`、参考答案、评分材料和终评分数不能通过这四个 Tool 读取。

本次私有证据中的 lab-server Pi＋GLM 补跑经历 10 次工具调用，包括首次公开仿真失败、
修改后再次仿真成功；本机 Codex＋GPT 经 SSH 的回合有 9 次 MCP 调用，包括一次错误路径
读取被拒。两者各自最终 `report.json` 为 `state=verified`、`execution=ok`、`verdict=pass`，
仅是固定版本 `v4-001` 的单题 development 结果。详细条件见[验证记录](VALIDATION.md)。

## 哪些轨迹真的保存了

| 证据 | 目前能回答 | 不能据此断言 |
| --- | --- | --- |
| Pi `pi-outcome.json`、`model-budget.jsonl` | 规范化消息/工具事件、逐次模型请求时间和 usage；预算记录还含请求体字节数 | 并未归档每次发送给服务商的完整请求体，也不是 Pi 原始 CLI JSONL 的逐字副本 |
| Codex `codex-events.jsonl`、`codex-outcome.json` | 原始 CLI JSONL、规范化事件、CLI 报告的 usage | 没有逐次模型 API 请求体或统一的模型请求起止时间 |
| `tools/<action-id>/{request,response,timing}.json` | 工具参数、响应、动作 ID、本机控制面耗时 | 不能直接拆出全部服务器排队/仿真/传输细项 |
| 公开 episode、终评 job 归档与 `report.json` | 任务文件、冻结候选、服务器动作、最终判定及同候选校验 | 不应回灌进同一回合的 Agent，也不自动成为下次回合的 Memory |

两种 Agent 运行位置决定原始 Agent 事件先落在服务器还是本机；仿真和评分原始记录都在服务器。
证据属于私有运行目录，不能随代码推送。特别是当前没有完整的“每次模型实际收到的
system/developer/user/tool 消息快照”，所以不能从现有事件反推服务商的完整上下文或
逐次精确 token 构成。

## 与 DeepSWE-Bench 的对应关系

这里比较的是 [Datacurve 的 DeepSWE benchmark](https://github.com/datacurve-ai/deep-swe)，
不是同名的 DeepSWE Agent。DeepSWE 任务采用 Harbor 格式：`task.toml` 固定元数据、环境和
限制，`instruction.md` 是 Agent 看到的题目，`environment/` 构建任务容器，`tests/` 与
`solution/` 不给 Agent；Pier 在隔离环境里运行 coding agent，收集其提交的补丁，
再在干净的 verifier 环境应用补丁并评分。其官方运行入口可选择 `mini-swe-agent`，Pier
也支持 Codex/Claude Code 等 Agent；**benchmark 题目文本并不是 Agent 的 system prompt**。

| 层次 | DeepSWE-Bench | Apollo Chips / VABench `v4-001` |
| --- | --- | --- |
| 题目 | `instruction.md`，连同可编辑代码仓库 | `task/instruction.md`、公开契约、测试、运行说明，经 `vabench_read` 按需获取 |
| Agent 提示 | 由选定 Agent 的配置生成；例如 [mini-swe-agent 源码](https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/agents/default.py) 分别渲染 system 与第一条 user 模板 | Apollo 构造 `AgentTask.system/prompt`；Pi 单独传 system，Codex 将其并入首条输入 |
| 行动接口 | 常见为编码 Agent 的 shell/编辑能力，在隔离代码工作区改文件 | 当前 Pi 仅四个结构化 Tool；原生 Codex 还保留其内建能力 |
| 公开反馈 | Agent 可在工作区自行运行可用检查 | 固定 `vabench_simulate` 公开 EVAS 诊断，返回结果可用于下一轮修改 |
| 提交与终评 | Pier 收集 Agent 提交的 Git patch，在独立 verifier 环境运行 held-out tests，输出 `reward.json` 等 | 冻结 `bbpd_ref.va` 后控制面调用原评分器，输出 `report.json` 和两份校验归档 |
| 轨迹 | Pier 提供 Agent trajectory 元数据与查看器 | Pi/Codex 事件、工具动作与归档已保存，但初始实际上下文和统一跨运行事件格式仍有缺口 |

DeepSWE 的具体 system/tool 说明由**所选 Agent 及其固定版本/配置**决定，不是
`instruction.md` 自带。当前 [mini-swe-agent 默认配置](https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/config/mini.yaml)
展示了 `system_template`、`instance_template` 和工具观察模板；不能据此声称某次 Pier
排行榜运行必然使用了今天 `main` 分支的同一份配置。对照实验必须保存实际版本和渲染结果。

## 下一步的最小标准化目标（尚未实现）

每个新 episode 固定并私有保存一份 `initial_context` 清单：Apollo 渲染的提示文本及其
实际传递角色、Agent/CLI 与模型版本、真实 `tools/list` schema、公开文件 SHA-256、
预算和执行位置。启动前比较工具集合与预期是否一致；Codex 要单独标出仍可用的内建工具。
再用统一的事件 ID/时间戳把模型响应、工具请求/结果、服务器动作、候选摘要和最终评分关联。
保留原始 Agent 事件，同时生成可分析的 JSONL 索引；完整服务商请求体若可抓取，也只在
明确定义的数据外发与私有存储条件下选择性保存，绝不把凭据或终评材料写进模型可见轨迹。
这样才能在固定任务与预算下可靠比较 Tool、Agent 和模型，并明确哪些差异来自提示或工具集合。
