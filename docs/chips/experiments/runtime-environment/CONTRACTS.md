# E1–E3：先证明接口与实验状态可靠

返回[方案入口](README.md)。本页保留接口验收设计；实际检查结果见[验证记录](../../VALIDATION.md)。
本页全部使用本地 fixtures／进程，不调用真实模型或服务器。
每条新行为先写失败测试再实现；已有通过测试记为回归，不倒写 Red 记录。

## E1：同一个任务环境能否被两种 Runtime 操作

**验证 C1。** 使用 vaBench `v4-001` 的构造公开材料与隔离临时会话；fixture 不代表真实电路解题。

| 组别 | 路径 | 固定内容 |
| --- | --- | --- |
| B0 基线 | 现有任务 session／公开工具入口 | 输入候选、动作序列、返回 fixture、工具与任务预算 |
| N1 | `AlphaApolloAgentRuntime`＋脚本 GenerationBackend＋新环境 | 与 B0 相同任务规则；backend 发真实结构的 tool calls |
| N2 | `ExternalAgentRuntime`＋测试用 ExternalAgentSession＋真实 MCP 子进程＋新环境 | 与 B0 相同规则；会话用 MCP 执行动作，而非伪造 transition |

N2 使用测试 session factory；当前配置中的 `agent=fake` 不允许授予环境工具，不能改掉该限制来过测试。
另用固定响应的本地 HTTP fixture 驱动真实 Pi CLI，验证 MCP 发现和数据回传；此项仍不是模型能力实验。

固定动作：读取说明／候选 → 写入错误候选 → 仿真得到预定错误 → 修复 → 再仿真 → 提交。
三组使用相互独立的新会话。捕获实际公开观察、候选字节、动作日志与冻结摘要，不只断言 Python 方法被调用。

**验收：**

1. N1、N2 与 B0 使用同一组工具名称、描述、参数 schema；实际 MCP `list_tools` 与授权一致。
   非法声明／未授权工具在本地拒绝；错误参数、越界文件访问由所属校验层拒绝，不产生未许可读写或仿真。
   任务规则仍以服务器 session 为权威，不为减少网络请求在客户端复制另一份规则。
2. 每个动作到达同一个 session 契约；候选摘要、公开结果、预算消费、冻结状态语义相同。
   比较前只剔除预先列明的 run ID、路径、时间等非语义字段；禁止剔除错误、数值、预算和提交状态。
3. N1 的 `project_response`／`continuation_messages` 正确回填 tool-call ID；N2 的模型侧收到相同公开反馈。
   对一条响应含多次调用、工具错误与普通文本分别验证：遵从明确的串行／拒绝策略，不能静默丢一半动作。
4. 原生 Runtime 与真实 Pi 的脚本路径都能读→写→失败→修复→提交；所有观察通过公共边界取得。
5. 原始模型／CLI 事件继续保留；标准化 `AgentResult` 不替代已有 raw 文件。

**输出：**JUnit、三组语义差异表、工具 schema 摘要、单条 Episode 的逐步对照记录。
非零语义差异需逐项解释；有权限、评分、冻结或动作次数差异即失败。

## E2：更换任务时保留不同的工具与仿真规则

**验证 C1、C2。** 对 vaBench 与 Analog `rlc-rf-bandpass-100mhz` 参数化测试；每个任务复用各自旧 session 为基线。

| 检查 | vaBench | Analog | 预期 |
| --- | --- | --- | --- |
| 可修改输入 | public contract 声明的候选文件 | 受限 `circuit.spi` | 仍按各任务校验；不强行统一写文件 API |
| 公开仿真 | EVAS 反馈 fixture | ngspice 反馈 fixture | 同名阶段可有不同日志、波形、单位、结构 |
| 工具发现 | `vabench_read/write/simulate/submit` | `analog_read/write/simulate/submit` | 命名、参数、描述兼容；只暴露该任务授权集合 |
| 提交 | 原 session 冻结逻辑 | 原 session 冻结逻辑 | Environment 不创建第二份“已提交”事实 |
| 最终判定 | 原 vaBench evaluator | 原 Analog scorer | Agent 接口不提供终评入口或隐藏输入 |

新增 Analog 时允许增加任务 ToolSpec／适配配置，**不增加新的 Pi／模型调用循环**。
数字 RTL 的 compile／simulate／synthesis／timing 仅做接口设计检查，不虚构第三个正式 benchmark。

同时放入人工构造的隐藏 canary，检查 system、初始 prompt、tool schema、公开响应、外部会话 metadata
和工作区公开文件均无泄漏。`task_payload`／Python 私有字段不是 OS 隔离：原生工具或工作区挂载
仍须遵守公开可见性；本测试不能宣称抵御同账号任意进程攻击。

**输出与判定：**两任务 B0／N1／N2 的同一份契约测试均通过；源文件 review 确认评分器未改写、
任务条件未写入共享模型循环。以责任和行为判断复用，不把删行数作为优化目标。

## E3：异常和退出后，实验仍然可信

**验证 C2。** 以可计数的真实本地后台子进程模拟仿真；注入只影响测试会话。
此处“求解次数”按 worker 启动记录／产物确认，不能只统计客户端调用。

| 编号 | 注入／动作 | 必须观察到的行为 |
| --- | --- | --- |
| F1 | 同 action ID 重发同 envelope；再尝试换参数 | 前者复用一次执行；后者冲突拒绝；新的合法 ID 是新动作 |
| F2 | 服务端接收后，客户端丢失响应并退出；重建控制进程 | 读取已有请求并查询原 ID；未确定状态前拒绝新动作；不产生第二个 worker |
| F3 | 仿真运行中关闭 Runtime／MCP | agent bridge 退出状态被记录；已接收 job 继续完成并可收取；不会因 close 隐式取消或重复启动 |
| F4 | 模型只输出“完成”，或超时／预算耗尽 | 标记未提交；无冻结确认不得启动终评；保留停止前的动作和候选 |
| F5 | submit 返回途中断线；确认冻结后再提交／写入 | 先核对原 action 和冻结哈希；不重复冻结；拒绝修改；终评只读取冻结版本 |
| F6 | 预算边界、多调用与 provider 自动重试 | 记录逻辑请求与实际 HTTP 尝试的区别；任务端强制动作／仿真预算；不把 runtime turn 数等同模型请求数 |
| F7 | scorer 执行失败、归档失败、下载损坏 | 与电路未达标分别记录；归档重试不重做仿真；校验失败不清理唯一证据 |
| F8 | 模型先提交，再发工具调用 | 终止后的调用被拒绝且留痕；没有后续候选或仿真变更 |
| F9 | 新运行与已有 run ID／配置冲突，或两个 Episode 并存 | 拒绝覆盖；各自环境、候选、动作 ID 和预算不串用 |

F3 特别检查现有 `RuntimeEnvironmentSocket.close()` 对未完成 step 的处理。
共享桥接关闭超时、任务仿真超时和服务器 job 期限不是一回事。不能靠无限延长等待通过测试；
若需要调整，只做可中止客户端等待、保留远端作业身份的最小变更，并补共享回归。
不要求本轮实现多 Agent 并发调度或机器重启后恢复整个模型上下文。

**终止映射：**

| 实际事件 | 环境／Agent 侧 | 操作者侧 |
| --- | --- | --- |
| 普通读写／仿真／可修复错误 | 非终止；公开诊断 | 保存动作和产物 |
| 服务端确认 submit | 环境终止，原因表示 submitted；`success=None` | 核对冻结，再按事先约定的策略终评 |
| 最终文本、模型预算、进程退出 | Agent 结束；不伪造 submit | 区分未提交、已提交、待恢复的动作 |
| final scorer 完成 | 不回注 Agent | 保存执行状态、有效性、原分数和可选达标判定 |

## 回归入口与保留证据

新增测试落点见[方案入口](README.md)。按变更选择已有：
[vaBench session](../../../../tests/chips/test_vabench_session.py)、
[Analog session](../../../../tests/chips/test_analog_session.py)、
[统一入口](../../../../tests/chips/test_experiment.py)、
[原生 Runtime](../../../../tests/reasoning/runtime/test_alphaapollo_agent_runtime.py)、
[外部 Runtime](../../../../tests/reasoning/runtime/test_external_agent_runtime.py)、
[MCP bridge](../../../../tests/reasoning/runtime/test_external_bridge.py)、
[组装契约](../../../../tests/workflows/resources/test_composition_contract.py)。

旧 `chips_vabench_agent`／`chips_analog_agent` CLI、配置与离线报告回归必须保留。
增加新证据时保留旧文件形状或做兼容读取；不改写历史 run。凭据、桥接 bearer token 不进入可归档快照。
采用[现有记录模板](../../../../tests/chips/RECORD_TEMPLATE.md)，记录实际执行、skip 与 fixture 范围。
E1–E3 全部必要检查通过才进入真实服务器；缺真实 Pi CLI 的检查明确 blocked，不能只用 mock 通过代替。
