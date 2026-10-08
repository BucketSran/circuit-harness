# 实验方案：Chips 接入 Apollo Runtime／Environment

> 发布说明：本文保留开源前的版本与实验记录。`demo/chips`、旧提交和历史议题归私有研发档案；
> 当前公开开发以 `main` 为基线，遵循[仓库范围](../../REPOSITORY_SCOPE.md)中的边界。

日期：2026-09-25。状态：**历史实验方案**。文中的“拟”“需要”和基线表描述制定方案时的状态，不代表当前代码；E1–E5 的执行结果和后续 Analog 修复回合以[验证记录](../../VALIDATION.md)为准。E5 原始 Analog 三回合均未提交，后续修复回合完成一次提交及 15/15 终评；两者不能合并为 E5 通过。

本轮只回答一个问题：能否参考 Robotics 的组装方式，让同一个 Chips 任务环境由不同 Agent 操作，
同时保留已有仿真、恢复、提交和评分规则？不以模型分数提升作为这次架构迁移的成立条件。

逐项 review：

1. 本页：目标、责任边界与修改顺序。
2. [契约与故障实验](CONTRACTS.md)：不用真实模型，先证明动作和状态正确。
3. [真实链路实验](LIVE_RUNS.md)：服务器、模型、预算、观测与结论范围。
4. [执行清单](TRACKER.md)：保留方案制定时的状态快照；实际执行证据见[验证记录](../../VALIDATION.md)。

本文修订[旧方案](../../HARNESS_ALIGNMENT_PLAN.md)中 G2／B／C 的近期优先级：
先复用现有 Runtime／Environment，而非先抽一个 Pi 启动 helper 或铺开配置 v2、批量评测、Memory。
相关讨论沿用 [#1](https://github.com/BucketSran/circuit-harness-private/issues/1)，本轮不重复开 issue。

## 1. 要证明的两项主张

| 主张 | 最小可信证据 | 不能据此声称 |
| --- | --- | --- |
| C1：Agent 启动与任务执行可以分离 | 同一 vaBench 环境契约，经 `AlphaApolloAgentRuntime` 与 `ExternalAgentRuntime` 接受相同动作；真实 Pi 与 Apollo 原生 Agent 各完成受控回合；Analog 复用相同组装路径 | 所有模型／CLI 已兼容，或新架构提高解题能力 |
| C2：接入共享 Runtime 不削弱实验可信度 | 旧／新路径的脚本动作语义等价；冻结、预算、重试与隐藏信息边界测试通过；真实断线同 ID 恢复与归档可核验 | 整段 Agent 可在服务器重启后恢复，或外部 CLI 内部完全可见 |

要排除的反例：新路径只是启动了 CLI，实际工具绕过 Environment；重复服务器作业；
Agent 说“完成”即被当成已提交；用默认零 reward 冒充评分；丢掉旧路径保存的原始轨迹。

## 2. 参考基线与已发现的约束

| 对象 | 固定提交／位置 | 结论 |
| --- | --- | --- |
| 当前 Chips | `dbbae43bef57e3267d11f464bdd33ec62139d5ae` | vaBench／Analog 脚本直接使用共享 Session，尚未接入本方案的环境层 |
| Robotics 参考 | `c502aba08e93e9ab2136529a1c7b0492825a5a6c` 的 `_robotics/methods.py::_build_agent_method` | 以 `environment_factory`、ToolCatalog 注入共享 Runtime；领域控制器持有 Episode |
| Main 参考 | `a3be42c14780a252b15e3eb492ad4f03c52eab6c` | 共享 Runtime 与 Codex 续接修复的固定参考；不要求整分支合并 |

本地关键代码：共享[构造器](../../../../alphaapollo/workflows/_resources/runtime.py)、
[Environment 契约](../../../../alphaapollo/common/environment/base.py)、
[外部 Runtime](../../../../alphaapollo/reasoning/runtime/external_agent_runtime.py)。
它们支持接入边界，但不是配置填好即可运行的保证：

- 当前 `_external_session_options` 明确拒绝原生 `codex` 的 Environment `tool_loop`，因为无法关闭全部原生执行工具。
  现有 Chips Codex＋SSH 链路仍保留；不能通过删除校验把它称为工具完全受控。
- Robotics 参考版本增加了 `codex_app_server`；需独立核对协议、版本、权限与记录，再决定是否移植。
  `codex_via_pi` 是另一种接法，不能冒充原生 Codex 验收。
- 共享 MCP `_tool_specs` 当前没有 `vabench_*`／`analog_*`。需要让声明与执行使用同一份 ToolSpec，
  不能只把 schema 交给模型而遗漏子进程的实际发现／授权。
- Apollo 原生 Agent 的 `project_response`／`continuation_messages`、外部 Runtime 的 MCP 动作和最终文本，
  都需映射到同一会话规则。`EnvironmentTransition.reward` 必填，而本任务终评在 Agent 退出后进行。

## 3. 拟采用的责任边界

```text
chips_experiment：冻结配置、建立 Episode、运行 Agent、终评、收取证据
  ├─ Apollo 原生 Runtime ────────────────────┐
  └─ 外部 Runtime → Pi／其他已验收接法 → MCP ──┤
                                             ↓
                              Chips Environment：公开动作与观察
                                             ↓
                              原任务 session + local／SSH 传输
                                             ↓
                         服务器持久作业 → 任务指定的仿真器／执行后端
                                             ↓
                                冻结候选 → 操作者侧独立终评
```

`local` 指 Agent 与执行入口同在服务器。Environment 的生命周期不等于服务器 job 的生命周期。
环境关闭只释放本次 Agent 桥接资源；已接收的服务器作业按既有规则完成、持久记录，随后可收取。

| 所有者 | 负责内容 | 此次变更方式 |
| --- | --- | --- |
| 共享 Runtime／Session | 模型交互、CLI、MCP 桥接与 Agent 结果 | 复用；必要缺口做有回归的最小补丁 |
| Chips Environment | 初始公开观察、响应投影、工具授权、状态转换 | 拟新增薄适配；调用原 session，不复制其状态机 |
| vaBench／Analog session | 候选、仿真次数、动作去重、提交冻结 | 保留权威来源；不同网表／文件／反馈规则仍由任务决定 |
| 实验协调入口 | 部署预检、运行身份、独立终评、归档 | 保留 `chips_experiment`；逐条替换内部组装 |
| 原评分器／仿真适配 | EVAS、ngspice、Spectre 等具体语义 | 保持各任务的固定版本与判定；不统一成一个评分公式 |

第一阶段环境转移的 `reward=0.0` 仅满足接口，公开元数据标记 `evaluation=not_performed`，
`success=None`；该字段不进入 benchmark 成绩汇总。只有操作者侧评分器拥有最终分数。
`submit` 的确认才代表候选冻结；模型的最终文本只结束 Agent 回合。

## 4. 按证据分次实现与提交

每项按行为 TDD，小改动单独 commit／push 到私有 `demo/chips`。以下是拟议落点，不是已创建的实现。

| 顺序 | 交付与拟议代码位置 | 完成门槛 |
| --- | --- | --- |
| S1 | vaBench ToolSpec 与薄环境：`common/execution/tools/chips_public.py`、`common/environment/chips.py` | E1 先验证最小投影／发现链路；再通过 E2 的 vaBench 契约 |
| S2 | 在 `chips_experiment.py` 接共享构造器；原 CLI 保持兼容；保存 Runtime 结果与原始事件 | E1 双 Runtime 通过，E3 退出／恢复和旧报告读取通过 |
| S3 | 同一环境适配接 Analog session；仅在确有重复时抽公共组合代码 | E2 的 Analog 参数化测试通过；启动代码不包含仿真器分支 |
| S4 | 真实服务器固定动作回放、断线与 12 个模型回合 | E4／E5 分阶段留证；失败按归因停止，不覆盖历史 attempt |
| 后续独立项 | 原生 Codex 的共享模式；Claude Code；数字 IC 新任务 | 按能力补验，不阻塞首个 Apollo＋Pi 实证，也不计作本轮已支持 |

S1 先用贯通单个工具的最小原型验证 E1 中 schema→MCP→Environment→session；不提前承诺某个文件拆分。
测试拟放 `tests/chips/test_runtime_environment.py`、`test_runtime_composition.py`，复用已有 fixtures。
共享改动只运行受影响的通用回归；不运行 Math／Robotics／Bio 领域实验。

## 5. 决策与退出条件

E1–E3 任何完整性失败，都先修复再进入真实实验。E4 证据闭环通过后才进入 E5。
真实模型未解出题可以形成有效的失败 Episode；但“真实模型完整提交闭环已验收”必须有实际提交、
独立终评和归档证据，不能由 fixture 代替。分数不达标与链路故障分开报告。

若原生 Agent 的 provider 适配缺字段，补最小映射并固定新配置版本；若共享 Runtime 强制了错误的终止／清理语义，
先验证最小补丁能否保住通用契约。无法小步满足时，保留旧路径并提交具体阻塞证据，停止扩大重构。
既有 CLI 的删除、配置 v2、批量调度、Memory、Spectre／EMX 新任务与数字 IC 真任务均另行评审。
