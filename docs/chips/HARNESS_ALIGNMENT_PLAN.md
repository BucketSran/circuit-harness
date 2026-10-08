# Chips Harness 改进方案：与 Math／Robotics 对齐

> 发布说明：本文保留开源前的版本与实验记录。`demo/chips`、旧提交和历史议题归私有研发档案；
> 当前公开开发以 `main` 为基线，遵循[仓库范围](REPOSITORY_SCOPE.md)中的边界。

> 范围更新（2026-09-26）：按用户最新决定，`demo/chips` 保留 Chips、
> Robotics 参考与共享内核，移除 Bio／Math 等领域 API。本文旧计划中有关保留
> Math API 的条件已被[当前分支范围](REPOSITORY_SCOPE.md)取代；其余比较作为设计参考。

> 参考分支现已从远端删除，所需材料保存在私有备份。下文分支名只标识当时的比较来源。

日期：2026-09-25。状态：**历史设计草案**。下文的基线、待办和实施范围保留为当时的决策记录；Runtime／Environment 实验及后续 Analog 修复的实际状态以[验证记录](VALIDATION.md)为准，不能把本页旧的“未执行”判断当作当前结论。

**后续讨论修订：**G2／B／C 的近期实施以
[Runtime／Environment 实验方案](experiments/runtime-environment/README.md)为准。
先验证 Chips 接入 Apollo 已有 Runtime／Environment，不另造 Agent 启动层；
本文其余工作包仍是候选路线，不代表一次性实施范围。

本文把跨分支比较收敛为实施方案，说明改什么、放在哪里、如何验收和如何分次提交。
它细化 [NEXT_WORK.md](NEXT_WORK.md) 中的配置、执行、轨迹与评测工作，不替代其中的长期任务接入路线。
相关讨论：[平台共建 #1](https://github.com/BucketSran/circuit-harness-private/issues/1)、
[模拟电路与双仿真器 #2](https://github.com/BucketSran/circuit-harness-private/issues/2)。

## 1. 建议采用的方向

**共享 Apollo 的 Agent、工具、执行与记录基础设施；Chips 保留任务生命周期、仿真器适配与电路评分。**

近期目标是让已有 vaBench 和 Analog 链路更容易配置、比较、复核与扩展。
不以目录相同或只剩一个 CLI 为完成标准，也不重写已经验证的服务器作业协议。

建议本轮实施范围为 A–E：共享修复、配置与能力声明、工具与传输边界、离线证据契约、批量实验。
F（Memory）作为后续独立里程碑。文档／图片接入、更多 PDK 任务和跨账号部署继续保留在长期路线中。
本次仅交付方案；提交本文不代表上述实现获验收，也不启动模型、SSH 或仿真。

## 2. 审查基线与已知缺口

### 2.1 比较版本

比较采用以下固定提交；它们是本方案的参考快照，不声明此后仍为最新版本。

| 对象 | 提交 | 用途 |
| --- | --- | --- |
| 私有 `demo/chips` | `6c3f0a8b4656c312d03b1f2f9b522e3acfec69bb` | 当前实现与历史报告基线 |
| 上游 Main／Math | `a3be42c14780a252b15e3eb492ad4f03c52eab6c` | 通用 Workflow、Math 统一评测和 Codex 修复 |
| 上游 `demo/robotics` | `c502aba08e93e9ab2136529a1c7b0492825a5a6c` | Episode 控制器、离线回读、执行记账和 Memory |

上游没有名为 `demo/maths` 的参考分支；Math 主要参考最新 Main 的 `examples/math/`。
实施时只移植所需变更，不整分支合并，不改私有参考分支，不向上游发布 Chips 内容。

### 2.2 现状与差距

| 编号 | 已有能力／实际差距 | 改进归属 |
| --- | --- | --- |
| G1 | Chips 已保留通用 Pi／Codex 会话；尚未包含 Main 的 Codex resume sandbox 修复 | A |
| G2 | vaBench／Analog 共用实验入口，但任务启动脚本直接构造共享 Pi／Codex Session，尚未接入共同的 Runtime／Environment 组装；支持组合写在入口中 | B、C |
| G3 | 模型设置已集中；任务、部署、认证引用与运行身份仍混在 operator 配置中 | B |
| G4 | EMX 已接通用 ToolSpec；vaBench／Analog 各自生成 MCP schema 与服务 | C |
| G5 | Analog 已复用 vaBench 的 local／SSH 传输；通用实现仍以 VABench 命名 | C |
| G6 | 已有完整单次记录与离线报告；没有统一的独立证据读取接口，旧记录形状也不完全相同 | D |
| G7 | Pi 有逐请求预算日志；原生 Codex 缺同等观测；两者内置工具权限与预算约束不同 | B、D |
| G8 | 统一入口目前只支持一题一次；没有任务集、独立重复与完整计划分母 | E |
| G9 | 共享记忆代码存在，但当前 Chips Agent 流程未接入跨 Episode Memory | F |
| G10 | 商业仿真器、开源仿真器及任务成熟度不同，不能换一个 simulator 名称就复用评分 | B、C 的边界 |

三个比较版本的 `common/trajectory/` 13 个文件、`common/execution/sandbox/` 10 个文件逐文件一致。
这表示代码基础共享，不表示各流程已经调用同一套记录器，也不表示三条线共用某个在线服务器。

当前能力边界保持如下，不因接口重构自动扩张：

| 条件 | 当前统一入口／已记录边界 | 本轮处理 |
| --- | --- | --- |
| vaBench + 服务器 Pi／GLM | 已有真实单题开发回合 | 保留并做新版本回归 |
| vaBench + 本机原生 Codex + SSH | 已有真实单题开发回合 | 保留并做新版本回归 |
| Analog 100 MHz RLC + 服务器 Pi／GLM | 已有真实单题回合；统一入口要求 local transport | 保留，不顺带声明 Analog Codex／SSH 已支持 |
| Spectre RC | 固定无 PDK RC 工具／作业基线 | 保留独立入口；不包装成通用 Agent benchmark |
| EMX | 既有布局到结果链路与通用工具接入 | 保留；真实设计评分仍依赖任务与专家指标 |

“已记录”指历史证据，详情以 [VALIDATION.md](VALIDATION.md) 为准；不能用历史回合替代改版后的集成验证。

## 3. 不变的契约

1. 支持服务器运行 Agent，以及本机 Agent 经 SSH 调用服务器；仿真仍在配置指定的服务器执行。
2. 公开 Tool 只提供任务允许的材料与反馈。提交冻结候选；独立终评不注册为 Agent 工具。
3. 原任务评分器与原始分数不变。执行完成、结果有效、设计达标、归档完整性分别记录。
4. 重复 action/job ID 不重复求解；执行状态未知时先查原任务，不因网络故障自动开启新仿真。
5. scratch 承担活跃小文件；持久归档经过校验才允许清理。SSH 断线恢复不等于服务器重启恢复。
6. 旧 CLI、配置和历史证据继续可读；新契约使用显式版本，原始归档不改写。
7. 保留共享内核及必要领域适配；不重新引入无关示例，也不删除内核中的 Math／Robotics API。
8. 默认 CI 不访问模型或实验室；不执行 Math／Robotics／Bio 领域任务。通用模块变动运行相关通用回归。
9. Key 内容不进入配置快照、Tool 子进程或 Git；模型、PDK 和原始轨迹继续遵循私有存储边界。

## 4. 目标结构与责任边界

```text
任务／评测配置 + Agent 配置 + 部署配置 + 本次运行计划
                           │
              解析、能力匹配、预检、冻结配置
                           │
                  Chips 实验协调入口
                    ┌──────┴──────┐
             Apollo Agent 会话    Chips 任务适配器
                    │             公开 Tool 与会话
                    └──────┬──────┘
                    local／SSH 控制协议
                           │
             服务器持久作业 → 任务固定的执行后端
                           │
              冻结候选 → 独立终评 → 私有归档
                           │
                 离线读取 → 报告／批量汇总
```

“local”表示 Agent 与执行入口位于服务器同机，不表示把仿真迁回个人电脑。
本机 Agent 的原始模型事件先写本机私有目录；服务器动作与仿真证据先写服务器私有目录。
使用同一 episode ID 和内容摘要关联，收取到操作者选择的归档位置后再生成完整清单。
缺少一侧证据时标记 partial，不把某一侧现有文件称为完整 Episode。
本轮不增加自动跨账号上传或集中接收原始轨迹的服务。

### 4.1 哪些共享，哪些仍由 Chips 拥有

| 责任 | 复用／保留位置 | 原则 |
| --- | --- | --- |
| Agent 进程、模型选择、MCP 桥接 | `alphaapollo/reasoning/runtime/` | 优先复用，不另造 Agent 主循环 |
| 工具基本类型、通用沙箱、通用轨迹 | `alphaapollo/common/` 的既有模块 | 只增加必要扩展；不塞入 task ID、PDK 路径 |
| 实验配置与批量协调 | `alphaapollo/workflows/chips*` | 负责调度与记录，不解释电路是否正确 |
| 服务器请求、作业、归档 | `alphaapollo/common/execution/chips/` | 先在 Chips 内共用；其他领域有实际需求后再上提 |
| vaBench／Analog 任务会话和评分桥接 | 现有 `vabench*`／`analog*` 模块 | 负责候选规则、公开反馈、独立终评 |
| Spectre／EMX／ngspice 适配 | 现有 `spectre.py`／`emx.py`／`ngspice.py` | 软件与工艺不同，不强制放进同一镜像 |
| 人可阅读的任务说明与示例 | `examples/chips/benchmarks/` | 固定 spec、来源、输出与评价方法 |

本轮不把所有任务模块搬到 `examples/`，也不新增一个同时管理仿真、评分和记忆的总类。
接口有独立用途才拆文件；下文新增模块名是建议落点，实施时可按实际代码量缩减。

### 4.2 与通用 Workflow 的关系

参考 Robotics：共享入口可以分派到领域评测器，不要求领域循环全部改写为普通 Workflow 步骤。
本轮保持 `chips_experiment` 为稳定入口，优先将任务环境注入现有 Runtime，参考 Robotics 的组装方式。
共享 Runtime 是此次实验目标；将来确有“规划→设计→审查”需求时，再增加多阶段 Workflow。
当前原生 Codex 的共享 `tool_loop` 存在原生工具权限限制，须独立验证接法；不能把注册了 Session 当作已接通。
不能让 Workflow 自动重试与服务器 action 去重、冻结终评分别控制同一任务。

## 5. A：同步共享 Codex 修复

**目标：**恢复会话时显式保留 sandbox，且不丢失 Chips 的原始事件记录。

**修改位置：**

- [codex.py](../../alphaapollo/reasoning/runtime/external/agents/codex.py)：移植 Main 对 resume 命令参数的修复。
- [test_external_codex.py](../../tests/reasoning/runtime/test_external_codex.py)：补充 read-only／workspace-write 续接回归。
- [.github/workflows/chips.yml](../../.github/workflows/chips.yml)：将该受影响通用回归纳入 CI。

**TDD 与验收：**先用两轮会话的构造 CLI 边界复现“续接命令未传 sandbox”，再修复参数位置；
断言初次与后续命令都带同一策略，模型与 MCP 配置保留，raw-event 归档仍有效。
验证现有 vaBench 单次 Codex 入口行为未变。本地命令构造测试不等于真实模型续接验收。

**交付：**一条独立修复提交。只移植涉及文件，不把整批上游 Math 示例带入 Chips。
这一修复针对共享续接能力；当前 vaBench 专用入口未启用该续接模式。

## 6. B：统一配置与能力声明

**目标：**换任务、Agent 或执行位置时，明确改变哪一项；拒绝不支持的组合，提前展示最终设置。

### 6.1 配置所有权

| 配置块 | 所拥有的字段 | 不应拥有的内容 |
| --- | --- | --- |
| benchmark/task | 任务 ID/版本、公开输入、候选规则、评分规则、最大仿真次数、任务时间限制、后端约束 | API Key、个人 SSH 别名 |
| agent | Agent 类型/CLI 版本、provider/model、thinking、模型预算、内置工具策略、提示版本 | 任务隐藏评分器、工艺文件内容 |
| deployment | Agent 所在位置、local/SSH、服务器运行路径、软件部署、scratch/归档根、认证引用 | 更改 benchmark 指标或阈值 |
| run | run ID、选定 profile、输出根、重复计划、终评启动策略 | 第二套互相矛盾的模型或任务设置 |

任务拥有“需要什么后端及版本约束”，部署拥有“软件实际在哪里”；不匹配在启动前阻断。
相同语义字段有唯一所有者，不设置多层静默覆盖。引用路径以配置文件所在目录解析，并冻结解析结果。
每次运行保存原始配置摘要、解析后的非敏感设置、源码/工具/任务版本、请求的模型以及实际报告的模型信息。
无法固定的服务端模型实现只记录可取得的版本/响应信息，不能把相同 model ID 当作后端权重从未变化的证明。

以下只展示拟议形状，**不是已经可运行的配置**：

```json
{
  "schema_version": 2,
  "run_id": "review-example-001",
  "benchmark": "vabench",
  "task_id": "v4-001",
  "agent_profile": "profiles/pi-glm.json",
  "deployment_profile": "profiles/server-local.json",
  "output": "/private/runs/review-example-001",
  "finalization": "after_confirmed_submit"
}
```

`finalization` 是操作者事先选择的策略，不授予 Agent 终评工具。
首版保留旧行为：vaBench 按原流程在确认提交后终评；Analog 默认仍显式 finalize。
只有明确配置自动终评的批量计划才允许协调器自动执行该步骤。

### 6.2 能力声明

将散落的条件判断收敛到小型、静态登记表，至少区分：

- 代码是否支持，以及哪组固定版本通过真实验收；未验收不伪装成已验证。
- Agent／任务／transport 组合、所需 Tool 与后端、是否可后台执行与收取。
- 模型预算项是运行时强制执行、仅请求 provider、仅观测，还是不支持。
- 工具集合是受限授权，还是包含 Agent 原生工具；保存实际暴露集合及可观测范围。

当前原生 Codex 缺少 Pi 同等的模型调用次数／输出预算控制。
要求严格相等预算的比较，若某端无法执行就应拒绝该实验条件；普通使用可以明确接受不等价条件。
不能仅在 JSON 中写同一个数值就称为公平对照，也不通过“同样 timeout”推导“同样 token 预算”。

多厂商凭据使用 provider 对应的私有认证引用，启动时只向所选模型进程传递必要凭据。
延续既有文件／环境变量入口，不把多家 Key 填进同一份可归档配置，也不在本轮建设 Key 共享服务。

### 6.3 落地与验收

建议新增 `alphaapollo/workflows/chips_config.py` 和小型 `chips_registry.py`，修改
[chips_experiment.py](../../alphaapollo/workflows/chips_experiment.py)、
[chips_experiment_settings.py](../../alphaapollo/workflows/chips_experiment_settings.py)。
现有任务 validator 保留任务专用检查；模型能力约束留在模型设置所有者，避免 registry 复制另一份规则。

先支持旧 v1→内部配置的投影，再添加 v2 profile。旧入口保持有效，不自动改写操作者文件。
增加无模型的 `plan/validate` 阶段，展示本次条件与阻塞项；纯配置验证不启动服务器进程，
实际主机预检另行显式执行。

验收至少覆盖：配置冲突在启动前被拒绝；Key 不进入快照；未知 Agent/后端组合不能启动；
v1 与等价 v2 产生相同工具与评分选择；模型推理档位必须满足原有显式设置要求；
不支持的预算不被标记为已强制执行；collect/finalize 遇配置漂移继续拒绝。

## 7. C：收敛任务适配、工具与持久传输

**目标：**vaBench 和 Analog 共用可靠基础设施，同时独立保留任务语义。

### 7.1 任务与工具

在静态任务注册项中引用小型适配器，负责 `validate/preflight`、公开工具、会话建立、
`finalize/collect` 以及结果投影。适配器调用原实现，不重写评分器；禁止配置任意 Python 导入路径。
首版只注册当前已支持的组合，不以“有适配器接口”作为新增模型或任务的验收。

公开工具由同一份 ToolSpec 生成模型 schema 与 MCP list_tools；调用处理仍交给各任务会话。
保留 `vabench_*`／`analog_*` 名称、参数与返回值，避免工具重命名影响模型表现。
共享的是注册、授权与投影方式，不把两个任务强制改成同一个 `simulate` 输出。
Tool 可脱离 Agent 直接调用与回放；终评和隐藏文件访问不能出现在任何公开注册表中。

建议新增 `common/execution/tools/chips_public.py` 作为公开工具目录；现有 EMX 工具职责保持。
优先验证现有 Runtime 的环境桥接能否服务这些 ToolSpec；不把模型凭据传给工具服务。
旧 `*_agent.py` 保持兼容，再逐条迁移内部组装；仅抽 Pi 启动 helper 不足以完成本轮 Runtime／Environment 目标。

### 7.2 传输与断线恢复

把 [vabench_remote.py](../../alphaapollo/common/execution/chips/vabench_remote.py) 中已被 Analog 共用的
请求/轮询/下载逻辑移到 Chips 内中性命名的 `transport.py`。
`RemoteVabench/LocalVabench/RemoteAnalog/LocalAnalog` 保留兼容薄包装，原 command 名与文件协议不变。

传输层只负责提交同一 envelope、查询同一 ID、取得响应和产物。
当前会话中的 `pending_action_id` 约束继续保留；新进程恢复必须从已存请求与服务端状态重建，
不能因为 Python 对象重建就假设前一动作不存在。若本轮发现该路径未实现，先补明确的恢复入口与测试，
不能在“重命名重构”提交中顺带引入不可审查的状态机变更。

仿真进程独立运行、去重、锁、冻结、终评意图和归档清理仍由原任务/作业模块负责。
不启用新的网络 daemon、HTTP 服务或 SSH 连接池；性能优化等分阶段计时提供证据后再做。

### 7.3 文件与验收

主要影响 `chips_vabench_agent.py`、`chips_analog_agent.py`、`vabench_session.py`、
`analog_session.py`、`vabench_remote.py`、`analog_remote.py` 及上述两个建议新增文件。

验收：原四工具名称/schema/语义兼容；实际 MCP list_tools 与声明一致；未授权工具不可调用；
直接调用与 MCP 调用对同一候选返回等价的公开结果；终评不可发现；
重复 ID 同 envelope 只执行一次、不同 envelope 被拒绝；响应丢失先查原 ID；
重建控制进程后不会绕过未知任务；模型凭据不进入工具/仿真进程。
冻结后不可再写候选、归档失败只重试归档等既有回归继续通过。

## 8. D：统一离线证据读取与观测

**目标：**提供稳定的 `read_episode` 概念，让单次报告、批量汇总和后续研究使用同一份可信投影。

参考 Robotics 的“先验证证据，再产生投影”原则；其具体记录绑定机器人控制器，不能直接复制成 Chips 数据。
以现有 `chips_episode_report.py`／`_chips_episode_report/evidence.py` 为基础拆分读取与展示；
建议新增 `alphaapollo/workflows/chips_episode.py`，原 CLI 和四类输出继续保留。
本轮先形成 Chips 契约；跨领域公共基类只在多个实际消费者需要后增加。

### 8.1 拟议读取结果

| 字段组 | 必须表达的事实 |
| --- | --- |
| identity | run/episode、task、benchmark、schema 和来源版本；缺项标明历史缺失 |
| settings | Agent/模型/推理、工具策略、预算执行能力、部署和后端身份 |
| execution | 运行进度、错误、停止原因；未知、阻塞、未提交不写成电路失败 |
| evaluation | 有效性、原始分数、单位/方向、可选二元成功、评分器版本与权威范围 |
| integrity | 已核验的对象和方法；reported hash、字节匹配、完整归档校验分别记录 |
| events/artifacts | 原始事件/产物引用、内容摘要、候选版本与终评关联 |
| accounting | 请求/工具/仿真计数、时间和 usage，每项附 coverage/source |
| rejection/gaps | 拒绝原因或缺失信息；不把损坏证据当作有效的零分记录 |

沿用旧状态字符串，先做无损投影，不强制修改所有历史结果 schema。
Analog 连续 reward 不转为通过率；`all_recorded_tests_passed` 与 `benchmark_success` 继续分开。

### 8.2 身份和计时

新记录使用 episode、agent call、server action、job 与候选摘要建立关联；
重试另记 attempt，但保持同一逻辑 action 身份。旧记录没有 ID 时标记无法关联，不按文本相似度猜测。

记录可观测阶段：模型请求、Tool 控制端、服务器动作、准备、仿真进程、归档与收取。
各主机使用自己的单调时钟计算持续时间，UTC 用于显示并标明主机；不能直接相减不同主机时间戳来推导网络延迟。
父子区间可能重叠，不直接相加作为总时间。求解器内部迭代只有原生日志提供时才展示，不虚构内部轨迹。

Pi 与 Codex 的观测能力分别标注；CLI 不提供的逐请求输入/token/时间继续为 unavailable。
保存 harness 实际提交的 system/user 内容与工具快照，不称为完整 provider 请求。
现有 raw-event 归档保持；新增请求捕获只使用可获得、可过滤凭据的接口，不为补指标引入 TLS 拦截。

### 8.3 验收

- 保留现有成功、修复后成功、预算耗尽与原生 Codex 四份历史证据，离线新旧投影的已知成绩与候选身份一致。
- 禁止网络与子进程后，仍能从完整归档生成报告；不读当前代码来补历史 prompt。
- 归档可迁移目录；产物引用必须在指定证据根内，拒绝越界路径与哈希不符。
- 故意损坏一个成员时，读取结果明确拒绝；批量报告保留该计划项及拒绝原因，不能漏掉样本。
- 缺失 token 不是零；重复事件不重复计数；只记录调用尝试时不伪称仿真已执行。
- v1 报告 CLI 兼容，输出目录权限/拒绝覆盖不变，读取前后源证据摘要一致。

## 9. E：任务目录与批量评测

**目标：**在单次实验之上添加薄协调器，复用同一执行、终评和读取逻辑，不再做第二套求解循环。

### 9.1 计划与运行

建议新增 `alphaapollo/workflows/chips_evaluate.py`，任务目录复用 B 的静态 registry。
首版明确列出已接入的 task ID 与 revision；不把 Analog 正例回放的三题都声明为 Agent 任务。
目录固定后支持显式子集与 repeats，默认串行，先不支持跨账号分布式调度。

执行前生成全部计划 cell：`benchmark/task_revision/task_id/condition_id/repeat`。
每个 cell 有独立 run ID、会话、候选目录、Agent 工作区与输出；不得由协调器注入上一轮候选或对话。
按适配器能力使用独立的临时 HOME/配置目录；认证可引用原私有凭据存储，不复制登录文件进证据目录。
目录分开只证明流程没有主动复用；若 Agent 的原生文件工具仍可读取其他目录，不能宣称操作系统层面已隔离。
严格独立评测需要额外核验实际可读范围；无法约束的条件只作为已声明限制的系统实验，不标为严格隔离条件。
配置阶段拒绝未知任务/重复 ID；主机预检阻塞的 cell 仍在清单中，不从分母消失。

协调器支持两种停止策略并预先记录：当前 cell 失败后继续其余计划，或停止并将剩余 cell 标为 not_run。
无论哪种都不自动补跑到成功。恢复批次仅收取已有任务、恢复归档，或启动确实尚未开始的 cell；
执行状态未知的 cell 必须先核对原 job。重跑失败模型回合是新的 attempt，保留原失败。

终评沿用 C 的任务适配器与事先声明的 finalization 策略。
批量入口不能把 Analog 的显式终评悄悄改成自动，也不能对超时但可能仍在运行的终评再启动一次。

### 9.2 汇总规则

批量根目录保存 `resolved_config.json`、`task_manifest.json`、`preflight.json`、`results.jsonl`、
`summary.json` 和独立 cell 目录；名称借鉴 Math，字段由 Chips 契约定义并带版本。

每个计划 cell 必有结果行；区分未启动、阻塞、超时、未提交、无效、有效失败、有效成功以及证据损坏。
汇总给出计划数、启动数、有效终评数、缺失数和每种失败原因。
二元成功率只对有明确 Boolean 成功语义的有效结果计算，并同时报告有效覆盖率。
可另列“计划样本中的成功比例”，但必须明确把它命名为端到端交付指标，不能混作电路正确率。

连续分数只在同一任务版本与评分规则下汇总，展示样本数和分布；单位不同的任务不平均。
模型 usage 缺失时报告已知量与覆盖，不把部分成本写成总成本。
Agent 与模型同时变化时标为“整套系统条件比较”；只有控制住其余变量才归因于某单项。

### 9.3 验收

用两个构造任务、两个 condition、两次重复验证 8 个 cell 全部存在；包含完成、阻塞、
未提交、未知执行和终评失败，不只测试全绿路径。
中断后重读清单不会重复已有求解；恢复后同一 cell 不出现两条互相覆盖的成功结果。
相同任务不同重复不共享候选、记忆或 Agent 会话；不同量纲不合并成总分。
声明严格隔离的条件还需在实际 Agent 权限下验证跨 cell 读取被拒绝，不能只检查输出目录名称不同。
离线汇总与逐个 D 读取结果一致，损坏/缺失 cell 保留在 coverage 中。

## 10. F：Memory 独立里程碑

**目标：**将经过检查的公开经验作为显式实验因素；不把完整归档直接发送给下一次 Agent。

此阶段依赖 B、D、E，不是 A–E 的交付门槛。建议沿用 `workflows/memory/adapter.py` 与
`evolving/`，新增 Chips 投影，并在 Agent 输入记录实际注入的条目 ID、内容摘要和检索耗时。

区分两种使用：正式独立评测默认 memory-off；经验迁移实验使用冻结、只读的 memory snapshot，
显式记录顺序、来源与数据划分。已有 vaBench r53 禁止跨任务 Memory 的条件继续拒绝开启。
当前 Episode 的公开对话历史不是跨实验 Memory，关闭 Memory 不意味着让 Agent 忘记刚收到的 Tool 反馈。

经验候选记录来源 task/revision、适用 simulator/PDK、复现步骤、失败边界及审核状态。
只有公开开发材料与允许用于后续任务的经验进入快照，隐藏答案和测试任务终评反馈不注入。
Agent 自述“修好了”不作为可信记忆；以可复核证据支持的公开结论或专家审核为依据。

验收包括：memory-off 没有检索/注入；不兼容任务版本不命中；禁止条件提前拒绝；
快照内容不能被评测过程修改；损坏快照不静默退化为对照组；检索日志可以还原实际注入。
memory-on 的效果需新对照实验验证，代码接通不代表有所提升。

## 11. 小提交顺序与回滚

每行是可单独 review 的提交单位；实际代码量过大时可继续拆，但不能把多个阶段压为一条大提交。
每次本地相关检查通过后提交并 push 到私有 `demo/chips`，确认对应 CI 后继续；修复提交保留历史，不强推。

| 次序 | 建议提交内容 | 可观察验收 | 依赖 |
| --- | --- | --- | --- |
| 1 | Codex resume sandbox 修复与相关 CI 测试 | 初始/续接参数与 raw-event 回归通过 | A |
| 2 | v1 配置规范化与能力声明 | 旧实验选择不变，不支持组合提前拒绝 | 1 |
| 3 | v2 profile 与 plan/validate | 所有权冲突、凭据过滤、预算能力可 review | 2 |
| 4 | 工具目录统一与 MCP 投影 | 同一 ToolSpec，原四工具接口不变 | 2 |
| 5 | 中性传输模块与兼容包装 | 两任务 local/SSH 构造协议回归通过 | 4 |
| 6 | 如有缺口，补控制进程重建后的未知 action 恢复 | 同 ID 查回，不能重复求解 | 5 |
| 7 | 小型 benchmark 适配器替换顶层分支 | 原 run/collect/finalize/archive 行为一致 | 3–6 |
| 8 | 独立 Episode 读取契约，报告调用它 | 四份历史证据离线等价、损坏明确拒绝 | 7 |
| 9 | 新事件关联与可观测阶段计时 | 重试不重计，不跨主机相减，旧证据可读 | 8 |
| 10 | 任务目录与批量计划生成 | 完整计划、身份唯一、未启动项保留 | 3、8 |
| 11 | 串行批量执行、收取与恢复 | 8-cell 失败混合夹具、不重复终评 | 7、10 |
| 12 | 批量离线汇总、配置示例与使用文档 | 单次与批次结果一致，覆盖与分数分开 | 9、11 |
| 后续 | Memory 快照、注入、隔离与对照记录分次实现 | 依第 10 节验收 | A–E 验收后 |

纯重构依靠既有行为回归；新增行为按 SOP 逐项 Red→Green→Refactor，不能把旧测试通过冒充 TDD 的 Red。
协议变更与文件迁移分开。若某提交回滚，保留其产生的版本化证据，不用旧 reader 覆盖新格式。
删除兼容入口必须另行评估消费者，本方案不预先批准删除 v1 CLI。
若重构引入状态/评分变化，先停止后续结构调整并复现原因；回滚代码不等于回滚外部作业，
已有服务器动作继续按原 ID 检查与收取，不能重放提交命令来验证回滚效果。

## 12. 测试与验收层次

### 12.1 需要长期保留的测试

| 测试位置 | 新增/扩展行为 |
| --- | --- |
| `tests/reasoning/runtime/test_external_codex.py` | 续接 sandbox、既有原始事件保存 |
| `tests/chips/test_experiment.py` | v1/v2、配置冻结、适配器、终评与归档状态兼容 |
| 建议 `tests/chips/test_experiment_config.py` | profile 所有权、能力矩阵、预算与 Key 引用 |
| 现有 `test_vabench_agent.py` / `test_vabench_codex.py` / `test_analog_agent.py` | 真实本地 MCP 进程、工具发现和权限；模型响应为夹具 |
| 现有 `test_vabench_session.py` / `test_analog_remote.py` / `test_analog_session.py` | 去重、断线构造、冻结与未知状态恢复 |
| `tests/chips/test_episode_report.py`，必要时新增 `test_episode_readback.py` | 历史读取兼容、缺失与损坏、离线路径边界 |
| 建议 `tests/chips/test_evaluate.py` | 计划、独立重复、中断恢复、coverage 和计分 |
| 后续建议 `tests/chips/test_memory.py` | 快照冻结、划分约束、注入记录与 memory-off |

静态配置/证据契约若采用项目的可导出组件 schema，则同步生成相邻 schema；
不为普通内部字典强行接入不相关 schema 框架。CI 只运行实际新增且适用的检查。

### 12.2 实施中的检查命令

以下是现有路径可用的受影响回归入口；实施时按提交选取子集，并添加新测试文件。
本方案编写时没有运行这些行为测试。

```bash
.venv/bin/python -m pytest -q tests/reasoning/runtime/test_external_codex.py tests/chips/test_vabench_codex.py
.venv/bin/python -m pytest -q tests/chips/test_experiment.py tests/chips/test_vabench_agent.py tests/chips/test_analog_agent.py
.venv/bin/python -m pytest -q tests/chips/test_episode_report.py tests/workflows/test_visualize.py
.venv/bin/ruff check alphaapollo tests
.venv/bin/ruff format --check alphaapollo tests
.venv/bin/python -m alphaapollo._schema_export
git diff --check
```

最终本地集成集合沿用 [Chips CI](../../.github/workflows/chips.yml) 和 [测试入口](../../tests/chips/README.md)，
不加入 Math／Robotics／Bio 的领域仿真。没有依赖的 skip 单列，不能算通过。

### 12.3 真实验收顺序

1. **历史离线兼容：**四份现有 Episode，核对候选、分数、类别、usage 已知量和缺口；源文件摘要不变。
2. **服务器工具直接回放：**固定公开候选与负例，分别验证现有 EVAS 和 ngspice 条件；不调用模型。
3. **控制端恢复：**只终止本次测试客户端/控制进程，确认同一 ID 的服务器动作继续、能被收取且没有第二次求解。
4. **三条真实 Agent smoke：**服务器 Pi＋GLM vaBench、本机 Codex＋SSH vaBench、服务器 Pi＋GLM Analog 各一条。
   运行前冻结 Agent/模型版本、thinking、工具策略、预算、任务/后端版本；全部记录，不要求 Agent 必须解对题才能承认协议正确。
5. **小批量验收：**建议先固定一种 Pi＋GLM 条件，对两道已接入任务各三次，共六个计划样本。
   用于证明调度、独立重复与汇总可用，不据此宣称稳定排名或精确尾延迟。后续 Agent/模型对照单独定计划。

真实运行按当时用户已授权的服务器、模型与费用范围执行；没有自动新增费用上限，也不无限重试。
阻塞项保留原记录与原因，不因测试计划要求凑齐成功样本而重跑。
模型没有正确提交属于模型回合结果；工具输入被错误拒绝、重复求解、隐藏终评泄漏则属于 Harness 验收失败。
完整请求捕获、服务器重启续跑、第二位成员复现不因以上通过而自动验收。

## 13. 阶段完成条件与暂缓项

### A–E 可以交付的条件

- 旧单次入口可用，新配置能明确区分任务、Agent 和部署；不支持组合在启动前阻断。
- vaBench 与 Analog 的公开工具、持久动作及终评语义保持，实际恢复不产生重复求解。
- 报告、单次结果和批量汇总从同一证据投影获得成绩；原始数据、缺失和拒绝原因可复核。
- 每个计划样本都有记录，各重复彼此隔离；不同分数单位、不同预算能力不被误称为相同条件。
- 本地回归/CI 与真实集成分别报告；三条真实路径未全部复验时，交付状态必须说明仅代码层完成。

### 本轮暂缓

| 暂缓项 | 原因／后续触发条件 |
| --- | --- |
| 全面改走通用 Workflow | 暂无必须重写任务控制器的需求；待多阶段 Agent 实验需要时接入 |
| 整理全部 `chips*` 文件到新包、删除旧 CLI | 容易混入行为变更；等接口稳定后单独做兼容迁移 |
| 全部仿真器统一 Docker/Podman | 软件、许可、工艺不同；按任务验证后端等价性 |
| Analog 原生 Codex／更多 Agent | 独立能力扩展，不能用接口重构代替真实验收 |
| Spectre 任意网表、晶体管/PDK、EMX 正式设计题 | 需要明确的任务输入、工艺、指标与评分证据 |
| 多用户调度、Key 池、公共服务、Web 管理后台 | 当前目标是可复现的个人私有运行，不需要先建设平台服务 |
| SSH 连接池、并发批次、缓存加速 | 先测分阶段耗时与资源瓶颈，避免改变对照条件 |
| 服务器重启后恢复整个 Agent 会话 | 超出当前持久动作恢复能力，另设状态/进程管理设计 |
| 文档/图片生成任务 | 继续按 NEXT_WORK 工作包 1；代表性资料与专家确认独立推进 |

## 14. 用户 review 清单

以下为本方案建议，可直接在对应章节修改；本次不因文档生成而开始实现。

| 决策 | 建议默认选择 | 影响 |
| --- | --- | --- |
| 近期范围 | A–E；F Memory 后续独立交付 | 优先稳定可比较评测 |
| 统一程度 | 统一接口与记录，保留 Chips 控制器 | 保留断线/冻结/归档能力 |
| 兼容策略 | v1 和旧 CLI 保留，v2 增量引入 | 已有实验不被迫迁移 |
| 工具策略 | 先保持各 Agent 现状并显式声明 | 不把重构与模型可见工具变化混在一起 |
| 仿真后端 | 每任务固定，暂不迁移现有后端 | 历史证据的解释不改变 |
| 批量规模 | 首版串行；真实试点两任务各三次 | 先验收生命周期，不做性能排名 |
| 实施节奏 | 小提交、逐次 push、对应 CI 通过后继续 | 每步可 review 与回滚 |

建议首先 review 第 3 节不变契约、第 6 节配置所有权、第 9 节批量统计和第 11 节提交顺序。
这些决定后续代码边界；模块名和具体 JSON 字段可在实现第一个行为切片时微调，但不能改变以上语义而不更新方案。

## 15. 参考实现

- [Main／Math 统一评测](https://github.com/AndrewZhou924/AlphaApollo-v3-dev/blob/a3be42c14780a252b15e3eb492ad4f03c52eab6c/examples/math/EVALUATION.md)：任务选择、独立重复、原评分与 coverage。
- [Main Codex 修复](https://github.com/AndrewZhou924/AlphaApollo-v3-dev/commit/a3be42c14780a252b15e3eb492ad4f03c52eab6c)：续接 sandbox。
- [Robotics 评测入口](https://github.com/AndrewZhou924/AlphaApollo-v3-dev/blob/c502aba08e93e9ab2136529a1c7b0492825a5a6c/alphaapollo/workflows/robotics.py)：共享入口分派到领域控制器。
- [Robotics 离线记录读取](https://github.com/AndrewZhou924/AlphaApollo-v3-dev/blob/c502aba08e93e9ab2136529a1c7b0492825a5a6c/alphaapollo/workflows/records.py)：验证身份与证据后投影。
- [Robotics Memory](https://github.com/AndrewZhou924/AlphaApollo-v3-dev/blob/c502aba08e93e9ab2136529a1c7b0492825a5a6c/alphaapollo/workflows/memory/episode.py)：可信结果落盘后写入记忆，限制正式评测条件。
- [Chips 单次入口](UNIFIED_EXPERIMENT.md)、[离线报告](EPISODE_REPORT.md)、[开发 SOP](DEVELOPMENT_SOP.md)：当前兼容与证据边界。
