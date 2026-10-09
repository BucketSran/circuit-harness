# 接入你的电路实验

本指南面向需要接入 benchmark、仿真器或实验环境的开发者。先完成 [AnalogBench RLC 示例](../../examples/analogbench/README.md)，确认 Docker、Harbor 和模型连接可用，再替换任务与工具条件。

## 1. 固定任务及其评测条件

为任务记录以下内容，可以沿用 [任务卡模板](../../examples/chips/benchmarks/TASK_TEMPLATE.md)：

| 项目 | 需要明确的内容 |
| --- | --- |
| 来源 | 仓库 URL、具体 commit、任务文件摘要、内容许可 |
| 候选 | Agent 可以修改哪些文件，电路端口、格式和大小限制 |
| 公开工具 | 可以执行的分析、公开材料、时间与仿真预算 |
| 终评 | checker 版本、测试条件、成绩含义、合法零分与不可评的区别 |
| 运行依赖 | 仿真器版本、PDK、镜像、许可证和部署位置 |

任务决定什么是正确电路，Harness 负责执行并保留证据。替换后端前，确认其支持任务需要的模型语法和分析类型。不能只替换命令名，就把另一仿真器的结果视为同一种实验条件。

## 2. 选择任务执行方式

两条路径都由 Harbor 安装 Agent、执行 Trial 和管理 Job。区别是任务是否接入 Harness 公开会话。这里的 session 指一次电路实验的工具访问、预算和候选收取过程，不是模型的聊天会话。

| 比较项 | 原生 Harbor 任务 | 使用 Harness 公开会话的任务 |
| --- | --- | --- |
| 仿真访问 | Agent 按原任务约定使用环境中的仿真工具 | Agent 通过 `harness-public` 访问允许的仿真后端 |
| 仿真预算 | 原任务与工具定义仿真限制，Harbor 管理执行阶段期限 | Harness 额外限制经过公开入口的动作和仿真次数 |
| 候选收取 | 按原任务约定收取文件 | Harness 关闭公开入口、冻结候选并记录摘要 |
| 最终评分 | Harbor 执行原任务的 verifier | `FrozenCandidateVerifier` 将冻结候选交给绑定的私有终评后端 |
| 执行证据 | 原生 Job、Trial、Agent 与 verifier 日志 | 在原生日志上增加会话记录、动作证据、冻结收据和终评关联 |

Harbor 原生任务同样可以隐藏测试并隔离 verifier，实际边界由任务和环境实现决定。Harness 公开会话提供统一的电路访问与冻结合同，不表示所有独立评分都需要它。Agent、模型和部署位置分别配置，两条路径都可以让 Pi + GLM 全部在服务器运行。

### 已有原生 Harbor 任务

如果 benchmark 已提供 `instruction.md`、`task.toml`、`environment/` 和 `tests/`，优先保留这些文件，使用 Harbor 自己的环境与 verifier。A、B 示例采用这条路径。

例如 AnalogBench RLC 使用任务内的 ngspice 与原 checker：

```text
Harbor 启动 Agent + Model
  → Agent 修改电路文件，调用任务环境中的 ngspice
  → 按原任务约定收取电路
  → Harbor 执行原 verifier → 成绩与原生日志
```

特定 benchmark 的准备与重放入口放在 `circuit_harness/benchmarks/`。AnalogBench 使用
`python -m circuit_harness.benchmarks.analogbench`，VA07 保存候选使用
`python -m circuit_harness.benchmarks.evas_va07`。Harbor 环境与 verifier 插件放在
`circuit_harness/harbor/`，仿真执行与固定任务源登记仍由 `circuit_harness/execution/` 维护。

用 `circuit_harness.harbor.profiles` 为原生 Job 编译 Agent 与 Model 配置。此时文件收取、测试注入和评分隔离遵循 Harbor 与该任务的实现，不会自动接入 Harness 的动作预算、冻结收据和私有终评接口。

任务仓库中的 `solution/` 用于操作者的 oracle 验收，不能放进 Agent 镜像或公开材料。发布示例准备代码与来源引用，下载后的完整任务保存在忽略目录。参考解和错误候选分别验收后，再开展真实 Agent 实验。

### 需要公开会话与独立终评

如果实验需要统一的受控仿真入口、候选冻结或私有远程终评，而原任务没有提供这些能力，使用 `CircuitDockerEnvironment` 或 `CircuitPodmanEnvironment`、`CircuitAgent` 和 `FrozenCandidateVerifier`。例如 Agent 用 EVAS 调试，最终只由私有 Spectre verifier 评分。

```text
公开 Harbor task
  → Harbor stock Agent + 独立 Model 配置
  → harness-public → 公开会话中读写候选、请求允许的仿真后端
  → 关闭公开入口、收取冻结候选
  → 私有 task package + verifier → 成绩与归档
```

当前 gateway 路径中的候选通过公开入口读写；Agent 容器内的普通文件不是终评提交对象。动作预算只统计经过该入口的调用，不限制所有 Agent shell 操作。Agent 沿用自己的工具循环和模型客户端。

按[公开会话合同](../reference/CURRENT_EVAS_PUBLIC_SESSION.md)准备公开材料、入口和预算。按[独立评测合同](../reference/BENCHMARK_EVALUATION.md)准备 final package 与 checker，再按[任务绑定](../reference/HARBOR_TASKS.md)连接它们。`task_id` 与 `task_version` 必须一致，配置和隐藏材料必须位于 task 与 Trial 导出目录之外。

当前实现提供具体后端与任务格式，没有任意 `SimulatorAdapter` 类的自动发现机制。新增接口时复用会话、进程管理和证据模块，扩展任务真正需要的输入输出。不要把 EVAS 的电压行为模型接口直接用于任意 SPICE 晶体管网表。

## 3. 准备工具与私有资源

| 环境 | 使用者提供 | 框架接入位置 |
| --- | --- | --- |
| 无 PDK 的 ngspice 任务 | ngspice 运行环境及任务文件 | [Analog 示例](../../examples/analogbench/README.md)或[ngspice 执行接口](../reference/NGSPICE.md) |
| ngspice 与开源 PDK | 固定版本的模型文件、支持该模型的仿真器 | [SKY130 OTA 示例](../../examples/analogbench/README.md) |
| EVAS | 固定源码、匹配平台的 kernel、公开材料和任务 checker | [EVAS 示例](../../examples/evas-va07/README.md)、[公开会话](../reference/CURRENT_EVAS_PUBLIC_SESSION.md) |
| 商业仿真器 | 已授权安装、PDK、许可证及本地或 SSH 访问 | [Spectre 示例](../reference/SPECTRE_RC.md)、[独立终评](../reference/BENCHMARK_EVALUATION.md) |

先按仿真器自己的安装说明建立可工作的环境。实际地址、安装路径和凭据写入 `private/`、`*.local.json` 或外部配置存储，公开模板仅展示字段。不要把模型 API key 传给 verifier，也不要把许可证配置挂载进公开解题环境。

公开仿真器禁网时，依赖网络许可证的商业仿真不能直接放进该容器。将它配置为受控远端执行或独立终评。服务器运行 Agent 时同样可选 Pi 等已支持的 Harbor Agent，不要求本地 Codex 参与。

## 4. 分别配置 Agent 与模型

复制[配置目录模板](../../examples/chips/harbor/profiles.example.json)。在 `agents` 中填写 Agent 设置，在 `models` 中填写模型 ID、服务地址、协议和密钥环境变量名。新增模型不需要复制整个 Agent 实现。

按 [Harbor 配置说明](../reference/HARBOR.md#编译-jobconfig-并交给-harbor)编译 Job，再由 Harbor 启动。配置编译不会发送模型请求。原生 Harbor 任务和公开会话任务都复用这个配置入口。

预算由实际任务与 Job 声明。动作预算只适用于提供该能力的公开会话，不能把它当作对所有原生 Agent shell 调用的统一限制。部署位置、网络和镜像检查见[部署指南](HARBOR_DEPLOYMENT.md)。

## 5. 验收并查看证据

先跑参考候选，再跑错误或不可评候选。分别核对执行成功、结果有效和电路达标。checker 出错、缺许可证或分析不受支持都不应被记录为模型零分。

按任务路径选择当前可用的证据入口：

| 路径 | 结果与轨迹入口 | 当前限制 |
| --- | --- | --- |
| 原生 Harbor | Job、Trial 结果，原 verifier 输出与 Agent 日志 | 当前 Harness 离线报告和正式 Pi 导出不接受仅有原生 reward、缺少公开会话冻结证据的任务 |
| Harness 公开会话 | [离线结果报告](HARBOR_RESULTS.md)，符合条件的 [Pi ATIF 导出](../reference/HARBOR.md#offline-pi-trajectory-export) | 报告核验冻结候选及独立凭据；正式轨迹导出当前仅支持完整的单个 Pi 会话，其他限制见导出合同 |

原生任务的成绩仍按原 checker 判断，不因尚未接入 Harness 报告而失效。统一两条路径的结果与评分关联属于[后续工作](../development/NEXT_WORK.md)，不能通过补造 session 或冻结收据来绕过当前限制。能读取或转换对话，不表示已经核验任务、最终候选和评分关联。

离线重放按示例检查 candidate、task、checker、engine 与配置身份，不能仅凭退出码判断成功。

最后用自己选择的 Agent 与模型运行任务，并检查工具调用和最终提交是否来自这次尝试。需要训练数据时，按 [ATIF 数据说明](../../circuit_harness/data/README.md)检查轨迹完整性、筛选条件与实际支持范围。参考候选验收和模型解题成绩分开报告。

服务器部署也按此顺序验收。先确认镜像架构、离线依赖和容器资源限制，
再运行参考候选与合法的性能负例，最后运行有预算限制的 Agent Trial。
已有的[服务器 SKY130 对照](../../examples/analogbench/README.md#linux-服务器上的性能对照)
展示了参考 7/7、性能负例 4/7，以及 CPU controller 缺失时如何保留基础设施失败。
公开会话路径先执行[静态与环境预检](HARBOR_DEPLOYMENT.md)；
预检通过仍不代表模型鉴权或商业终评许可证已验收。
另见 [EVAS 与独立 Spectre 的服务器实验](../../examples/evas-va07/README.md#服务器公开会话实验)。
该实验分别保留了超时的模型 Trial、事后恢复的独立评分和成功的参考候选控制，
没有把后续恢复结果改写为原 Trial 的成功。
