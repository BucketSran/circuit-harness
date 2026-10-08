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

### 已有原生 Harbor 任务

如果 benchmark 已提供 `instruction.md`、`task.toml`、`environment/` 和 `tests/`，优先保留这些文件，使用 Harbor 自己的环境与 verifier。A、B 示例采用这条路径。

特定 benchmark 的准备与重放入口放在 `circuit_harness/benchmarks/`。AnalogBench 使用
`python -m circuit_harness.benchmarks.analogbench`，VA07 保存候选使用
`python -m circuit_harness.benchmarks.evas_va07`。Harbor 环境与 verifier 插件放在
`circuit_harness/harbor/`，仿真执行与固定任务源登记仍由 `circuit_harness/execution/` 维护。

用 `circuit_harness.harbor.profiles` 为原生 Job 编译 Agent 与 Model 配置。此时任务的文件收取、测试注入和评分隔离遵循 Harbor 与该任务的实现；它没有自动获得 Harness 公开会话的动作预算、冻结收据或私有远程终评能力。

任务仓库中的 `solution/` 用于操作者的 oracle 验收，不能放进 Agent 镜像或公开材料。发布示例准备代码与来源引用，下载后的完整任务保存在忽略目录。参考解和错误候选分别验收后，再开展真实 Agent 实验。

### 需要公开会话与独立终评

如果 Agent 只应接触有限公开材料，或最终评分在独立服务器执行，使用 `CircuitDockerEnvironment` 或 `CircuitPodmanEnvironment`、`CircuitAgent` 和 `FrozenCandidateVerifier`。

```text
公开 Harbor task
  → Harbor stock Agent + 独立 Model 配置
  → harness-public → 公开会话 → 允许的仿真后端
  → 关闭公开入口、收取冻结候选
  → 私有 task package + verifier → 成绩与归档
```

按[公开会话合同](../chips/CURRENT_EVAS_PUBLIC_SESSION.md)准备公开材料、入口和预算。按[独立评测合同](../chips/BENCHMARK_EVALUATION.md)准备 final package 与 checker，再按[任务绑定](../chips/HARBOR_TASKS.md)连接它们。`task_id` 与 `task_version` 必须一致，配置和隐藏材料必须位于 task 与 Trial 导出目录之外。

当前实现提供具体后端与任务格式，没有任意 `SimulatorAdapter` 类的自动发现机制。新增接口时复用会话、进程管理和证据模块，扩展任务真正需要的输入输出。不要把 EVAS 的电压行为模型接口直接用于任意 SPICE 晶体管网表。

## 3. 准备工具与私有资源

| 环境 | 使用者提供 | 框架接入位置 |
| --- | --- | --- |
| 无 PDK 的 ngspice 任务 | ngspice 运行环境及任务文件 | [Analog 示例](../../examples/analogbench/README.md)或[ngspice 执行接口](../chips/NGSPICE.md) |
| ngspice 与开源 PDK | 固定版本的模型文件、支持该模型的仿真器 | [SKY130 OTA 示例](../../examples/analogbench/README.md) |
| EVAS | 固定源码、匹配平台的 kernel、公开材料和任务 checker | [EVAS 示例](../../examples/evas-va07/README.md)、[公开会话](../chips/CURRENT_EVAS_PUBLIC_SESSION.md) |
| 商业仿真器 | 已授权安装、PDK、许可证及本地或 SSH 访问 | [Spectre 示例](../chips/SPECTRE_RC.md)、[独立终评](../chips/BENCHMARK_EVALUATION.md) |

先按仿真器自己的安装说明建立可工作的环境。实际地址、安装路径和凭据写入 `private/`、`*.local.json` 或外部配置存储，公开模板仅展示字段。不要把模型 API key 传给 verifier，也不要把许可证配置挂载进公开解题环境。

公开仿真器禁网时，依赖网络许可证的商业仿真不能直接放进该容器。将它配置为受控远端执行或独立终评。服务器运行 Agent 时同样可选 Pi 等已支持的 Harbor Agent，不要求本地 Codex 参与。

## 4. 分别配置 Agent 与模型

复制[配置目录模板](../../examples/chips/harbor/profiles.example.json)。在 `agents` 中填写 Agent 设置，在 `models` 中填写模型 ID、服务地址、协议和密钥环境变量名。新增模型不需要复制整个 Agent 实现。

按 [Harbor 配置说明](../chips/HARBOR.md#编译-jobconfig-并交给-harbor)编译 Job，再由 Harbor 启动。配置编译不会发送模型请求。原生 Harbor 任务和公开会话任务都复用这个配置入口。

预算由实际任务与 Job 声明。动作预算只适用于提供该能力的公开会话，不能把它当作对所有原生 Agent shell 调用的统一限制。部署位置、网络和镜像检查见[部署指南](../chips/HARBOR_DEPLOYMENT.md)。

## 5. 验收并查看证据

先跑参考候选，再跑错误或不可评候选。分别核对执行成功、结果有效和电路达标。checker 出错、缺许可证或分析不受支持都不应被记录为模型零分。

对原生 Harbor 示例，查看其 Job、Trial、verifier 和 Agent 日志。对 Harness 公开会话路径，按[结果报告](../chips/HARBOR_RESULTS.md)核验冻结候选、独立评分和归档。离线重放按示例检查 candidate、task、checker、engine 与配置身份，不能仅凭退出码判断成功。

最后用自己选择的 Agent 与模型运行任务，并检查工具调用和最终提交是否来自这次尝试。需要训练数据时，按 [ATIF 数据说明](../../circuit_harness/data/README.md)检查轨迹完整性、筛选条件与实际支持范围。参考候选验收和模型解题成绩分开报告。
