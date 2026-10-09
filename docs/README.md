# Circuit Harness 文档

从运行一个已有任务开始，再接入自己的 benchmark、仿真器与部署环境。
Harbor 负责 Agent、Job 和 Trial；Harness 补充电路工具访问、候选收取、独立终评和证据处理。

## 运行实验

| 要完成的工作 | 文档 |
| --- | --- |
| 安装 Python 包与所需 extras | [安装](../INSTALL.md) |
| 首次运行，无 PDK 与商业许可证 | [AnalogBench RLC](../examples/analogbench/README.md) |
| 使用开源工艺模型 | [AnalogBench SKY130 OTA](../examples/analogbench/README.md#b验证开放-sky130-路径) |
| 用 EVAS 重放保存候选 | [EVAS VA07](../examples/evas-va07/README.md) |
| 接入自己的任务、后端和环境 | [开发者接入指南](guides/integration.md) |

已有完整 Harbor task 时，保留原环境与 verifier。需要受控仿真入口、动作预算和
候选冻结时，使用 Harness 公开会话。两条路径的选择及结果入口以接入指南为准。

## 配置与协议

| 需要查阅的合同 | 文档 |
| --- | --- |
| Agent 与模型连接、Job 配置 | [Harbor](chips/HARBOR.md) |
| Docker、Podman 与服务器预检 | [Harbor 部署](chips/HARBOR_DEPLOYMENT.md) |
| 私有目录、凭据与访问边界 | [部署规范](chips/DEPLOYMENT.md) |
| 公开材料、仿真动作与预算 | [EVAS 公开会话](chips/CURRENT_EVAS_PUBLIC_SESSION.md) |
| 多任务公开/私有配置绑定 | [任务绑定](chips/HARBOR_TASKS.md) |
| 冻结候选、Spectre 与开源终评 | [独立评测](chips/BENCHMARK_EVALUATION.md) |
| 独立 RC 仿真与解析解验收 | [ngspice](chips/NGSPICE.md)、[Spectre](chips/SPECTRE_RC.md) |
| 固定任务的操作者协议 | [VABench r53](chips/VABENCH.md)、[Analog](chips/ANALOG_DESIGN_BENCH.md) |
| 后台作业、归档与清理 | [存储使用](chips/STORAGE.md) |

仿真器与模型服务由使用者提供。任务决定允许的模型语法、分析类型和评分条件；
提供适配接口不包含软件、PDK 或许可证。能力范围见[仿真器接入](chips/OPEN_SIMULATORS.md)。

## 查看结果与准备数据

- [实验条件与比较](chips/EXPERIMENT_PROTOCOL.md)：固定哪些条件，如何解释失败和成绩。
- [Harbor 离线报告](chips/HARBOR_RESULTS.md)：公开会话冻结证据与独立评分核验。
- [保存 Episode 查看器](chips/EPISODE_REPORT.md)：读取兼容的历史归档。
- [ATIF 数据](../circuit_harness/data/README.md)：正式 Pi 导出、数据筛选与外部训练器加载。
- [验证记录](chips/VALIDATION.md)：已观察结果、版本和证据限制。
- [后续工作](chips/NEXT_WORK.md)：尚未完成的接入与验收。

## 开发框架

模块入口是 [benchmarks 示例](../examples/README.md)、[execution](../circuit_harness/execution/README.md)、
[harbor](../circuit_harness/harbor/README.md)与 [data](../circuit_harness/data/README.md)。
开发按[贡献指南](chips/CONTRIBUTING.md)、[SOP](chips/DEVELOPMENT_SOP.md)与[测试入口](../tests/chips/README.md)进行。
术语见[词汇表](../GLOSSARY.md)，职责与发布边界见[仓库范围](chips/REPOSITORY_SCOPE.md)。
