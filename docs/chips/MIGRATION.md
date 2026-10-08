# 迁移到独立 Circuit Harness

本次是有意的包名与入口变更。Python distribution 改为 `circuit-harness`，Python 包改为 `circuit_harness`，首个独立版本为 `0.1.0`。完整安装要求 Python 3.12+。不再提供 `alphaapollo` 导入别名，也不再加载 Apollo。

## 更新调用方

| 原入口 | 新入口 |
| --- | --- |
| `alphaapollo.common.execution.chips.*` | `circuit_harness.execution.*` |
| `alphaapollo.workflows.harbor_chips.*` | `circuit_harness.harbor.*` |
| `python -m alphaapollo.workflows.chips` | `circuit-harness` 或 `python -m circuit_harness` |
| `alphaapollo.workflows.chips_public_mcp` | `circuit_harness.public_mcp` |
| `alphaapollo.workflows.chips_task_authoring` | `circuit_harness.task_authoring` |
| `alphaapollo.workflows.chips_episode_report` | `circuit_harness.reporting.episode` |
| `alphaapollo.data_preprocess.prepare_atif` | `circuit_harness.data.prepare_atif` |
| `alphaapollo.data_preprocess.atif_dataset` | `circuit_harness.data.atif_dataset` |
| `alphaapollo.data_preprocess.validate_atif` | `circuit_harness.data.validate_atif` |

优先使用新的 checkout 与虚拟环境，安装 `.[harbor,data]`。若在旧目录构建 wheel，先移走旧的生成目录 `build/` 和 `*.egg-info`，再运行 wheel smoke；setuptools 可能把旧构建缓存混入新包。同时更新 Python imports、`python -m` 命令、Harbor JobConfig 的 `import_path` 和检查源码位置的相对路径。新模板已使用新名称。不要只改包安装名而继续使用旧 Job JSON。

vaEVAS 等调用方若仍固定在旧 checkout，可继续按原版本运行。切换到本版本前，须按上表更新其执行模块、Harbor 插件和源码位置校验，并复跑调用方验收。本仓库不替其他项目切换 checkout 或重写其配置。

重新生成和部署 CLI zipapp，同时记录新摘要。源文件路径和 Harness 源码摘要会变化；候选摘要、任务协议、评分结果与归档格式保持原有语义。进行中的会话继续使用创建它的版本，完成归档后再切换；不要覆盖旧 bundle 或改写历史证据的路径/哈希。

## 退役的能力

Apollo 的通用 runtime、Workflow YAML 引擎、Robotics、evolving/Memory、PPO/DPO/Megatron/verl 训练集成不再随包提供。旧 `chips_*_agent`、`chips_pi_runtime`、`chips_experiment`、`chips_evaluate` 与 `chips_vabench_deployment` 启动器及其配置退役。

新的自动 Agent 实验使用 [Harbor](HARBOR.md)。旧 VABench/Analog runner 配置不是 Harbor JobConfig，不能直接改名字继续运行。这些 benchmark 的固定操作者会话、仿真、冻结、评分与归档仍在 `circuit-harness --help` 中；完整迁入 Harbor 需要任务方提供 [task bindings](HARBOR_TASKS.md) 与公开/私有材料，并单独验收。EMX 的直接操作者 CLI 保留，Apollo `emx_simulate` 工具桥退役。图文任务的确认与有界构建命令保留，Apollo Codex 两阶段自动 runner 退役。

旧 Pi/Codex 事件与 Episode 归档仍可用离线报告读取。训练数据继续通过 ATIF 导出，`AtifSFTDataset` 可由外部训练器消费。验证器直接调用该 PyTorch Dataset，不依赖 verl；本仓库不再固定训练器版本或启动训练。

历史实现可在本公开仓库迁移前的 `190719c2f237ece7008c63918a1a523c9bbe8ce1` 查阅。真实旧运行仍需其原有私有材料和部署环境；检出历史源码不等于恢复运行条件。
