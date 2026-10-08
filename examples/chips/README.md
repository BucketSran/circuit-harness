# Chips examples: choose a task or direct simulator path

当前最完整的 Agent 示例是 [VABench `v4-001` 双路径运行](benchmarks/vabench/README.md)：
服务器同机 Pi＋GLM，以及本机原生 Codex＋GPT 经 SSH 调用服务器 EVAS。
两者都保存轨迹、冻结候选、独立终评和归档；已实测单题，尚未验收整个任务集或跨账号复现。
其他入口见 [benchmark 目录](benchmarks/README.md)、
[ngspice RC](../../docs/chips/NGSPICE.md)、[Spectre RC](../../docs/chips/SPECTRE_RC.md)。

| 目标 | 从这里开始 | 运行入口 |
| --- | --- | --- |
| 单次可评分 Agent 实验 | [VABench](benchmarks/vabench/README.md) 或 [Analog RLC](benchmarks/analog_design_bench/TASK.md) | `python -m alphaapollo.workflows.chips_experiment`；配置与终评见[实验协议](../../docs/chips/UNIFIED_EXPERIMENT.md) |
| 直接运行、恢复或检查服务器仿真 | [EMX](emx/README.md)、[ngspice](ngspice/)、[Cadence](cadence/README.md) | `python -m alphaapollo.workflows.chips`；它是服务器离线包的命令入口，不负责 Agent 实验编排 |
| 离线复盘一次实验 | [Episode 报告](../../docs/chips/EPISODE_REPORT.md) | `python -m alphaapollo.workflows.chips_episode_report`；只读现有证据 |
| 从材料形成任务草案 | [任务建模](task_authoring/README.md) | 独立试点，不自动纳入可评分 benchmark |

`emx/config.yaml`、`emx/workflow.yaml` 和 `emx/tasks.jsonl` 仅是 EMX smoke 配方，
不是整个 Chips 平台的默认任务、数据集或配置。旧的根目录示例路径已迁至 `emx/`；
私有配置中的 converter 路径也应随之更新。

已有 VABench/Analog 实验可离线查看，命令不会启动模型或仿真：

```bash
python -m alphaapollo.workflows.chips_episode_report /absolute/private/experiment \
  --output /absolute/private/reports/episode-001
```

支持的目录、计时含义及证据限制见[Episode 报告](../../docs/chips/EPISODE_REPORT.md)。

定义新任务和工具的落点见[协作指南](../../docs/chips/CONTRIBUTING.md)，当前验收边界见[验证记录](../../docs/chips/VALIDATION.md)。

Icarus Verilog 可用于本地执行生命周期 smoke，不代表实验室后端：

```bash
python -m alphaapollo.workflows.chips rtl-smoke \
  --design examples/chips/rtl/design.sv --testbench examples/chips/rtl/tb.sv \
  --output runs/chips/rtl-smoke
```
