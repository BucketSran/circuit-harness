# 固定 VABench 后端

原始 r53 / EVAS 0.8.7 复现路径保留，直接操作者命令见 [VABench](../../../../docs/chips/VABENCH.md) 和 `circuit-harness --help`。任务卡见 [TASK.md](TASK.md)，固定后端依赖见 [lock 文件](requirements-linux-py312.lock)。

旧 Pi/Codex runner 配置已退役。新 Agent 实验采用 [Harbor task](../../../../docs/chips/HARBOR_TASKS.md)；旧 VABench session 尚不是开箱即用的 Harbor task，需任务方转换和验收，见[迁移说明](../../../../docs/chips/MIGRATION.md)。
