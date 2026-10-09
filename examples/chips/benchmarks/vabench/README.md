# 固定 VABench 后端

原始 r53 / EVAS 0.8.7 复现路径保留，直接操作者命令见 [VABench](../../../../docs/chips/VABENCH.md) 和 `circuit-harness --help`。任务卡见 [TASK.md](TASK.md)，固定后端依赖见 [lock 文件](requirements-linux-py312.lock)。

这里的 VABench session 是操作者协议，不是开箱即用的 Harbor task。新的 Agent 实验按[接入指南](../../../../docs/guides/integration.md)准备任务、公开材料和 verifier；需要公开会话时使用 [task bindings](../../../../docs/chips/HARBOR_TASKS.md)，并独立验收。
