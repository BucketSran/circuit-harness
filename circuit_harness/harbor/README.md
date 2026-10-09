# Harbor 电路环境与配置组合

可选集成要求 Python 3.12 或更新版本，以及 Harbor 0.23.0。
`profiles` 把独立的 Agent 设置和 Model 协议连接编译为 Harbor JobConfig。
原生 Harbor 任务保留自身环境与 verifier；下面的会话插件用于需要受控仿真和候选冻结的任务。
`CircuitDockerEnvironment` 和 `CircuitPodmanEnvironment` 提供公开电路工具，`CircuitAgent` 委托 stock Harbor Agent，
并在 Agent 阶段结束时关闭公开动作入口、冻结候选。
`FrozenCandidateVerifier` 使用独立私有配置评价已冻结候选。

协议支持、部署步骤与证据边界以 [Harbor 文档](../../docs/reference/HARBOR.md) 为准。
首次实验见 [AnalogBench 快速开始](../../examples/analogbench/README.md)，
接入自己的任务见[开发者指南](../../docs/guides/integration.md)。
配置与任务模板见 [examples/chips/harbor](../../examples/chips/harbor/README.md)。
相邻 `profiles.schema.json` 描述 Agent 与 Model 目录。
`config.schema.json` 保留可选 `NativeCodexAgent` 主机原生路径的兼容配置。
Harbor 管理 Trial 与 Agent loop；这些模块不创建另一个评测控制器。
