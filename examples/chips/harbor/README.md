# Harbor 配置与任务模板

这里提供可编辑的配置结构。模型名、endpoint、私有路径和零值镜像摘要都是占位内容，
不表示真实服务或电路任务已经验证。
协议支持与完整操作说明以 [Harbor 文档](../../../docs/chips/HARBOR.md) 为准。

- `profiles.example.json` 分别声明 Agent 设置和 Model 协议连接。
- `job.example.json` 声明 Harbor task、公开电路环境和私有 verifier。
- `public-session.example.json` 声明公开会话、材料、后端与预算。
  示例 manifest 仅展示字段形状；替换为任务实际的固定实验条件。
- `final-evaluation.example.json` 声明独立终评 package 和远端配置。
- `task.example/` 提供单步 Harbor task 结构与公开工具用法。

把 task 复制到实际公开任务目录。把编辑后的 Job、公开会话配置和终评配置保存到私有操作目录，
并更新 Job 中的绝对路径。私有操作目录必须位于 task 和 Trial 导出目录之外。
`task.example/tests/test.sh` 只满足 Harbor 的任务文件检查；它故意失败，
以免误用默认 verifier 时产生成功结果。电路评分由 Job 指定的 `FrozenCandidateVerifier` 完成。

编译与启动步骤见 [Harbor 操作说明](../../../docs/chips/HARBOR.md#编译-jobconfig-并交给-harbor)。
生成的输出文件必须尚不存在。密钥通过 `key_env` 指定的环境变量供 Harbor 在运行时解析。
根据 Agent 容器实际网络选择 `gateway_host`；宿主机别名可能无法在容器内解析。

多任务配置见 [task-bindings 示例](task-bindings.example.json) 和
[接入步骤](../../../docs/chips/HARBOR_TASKS.md)。示例是构造任务路径，必须由操作者替换，
不表示真实 vaEVAS 七题都已支持当前 backend。
