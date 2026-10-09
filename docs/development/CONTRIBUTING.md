# 参与开发

先从 [README](../../README.md) 找到所属能力，并沿用已确认的需求。实现与交付遵守 [开发 SOP](DEVELOPMENT_SOP.md)，默认准备可 review 的 PR。

| 修改内容 | 所属位置 |
| --- | --- |
| 仿真执行、SSH、公开会话、冻结与归档 | [execution](../../circuit_harness/execution/README.md) |
| Agent/Model 配置、任务绑定、容器和 verifier | [harbor](../../circuit_harness/harbor/README.md) |
| 轨迹筛选、SFT 数据与 loss mask | [data](../../circuit_harness/data/README.md) |
| 读取旧实验记录 | [Episode 报告](../guides/EPISODE_REPORT.md) |
| 任务题面、参考设计和判据 | benchmark 所属项目，Harness 只提供执行集成 |

电路专家提供单位、端口、可修改范围、参考案例和判定规则。实现者先完成直接执行和独立检查，再定义 Agent 可见反馈；新增工具使用 [工具规格](TOOL_TEMPLATE.md)。测试入口见 [tests/chips](../../tests/chips/README.md)。

公开贡献只包含通用代码、安全夹具与脱敏模板，材料范围见 [发布边界](REPOSITORY_SCOPE.md#publication-boundary)。
