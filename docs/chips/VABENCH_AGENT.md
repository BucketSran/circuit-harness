# VABench Agent 入口迁移

Apollo 专用 Agent/实验 runner 已退役，详见[迁移说明](MIGRATION.md)。原指南与实现保存在迁移前的 Git 提交中，旧运行记录不代表新入口已完成同样验收。

新 Agent 实验使用 [Harbor](HARBOR.md)、[部署步骤](HARBOR_DEPLOYMENT.md)和[任务接入](HARBOR_TASKS.md)。固定 VABench 与 Analog 的直接会话、冻结、评分和归档仍由[操作者执行层](../../circuit_harness/execution/README.md)提供；旧配置不是可直接运行的 Harbor task。
