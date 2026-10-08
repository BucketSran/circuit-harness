# 实验条件与数据流

Harbor 负责 Agent/Job/Trial，Harness 提供电路环境、公开反馈、候选冻结和独立 verifier。Agent 与 Model 分别配置，支持的协议由 [profiles](HARBOR.md) 编译器验证。

```text
操作者固定 task、Agent/Model、预算与后端
  → Harbor 创建 Trial → Agent 调用 circuit-public
  → 公开 EVAS / 测量 → 反馈与候选修改
  → 关闭公开动作入口 → 冻结最后完整候选
  → 私有 Spectre / 开源 checker → 校验评分证据
  → 结果报告 / ATIF 导出
```

runner 可在本机或服务器，独立终评按部署配置连接。服务器使用 Pi/GLM 不要求安装 Codex。原生 Codex 路径通过受控 MCP 暴露任务会话，保留其原生工具隔离要求。具体网络、容器、目录与许可证条件见 [Harbor 部署](HARBOR_DEPLOYMENT.md)。

## 开跑前固定什么

记录 task/checker 版本、公开材料、可修改范围、Agent 与 Model 协议、工具和总预算、EVAS 源码与 kernel 摘要、终评后端、镜像及运行位置。不同部署、后端或提示词属于不同实验条件，不自动合并比较。

凭据留在私有环境或配置中，不能进入 task 或 Agent 可读目录。公开会话目录与 verifier 私有材料分离。先完成对应后端的环境检查与任务正负控，再将真实 Agent 结果用于分析；预检成功不等于模型认证或电路成功。

## 状态与证据

run、action、job ID 和候选摘要关联请求、响应、冻结与终评。断线或超时后先查询原作业；未知状态不能当作可以重新执行的许可。评分采用冻结的实际提交，不自动挑选历史最佳版本。独立评分未执行或基础设施错误时保持未评分，不能填成零分。

轨迹仅包含 Agent 实际可见的内容。私有 checker、参考答案与终评分数作为结果证据关联，不能混入训练对话。ATIF 的重建上下文、未评分记录和 reasoning 选择见 [数据合同](../../circuit_harness/data/README.md)。

旧 Apollo 两条 Agent 链路已退役；固定后端和旧归档可按[迁移说明](MIGRATION.md)继续使用。真实历史验收见 [VALIDATION](VALIDATION.md)，新任务与部署需重新留证。
