# 后续验收

按[接入指南](../guides/integration.md#2-选择任务执行方式)分别验收原生 Harbor 与 Harness 公开会话路径。
已有完整任务优先保留原环境与 checker；公开 EVAS → 冻结候选 → 独立 verifier 验收受控仿真与私有终评能力。

- 调用方迁移到 `circuit_harness` 后，按自己的真实任务重新检查源码身份、工具协议和评分关联。
- 新原生 Harbor 任务保留自身环境、候选接口与 verifier。需要公开会话的任务使用 [task bindings](HARBOR_TASKS.md)，固定公开材料与私有 checker。VABench/Analog 的旧直接 runner 配置按实际需求迁移并单独验收。
- 本机原生 Codex、服务器 Pi/GLM、Docker/Podman 等组合分别验证。已有条件成功不能认证其他条件。
- 任务集实验保留全部计划尝试、失败和预算。公开会话任务使用 [Harbor 结果报告](HARBOR_RESULTS.md)区分基础设施失败与有效零分；原生任务保留 Harbor 自身结果和原 verifier 输出。
- 补齐原生 Harbor 任务的统一报告与正式轨迹导出，关联任务版本、最终候选、checker、成绩与原生轨迹。复用 Harbor 已有结果，保留两条路径各自的证据与隔离条件，不强制原生任务接入公开会话。
- [ATIF/SFT](../../circuit_harness/data/README.md)继续验证外部训练器加载、真实模型 tokenizer 和训练效果。格式检查不证明学习质量。

实际工作沿用已确认的需求和证据，默认交付可 review 的 PR。未获得对应主机、模型和预算授权时，不启动新的真实实验。历史结果及其限制见 [VALIDATION](VALIDATION.md)。
