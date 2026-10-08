# 后续验收

以 [Harbor](HARBOR.md) 的公开 EVAS → 冻结候选 → 独立 verifier 链路作为完整流程验收。

- 调用方迁移到 `circuit_harness` 后，按自己的真实任务重新检查源码身份、工具协议和评分关联。
- 新任务使用 [task bindings](HARBOR_TASKS.md)，固定公开材料与私有 checker。VABench/Analog 的旧直接 runner 配置需转换和单独验收。
- 本机原生 Codex、服务器 Pi/GLM、Docker/Podman 等组合分别验证。已有条件成功不能认证其他条件。
- 任务集实验保留全部计划尝试、失败和预算，使用 [Harbor 结果报告](HARBOR_RESULTS.md)区分基础设施失败与有效零分。
- [ATIF/SFT](../../circuit_harness/data/README.md)继续验证外部训练器加载、真实模型 tokenizer 和训练效果。格式检查不证明学习质量。

实际工作沿用已确认的需求和证据，默认交付可 review 的 PR。未获得对应主机、模型和预算授权时，不启动新的真实实验。历史结果及其限制见 [VALIDATION](VALIDATION.md)。
