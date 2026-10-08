# circuit-harness

Circuit Harness 是面向 AI Agent 电路设计与评测的独立实验平台。它连接仿真器和电路工具，管理公开任务会话、候选冻结、独立评分与轨迹数据。Harbor 负责 Agent、Job 和 Trial 的运行与评测调度。

Agent 与模型分别配置，便于比较不同组合。runner 可部署在本机或服务器，仿真器通过本地或 SSH 接口接入。

```text
Harbor task + Agent/Model 配置
  → Agent 调用公开电路工具 → EVAS / 公开测量 → 修改候选
  → 会话结束、冻结候选 → 私有 verifier → 评分与证据
  → ATIF 导出 → 用户选择数据 → 外部训练器
```

Spectre 可用于独立终评，开源 checker 也可按任务配置选择。公开反馈与隐藏评分材料隔离；运行成功、结果有效和电路达标分别记录。任务内容与判据归 benchmark，EVAS 算法归 vaEVAS。

## 开始使用

完整平台要求 Python 3.12+，Harbor 固定为 0.23.0：

```bash
python -m pip install -e '.[harbor,data]'
circuit-harness --help
python -m circuit_harness.harbor.profiles --help
```

根据 [Harbor 配置说明](docs/chips/HARBOR.md)、[任务接入](docs/chips/HARBOR_TASKS.md)和[部署步骤](docs/chips/HARBOR_DEPLOYMENT.md)填写私有配置。模板中的路径、模型与镜像摘要都是占位值，不是可直接运行的实验。模型凭据、许可证和 benchmark 私有材料由操作者配置。

| 需要 | 入口 |
| --- | --- |
| Pi / Codex 等 Agent 与模型组合、Job 编译 | [Harbor](docs/chips/HARBOR.md)、[配置模板](examples/chips/harbor/README.md) |
| 当前 EVAS 公开会话 | [会话合同](docs/chips/CURRENT_EVAS_PUBLIC_SESSION.md) |
| 冻结候选、Spectre / 开源 checker、重放 | [独立终评](docs/chips/BENCHMARK_EVALUATION.md) |
| 直接运行固定 VABench、Analog、ngspice、Spectre、EMX | [执行模块](circuit_harness/execution/README.md)、[示例](examples/chips/README.md) |
| Harbor 结果与轨迹 | [结果报告](docs/chips/HARBOR_RESULTS.md)、[ATIF 数据](circuit_harness/data/README.md) |
| 阅读旧实验归档 | [离线 Episode 报告](docs/chips/EPISODE_REPORT.md) |

## 代码边界

- `circuit_harness/execution/` 管理仿真、进程、SSH、公开会话、冻结与归档。
- `circuit_harness/harbor/` 对接 Harbor，编译配置、控制公开入口、执行私有 verifier、导出结果与 ATIF。
- `circuit_harness/data/` 准备训练数据并验证 token 和 loss mask。训练器在项目外运行。
- `circuit_harness/reporting/` 离线读取保存的实验记录。
- `circuit_harness/cli.py` 提供直接操作者命令，`public_mcp.py` 提供原生 Agent 的公开会话桥。

Python 包名为 `circuit_harness`，命令行入口为 `circuit-harness`。固定 VABench r53 和 Analog 提供直接会话及评分接口；通过 Harbor 运行时，还需要准备对应的 task。旧版本调用方与任务模板的适配见[迁移说明](docs/chips/MIGRATION.md)。

## 开发与证据

默认分支为 `main`。开发按 [AGENTS.md](AGENTS.md) 和[开发 SOP](docs/chips/DEVELOPMENT_SOP.md)进行，默认交付可 review 的 PR，确认后合并。[测试入口](tests/chips/README.md)说明本地检查；[历史验证记录](docs/chips/VALIDATION.md)只证明当时版本与条件，不能当作新部署的验收。

公开仓库保存通用源码、合成测试和脱敏模板。真实轨迹、模型密钥、机器配置、PDK 和隐藏评分材料留在私有存储，见[发布边界](docs/chips/REPOSITORY_SCOPE.md#publication-boundary)。

## 许可证与历史来源

本项目采用 Apache License 2.0，见 [LICENSE](LICENSE)。部分代码源自 AlphaApollo，来源归属见 [Notice.txt](Notice.txt)。当前版本独立维护，安装和运行均不依赖 Apollo，也不提供 Apollo 集成。
