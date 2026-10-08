# circuit-harness

基于 [AlphaApollo v3](https://github.com/AndrewZhou924/AlphaApollo-v3-dev) 建设的电路实验平台。我们先把任务、仿真、公开工具、独立评分和运行轨迹接成可复核的闭环，再在相同任务与预算下比较 Tools、Agent、模型和经验库。**运行成功、结果有效、电路达标是三个不同的判断。**

本仓库的默认开发分支是 `main`。它保留 Chips、Robotics 参考适配和 Apollo 共享内核；Bio、Math 及其他任务专用代码已从当前版本移除，详见[仓库范围与发布边界](docs/chips/REPOSITORY_SCOPE.md)。Python 包名仍为 `alphaapollo`，现有导入、CLI 和配置继续兼容。

公开仓库从审查后的源码快照开始，完整研发历史、旧 Issue/PR 与真实实验档案保留在私有存储。以下验证记录中的 `lab-server` 是匿名部署标识，不是可直接连接的主机。

开发从 [AGENTS.md](AGENTS.md) 和 [harness-workflow](.agents/skills/harness-workflow/SKILL.md) 进入，
按需使用共享 Matt、pstack skills。默认一个可独立验收的 ticket 对应一个 PR，完成检查与审查后交给用户 review；
确认后再合并并同步日常目录。具体约定见[开发 SOP](docs/chips/DEVELOPMENT_SOP.md#development-workflow)，
产品目标与后续验收见[路线图](docs/chips/ROADMAP.md)和[后续工作](docs/chips/NEXT_WORK.md)。

## 现在能做什么

| 任务或链路 | 执行后端 | 已有证据 | 当前边界 |
| --- | --- | --- | --- |
| [VABench r53](docs/chips/VABENCH_AGENT.md) | lab-server 上固定 EVAS 0.8.7 与 Bubblewrap | 三任务正负控；`v4-001` 已分别完成服务器 Pi＋GLM、本机 Codex＋GPT 经 SSH 的真实单题回合、冻结、原评分器终评和归档 | 不是完整任务集的 Agent 成功率，也不是两种 Agent 的公平排名 |
| [Analog Design Bench](docs/chips/ANALOG_DESIGN_BENCH.md) | lab-server 上固定源码、Podman 镜像与 ngspice | 三题参考解/starter 正反控；两道 RLC 共用 Tool 与 Pi＋GLM 入口，宽带题实测自主仿真、恢复、提交、独立终评及两端归档核验（6/7，0.9） | 宽带电路仍缺一项指标；未形成稳定成功率或本机 Codex 条件 |
| [图文任务构建预试](examples/chips/task_authoring/README.md) | 本机 Codex＋MCP/SSH → lab-server Spectre | 构造图文提取、确认、有界测试台构建、公开仿真、四个独立参考实例及冻结包两次重放 | 单个 D0 开发回合；未验收通用文档、真实专家确认或方法比较 |
| [Spectre RC](docs/chips/SPECTRE_RC.md) | lab-server 上无 PDK 理想 RC、后台 Spectre | 真实仿真、PSF 解析、独立 RC 检查与归档 | 尚无晶体管/PDK 设计任务的 Agent 验收 |
| [EMX JSON→GDS](examples/chips/README.md) | 本地 gdstk、SSH、实验室 EMX | 厂商样例的生成、提交、下载、同一作业恢复与日志校验 | 几何示例不是工艺合格布局；`verdict=not_evaluated` |
| [ngspice RC](docs/chips/NGSPICE.md) | lab-server 上理想 RC 后台作业 | AC/瞬态独立检查、SSH 断线后继续、归档与下载校验 | 执行基线，不等于完整电路设计 benchmark |

上述是真实运行与[验证记录](docs/chips/VALIDATION.md)中对应条件的结论；CI 使用本地回归与构造边界，不会自动调用实验室仿真器或模型。一次单题通过不能推断任务集成功率。

## 一次 Agent 实验怎样运行

~~~text
任务与私有配置 → Agent（服务器 Pi 或本机 Pi / Codex）
             → 公开 MCP Tools → 服务器任务会话 → 固定仿真器 → 公开反馈
             → 候选冻结 → 操作者侧独立终评 → 轨迹、评分与归档校验
~~~

Agent 能看到本次任务材料、自己的对话和公开 Tool 反馈；最终评分器与隐藏答案不提供给 Agent。服务器同机模式把模型请求和仿真放在 lab-server；本机模式把模型请求留在本机，通过 SSH 调用服务器上的公开会话。两种方式都保存 Agent 事件、Tool 请求/响应与服务器作业证据；本机模式的两端证据仍需按 run ID 和哈希核对。[调用链与开跑约定](docs/chips/EXPERIMENT_PROTOCOL.md)说明凭据、预算、推理档位、断线和恢复边界。

[共同单次实验入口](docs/chips/UNIFIED_EXPERIMENT.md)目前支持 **VABench 或 Analog 的一题 × 一次**：私有实验 JSON 引用该任务的 operator JSON，记录 `experiment_manifest.json`、一行 `results.jsonl` 和 `summary.json`。仿真与评分仍由各任务实现；VABench 的 pass/fail 与 Analog 的连续 reward 不混成同一种成功率。Spectre 和 EMX 仍使用各自入口。

固定任务集可用同页的 `chips_evaluate` 准备批次、顺序启动单次实验，并按任务与模型条件汇总已有结果；异步终评继续由各任务入口完成。批次支持断点继续，但执行状态不明的 cell 需要先核对原会话。

已有实验可以通过[离线 Episode 报告](docs/chips/EPISODE_REPORT.md)查看保存的提示、Agent 事件、Tool 时间线、候选 diff 与终评关联。它读取已有证据，不重新运行实验；目前覆盖 VABench 与 Analog 的成功、修复和预算耗尽等记录。

## 从哪里开始

需要 Python 3.10+ 和与所选任务相符的服务器/仿真器权限。本地 macOS 与 CI Linux 有检查记录；实验室许可证、固定任务源码和模型凭据需在本人获准的环境配置。

~~~bash
git clone --branch main https://github.com/BucketSran/circuit-harness.git
cd circuit-harness
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[chips,mcp]'
python -m alphaapollo.workflows.chips_experiment --help
~~~

选 [VABench 示例](examples/chips/benchmarks/vabench/README.md)或 [Analog 任务卡](examples/chips/benchmarks/analog_design_bench/TASK.md)，先按任务文档部署固定版本并完成预检。把对应的 [VABench 实验模板](examples/chips/benchmarks/vabench/experiment.example.json)或 [Analog 实验模板](examples/chips/benchmarks/analog_design_bench/experiment.example.json)及 operator 模板复制到 Git 忽略的私有目录，填入本人的绝对路径、**新的** run ID、模型与预算；Key 不写进 JSON 或仓库。确认公开工具正负控后才启动真实模型回合：

~~~bash
python -m alphaapollo.workflows.chips_experiment run --config /absolute/private/experiment.json
~~~

VABench 的异步终评用同一配置 `collect`；Analog 在确认冻结提交后用同一配置 `finalize`，归档故障用 `archive` 恢复。命令、配置字段和状态含义见[单次实验说明](docs/chips/UNIFIED_EXPERIMENT.md)。只想检查工具链时，可按 [EMX 示例](examples/chips/README.md)、[ngspice RC](docs/chips/NGSPICE.md)或 [Spectre RC](docs/chips/SPECTRE_RC.md)直接调用，不把工具正负控算作 Agent 成绩。

## 在哪里修改与验收

| 内容 | 入口 |
| --- | --- |
| 任务规格、可调变量、参考案例 | [Chips benchmark 目录](examples/chips/benchmarks/README.md)与[任务卡模板](examples/chips/benchmarks/TASK_TEMPLATE.md) |
| 受限候选、仿真执行、后台作业和归档 | [Chips 执行层](alphaapollo/common/execution/chips/) |
| Agent 可见的 Tool 接口 | [Chips Tools](alphaapollo/common/execution/tools/chips.py)及各任务公开会话 |
| Pi / Codex、任务操作与单次实验编排 | [Chips Workflow 模块](alphaapollo/workflows/chips_experiment.py) |
| 开发顺序、测试与真实证据 | [开发 SOP](docs/chips/DEVELOPMENT_SOP.md)、[测试入口](tests/chips/README.md)、[验证记录](docs/chips/VALIDATION.md) |

电路专家可以先给出规格、允许的修改范围、手工仿真步骤与判定规则，再与协作 Agent 按[协作指南](docs/chips/CONTRIBUTING.md)落到任务、Tool、执行器和测试。新增 Tool 先能直接调用和独立检查，再暴露给 Agent；运行产物、工艺资料、模型轨迹与凭据保存在各自的私有环境。

接下来要完成跨账号复现、更多任务和重复实验、跨主机证据自动核验，以及可控的经验/Memory 对照。图文任务构建已有受限的合成材料预试和操作者确认入口；通用 PDF/Word 解析、真实专家澄清与方法比较仍待验收；详见[后续工作](docs/chips/NEXT_WORK.md)。

## 发布范围与许可

源码、通用 verifier、仿真器适配器、合成测试和脱敏模板随源码发布。真实轨迹、机器配置、凭据和受限工艺材料单独管理；外部 benchmark 的任务和隐藏评分材料不随 Harness 自动发布。具体规则和快照交付方式见[发布边界](docs/chips/REPOSITORY_SCOPE.md#publication-boundary)。

旧研发仓库保留为私有的 `circuit-harness-private`，它的提交历史、Issue/PR 和 Actions 记录没有迁入本仓库。文档中指向该仓库的历史讨论链接需要原有访问权限；公开使用与贡献请从本页和[协作指南](docs/chips/CONTRIBUTING.md)进入。

沿用 AlphaApollo v3 的 Apache License 2.0，见 [LICENSE](LICENSE) 和 [Notice.txt](Notice.txt)。
