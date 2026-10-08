# 电路专家与 Agent 的 Chips 协作入口

本项目首先建立可复现的 IC 仿真实验平台，在此基础上研究任务、Tools、Agent 与 Memory。
电路专家负责确认物理意义、有效范围和判定规则；协作 Agent 负责将其落成任务文件、代码和检查。
[路线图](ROADMAP.md) 是研究方向，[后续工作与验收计划](NEXT_WORK.md) 是候选工作索引；
[第一阶段状态](PHASE1_HARNESS.md) 与 [验证记录](VALIDATION.md) 是已完成工作的证据。
服务器目录、多人访问边界、凭据和模型网络按 [部署规范](DEPLOYMENT.md) 约定；
VABench 已有专用公开环境预检，统一跨后端 preflight 尚待实现。
开发阶段由 [harness-workflow](../../.agents/skills/harness-workflow/SKILL.md) 选择，
已知组件任务可直接使用项目 skill。共享 Matt、pstack 方法与本地约束的分工见
[开发 SOP](DEVELOPMENT_SOP.md#development-workflow)；[测试入口](../../tests/chips/README.md)
集中列出长期保留的回归、探针、案例和记录模板。
草稿、讨论、正式文档与原始证据按 SOP 的[文档入库规则](DEVELOPMENT_SOP.md#documentation-policy)分别管理。
近期真实实验优先采用[本机 Codex＋SSH 主线](DEVELOPMENT_SOP.md#current-experiment-path)；
旧路线／计划不能替代本次任务范围和运行条件。

## 1. 取得正确版本

```bash
git clone --branch main https://github.com/BucketSran/circuit-harness.git
cd circuit-harness
git status --short --branch
git remote -v
```

公开源码无需私有仓库权限。已有工作目录先检查未提交修改，不用覆盖/重置方式同步。
开发与 PR 以 `BucketSran/circuit-harness` 的 `main` 为目标。没有写权限时使用自己的 fork；
核对 `origin` 的实际归属，再从最新 `main` 建立功能分支。旧私有仓库不能直接改 remote 后推送到公开仓库。
旧参考分支已从远端删除；参考历史保存在私有备份，见[仓库范围](REPOSITORY_SCOPE.md)。
不向原 AlphaApollo 上游推送或创建 PR。公开源码不授予实验室账号、商业许可证或受限数据的访问权限。

## 2. 具体修改位置

“已有”指文件或模块存在，不等于已经完成真实实验室验收；“拟新增”指尚未实现。

| 想完善的内容 | 代码/资料位置 | 状态与边界 |
| --- | --- | --- |
| 任务说明、约束、数据来源、参考结果和验收规则 | [examples/chips/benchmarks/](../../examples/chips/benchmarks/README.md) | 已登记 RC、VABench 与 Analog 任务；非结构化材料接入与通用 loader 待实现 |
| 几何 JSON 与 GDS 转换 | [json_to_gds.py](../../examples/chips/emx/json_to_gds.py)、[layout.json](../../examples/chips/emx/layout.json) | 已有最小多边形/标签示例；真实工艺映射与参数化设计待补 |
| 实验室连接、EMX wrapper 和结果文件约定 | [emx.config.example.json](../../examples/chips/emx/emx.config.example.json)、[EMX 操作说明](../../examples/chips/emx/README.md) | 共享无敏感信息的模板；每人的实际设置写在忽略的 `examples/chips/emx/*.local.json` |
| 转换、传输、作业状态、恢复下载 | [emx.py](../../alphaapollo/common/execution/chips/emx.py) | 已有执行器；支持直接调用，不依赖模型作决策 |
| VABench／Analog／Gain 公开动作的同机或 SSH 传输 | [session_transport.py](../../alphaapollo/common/execution/chips/session_transport.py) | 共用动作 ID、重试、恢复和证据写入；每项任务在自己的适配层定义命令名与模拟／评分语义 |
| 远端启动/取消/日志收集、进程超时 | [ssh_worker.py](../../alphaapollo/common/execution/chips/ssh_worker.py)、[process.py](../../alphaapollo/common/execution/chips/process.py) | 已有；实验室 worker 只用 Python 标准库 |
| 模型可见的 Tool 名称、参数说明和反馈 | [tools/chips.py](../../alphaapollo/common/execution/tools/chips.py) | 已有 `emx_simulate`；新工具先填写 [工具规格](TOOL_TEMPLATE.md) |
| EMX Agent 角色、提示和调用步骤 | [workflow.yaml](../../examples/chips/emx/workflow.yaml) | 已有单步示例；这里改流程，不把流程策略塞进 EMX 执行器 |
| EMX Agent / 模型 / 超时等运行配置 | [config.yaml](../../examples/chips/emx/config.yaml) | 此示例目前选 Codex；VABench 已有独立 Pi 入口，EMX 示例的 Pi 配方仍待验收 |
| 阶段事件、CLI 状态与报告 | [journal.py](../../alphaapollo/common/execution/chips/journal.py)、[workflows/chips.py](../../alphaapollo/workflows/chips.py) | 已有；跨 Agent→MCP→job 的关联尚待真实运行验证 |
| 单次 Agent 实验的启动、续收、终评和归档 | [chips_experiment.py](../../alphaapollo/workflows/chips_experiment.py)、[内部任务策略](../../alphaapollo/workflows/_chips_experiment/tasks.py) | 公开 CLI 负责实验生命周期；任务策略决定 VABench／Analog 的具体动作 |
| 已保存轨迹的离线复盘 | [chips_episode_report.py](../../alphaapollo/workflows/chips_episode_report.py)、[证据关联](../../alphaapollo/workflows/_chips_episode_report/evidence.py) | 只读归档，不启动模型、SSH 或仿真 |
| 数据规范化与 public/private manifest | `alphaapollo/data_preprocess/chips/` | 拟新增；复用现有数据准备机制 |
| 结果解析、有效性检查与电路评分 | [执行模块地图](../../alphaapollo/common/execution/chips/README.md)；未来通用 Grader 可落在 `alphaapollo/common/grader/chips/` | 已有 RC、Spectre、VABench 和 Analog 的任务专用检查；任意电路/工艺的通用评分尚未实现 |
| 任务内 memory 与跨任务经验检索 | `alphaapollo/workflows/memory/chips.py`；复用 `alphaapollo/evolving/memory/` | Chips adapter 拟新增；跨任务策略不是现成开关 |
| 行为测试 | [test_chips_harness.py](../../tests/common/execution/test_chips_harness.py)、[test_chips.py](../../tests/workflows/test_chips.py) | 已有；任务夹具/数据准备/评分测试随对应模块增加 |

如果只是增加电路案例或修改一个工具的参数/算法，通常无需动通用 runtime。
若需要注册新 Tool，再联动 `workflows/resources.py`、`workflows/_resources/runtime.py` 和
`reasoning/runtime/external/bridge/mcp_server.py` 的既有接线，并做受影响检查。
将来迁移到专属电路 Agent 时，尽量保留领域实现与 Tool 契约，只替换 runtime/传输适配。

## 3. 专家与 Agent 怎样交接

1. 专家先给一个实际工作案例：输入、手动步骤、输出、判定方式，以及允许变化的设计条件。
2. Agent 用 [任务卡](../../examples/chips/benchmarks/TASK_TEMPLATE.md) 整理，所有未知项标为待确认；
   不能自行猜测工艺层号、端口、单位、目标阈值或 EMX 选项。
3. 专家确认任务卡及结果解释后，Agent 在上表对应路径实现最小闭环与必要测试。
4. 先直接调用工具重现专家结果，再接模型。专家查看输入、输出、指标和失败判定，无需先审阅全部代码。
5. 交付可供 review 的 PR、复现步骤、检查结果及未验证项；只在获准环境做实测。
   默认用户确认后合并；明确授权自主合并时直接完成整合，详见 [PR 交付规则](DEVELOPMENT_SOP.md#delivery-and-review)。

一个任务或 Tool 的第一次贡献可以只有完整规格与真实样例；明确标为未接入即可。
成功案例、设计不达标案例、执行故障案例分别保存和判断。模型总结不代替独立的验收规则。

## 4. 可以直接发给协作 Agent 的提示

```text
我们在公开仓库 BucketSran/circuit-harness 上开发，以 main 为基线与集成目标。
先读取根目录 AGENTS.md、当前 Issue/PR 和所属组件入口，按任务需要加载 harness-workflow 或组件 skill。
采用开发 SOP 的文档、验收和 PR 交付规则；沿用已确定的范围、测试入口和授权。
检查当前分支、remote 和未提交修改；保护已有工作和私有参考备份。不向 AlphaApollo 上游推送。

我要贡献的电路任务或工具是：[填写目标]。
可用材料和本机路径是：[填写案例、输入、结果及专家说明]。
此次工作范围是：[仅整理规格 / 实现并做本地测试 / 在明确给出的实验室环境验收]。

先确定任务拥有者。vaEVAS benchmark 的题目、参考答案与评分仍由该组件维护；本仓保留既有任务入口。
本仓新增直接工具案例时，按 examples/chips/benchmarks/TASK_TEMPLATE.md 整理必要的专家知识。
Tool 规范依据 docs/chips/TOOL_TEMPLATE.md，实现按协作指南的职责表落点。
缺失的电路规则和验收阈值明确列出并询问，不编造。
保持直接工具入口，复用现有 runtime、MCP 和执行框架，不重写通用 Agent。
原始运行产物留在 runs/ 或指定本地目录，任务目录保存来源和可共享的精简夹具。
未定稿计划和临时清单留在忽略的 .planning/chips/；正式内容优先更新现有文档，不把所有规划产物入库。
当前真实实验优先用本机原生 Codex＋SSH 调服务器仿真；已有方案不代表授权开跑或其他接法已验收。
只运行本改动相关检查，不启动 Bio/Robotics/Math 的领域评测。
完成后按范围交付：讨论给出决定；开发给出经过检查的 PR；只读审查给出发现。
PR 正文使用共享 pr skill，区分本地替身、真实软件、真实模型及未验收条件。
```

以上提示的方括号内容由贡献者填写；实验室账号、许可与受限资料由各成员按所属环境配置。
本机接入步骤见 [Chips 操作说明](../../examples/chips/README.md)。
