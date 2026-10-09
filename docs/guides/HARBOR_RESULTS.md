# Harbor 离线实验报告

报告入口只读取保存的 Harbor 0.23 Job、Trial、冻结候选及独立终评凭据。
Harbor 继续负责执行和任务计划；导出器不创建 Job，不调用 Agent、SSH、Docker 或评分器。
[Harbor 配置](../reference/HARBOR.md)和[任务绑定](../reference/HARBOR_TASKS.md)仍是运行入口。

## 支持的任务路径

当前导出器面向 Harness 公开会话与独立终评路径，依赖候选冻结及终评凭据。
它读取原生 Harbor 的任务计划和 Trial 记录，但不会把任意原生任务的 reward
直接当作已经核验的 Harness 分数。

AnalogBench 等原生任务继续使用自身的 Trial 结果、原 verifier 输出和 Agent 日志。
原任务的独立评分不要求 Harness session；缺少 Harness 冻结证据也不表示原 checker
成绩无效。两条路径的选择见[接入指南](integration.md#2-选择任务执行方式)。
原生任务的统一结果关联仍是[后续工作](../development/NEXT_WORK.md)，不能补造冻结收据来套用此入口。

## 导出报告

```bash
python -m circuit_harness.harbor.reporting \
  --job /private/saved-harbor-job \
  --output /private/new-report-directory
```

输出为新目录内的 `report.json`。目录权限为 `0700`，文件权限为 `0600`。
已有输出目录及 Job 内的输出路径会被拒绝。来源不被修复、覆盖或更新。
报告包含私有任务路径，只保存在操作者私有目录；共享前须审查。

退出码 `0` 表示证据没有发现冲突，允许计划中有失败、缺失及未启动项。
退出码 `1` 表示发现 Trial 或 Job 证据损坏、重复或冲突，报告保留错误和无可信分数的记录。
退出码 `2` 表示参数、来源配置或计划无法解析，此时不生成报告。
缺少当前配置或所引用资源也会报错，不能把变化后的资源当成历史运行身份。

## 计划与统计

`schema_version=1` 的契约由
[`ExperimentReport`](../../circuit_harness/harbor/reporting.py)与相邻的
[`reporting.schema.json`](../../circuit_harness/harbor/reporting.schema.json)维护。
Python 入口 `read_job(path)` 返回同一记录模型。

导出器读取原生 `JobConfig.tasks`、`agents`、`n_attempts`，为每个任务与 Agent 配置保留
`n_attempts` 行。本地 dataset 只通过 Harbor 原生本地 resolver 展开；拒绝远程
Git、registry/package dataset、regrade `source_jobs`，不联网补齐任务。
它复用私有 task binding 的选择与摘要，未支持任务保留 `not_run` 与明确原因。
`attempt` 是按保存的 Trial 目录排序得到的离线行号，不是 Harbor 重试序号。
无法关联计划的坏记录另外保留为 `plan_id=unknown`，不扩大计划分母。

每组分别记录任务 ID/version、原生任务配置、Agent/Model、配置摘要、公开工具与预算、
终评后端以及任务包、判据和条件身份。记录中还保存实际 `agent_info`、公开会话运行身份、
终评 profile/tools 或 replay configuration 摘要。不同任务、模型或绑定配置分别统计。
同一计划组内的实际 Agent 名称、版本、模型名称、provider 与运行资源身份一起核验；
任一实际条件不同，该组分数全部失效。缺失信息保留 `null`，实际名称原样保存，
不根据计划中的模型前缀猜测 provider 或服务身份；缺失尝试的条件分母不被猜测。

| 字段 | 口径 |
| --- | --- |
| `planned` | 一个任务与一个 Agent 配置的 `n_attempts`，含缺失和失败 |
| `started` | 原生 `TrialResult.started_at` 已存在的尝试数 |
| `started_unknown` | 保存了 Trial 配置但无法确认开始状态的尝试数 |
| `metric_valid` | 冻结候选与独立终评通过身份和完整性核验的有效分数个数，包含零分 |
| `missing` | `missing` 与 `not_run` 的数量之和 |
| `coverage` | `metric_valid / planned` |
| `score_denominator` | 有效分数个数，与 `metric_valid` 一致 |
| `mean_score` | 有效分数之和除以 `score_denominator`；分母为零时为 `null` |

例如计划六次，正例一分、有效零分各一次，另有执行失败、缺失结果、运行中、未启动各一次，
则 `planned=6`、`metric_valid=2`、`coverage=1/3`、`score_denominator=2`、`mean_score=0.5`。
报告不计算跨任务总榜、成本估算或显著性结论。原生 usage/cost 缺失时保持 `null`。

## 状态与独立评分

| `status` | 可观察证据 |
| --- | --- |
| `not_run` | 计划槽位没有对应的保存 Trial 配置 |
| `missing` | 有 Trial 配置，缺少 TrialResult；开始状态为 `null` |
| `running` | 有开始记录，尚无结束记录且无平台异常 |
| `failed` | Harbor 记录平台或 Agent 异常；异常类型单独保留 |
| `unevaluable` | 已结束，但没有经核验的有效分数 |
| `graded` | 已核验独立终评有效分数且无平台异常 |
| `invalid` | 来源损坏、重复或身份/条件矛盾，`score=null` |

Agent 异常、候选收取和任务正确性分别保留。Agent 超时后若仍收取最后候选，
且独立终评有合法分数，行可同时为 `failed`、`agent_exception=AgentTimeoutError`、
`score=1`。这个分数进入终评的有效分母，失败状态仍然存在，不被评分覆盖。
`termination_reason` 保留 episode-end 中的结束原因；`collection_source` 标明 Agent 提交或 Runner 收取。
两者本身不证明正确。

Spectre 分数来自 `verifier/transport/*/archive`，通过现有 `verify_archive` 重校
归档成员、sealed Job 和原 checker 的结果。开源终评分数来自 `verifier/replay`，
通过现有 `verify_replay` 重校完整证据与原 checker 结果。两种凭据必须唯一。
之后核对本地冻结候选字节、task ID/version、purpose、任务包、criteria 和 condition。
有效分数必须满足 `execution=ok`、合法 pass/fail verdict、有限的 `[0,1]` 数字。
基础设施故障与不支持分析的分数保持 `null`。

`evaluation.json` 只作为包装结果的一致性校验。存在时必须与独立凭据完全一致，
孤立的 evaluation 文件或 Harbor reward 不产生可信分数。取消后即使包装文件未写完，
已有的 sealed replay 执行故障仍可被报告。存在的原生 reward 必须与独立分数相符。
报告保留来源文件摘要；核验归档使用新的临时目录，不修改来源树。

## 与旧实验字段的对应

| 旧入口字段 | Harbor 报告字段 | 差异 |
| --- | --- | --- |
| 任务与实验 `conditions` | `task`、`conditions`、`actual_conditions` | 读取原生配置与实际凭据，不导入旧控制器 |
| 单次结果 `score` | `records[].score` | 仅接受已核验独立终评；保留有效零分 |
| 单次状态与失败原因 | `status`、`agent_exception`、`evaluation.execution`、`errors` | 平台失败与终评分数可同时存在 |
| 候选身份 | `candidate_sha256`、`collection_source` | 重校冻结文件字节及终评绑定 |
| 批次 `planned` / `started` / `metric_valid` | `groups` 同名字段 | 缺失开始证据另外记录 `started_unknown` |
| 批次 `metric_coverage` | `coverage` | 显式使用计划分母 |
| 批次 `score_count` / `score_mean` | `score_denominator` / `mean_score` | 均只使用有效独立分数 |
| 用量和费用 | `usage` | 采用原生保存值，缺失保持未知 |

以上是保存记录的字段对应，不是启动命令或配置转换器。当前运行由 Harbor 管理；保留原始证据，不改写旧记录。

## 验证范围

[`test_harbor_reporting.py`](../../tests/chips/test_harbor_reporting.py)通过公开 CLI 和真实本地文件
验证完整计划、有效零分、未知开始状态、条件分组、损坏/重复/陈旧证据、预算漂移、输出拒绝覆盖、
task binding 及未完成的 replay。独立 checker 与仿真器回复为构造夹具。
禁止主机密钥查询、socket 和新子进程的检查只限制导出阶段；夹具准备使用真实本地 checker 进程。

2026-10-07 已对两份保存的真实 Harbor Job 做只读导出。正常 Job 的
`planned/started/metric_valid=1/1/1`，独立 Spectre 终评一分；超时对照同样保留一分，
同时保留 `failed` 与 `AgentTimeoutError`。两份输入树的完整文件摘要前后相同。
私有输出与复核命令保留在实施记录中。这是旧证据的离线重分析，不构成新的模型、服务器或电路运行。
