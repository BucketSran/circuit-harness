# Chips 单次实验入口

`alphaapollo.workflows.chips_experiment` 用一份私有 JSON 选择任务、Agent 和本次运行目录，
在同一位置写出 `experiment_manifest.json`、单行 `results.jsonl` 和 `summary.json`。
准备或运行前可用 `python -m alphaapollo.workflows.chips_experiment validate --config <experiment.json>`
检查任务／Agent 组合、plan 与 operator 字段归属和显式模型设置。它不创建运行目录、
读取 Key、调用模型或联系服务器；`config_valid` 仅表示配置校验通过，运行环境仍须预检。
它调用已有的 VABench 或 Analog 公开会话；仿真器、候选约束和最终评分仍由各任务实现。
这是 **一题 × 一次** 的执行单元；固定批次由下文的 `chips_evaluate` 顺序调用。

Agent、任务和部署是三个独立维度。当前实现只接通以下组合；换任务不会自动生成一套新 Agent，
而是由任务提供候选规则、公开 Tool、仿真与终评策略：

| 任务 | 共同入口允许的 Agent | 当前执行位置与边界 |
| --- | --- | --- |
| VABench | Pi、原生 Codex | Pi 可在服务器同机或控制机；Codex 已验收本机经 SSH 调服务器 |
| Analog Design Bench 两道 RLC 题 | Pi | 服务器同机已有两题真实模型证据；宽带题完成自主仿真、恢复、提交、独立终评与归档核验（6/7，0.9）；原生 Codex＋SSH 尚未接入共同入口 |

模型 ID、推理档位和预算属于每次 Agent 配置，不由任务名决定。Apollo 原生 Runtime 的
VABench 对照已有单独运行证据，但还不是上表共同入口的 `agent` 选项。

从 [VABench 配置](../../examples/chips/benchmarks/vabench/experiment.example.json)
或 [Analog 配置](../../examples/chips/benchmarks/analog_design_bench/experiment.example.json)
复制到本人私有目录，填入绝对路径。`operator_config` 指向该任务原有的私有 operator JSON，
模型、推理档位、预算、服务器会话和传输方式仍在其中；实验 JSON 不存 Key。
新 run 必须使用新的 `run_id`、会话、输出目录和终评目录。

Analog 宽带题可从 [宽带模板](../../examples/chips/benchmarks/analog_design_bench/experiment.broadband.example.json)
开始；实验、operator 和新建会话的 `task_id` 必须一致。两题共用 RLC Tool 与 Agent 入口，
由任务契约决定公开文件、候选接口和诊断，详见 [RLC 运行说明](ANALOG_DESIGN_BENCH.md#用任务配置复用同一条-rlc-链路)。

预算是实验条件，模板值不保证任务完成。2026-09-26 宽带首回合在第 11 次请求发送前
因 `65259 > 64000` 字节停止；新的 16 次请求／131072 字节条件完成了主动提交。
第二回合实际最大请求仅 46237 字节，因此这两个样本不能证明扩大请求预算导致了成功。
两个回合、候选收取方式和电路成绩均保留在[验证记录](VALIDATION.md#broadband-piglm-episode-acceptance-2026-09-26)。

```bash
python -m alphaapollo.workflows.chips_experiment run --config /private/config/experiment.json
```

VABench 支持 Pi 或本机原生 Codex，复用原 [Agent 入口](VABENCH_AGENT.md)。
`task_id` 必须等于 operator 中的 `task_id`，`run_id` 必须等于其 `job_id`。
公开提交后，原入口会启动独立终评；若结果尚在运行，读取 `results.jsonl` 的 `pending`，
稍后用**同一份未修改的配置**收取，不会重新调用 Agent：

```bash
python -m alphaapollo.workflows.chips_experiment collect --config /private/config/experiment.json
```

Analog 当前接入服务器同机 Pi 与上述两道固定版本 RLC 公开会话；运行后若结果为
`awaiting_final`，操作者用同一配置启动原评分器并归档：

```bash
python -m alphaapollo.workflows.chips_experiment finalize --config /private/config/experiment.json
```

`final_output` 是服务器上新的评分工作目录；`archive_root` 是服务器上的私有持久目录。
`key_file` 可选，只供 Analog Pi 读取本人拥有的 `0700` 目录中的 `0600` Key 文件；
也可在启动进程的环境中提供 `CHIPS_MODEL_KEY`，两者不可同时使用。Key 值不会进入实验配置或归档。
混合 VABench／Analog 批次若由同一进程执行，VABench 当前使用环境变量中的 Key；
此时 Analog 实验 JSON 应省略 `key_file`，让两类任务使用同一个私有环境变量。
若 Analog 需要使用 `key_file`，应在不带 `CHIPS_MODEL_KEY` 的独立进程中运行；
同时提供两种来源会在 Analog 预检阶段被拒绝，不会启动模型或仿真。
终评只接受已确认冻结、哈希一致的候选。评分使用固定上游 ngspice/Podman 流程，
成功后调用既有 `analog-archive` 并再次校验 Episode。若只有归档失败，
`results.jsonl` 保留分数并标记 `archive_failed`；若进程在评分已记录、归档完成前中断，
则保留 `final_graded`。两种情况均可用以下命令恢复**归档**，不重跑评分：

```bash
python -m alphaapollo.workflows.chips_experiment archive --config /private/config/experiment.json
```

结果行的 `status` 描述整次实验的进度；`final_execution`、`final_validity` 与
`score`/`verdict` 分开描述终评。VABench `pass/fail` 才投影为二元 `benchmark_success`；
Analog 保留原 reward，哪怕为 `0.0` 也是有效分数，`benchmark_success` 与
`summary.success_rate` 仍为 `null`，不虚构跨任务的通过阈值。
`unsubmitted`、`blocked`、`archive_failed` 和 `unknown_execution` 不进入完成次数。
如果终评调用超时或回复不明，保留 `finalize-intent.json`，必须检查原服务器目录；
共同入口不会用同一 ID 再启动一次评分。

顶层清单记录任务、run ID、Agent、模型实验设置及实验/操作者配置哈希。
原始模型和 Tool 轨迹在 `output/agent/`；VABench 的原评分与公开归档沿用该目录下的
既有路径，Analog 的私有 Episode 在 `archive_root/episodes/<run_id>/`。
运行后更改实验或 operator JSON，`collect`、`finalize`、`archive` 都会拒绝继续。
本入口不代替 EVAS、ngspice、Spectre 或 EMX 的版本和工艺固定证据；这些仍由各任务记录。

当前本地回归使用构造 Agent/服务器回复，验证了两类任务的状态投影、配置冻结、
0 分、归档恢复和超时不重跑。它不构成新的真实模型、SSH 或仿真器验收。
Spectre RC 与 EMX 仍保留各自入口，待真实任务具备独立指标后再按同一实验记录接入。

完成或停止后的实验目录可直接交给[离线 Episode 报告](EPISODE_REPORT.md)，查看历史设置、轨迹和已有终评；不触发 `collect`、`finalize` 或新的模型调用。

## 固定批次的准备、运行与汇总

`chips_evaluate` 只编排已有单次实验。`prepare`／`report` 不调用模型或服务器；
`run` 会按清单顺序启动尚未运行的 cell，是否联系模型和仿真器取决于该 cell 的配置。
兼容的 v1 批次配置只引用已有单次实验 JSON，不复制任务或模型字段：

```json
{
  "schema_version": 1,
  "output": "/absolute/private/batch-001",
  "cells": [
    {"condition_id": "pi-glm-low", "experiment": "/absolute/private/experiment-001.json"},
    {"condition_id": "pi-glm-low", "experiment": "/absolute/private/experiment-002.json"}
  ]
}
```

需要做可比较评测时使用 **v2**：保留相同的 `output` 和 `cells`，把 `schema_version` 改为 `2`，
并增加 `conditions`。每个“任务 ID＋条件”恰有一条版本声明；例如：

```json
{
  "benchmark": "vabench",
  "task_id": "v4-001",
  "condition_id": "pi-glm-low",
  "agent_revision": "pi-0.87.0",
  "harness_revision": "code-commit-and-bundle-digest",
  "task_revision": "vabench-r53-and-task-digest",
  "prompt_revision": "reviewed-prompt-digest",
  "toolset_revision": "reviewed-toolset-digest",
  "simulator_revision": "EVAS-0.8.7-and-executor-pin",
  "scorer_revision": "original-r53-scorer-pin",
  "memory_snapshot": null
}
```

将这样的对象放入 `conditions` 数组，Analog 等其他任务各填一条。以上是字段示例，实际值须从
预检、部署包和任务记录核对；程序冻结并校验**声明值**，不凭字符串自行证明服务器正在使用该版本。
同一 `condition_id` 在不同任务中必须保持 Agent、Harness、模型预算和 Memory 设置一致；
同一任务／条件不能有不同的版本声明。`memory_snapshot: null` 表示关闭跨 Episode Memory。
v1 保留用于旧批次和探索性汇总；正式条件比较应使用 v2，并核对每次运行证据中的实际版本。

每个实验 JSON 的 `output` 必须是批次 `output/cells/<run_id>`；run ID 和 operator 会话
不能重复。相同 `condition_id` 必须有相同的 Agent、传输、模型与预算设置。准备时先通过
单次实验原有校验，再冻结批次、实验和 operator 文件哈希，并为所有计划项写入 `not_run`：

```bash
python -m alphaapollo.workflows.chips_evaluate prepare --config /absolute/private/batch.json
python -m alphaapollo.workflows.chips_evaluate preflight --config /absolute/private/batch.json
python -m alphaapollo.workflows.chips_evaluate run --config /absolute/private/batch.json
python -m alphaapollo.workflows.chips_evaluate reconcile --config /absolute/private/batch.json
python -m alphaapollo.workflows.chips_evaluate report --config /absolute/private/batch.json
```

`preflight` 检查所有尚未启动的真实模型 cell，结果保存在批次私有目录的 `preflight.json`。
它先检查凭据来源冲突或缺失，再检查 VABench 的 Pi 启动器、依赖、模型 HTTPS
路径及远端公开会话，和 Analog 的既有 Python、Pi、Podman、公开文件、模型路径预检。
VABench 的公开会话预检会运行其隔离环境检查；此命令不发送模型请求或进行终评。
任一 cell 未就绪时返回 `2`，不会启动任何 Agent。`run` 会在启动第一个新 cell
之前重新执行整批预检，所以独立运行 `preflight` 只是供操作者提前查看。
预检通过不保证后续模型授权、网络持续可用或电路得分；这些仍由单次轨迹和终评记录。

`run` 在批次中一次只启动一个 Agent，单次命令返回非零就停止；再次调用只启动仍为 `not_run`
的 cell，不重复已创建输出的实验。它为每次启动先写私有 intent；如果进程在单次实验
落盘前中断，该 cell 记为 `unknown_execution` 并停止后续启动，须先人工核对服务器会话与
原输出，不能直接重跑。同一批次同时只能有一个 `run`／`reconcile` 进程。
`reconcile` 按单次结果状态调用既有 VABench `collect` 或 Analog 的 `collect`、`finalize`、
`archive`；Analog 的独立终评仍由单次入口核对已确认的冻结候选后才启动。
每次收取／终评动作先写私有记录；动作中断后若单次状态没有前进，批次标为
`unknown_execution`，不能自动重试。VABench 的一次正常收取若仍为 `pending`，
下次可以再次查询；Analog 的评分后归档故障只恢复 `archive`，不重复评分。
`run` 不代替终评，也不会把 Agent 退出码当成解题成绩。操作者仍可按上面的单次命令
处理特定 cell，再运行 `report` 刷新批次 `results.jsonl` 和 `summary.json`。

配置漂移会拒绝运行或汇总；已经创建但证据损坏的 cell 记为 `evidence_error`，
返回非零状态，不从计划数中消失。汇总按**任务 ID＋条件**分组：
VABench 的有效二元结果计算成功率，Analog 的有效连续分数单独统计，0.0 保留为有效分数。
每组同时给出 `planned`、`started`、`metric_valid` 与 `metric_coverage`（有效指标数／计划数）；
`binary_success_rate` 的分母是有效二元结果，`binary_successes_per_planned` 的分母是全部计划样本，
两者不能互换。连续分数列出有效样本数、均值、中位数、最小／最大值；至少两个有效样本才给出
样本标准差。没有有效分母时写 `null`；未完成项仍在计划数里。
当前不提供跨任务的合并总分。
批次及每个任务／条件组另列 `agent_submission_counts`（submitted／not_submitted／unknown）、
`collection_source_counts`、`termination_counts` 和 `final_candidate_publicly_simulated_counts`
（yes／no／unknown）。这些计数覆盖全部计划项，与 `completed` 和电路得分分别解释。
单次结果行及其 summary 的 `episode` 使用相同的终止／提交字段；批次 `report` 可从已有
私有证据派生这些字段，不改写历史单次记录。缺证据为 `null`／unknown，不当作没有发生。
提交请求返回 `unknown_execution` 时也保留未知，不因缺少确认或 `unsubmitted` 标签推断它未执行。
`final_candidate_publicly_simulated=true` 仅表示最终摘要对应一次成功执行的公开仿真，
不证明其指标达标；`false` 表示已有动作记录中没有对应成功仿真，记录不完整时保留未知。
新的 cell 不注入旧实验轨迹；跨 Episode Memory 仍是独立、默认关闭的实验条件。

### Analog 结束与验收

新会话在正常结束、预算耗尽或截止时间后，由 operator 收取最后完整候选；
`agent_submitted=false` 与 `collection_source=episode_end` 明确区分自动收取和主动提交。
有候选时仍单独运行 `finalize`，按候选摘要核对独立终评及归档。
没有候选时记录 `missing_candidate`，可运行 `archive` 保存失败轨迹，
此时 `final_execution`／`final_validity` 为 `not_applicable`，分数为空，不启动评分器。
`completed` 表示本次记录已收尾，不等于 Agent 解题成功。
未完成或状态未知的服务器动作须先恢复，不在其执行中冻结候选。
旧会话和原有实验结果不追溯改写。
