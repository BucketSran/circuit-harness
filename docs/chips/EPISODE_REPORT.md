# Chips 离线 Episode 报告

把一次已保存的实验整理成可展开的 HTML、JSON 摘要、Markdown 和 CSV 时间线。
报告读取既有文件，不请求模型、不连接 SSH、不重跑仿真或评分，不修改原始实验。
关联 [共建讨论 #1](https://github.com/BucketSran/circuit-harness-private/issues/1)。

## 使用

在安装项目依赖的环境中，选择一条实验及一个**不存在**的新输出目录：

```bash
python -m alphaapollo.workflows.chips_episode_report \
  /absolute/private/experiment \
  --output /absolute/private/reports/episode-001
```

打开输出中的 `report.html`。时间线的 `details` 链接定位对应记录，点击记录可展开；Agent 的工具返回事件
在 action ID 和响应内容匹配时，可以跳到对应 Tool 请求、反馈和计时。
候选修改显示相邻已观测写入的 diff；第一份写入不被当作官方 starter 的 diff。
Analog 的恢复动作也进入候选变化序列；只有记录中已有该版本内容时才生成恢复 diff。
`final_candidate_publicly_simulated` 对照主动提交／收取摘要与成功执行的公开仿真摘要；
它不表示电路通过，缺少摘要或响应时保留未知。

支持以下已有布局：

| 输入目录 | 读取方式 |
| --- | --- |
| 共同实验入口的 `output/` | 读取顶层清单、单行结果、`agent/`，以及已保存的 `final-result.json` |
| 旧的 Agent evidence 目录 | 读取 `pi-outcome.json` 或 `codex-outcome.json`、`tools/`、`model-budget.jsonl`、`report.json` 等已存在证据 |
| 原生 Apollo 验证探针 evidence | 读取 `native-outcome.json`、`runtime-result.json`、`http/`，展示已记录的 Runtime 回合和 HTTP 请求／返回 |
| 已展开的 Analog Episode | 根目录包含 `agent/`、`session/`、`final/`；核对冻结候选与评分输入字节 |

只接受单个 Episode，不是批量排名入口。已有的 EMX/RC 作业报告仍使用各自入口。
不从配置中的远端路径自动下载文件；也不在没有历史快照时读取当前代码来补写提示或工具定义。
原生 Codex 当前缺少与 Pi 同等的逐模型请求时间账本，相关总量与耗时留空。

## 输出与复用边界

- `episode-report.json`：版本化派生记录、来源文件 SHA-256/大小、证据缺口、设置、事件、动作、候选变化和终评。
- `report.html`：复用 Apollo `workflows/visualize.py` 的转义渲染、折叠块和样式，不依赖外部网页资源。
- `report.md`：结果、计时、用量与缺口的简要摘要。
- `timeline.csv`：模型请求与 Tool 的控制端时间区间，缺失值为空。

原始 Chips 外部 Agent 记录不等同于共享 `traj.jsonl` 契约，因此不伪造 Workflow 回合后调用
通用 reducer。Chips 适配器负责来源语义；共享查看器负责已有的展示能力。
Math 批量统计仍是另一层，不把不同任务的连续分数混合求平均。

所有输出保存在新建的 `0700` 目录，文件为 `0600`；拒绝覆盖已有目录或写入源证据目录。
报告包含原始 prompt、代码及反馈，属于私有产物，不是自动脱敏的分享版。
不要把报告加入 Git。Key 文件不会被读取；operator 仅投影模型、推理、预算和传输等选定字段。

## 如何解释结果

实验 `status`、终评分数及派生的过程类别分别显示。报告生成命令退出 `0` 只表示报告已生成，
也可以对应一次失败、阻塞或未提交实验；无可识别输入、JSON 损坏、候选不一致或归档损坏退出 `2`。

`harness_status` 表示收取／终评进度，`agent_submitted` 表示是否有主动提交证据，
`collection_source` 区分 `agent_submit` 和 `episode_end`，设计分数单列。
自动收取后电路通过，类别为 `collected_design_pass`；未全部通过为 `collected_design_fail`。
这两者都不算 Agent 主动提交。`closed_without_candidate` 表示已结束但没有候选，分数为空。
`finalized` 仅说明终评已有结果，不证明全部轨迹归档已完成。
历史记录缺少提交证据时保留未知，不从一句“完成了”推断提交。
报告同时保留原始终止状态和 Harness 分类，并展示已记录的逐请求预算提示。

| 类别 | 判定依据 |
| --- | --- |
| `success` | VABench 记录有效终评 `pass`；或 Analog 有有效终评且已记录的测试全部通过 |
| `success_after_repair` | 除终评通过外，还观察到失败仿真 → 不同哈希候选的成功仿真，且提交/终评对应修复后的候选 |
| `budget_exhausted_unsubmitted` | 未提交，且有明确 `budget_stop` 或最后一次模型响应为 `length` |
| `graded_failure` | 有效 VABench `fail`，或 Analog 已记录的测试未全部通过 |
| `graded` | 有有效连续分数，但缺少可判定“所有测试通过”的记录 |
| 其它状态 | 保留 `unsubmitted`、`blocked`、`pending` 等记录，不猜测未记录的原因 |

“修复”仅描述有证据的动作序列，不自动判断模型的诊断理由正确，也不代表全部电路错误都已修复。
一次成功可能包含读取失败等已恢复的工具错误，失败次数另列。

Analog 的原连续 `score` 保留；`benchmark_success` 仍为空，不更改统一实验入口的评分语义。
`all_recorded_tests_passed` 是单独的查看指标，不能据此构造跨任务成功率。
VABench 的 `development_only` 等评分权威原样保留，不升级为正式认证。

## 计时、用量和完整性

- 模型时间来自 Pi 请求与响应的控制端时间戳，包含网络、排队、生成和流式输出；不是纯模型计算时间。
  单列最长请求及 `length` 停止次数；逐请求的 `output_budget` 展示已记录的配置上限、
  reported reasoning/output 比例与 output−reasoning。它们不是可见文本长度或思考时长。
  缺用量或相互矛盾的数字保留空值，不补成零。
- 存在 `pi-wire-requests.jsonl` 时按请求 ID 关联实际 provider payload；重复或无法匹配的 ID 会报错。
  原生探针的 `http/` 保留完整请求／返回和相对时长；由于没有绝对时钟锚点，不推算它与 Tool 的时间交错。
  原生 provider 用量保留在每次返回中，不强行转换成 Pi 的 token 字段。`runtime-result.json` 的回合原样展示。
- Tool 时间包含控制端/SSH 轮询等开销。若服务器归档有对应 `measurement.json`，另列服务器 action 时间；
  公开仿真的调用数、控制端区间及有证据的服务器 action 区间另行汇总。
  它也包含服务器封装开销，不能当作纯求解器时间。缺少独立测量时不拆出“网络耗时”。
- 区间合计不是重叠任务的墙钟时间。终评计时单列，不加到 Agent 窗口里冒充总实验耗时。
- Pi usage 只累计记录到的响应；每个字段附覆盖条数。缺失字段为 `null`，不填零。
  reasoning 是 output 的细项，不重复相加；累计请求 token 不是单次上下文长度或账单。
- Codex/Pi outcome 原报用量单列，不默认将最后一条响应的用量解释成全 Episode 总量。
- VABench 优先使用原归档校验器重新检查包与成员，再核对提交、冻结候选、终评及客户端/服务器动作。
  `verified_archives` 表示本地已有两个归档的字节及关联校验通过，不是外部认证。
- Analog 已展开数据的 `bytes_matched` 表示冻结与评分输入的字节匹配；它不证明整个展开目录有归档封印。
  统一入口仅有服务器返回结果时为 `reported_match`，明确记录没有读取远端字节。

未被记录的 provider 请求、CLI 自带系统提示、未输出的内部推理和工具发现内容无法追溯重建。
报告不重新发送 Episode 给 Agent，也不自动写入 Memory。

## 代码与测试

| 文件 | 职责 |
| --- | --- |
| [chips_episode_report.py](../../alphaapollo/workflows/chips_episode_report.py) | 输入选择、用量与时间线投影、CLI 和私有输出 |
| [evidence.py](../../alphaapollo/workflows/_chips_episode_report/evidence.py) | action ID 关联、候选 diff、修复序列、归档与终评核对 |
| [view.py](../../alphaapollo/workflows/_chips_episode_report/view.py) | 复用共享 HTML 原语，渲染 Chips 报告 |
| [test_episode_report.py](../../tests/chips/test_episode_report.py) | 构造证据边界的回归测试，不冒充真实仿真 |

```bash
python -m pytest -q tests/chips/test_episode_report.py tests/workflows/test_visualize.py
```

真实历史归档的离线验收见 [验证记录](VALIDATION.md)。后续仍需独立完善逐请求上下文捕获、
原生 Codex 的请求级计时、更多任务的报告适配及跨实验统计。
