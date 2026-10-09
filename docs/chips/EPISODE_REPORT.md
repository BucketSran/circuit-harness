# 保存 Episode 的离线报告

`circuit_harness.reporting.episode` 读取兼容布局的已保存实验，生成 HTML、JSON、
Markdown 与 CSV 时间线。它不执行 Agent、模型、SSH、仿真或评分，也不修改来源。
这是历史归档查看器；新的 Harbor 公开会话结果使用[Harbor 报告](HARBOR_RESULTS.md)。

## 生成报告

选择单个实验和一个不存在的新输出目录：

```bash
python -m circuit_harness.reporting.episode \
  /private/saved-episode --output /private/reports/episode-001
```

打开 `report.html`，展开事件、工具反馈和候选变化。仅在动作 ID 与响应匹配时，
工具返回才关联到对应请求；没有历史快照时，不读取当前代码来补造当时内容。

| 来源布局 | 读取内容 |
| --- | --- |
| 保存的实验 `output/` | 顶层清单、结果、`agent/` 和已有 `final-result.json` |
| Pi/Codex evidence | outcome、`tools/`、模型预算和已有报告 |
| 保存的 runtime evidence | `native-outcome.json`、`runtime-result.json` 与 `http/` |
| 展开的 Analog Episode | `agent/`、`session/`、`final/`，及冻结/评分输入字节 |

并非任意 Harbor Trial 都符合这些布局。报告不下载配置中引用的远端文件，
也不把旧归档当成当前执行入口已验收的证据。

## 输出与隐私

- `episode-report.json` 保存来源摘要、证据缺口、设置、事件与终评关联。
- `report.html` 使用本地转义渲染和折叠块，不加载外部资源。
- `report.md` 汇总状态、计时、用量和缺口。
- `timeline.csv` 保存已观察时间区间，未知值留空。

输出目录为 `0700`，文件为 `0600`，拒绝覆盖或写入来源目录。报告包含原始
提示、代码与工具反馈，属于私有运行产物，并未自动脱敏。不要提交到 Git。

## 解释结果

报告命令退出 `0` 只表示报告生成成功，原实验仍可能失败、阻塞或没有提交。
无可识别来源、损坏 JSON、候选不一致或归档损坏退出 `2`。
`harness_status`、`agent_submitted`、`collection_source` 与电路成绩分别保留；
自动收取的候选通过不等于 Agent 主动提交。

`final_candidate_publicly_simulated` 只核对提交与成功公开仿真的摘要，不证明终评通过。
缺少内容或摘要时保持未知。候选 diff 只比较已记录版本，第一份写入不假定为
官方 starter。独立评分遵循原归档核验器，不将 Agent 自报或工具退出码当成成绩。

Pi 逐响应 usage 与 Codex 回合累计 usage 分别解释。reasoning 属于 output 的细项，
不重复相加；缺少账单或逐请求信息时，用量和费用保持未知。

对应检查是 `tests/chips/test_episode_report.py`。
它们验证保存文件的读取和展示，不调用真实模型或仿真器。
