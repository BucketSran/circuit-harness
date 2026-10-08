# Runtime／Environment 实验执行清单

日期：2026-09-25。以下保留**制定方案时的清单快照**，`not_run` 不表示当前状态。
实际执行状态、逐回合结果与失败统一维护在[验证记录](../../VALIDATION.md)，不在两处重复更新。
入口：[方案](README.md)、[E1–E3](CONTRACTS.md)、[E4–E5](LIVE_RUNS.md)。

| ID | 条件／交付 | 依赖 | 状态 | 证据／阻塞原因 |
| --- | --- | --- | --- | --- |
| E1 | vaBench B0／N1／N2；真实 Pi CLI＋HTTP fixture | 薄环境、工具发现和组装实现 | not_run | 未实现 |
| E2 | 两任务参数化契约与公开材料隔离 | E1、Analog 薄适配 | not_run | 未实现 |
| E3 | F1–F9、旧 CLI／报告兼容 | E1、受影响生命周期实现 | not_run | 未实现 |
| E4-R1 | 2 任务 × 2 正负候选 × 2 路径，8 会话 | E1–E3 | not_run | 需冻结部署与 fixture 版本 |
| E4-R2 | vaBench local／SSH，2 会话 | E4-R1 | not_run | 需新环境部署 |
| E4-R3 | 在途断线与重建收取，1 会话 | E4-R2 | not_run | 需隔离作业与进程证据 |
| E5-M1 | 旧 Pi＋GLM／vaBench，重复 1／2／3 | E4 | not_run | 需冻结最终运行设置 |
| E5-M2 | 新 Pi＋GLM／vaBench，重复 1／2／3 | E4、M1 首次 | not_run | 同上 |
| E5-M3 | 原生 Apollo＋GLM／vaBench，重复 1／2／3 | E4、provider／预算映射 | not_run | 需验证实际设置可执行 |
| E5-M4 | 新 Pi＋GLM／Analog，重复 1／2／3 | E4、新路径前置回合 | not_run | 同上 |

开始执行时把 E5 每次重复展开为独立行，使用不可复用 run ID；不改写失败行来放置成功重试。
状态：`not_run` 尚未执行、`blocked` 缺前置条件、`passed` 满足本项验收、`failed` 未满足。
对 E5 另列链路状态、是否提交、评价有效性、原分数，避免把 `passed` 误读为电路达标。
首轮 4 条均取得可信集成证据后再继续剩余 8 条；中途改动条件时创建新条件版本。

每条证据填写：实际代码／任务／软件版本、私有记录引用及摘要、真实／fixture 组件、
执行数与跳过数、停止原因、候选／冻结／终评关联、观测缺口。
记录模板沿用 [RECORD_TEMPLATE.md](../../../../tests/chips/RECORD_TEMPLATE.md)，
完成后只将脱敏结论追加到 [VALIDATION.md](../../VALIDATION.md)。

本轮验收输出是一份逐项证据表及“C1／C2 支持到什么范围”的结论。
若实测否定某个设计假设，也完成该项调查；保留旧路径、报告阻塞，不将它写成全迁移完成。
