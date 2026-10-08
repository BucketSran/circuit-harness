# VABench Agent／模型对照协议

本页只比较同一公开任务、同一原评分器下的完整 episode。每个回合单独建服务器会话、
job ID 和私有 evidence 目录；不得把一个回合的候选或工具反馈带入另一个回合。
`pass` 是原 EVAS 独立终评的 `development_only` 结果，不是模型能力排名。

## 固定条件与记录

首批任务固定为 VABench r53 的 `v4-001`。运行前核对公开 `instruction.md`、
`public_contract.json`、`visible_test.scs` 的 SHA-256，分别为
`9be58adf…d1796e`、`390fac62…0c5781d1`、`41cd2e82…94b113c`。
固定 EVAS 0.8.7、相同原评分器、Apollo 提交、Pi／Codex CLI 版本、公开工具实现、
服务端隔离后端与控制机到服务器的传输方式。会话使用相同的
`max_actions=24`、`max_simulations=4`、`timeout_s=120`；回合墙钟设为 600 秒。
Pi 两条件都设 `max_model_calls=12`、`max_request_bytes=64000`、
`max_output_tokens=4096`，并在每次运行前显式选择、核对 `thinking` 档位；
原生 Codex 同样显式选择 `reasoning_effort`。档位变化是新的实验条件，不与原回合混算。
每次保存 `operator.json`、`preflight.json`、`episode-budget.json`、原始 Agent 事件、
`tools/*`、已验证公开 episode、独立终评归档和 `report.json`。模型费用没有厂商账单
证据时只能记 `not_measured`。服务器预检或许可证失败应单列为基础设施失败，不算模型答错。

三种实验条件与能支持的推断：

| 条件 | 固定／变化 | 可讨论的问题 |
| --- | --- | --- |
| Pi＋GLM | Pi、四个公开工具、任务与预算固定；GLM 服务 | 基准条件 |
| Pi＋GPT (`openai-codex`) | 与上行同一 Pi 入口与工具限制；更换模型和服务商 | 模型／服务商组合差异，不能只归因于权重 |
| 原生 Codex＋同一 GPT | 相同任务、公开工具和 GPT 型号；更换 Agent 运行器 | 运行器观察；原生内建工具仍可用，不能当作纯 Agent 效应 |

原生 Codex CLI 当前只提供回合累计 token usage 和墙钟超时；没有与 Pi 的
`before_provider_request` 等价的请求数硬拦截。因此第三行的模型请求预算不能宣称
与前两行严格相等。即使某次原生回合没有实际调用内建工具，它们仍在可选集合中。
若要研究单独的 Agent 效应，仍需可证明相同的工具可用集合与请求预算控制；
在此之前把 Pi／原生 Codex 差异标成混杂因素。

## 已有真实样本的回顾性对照

2026-09-23 本机 Pi＋`glm-5.3-flash` 与 2026-09-24 本机原生
Codex＋`gpt-5.6-luna`／`medium` 两份公开 episode 归档再次校验通过。
两者上述三份公开任务文件的哈希相同；归档内服务器会话限制均为 24 个动作、
4 次公开仿真、单次 120 秒，终评均为 `pass`。

| 实测指标 | Pi＋GLM | 原生 Codex＋GPT |
| --- | ---: | ---: |
| 已观察模型请求 | 7 | 不可从现有 CLI 事件逐次核对 |
| 公开工具调用 | 8 | 9 |
| 回合墙钟上限 | 1800 秒 | 600 秒 |
| 模型请求数硬限额 | 24 | 无 |
| 候选 SHA-256 前缀 | `f6d5a192` | `3d0747f1` |
| 原 EVAS 独立终评 | `pass` | `pass` |

这两份样本的 Agent、模型、可用工具和模型预算同时变化，且每种只有一次真实回合；
它们证明两条端到端链路可用，**不能**用工具调用数或 token 数判断哪个模型／Agent 更好。
Pi 的逐响应 usage 与 Codex 的回合累计 usage 口径也不同，暂不计算跨行 token 比值。
私有原始记录在 Git 忽略的 `runs/chips/validation/20260923-vabench-agent/`
和 `runs/chips/validation/20260924-codex-luna-medium-002/evidence/`。

Pi＋GPT 入口、配置模板、四工具限制和预算回归已具备；本机 Pi 的
`openai-codex` OAuth 认证文件目前不存在，故 **Pi＋GPT 真实回合尚未运行**。
完成该登录后，先用全新会话按固定条件各跑 Pi＋GLM 与 Pi＋GPT，至少覆盖一个
成功任务和一个会触发诊断／修复的任务，再增加重复回合与任务数。报告每个 episode 的
完整轨迹和失败类型，最后再汇总成功率、调用量和实际账单；单个 `v4-001` 不能代表总体表现。
