# VABench `v4-001`：两种 Agent 位置的实验示例

这两个示例使用同一份 VABench r53 任务 pin、服务器上的 EVAS 0.8.7、四个公开 Tool 和原评分器。
区别是 Agent 与模型请求在哪里运行。每次实验必须使用**新的**公开会话、job ID 和私有证据目录；
不要复用另一回合的候选。下列 `/private/...`、`lab-host` 和模型路径都是占位符，须替换为本人环境。
真实运行会调用模型与仿真器；先按 [环境与 pin 准备](../../../../docs/chips/VABENCH.md)
部署固定源码、EVAS、Apollo bundle 和各自的 Agent 运行时。

| 示例 | Agent / 模型请求 | 公开 Tool、仿真、终评 | 已有实测 |
| --- | --- | --- | --- |
| A：服务器同机 Pi＋GLM | 服务器，Key 留在服务器个人账号 | 服务器本地 MCP → EVAS | `v4-001` 真模型回合、冻结、独立终评与归档通过 |
| B：本机原生 Codex＋GPT | 本机，ChatGPT 登录或 API Key 留在本机 | 本机 MCP → SSH → 服务器 EVAS | `v4-001` 真模型回合、冻结、独立终评与归档通过 |

两条路径都只在 **单题 development 条件**下验收；不是整个 VABench 成绩，也不是
Pi 与 Codex 的公平能力排名。[逐次证据与限制](../../../../docs/chips/VALIDATION.md)、
[Agent/模型对照条件](../../../../docs/chips/VABENCH_COMPARISON.md)另行记录。
两条路径实际传给 Agent 的提示和工具 schema、以及任务文件进入对话的时机，
见[输入与轨迹数据流](../../../../docs/chips/VABENCH_AGENT_DATA_FLOW.md)。

## A. Agent、MCP 与仿真均在服务器

在服务器个人账号的 Bash 终端中，将
[profile 模板](server-deployment.profile.example.json)复制为个人持久目录内的
`deployment.profile.json`，设置权限 `0600`。将 `pin`、`run_root`、`job_root`、
`archive_root`、`python`、`bundle`、`pi_cli`、BigModel Coding Plan 的 `base_url`
和 `model` 改为实际值。三个根目录必须预先存在、本人持有且权限为 `0700`；
个人配置目录也须先创建为 `0700`。下文 B 的 pin 路径应与此 profile 的 `pin` 指向同一文件。
密钥不写入 JSON。`prepare` 会生成一次性公开会话与配置，并执行服务器预检，
但不会调用模型或仿真：

```bash
umask 077
cp examples/chips/benchmarks/vabench/server-deployment.profile.example.json \
  /private/persistent/config/deployment.profile.json
chmod 600 /private/persistent/config/deployment.profile.json
# 编辑 profile，将示例路径、服务地址和模型 ID 改为实际值。
/private/pi-venv/bin/python -m alphaapollo.workflows.chips_vabench_deployment prepare \
  --profile /private/persistent/config/deployment.profile.json \
  --run-id pi-v4-001-001
```

成功返回 `config` 和 `evidence` 两个绝对路径。用返回值替换以下示例路径；
`detach` 在服务器启动独立 Pi 进程后返回，本机 SSH 断开不会终止该回合：

```bash
read -r -s -p 'Model key: ' CHIPS_MODEL_KEY
printf '\n'
export CHIPS_MODEL_KEY
/private/pi-venv/bin/python -m alphaapollo.workflows.chips_vabench_agent detach \
  --config /private/scratch/experiments/pi-v4-001-001/operator.json \
  --evidence /private/scratch/experiments/pi-v4-001-001/evidence
unset CHIPS_MODEL_KEY
```

在服务器私有 `evidence/` 中查看 `operator.log`、`model-budget.jsonl`、
`pi-outcome.json`、`tools/*`、单次实验账本 `results.jsonl` 和最终 `report.json`。`report.json` 中的
`state=verified`、`result.execution=ok`、`result.verdict=pass` 分别表示归档校验、
评分器执行和电路判定；仅看到启动 PID 不表示已完成。服务器进程或机器重启后，
整段模型会话不会自动续跑。详细配置和恢复边界见[服务器部署说明](../../../../docs/chips/SERVER_LOCAL_DEPLOYMENT.md)。

## B. Agent 与模型在本机，仿真仍在服务器

先在服务器用同一个任务 pin 创建**另一个**公开会话，并确认预检 `state=ready`。
下例中的 `chips-agent.pyz` 是本分支生成并已部署的 bundle：

```bash
ssh -T lab-host python3.12 /private/harness/chips-agent.pyz vabench-session \
  --pin /private/pins/dut.json --output /private/scratch/session-codex-001
ssh -T lab-host python3.12 /private/harness/chips-agent.pyz vabench-preflight \
  --session /private/scratch/session-codex-001
```

在本机将 [Codex 模板](agent.codex.config.example.json)复制到 Git 忽略的
`runs/chips/private/`，设置权限 `0600`。填入真实 SSH 别名、服务器 Python/bundle/
session/job/archive 路径、本机 Codex CLI 路径，以及明确的模型 ID 和
`reasoning_effort`；`job_id` 必须全新。模板的 `auth_kind=chatgpt` 使用本机登录，
运行前检查 `codex login status`。若改用 `auth_kind=api_key`，只在本机私有进程环境
提供 `OPENAI_API_KEY`，不要写进配置或仓库。

```bash
umask 077
mkdir -p runs/chips/private
chmod 700 runs/chips/private
cp examples/chips/benchmarks/vabench/agent.codex.config.example.json \
  runs/chips/private/codex-v4-001-001.json
chmod 600 runs/chips/private/codex-v4-001-001.json
# 编辑本机 JSON，将示例路径、会话、job ID 和模型 ID 改为实际值。
codex --version
codex login status
python -m alphaapollo.workflows.chips_vabench_agent codex \
  --config runs/chips/private/codex-v4-001-001.json \
  --evidence runs/chips/private/codex-v4-001-001-evidence
```

本机 `evidence/` 保存 `auth-preflight.json`、`preflight.json`、
`codex-events.jsonl`、`codex-outcome.json`、`tools/*`、`report.json` 和已校验的
公开/终评归档；服务器保留公开动作、EVAS 和评分原始记录。只有收到
`vabench_submit` 成功回执才会启动独立终评。若本机断线，已提交的服务器动作
继续运行，但 Codex 不会自动接着思考；可按原 action ID 用 `recover` 查询，
或用 `collect` 收取已启动终评的结果，**不要用新 ID 重跑同一仿真**。
两条 Agent 路径的新 evidence 目录还会保存单次实验的
`task_manifest.json`、`resolved_config.json`、`agent-input.json`、一行 `results.jsonl`
及 `summary.json`。这样一次运行就是 1 题 × 1 次的标准单元；原始动作与仿真证据仍保留在
既有路径中。[状态与字段说明](../../../../docs/chips/VABENCH_AGENT.md)。
命令和限制见 [VABench Agent 说明](../../../../docs/chips/VABENCH_AGENT.md)。

## 本次能够检查什么

两种路径都应核对公开工具请求/响应、模型轨迹、冻结候选摘要、服务器仿真记录、
终评结果及两份归档校验。公开仿真成功只表示这次测试可运行；最终判定以独立
`report.json` 的 `result` 为准。原始轨迹和评分包属于个人私有证据，不能提交到 Git。
其他现有案例包括 [Analog Design Bench](../analog_design_bench/TASK.md)、
[ngspice RC](../../../../docs/chips/NGSPICE.md)、[Spectre RC](../../../../docs/chips/SPECTRE_RC.md)
和 [EMX JSON→GDS](../../README.md)；它们各自的 Agent/模型验收范围不同，不能套用本例成绩。
