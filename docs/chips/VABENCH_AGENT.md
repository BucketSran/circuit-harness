# VABench 公开工具与 Agent 闭环

提示文本、MCP 工具说明如何进入 Agent，以及与 DeepSWE-Bench 的任务/评分数据流对照，
见[Agent 输入与轨迹说明](VABENCH_AGENT_DATA_FLOW.md)。

Chips 现在提供公开任务 → 修改候选 → 公开 EVAS 反馈 → 冻结 → 原评分器独立终评 → 归档下载的入口。
它复用 Apollo 的 `PiSession` / `CodexSession`、MCP bridge，以及 vaEVAS 的公开导出、Bubblewrap 执行器、波形摘要和最终评分器。
这是独立的 **Chips development 条件**；没有复现或认证原 VABench 的完整 G2/native campaign。
当前 Bubblewrap 是这条已验证公开仿真链路的实际后端，不是仅供 smoke 使用的临时选项。
若将来为 VABench 增加 Podman，应固定 EVAS 与依赖镜像，核对公开反馈、隔离边界及原评分器正负例，
并作为新的执行条件单独报告；在此之前不改变现有闭环的后端或混算两种条件的成绩。

已通过真实 Pi 0.87.0 + MCP + SSH + EVAS 的脚本模型响应测试，以及 family 001 三种任务的脚本修复闭环。
GLM 5.3 Flash 也在 `v4-001` 完成一次真实回合：7 次模型请求、8 次公开工具调用、
公开仿真成功、候选冻结、原评分器 `pass`，两份归档下载校验通过。
这是一个 development 条件下的单任务结果，不能推断其它任务成功率或原 VABench campaign 分数。
此前还单独完成了 BigModel 最小请求和本地 Pi/MCP smoke。
参考候选仅用于 operator 集成测试，不会被接入真实模型会话。

## 组件与编辑位置

| 位置 | 责任 |
| --- | --- |
| `alphaapollo/common/execution/chips/vabench_session.py` | 四个公开工具、调用去重、冻结、公开轨迹归档 |
| `alphaapollo/common/execution/chips/vabench_public_worker.py` | 用固定 vaEVAS Python 运行原隔离执行器，公开仿真与波形摘要 |
| `alphaapollo/common/execution/chips/session_transport.py` | 各 Chips 任务共用的 SSH／同机提交、查询、同 ID 恢复与 operator 下载/拷贝；`vabench_remote.py` 保留 VABench 兼容类和命令名 |
| `alphaapollo/workflows/chips_vabench_agent.py` | MCP / Pi / Codex 入口、终评收尾与报告 |
| `alphaapollo/reasoning/runtime/external/bridge/pi_budget.ts` | Pi pilot 请求次数、请求体大小和输出上限；逐条 usage 记录 |
| `tests/chips/test_vabench_{session,agent}.py` | 本地边界、真实后台进程、归档与构造网络回归 |
| `tests/chips/probes/pi_vabench.py` | 显式启动的真实 Pi/SSH/EVAS 探针；本机 HTTP 脚本响应，不是模型能力实验 |
| `tests/chips/test_vabench_codex.py` | 原生 Codex 入口、认证隔离及真实 MCP 子进程的构造服务器回归 |

电路任务契约仍由固定版本的 vaEVAS 提供；新增任务/Tool 时先说明公开输入、允许修改的候选和判定规则。
不要通过修改封存参考答案或原评分器来让测试通过。最终评分不注册为模型 Tool。

## 公开工具契约

- `vabench_read({path})`：只读 `task/`、`submission/`；空路径列出文件，拒绝路径穿越、符号链接及超限文件。
- `vabench_write({path, content})`：只写原公开契约的候选文件，每文件上限 100 KB。
- `vabench_simulate({})`：固定公开命令，返回退出状态、诊断、候选摘要和原波形摘要；不是最终正确性分数。
- `vabench_submit({})`：冻结候选并终止修改，返回提交身份，不返回终评分数。

会话默认最多 24 个动作、4 次公开仿真，每次仿真预算 120 秒；`vabench-session`
也接受 `--max-actions`、`--max-simulations`、`--timeout-s`，并在服务端预检中回报实际限制。
每次尝试的 `episode-budget.json` 保存预检中的服务端限制与 Agent 侧限制，便于复核。
Testbench 反馈仅来自公开参考 DUT。
Pi 禁用内建工具、自动扩展/Skill/模板加载；实际工具集合已在协议探针中检查。
模型凭据仅在 Pi 所在机器的进程环境中；MCP 子进程经 `env -i` 启动，候选沙箱也清空环境。
0700 保护其他普通账号的访问；模型候选执行的隔离来自 Bubblewrap 的文件/网络命名空间。

## 配置与运行

先按 [VABench 环境说明](VABENCH.md) 安装 EVAS 0.8.7、固定源码和任务 pin。
服务端需要 Linux Bubblewrap；旧版不支持 `--clearenv` 时，适配器通过 `env -i` 实现相同的空环境启动，
保留原执行器的挂载和网络隔离参数。没有修改实验室系统配置或 vaEVAS 源码。

在控制机生成新版标准库包，然后部署到自己的服务器私有目录：

```bash
python -m alphaapollo.workflows.chips bundle --output chips-agent.pyz
```

在服务器上创建新会话，所有路径均为占位符：

```bash
umask 077
python3.12 chips-agent.pyz vabench-session \
  --pin /private/pins/dut.json --output /private/scratch/session-001
python3.12 chips-agent.pyz vabench-preflight --session /private/scratch/session-001
```

在控制机安装 `@earendil-works/pi-coding-agent@0.87.0`，复制
[配置模板](../../examples/chips/benchmarks/vabench/agent.config.example.json) 到 Git 忽略的私有位置。
填写 SSH 别名、路径和经确认的服务商 API/model ID；**配置文件里不放 Key**。
BigModel 中国区 Coding Plan 的本次配置使用
`https://open.bigmodel.cn/api/coding/paas/v4` 和 `glm-5.3-flash`，
与 [Pi 模型目录](https://pi.dev/models/zai-coding-cn/glm-5-3-flash) 一致。
通过本机秘密管理方式设置 `CHIPS_MODEL_KEY` 环境变量后执行：

```bash
umask 077
python -m alphaapollo.workflows.chips_vabench_agent pi \
  --config /private/operator.json --evidence runs/chips/private/new-attempt
```

入口先检查公开执行环境，再启动 Pi；结束后只有已冻结候选才可交给原评分器。
若 Agent 未确认提交，入口记录 `unsubmitted` 并停止，不调用终评。
每次模型尝试使用新会话、新 job ID 和新 evidence 目录。请求体/次数/输出及墙钟有上限，
**不是人民币金额硬限额**；按服务商计费规则事先选定预算。Pi 的默认零价格不能当作免费证明。
请求扩展抛异常会被 Pi 捕获后继续执行，因此达到次数/体积限制时专用 pilot 进程直接退出 73；
已用模拟服务证实第二次请求不会发送。HTTP 库内部重试和服务商实际账单需另行核对。

`report.json` 分开记录 Agent 终止原因、公开候选和原评分 verdict。电路失败是有效评测结果；
基础设施失败不能算作模型答错。模型费用未核实时记 `not_measured`。
`model-budget.jsonl` 保存请求数量与逐响应 usage；`PiSession.usage` 当前只代表最后一个 assistant 消息，
不能拿它当整次实验的总 token 数。原始轨迹及归档均为私有数据。

## 单次实验记录（Math 风格的 1 题 × 1 次）

`pi` 和 `codex` 入口现在把每个新 evidence 目录当作一个最小评测单元。模型启动前写入
`task_manifest.json`（本次唯一任务和 repeat=1）、`resolved_config.json`（本次实际 operator
配置和 Agent 种类）、`agent-input.json`（Apollo 给 Agent 的 system/prompt、四个公开 Tool
schema，以及是否仍可使用原生工具）。`preflight.json` 保存服务器检查结果。
这些文件和现有的 `operator.json`、`tools/*`、Pi/Codex 轨迹、仿真与终评归档都留在同一
私有 evidence 目录，不搬动既有路径；便于旧的 `recover`/`collect` 命令继续工作。
新目录必须由当前用户持有且权限为 `0700`，已有旧证据文件的目录不能复用。

`results.jsonl` 始终是一行，代表这一题的一次尝试；先登记 `pending`，随后按预检、
Agent 和终评的实际进度原子替换。`summary.json` 从该行得到计划数、完成数和成功率。
只有原评分器返回 `execution=ok` 且 verdict 为 `pass`/`fail`，这一行才是 `completed`，
其中 `benchmark_success` 为 true/false。预检未就绪为 `blocked`，Agent 未确认冻结为
`unsubmitted`，终评仍在运行则为 `pending`；这些情况下分数和成功率均为 null，
不能把未评分当作电路失败。VABench 此处只有 pass/fail 判定，数值 `score` 始终为 null，
不把布尔判定伪装成数学分数。异常只记录类型，不把异常原文或密钥写入账本。
结果行另计 `tool_calls`（有请求记录的公开动作数）和完成时的 `wall_elapsed_seconds`
（从登记到终评收取的墙钟时间，可能包括等待远端作业的时间）；逐动作耗时仍看
`tools/<action-id>/timing.json`。当前不把 Pi 最后一条消息的 usage 误算成整次 token 总量。
进程被强制终止时，账本可能停在 `pending`/`running`；该状态表示最后一次已持久化的
观察，**不证明进程仍在运行**。已有终评作业可用 `collect` 更新同一行，不重跑模型。
对新格式运行，后续 `collect`/`finalize`/`recover` 必须使用与记录完全一致的 operator
配置；若 job ID 或其他字段发生变化，入口会在联系服务器前拒绝，避免把别的作业写入本次记录。

`trajectory_refs` 指向现有证据：Codex 原始 `codex-events.jsonl` 与规范化 outcome，
Pi 的 outcome 和请求预算日志，以及 `tools/*` 的逐动作请求/响应/耗时。
Pi 的 outcome 不是原始服务商请求流；即使保存了 system/prompt 和 Tool schema，
也不能据此声称已还原服务商实际收到的每条请求。`task_manifest.json` 中的任务 ID
先来自 operator 配置，标记为 `declared_by_operator_config`；新版服务器预检回传真实会话的
任务 ID 和 `session.json` SHA-256。若与 operator 声明不符，会在模型启动前阻断；
匹配状态和会话摘要写入结果行。旧部署若未回传这两个字段，状态仍为 `unverified`，
不能把 operator 声明当作服务器证明。旧配置未声明任务 ID 时保留 null，
另记服务器返回的 ID 为 `server_reported`，不冒充预先选定的题目；任务真身还可结合固定 pin
和归档复核。
未来批量入口可以汇总多个这样的单次结果，不需改变 episode 内的证据格式。

## 本机原生 Codex 选项

复制 [Codex 模板](../../examples/chips/benchmarks/vabench/agent.codex.config.example.json)
到 Git 忽略的私有目录，填入本机 Codex CLI、显式模型 ID，以及新建的服务器会话、job ID 和归档路径。
`reasoning_effort` 可显式指定推理档位；入口忽略本机 `config.toml`，因此对照实验不要依赖其中的默认值。
`auth_kind=chatgpt` 用本机 Codex 登录，先运行 `codex --version`、`codex login status`；
`auth_kind=api_key` 则在本机私有环境设置 `OPENAI_API_KEY`。配置 JSON 不放密钥。
与 Pi 一样先在服务器创建新任务会话并执行公开预检，然后在本机运行：

```bash
umask 077
source .venv/bin/activate
python -m alphaapollo.workflows.chips_vabench_agent codex \
  --config /private/codex-operator.json --evidence runs/chips/private/codex-new-attempt
```

入口保存 `auth-preflight.json`、服务器 `preflight.json`、原始 `codex-events.jsonl`、
规范化 `codex-outcome.json` 及 `tools/*` 动作日志。MCP 子进程通过 SSH 调服务器的公开会话；
原始 Codex JSONL 在回合运行中直接写入 0600 私有文件，控制进程中断后已写出的事件仍可检查；
这不等于模型会话可自动续跑，也不保证断电前最后一条记录已落盘。
只有收到 `vabench_submit` 的成功确认才启动独立终评，再下载并校验公开/终评归档。
本机 Agent 中断时，已提交的服务器动作仍可按动作 ID 恢复；整个 Codex 回合不会自动续跑。
若 Codex 超时导致最后一条 JSON 被截断，原始文件仍保留该片段；规范化结果只解析完整记录。

Codex 当前以 `read-only` 本机写入沙箱运行，但其内建读取/命令能力并未像 Pi 一样禁用；
提示词要求使用四个公开 MCP Tool 不构成强制工具隔离。不要把它与 Pi 的仅四 Tool 条件
直接当作单变量对照，也不要在未核对数据外发范围前给它可读的私有材料。
此入口只有回合墙钟超时和服务端公开动作/仿真限制，没有 Pi 扩展的模型请求数/体积硬限额；
`episode-budget.json` 明确将原生 Codex 的模型请求上限标为未执行，不以回合超时冒充请求预算。
`report.json` 的
`model_cost` 仍为 `not_measured`。当前自动化测试只验证构造 CLI/服务器回复与真实 MCP
进程/日志；真实模型结果须另看独立运行记录。2026-09-24 的本机原生 Codex
`gpt-5.6-luna` / `medium` 单任务回合已通过 SSH、EVAS 和独立终评，
详见[验证记录](VALIDATION.md)。这仍是 `development_only` 的一次样本；其冻结候选随后单独
完成了 Spectre 公开波形对照，未改变 EVAS 终评成绩。

## 本机 Pi＋GPT 对照入口

同一 `pi` 入口现在可选 `provider_kind=openai_codex`，使用 Pi 自己的 `openai-codex`
OAuth 登录与显式模型、`thinking` 档位；[配置模板](../../examples/chips/benchmarks/vabench/agent.pi-codex.config.example.json)
没有 `base_url` 或 API Key。`pi_auth_dir` 指向**证据目录之外**、由本人持有且权限为
`0700` 的 Pi 认证目录，其中 `auth.json` 必须为 `0600` 并包含 `openai-codex` 项。
先用相同的 `PI_CODING_AGENT_DIR` 在 Pi 交互界面完成 `/login`，再用
`pi auth check --provider openai-codex --no-refresh` 检查状态；不要输出 token。
本机 Codex CLI 的 ChatGPT 登录不会自动写入 Pi 的 `auth.json`，不能直接替代这一步。
此模式仍禁用 Pi 内建工具，仅通过 MCP 暴露四个 VABench 公开工具；Pi 预算扩展和
`model-budget.jsonl` 已接入，但在 `openai-codex` 服务上的实际请求拦截仍需真实回合验证。
执行命令仍为上面的 `... chips_vabench_agent pi`，
但每个模型回合必须使用新的服务器会话、job ID 和私有 evidence 目录。
截至 2026-09-24，本机尚无 Pi 的 `openai-codex` OAuth 凭据，未声称该条件已跑通真实模型。

## 冻结候选的 Spectre 波形对照

只从已验证的公开 episode 归档取 `candidate/bbpd_ref.va` 与
`public/task/visible_test.scs`，在服务器个人私有目录运行 Spectre；不把 Spectre
结果写回候选或原评分器。下载 PSFASCII 后，用
`python -m alphaapollo.common.execution.chips.vabench_spectre_parity`
的 `--evas-csv`、`--spectre-psf`、`--output` 参数复算公开波形差异。
该工具要求固定的 `data/clk/retimed_data/up/down` 信号集合，先把 Spectre 波形线性插值到
EVAS CSV 时间点，再报告全时段误差与距输入阈值切换至少 0.2 ns 的稳定区误差。
它是额外的跨仿真器观察，不是 VABench 的另一套评分规则；真实样本结果见[验证记录](VALIDATION.md)。

## lab-server 同机 Pi 选项

日常新实验优先使用[服务器端统一配置与准备入口](SERVER_LOCAL_DEPLOYMENT.md)；以下是手工配置方式。

Pi 可以与公开会话、EVAS、最终评分器一起运行在服务器上。此时先在服务器的私有 scratch 中安装
Node/Pi 和隔离 Python 环境（含 `mcp>=1.27,<2` 与 Apollo 基础依赖），部署本分支的 `alphaapollo/`
源码；仿真仍由固定 EVAS 0.8.7 环境执行。参照
[同机配置模板](../../examples/chips/benchmarks/vabench/agent.server-local.config.example.json)
设置 `"transport": "local"`，`python`/`bundle`/`pi_cli`/`session` 都填服务器绝对路径，不填 `host`。
每次新实验仍需新会话、新 job ID、新 evidence 目录。

在服务器自己的交互终端读取模型 Key 到环境变量，再以 `detach` 启动：

```bash
umask 077
read -r -s -p 'Model key: ' CHIPS_MODEL_KEY
printf '\n'
export CHIPS_MODEL_KEY
PYTHONPATH=/private/apollo-source PATH=/private/node/bin:$PATH \
  /private/pi-venv/bin/python -m alphaapollo.workflows.chips_vabench_agent detach \
  --config /private/operator.json --evidence /private/scratch/new-attempt
unset CHIPS_MODEL_KEY
```

`detach` 返回 PID；模型、MCP 子进程及工具调用在服务器继续推进，SSH 断开不终止该次运行。
日志在 `operator.log`，结束标志在 `report.json`；中途可读取 `model-budget.jsonl`、`tools/*/timing.json`
和服务器 job 状态。凭据不写入配置或 evidence，但 Pi 运行期间会出现在该进程的环境中，
所以只在独用账号/私有目录下使用。MCP 子进程不继承凭据。
这仍不是机器重启后的自动恢复：Agent/Pi 进程被杀或服务器重启时，当前已提交的公开动作按同 ID
恢复，整段模型会话需另行设计 checkpoint/续跑。

若服务器到模型服务的 HTTPS 不可达，同机 Pi 无法完成模型请求；此时保留控制机 Pi + SSH 模式。
用 `transport=local` 与 `transport=ssh` 的相同任务 pin、模型配置和新会话分别测量；
每次运行的 `model-budget.jsonl` 含请求/响应时间戳，`tools/*/timing.json` 含工具墙钟时间。
不同模型轨迹的总耗时不等价于纯网络加速，须分别报告模型请求、工具执行和评分/归档耗时。

2026-09-23 的 `v4-001` 单任务探索性对照（GLM 5.3 Flash / Pi 0.87.0 / EVAS 0.8.7）：

| 位置 | 模型请求 | 公开工具 / 仿真 | 模型请求合计 | 工具调用合计 | 自动 Agent / 全程 | 终评 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 本机 Pi + SSH | 7 | 8 / 1 | 131.26 s | 24.10 s | 156.56 / 168.77 s | pass |
| lab-server 同机首轮 | 7 | 8 / 1 | 169.91 s | 5.04 s | 未记录 / 319.00 s* | pass |
| lab-server 同机补跑 | 9 | 10 / 2 | 262.08 s | 7.74 s | 271.00 / 277.00 s | pass |

*首轮在已确认提交后碰到服务器默认 GBK 对 Pi UTF-8 输出的解码错误；人工只执行终评收尾，
未重跑模型或公开仿真。因此 319 秒包含诊断与人工等待，不能拿来比较自动流程。
已在 CLI 输出解码与同机启动环境明确使用 UTF-8；补跑覆盖了自动收尾。
两组 7 请求 / 8 工具的轨迹规模相同，但模型生成内容与服务商延迟不同；补跑还多一次失败仿真和修复。
这些不是随机化重复试验，不能从总墙钟得出网络方案优劣或成功率结论。
所有三组最终公开/终评归档离线校验通过，最终评分都属于 `development_only`。
私有证据保存在 `runs/chips/validation/20260923-vabench-agent/`（Git 忽略）；三组文件中未发现模型 Key 明文。

lab-server 运行中一次 RSS 快照：operator Python 55 MiB、Pi Node 128 MiB、MCP Python 73 MiB；
RSS 含共享页，三者不能简单当成独占内存之和。安装占用约 Pi 426 MB、隔离 Python 环境 132 MB、
Apollo 源码 14 MB，均放在私有 scratch。服务器重启可能清除该 scratch，正式常用部署应按存储规范
把可重建的安装说明与必要版本信息留在持久区，再在 scratch 展开。

## 断线、恢复与存储

每个公开动作先保存请求并启动服务器独立进程，再确认接收。SSH 中断后当前动作继续，
客户端重试同一个 ID，只读取既有结果；执行状态未知时禁止开始另一个动作，不自动重跑仿真。

```bash
python -m alphaapollo.workflows.chips_vabench_agent recover \
  --config /private/operator.json --evidence runs/chips/private/new-attempt --action-id RECORDED_ID
python -m alphaapollo.workflows.chips_vabench_agent collect \
  --config /private/operator.json --evidence runs/chips/private/new-attempt
```

`recover` 读取本地已保存的请求；`collect` 查询终评并下载、离线校验归档，不运行新模型或新仿真。
在 SSH 模式中，模型在控制机运行，因此控制机休眠会暂停后续模型决策；服务器保证当前已提交动作
与已启动终评继续，不承诺整段控制机 Agent 会话在断网后自行推进。同机 `detach` 模式消除这一
控制机依赖；重启/OOM/服务器回收仍需另行调度器方案。

候选与大量小文件在服务器 scratch，公开轨迹先封存至 `archive_root/episodes/<job-id>/`，
最终评分由现有独立 job 写入 `archive_root/<job-id>/`。两份收据通过同一候选摘要关联。
终评归档失败时可用 `job-archive` 单独重试；公开轨迹归档失败时保留冻结候选，修复存储后重试 `finalize`。
下载保留 `report.json`、公开轨迹归档、最终评分归档以及 Pi/工具记录。含隐藏评分的最终包不能给模型读取。
本入口不自动清除 scratch；沿用 [存储规范](STORAGE.md) 的校验后清理原则。

## 可复用的无费用验收

使用新的公开会话与 operator 参考候选，显式运行：

```bash
python tests/chips/probes/pi_vabench.py \
  --config /private/operator.json --pi /absolute/path/to/pi \
  --candidate /private/reference.va --artifact bbpd_ref.va \
  --output runs/chips/validation/new-pi-fixture
```

模拟服务给出固定动作：读取 → 写入错误 → 仿真失败 → 修复 → 仿真成功 → 提交。
Pi、MCP、SSH、EVAS、最终评分及下载都真实执行。模型响应是 fixture，不能计入模型成功率。
`--budget-only` 不需候选，用一次请求的上限验证第二次请求被拦截；也必须提供新的 output 目录。
普通 CI 只执行本地回归，不连接实验室或模型服务。
