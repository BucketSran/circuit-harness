# Analog Design Bench 三题评分接入

`circuit_harness.cli analog-bench` 是操作者侧的固定评分入口。它核对上游任务树的内容哈希，冻结一份候选 `circuit.spi`，在无网络隔离环境中运行该任务原版 `tests/test.sh`，保存输入、评分器日志、`reward.json`、清单和结果，再校验并复制到持久归档。结果区分 `graded`、`verifier_error` 与 `timeout`；`graded` 的 0 分表示评分器真实执行并判定候选不达标。一次命令只评一个候选，不调用模型。

`analog-public` 提供两道 RLC 题的公开诊断入口，由 `--task-id` 选择。它只接收有限的 R/L/C 子电路，核对同一上游任务树的哈希，然后仅挂载公开 testbench、该题声明的公开分析脚本与候选到固定 Podman 环境；`solution/`、`tests/` 和模型密钥不会进入容器。输出保存输入快照、公开仿真日志及测量值，`authority` 为 `public_diagnostic`，`task_correctness` 为 `not_evaluated`。测量缺失或仿真失败会明确报错，不会推算 reward。2026-09-24 在 lab-server 用固定 RLC 镜像运行了 100 MHz 题的参考候选与一个低性能候选，两者均返回 15 个公开测量值；此验证未调用 Agent。2026-09-26 宽带新公开入口也完成了固定镜像下的参考候选和构造低性能候选仿真，两者均返回完整 11 点；100 MHz 参考候选同时复跑并返回 15 项测量。模型闭环的验收与这些操作者正负控分别记录。

## 版本边界

两道 RLC 题固定于 Analog Design Bench 的[初始公开提交](https://github.com/Arcadia-1/analog-design-bench/tree/fb0ec30463d005d3e463caf4e48ab9a26008e869)，OTA 固定于[当前所审查提交](https://github.com/Arcadia-1/analog-design-bench/tree/c23f124de1e461655d2e02ce6cfae2654ccea0d3)。当前 50 题目录已移除这两道 RLC 题，所以这里是三个**分别固定版本的试验**，不是同一榜单的三题成绩。运行器把三个任务树的 SHA-256 和相应上游基础镜像 digest 固定在 `analog_design_bench.py` 中；内容不符即拒绝执行。上游[许可说明](https://github.com/Arcadia-1/analog-design-bench/blob/main/LICENSE)对软件和 benchmark 内容分别规定 Apache-2.0 和 CC BY-NC 4.0；本仓库不复制原题、参考网表或评分器。

准备上游源码时，把两次 `git archive` 展开到操作者私有的同一个 `$CHIPS_SOURCE_ROOT`，形成 `tasks/<task-id>/`；保留原始提交和导出的 tar 校验和。不要把 `solution/`、`tests/` 或归档目录挂载进 Agent 工作区。macOS 二次打 tar 可能额外生成 `._*` 文件并导致哈希不符；应直接转移 `git archive` 的 tar，或以只收录普通文件的脚本重新打包。

## 运行

在 Linux 仿真服务器安装 Python 3.10+、rootless Podman；原生模式还需要 `bwrap` 和 ngspice。构建并部署本仓库的标准库 zipapp：

```bash
python -m circuit_harness.cli bundle --output chips-agent.pyz
```

每次使用新的 `$CHIPS_RUN_ROOT/<run-id>`；`$CHIPS_ARCHIVE_ROOT` 是本人可写、他人不可读的持久目录。下面以固定基础镜像在服务器的离线导入副本为例。先按 `TASKS` 固定的上游 digest 下载正确架构的镜像，保存 tar 并核对传输 SHA-256；在服务器用 `podman load -i "$IMAGE_TAR"` 导入，查得本地镜像的 `sha256:` ID，再运行：

```bash
umask 077
python3.12 chips-agent.pyz analog-bench \
  --task-id rlc-rf-bandpass-100mhz \
  --source-root "$CHIPS_SOURCE_ROOT" \
  --candidate "$CHIPS_CANDIDATE" \
  --output "$CHIPS_RUN_ROOT/rf-run-001" \
  --archive-root "$CHIPS_ARCHIVE_ROOT" \
  --backend podman \
  --runtime-image "$RUNTIME_IMAGE_ID" \
  --offline-image-archive "$IMAGE_TAR" \
  --podman-single-id --podman-no-cpu-limit
```

Agent 可以反复调用的公开诊断命令使用另一块全新输出目录，不带评分器和终评归档：

```bash
python3.12 chips-agent.pyz analog-public \
  --task-id rlc-rf-bandpass-100mhz \
  --source-root "$CHIPS_SOURCE_ROOT" \
  --candidate "$CHIPS_CANDIDATE" \
  --output "$CHIPS_RUN_ROOT/public-001" \
  --runtime-image "$RUNTIME_IMAGE_ID" \
  --offline-image-archive "$IMAGE_TAR" \
  --podman-single-id --podman-no-cpu-limit
```

`analog-public` 是供操作者直接调用的单次诊断命令；下文的受控会话另有 Pi Tool 桥接。
标准 `chips.pyz` 的此入口已在 lab-server 用固定镜像和低性能候选执行，返回 15 个公开测量值，且结果归档保存在本人私有目录；这不代表 Agent 解题成功。

### 用任务配置复用同一条 RLC 链路

| `task_id` | 候选接口 | 公开诊断 |
| --- | --- | --- |
| `rlc-rf-bandpass-100mhz` | `.subckt rlc_rf_bandpass IN OUT COM` | 上游 `tb_ac.spi`、`tb_stopband.spi` 的标量测量 |
| `rlc-broadband-50-to-200-match` | `.subckt rlc_broadband_match IN OUT COM` | 上游 `analyze_broadband.py`：3.3–3.8 GHz 的 11 点有限 Q 扫频 |

两题复用同一套六个 Tool、会话实现、英文 system prompt 和 Pi 启动能力。
宽带题执行原公开分析脚本的有限 Q 转换，不用理想器件直接跑 testbench 来代替它；
公开反馈给出逐点 `gamma`、`transducer_gain` 和汇总指标。扫描点缺失、非法值或频率不符
记为 `simulation_error`。101 点扫描及容差组合仍由原隐藏评分器在冻结后执行。
公开诊断执行成功只说明数据完整，不说明电路达标。

任务契约在 [任务注册表](../../circuit_harness/execution/analog_design_bench.py) 的
`TASKS[task_id].public_rlc` 声明子电路名、端口、公开文件和诊断入口；
[公开运行器](../../circuit_harness/execution/analog_public.py) 执行并解析诊断，
[会话](../../circuit_harness/execution/analog_session.py) 管理候选、动作、冻结和恢复，
自动 Agent runner 已退役，见[迁移说明](MIGRATION.md)；下面描述保留的操作者会话。

创建宽带会话时，在 `analog-session` 中显式使用
`--task-id rlc-broadband-50-to-200-match`，其余镜像与预算参数同原 RLC 会话。
把同机 operator 模板复制到私有目录，设置同样的 `task_id` 和新会话路径；
再复制 旧实验模板已退役，参见迁移说明，
填写 operator 路径及新的运行、终评和归档目录。实验配置、operator、会话的任务 ID
必须一致，否则在启动模型前拒绝。旧 operator 未填写 `task_id` 时保留 100 MHz 默认。

新会话为 schema v3，记录 `task_contract_sha256`；任务接口发生漂移时拒绝继续。
旧 v1/v2 仅兼容原 100 MHz 题，不允许把历史会话改名为宽带题。
两题的真实上游文件已通过本地离线部署包的建会话、读写、历史恢复、提交和归档校验；
此本地检查没有仿真或模型请求。2026-09-26 同版部署包的真实服务器公开仿真另行通过，
宽带参考候选原终评为 7/7、1.0 分，构造低性能候选为 1/7、0.1 分；
后者有结构检查的部分分，不能把“负控”一律解释成零分。
OTA 仍只有操作者评分入口，未接入该被动 RLC 协议。

`analog-session` / `analog-action` 提供会话层的六个受控动作：读公开题面与 testbench、写入受限 R/L/C 候选、运行上述公开诊断、查看候选历史、恢复候选、冻结提交。每个动作先保存带 ID 的请求再执行；同 ID 重放返回原响应，执行状态不明时阻止后续修改。会话限制总动作数及仿真次数，保存每次输入、响应、耗时和仿真原始记录。创建命令使用与 `analog-public` 相同的固定镜像选项，并额外指定 `--max-actions`、`--max-simulations`；随后每次通过 `analog-action --session "$SESSION" --request request.json` 调用，其中请求必须包含 `id`、`tool`、`arguments`，Tool 名为 `analog_read`、`analog_write`、`analog_simulate`、`analog_history`、`analog_restore`、`analog_submit`。也可用 `--request -` 从标准输入接收 JSON。

跨 SSH 使用时，`analog-request --session "$SESSION" --request request.json` 先把请求留在服务器并启动独立子进程，返回 `accepted`；之后在新的连接中用 `analog-response --session "$SESSION" --id "$ACTION_ID"` 查询。重试必须使用同一 ID 和完全相同的请求；若启动状态不明，返回 `unknown_execution` 并停止新动作。2026-09-24 在 lab-server 完成了独立写入、公开仿真和冻结；重发同一仿真请求后，服务器仅有一个仿真目录，轨迹已校验并保存在私有归档。测试覆盖了 SSH 命令退出后的服务器继续执行，未人为切断正在运行的连接。

2026-09-24 在 lab-server 用标准离线 bundle 完成了一次操作者驱动的 `read → write → simulate → submit`：公开仿真返回 15 个测量值，提交后保存了候选 SHA-256；会话轨迹已逐文件核对并保存在本人私有目录。这验证了服务器上的会话命令链，但**不是** Pi/GLM 自主解题，终评也未运行。

`analog_history` 无参数，返回本会话保存的候选 SHA-256、当前候选摘要及各版本的公开仿真反馈；列表按摘要排序，不是质量排名，也不含隐藏评分。`analog_restore` 接收 `{"candidate_sha256":"<64 位摘要>"}`，核验已有快照后恢复完全相同的字节。两个动作消耗总动作预算，恢复不消耗仿真次数；提交或结束后拒绝修改。Agent 应根据全部公开指标比较候选，而非只看中心增益。平台仍收取当前版本，不自动用历史版本替换它。历史会话没有快照的版本不支持直接恢复。

### 独立终评

会话必须已确认 `analog_submit` 或具有服务器签发的结束收取回执、且 `frozen/circuit.spi` 与冻结哈希一致时，操作者才运行原评分器：

```bash
python3.12 chips-agent.pyz analog-finalize \
  --session "$SESSION" \
  --output "$CHIPS_RUN_ROOT/final-001" \
  --archive-root "$CHIPS_ARCHIVE_ROOT"
```

该入口调用固定上游 `tests/test.sh`，评分输入快照必须与冻结候选逐字节一致；结果不会回送 Agent。2026-09-24 对上述操作者驱动的低性能候选实跑，原评分器返回 `graded`、0/15、0.0 分，容器退出码为 0；冻结候选与评分输入 SHA-256 一致，私有归档逐文件比对通过。这是有效负例，不是模型成绩。

冻结后可用 `analog-archive --session "$SESSION" --archive-root "$CHIPS_ARCHIVE_ROOT" --episode-id <new-id> --final-output <final-run>` 封存会话；真实 Agent 回合再加 `--agent-evidence <pi-evidence>`。归档只选公开任务材料、各动作请求/响应/耗时、仿真记录、历史候选快照、冻结候选、Pi 原始 JSONL 事件流与规范化结果、预算记录、终评产物，不复制 Pi home、工作目录或密钥文件。原始流在解析前保存到私有 `pi-events.jsonl`，并对当前模型 Key 作文本替换；它记录 Pi 实际输出的事件，不声称能观察未输出的模型内部推理。`verify-analog-episode <archive-dir>` 会离线核对包哈希与每个成员，归档后才考虑清理 scratch。2026-09-24 的操作者负例封存 41 个文件，服务器端与下载至本机后均通过同一验证；此包不含真实模型轨迹。归档内含原评分结果和可能敏感的服务器路径，只供操作者私有保存，不能作为 Agent 公开输入。

把 `--task-id` 换为任务卡中的另两个 ID 即可选择对应评分器；RLC 与 OTA 使用不同的上游基础镜像。`--podman-single-id` 只用于 rootless 账户没有多 UID 映射的主机，`--podman-no-cpu-limit` 只用于 CPU cgroup 未下放的主机；后者保留 2 GiB 内存限制，但不强制 4 CPU 配额。若 Podman 的存储放在本地盘，还需同时传 `--podman-root` 和 `--podman-runroot`；加载镜像时也要用相同存储参数。没有离线 tar 时，可直接让 Podman 使用固定的上游镜像 digest，省略 `--runtime-image` 和 `--offline-image-archive`。若使用离线 tar，运行器会记录上游 digest、本地镜像 ID 和 tar SHA-256；操作者仍须核对该 tar 确实来自固定 digest。

Podman 在无网络容器内挂载原版 `tests/test.sh`、候选快照和专用结果目录，并用只读根文件系统运行。这里使用的是**固定的上游基础镜像加挂载的原版任务脚本**，不是上游逐题构建的完整验证镜像。`--backend bubblewrap --ngspice "$NGSPICE_BIN"` 是无容器的替代路径；它保留上游评分脚本但使用指定的本机 ngspice 版本，不等同于固定镜像环境。OTA 的原生模式还要求上游 Sky130 模型和 checker 已在预期 `/opt` 路径，缺失时在仿真前报错。

本任务的开发约定：Bubblewrap 用于快速 smoke、故障定位和与镜像结果核对；正式对照固定使用上述 Podman 镜像路径。每次运行只选一个 `--backend`，并在报告中保留后端与版本。Agent 不选择后端；`analog-bench` 始终是操作者侧评分器，公开诊断与会话动作另走上述入口。参见 [Chips 开发 SOP](DEVELOPMENT_SOP.md#3-区分测试层次)。

`<run-id>/manifest.json` 记录源码提交/哈希、候选哈希、后端、镜像 digest 和原生仿真器版本；`inputs/circuit.spi` 是实际输入快照；`verifier.log` 与 `verifier/` 保留原始诊断和评分文件；`result.json` 给出执行状态、得分和归档位置。归档复制后逐文件校验；失败时保留 scratch 供恢复，不自动删除。原始输入、日志和参考电路留在私有归档，公开文档只发布经审核的摘要。

## 2026-09-23 lab-server 试验

本次用 rootless Podman、固定的上游基础镜像和原版评分脚本，在 lab-server 对三道题各跑参考解与公开 starter。参考解是环境正控，绝不是 Agent 成绩；starter 是负控。服务器直连 GHCR 拉取镜像曾超时，随后改为下载固定 digest 的 amd64 镜像、校验 tar 后离线导入。lab-server 的 rootless Podman 仅有单 UID/GID 映射且 CPU cgroup 未下放，因此本次启用了上述两个兼容选项。两道 RLC 题也在服务器用原生 ngspice 47 + bubblewrap 单独复现过相同分数。

| 任务 | 参考解 | starter | 证据级别 |
| --- | --- | --- | --- |
| 100 MHz RLC 带通 | 15/15，1.0 | 0/15，0.0 | 原评分脚本 + 固定 RLC 基础镜像 |
| 3.3–3.8 GHz 匹配 | 7/7，1.0 | 0/1，0.0 | 原评分脚本 + 固定 RLC 基础镜像 |
| 五管 Sky130 OTA | 7/7，1.0 | 0/7，0.0 | 原评分脚本 + 固定 Sky130 基础镜像；63 个分析点 |

六次固定镜像运行的原始归档在 Git 忽略的 `runs/chips/validation/20260923-analog-design-bench/server-archives/`；两个 RLC 任务的原生运行及早期环境故障日志也保存在该私有验证目录。六次结果状态均为 `graded`，而不是评分器故障。此处没有 Pi/Codex 轨迹或模型预算比较。此后另行完成了公开诊断、冻结会话、Pi Tool 桥接、后台动作恢复及私有 Episode 归档；2026-09-24 又实测了 broadband 任务的原评分器正负控，以及 100 MHz 任务的真实 Pi/GLM 回合：三次截断未提交、一次低推理档位完成自主仿真与提交并通过终评、一次低推理档位因总预算耗尽未提交。各回合必须按模型成绩与环境对照分别记录。


v2/v3 会话采用 `collection_policy=episode_end`：Agent 主动提交记为
`collection_source=agent_submit`；正常结束或声明的预算停止后，由操作者通道
`analog-close --session <session> --reason <reason>` 冻结最后完整候选，记为
`collection_source=episode_end`、`agent_submitted=false`。没有候选时记为
`missing_candidate`，不构造分数。收取不向 Agent 暴露评分器，也不伪造工具调用。
原题说明单独保留；新建会话的可读说明在完整原题前后加入明确的 Harness 适配说明，
声明受控工具路径与结束规则，不再依赖替换原题中的特定句子。

会话有运行中或状态未知的动作时返回 `awaiting_action_recovery`，不冻结候选。
操作者恢复原动作后可用 `analog-close` 重试收取；该操作不会启动模型。
冻结／收取及原始 Agent 轨迹分别归档，独立终评仍需 `finalize`。
旧 v1 会话保留显式提交规则；已完成的历史实验不会自动重新评分。
