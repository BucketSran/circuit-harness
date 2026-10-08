# Chips 测试入口

这里集中保存验收案例、结果模板和可重复使用的巡检探针。
已有行为测试保留在其模块旁边；本目录是统一导航和新增部署测试的归属，不复制同一套断言。
开发遵循 [SOP](../../docs/chips/DEVELOPMENT_SOP.md)，部署遵循 [目录与权限规范](../../docs/chips/DEPLOYMENT.md)。

## 文件与执行边界

| 位置 | 内容 | 默认接触外部服务？ |
| --- | --- | --- |
| [test_spectre_testbench.py](test_spectre_testbench.py) | 确认后编译差分增益候选、原始 AC PSF 独立评分、离线包后台执行和归档校验；构造 Spectre 进程 | 否；真实 Spectre 验证单独记录 |
| [test_task_authoring_agent.py](test_task_authoring_agent.py) | 图像输入/草案阶段边界、确认前拒绝启动、真实 MCP 工具发现；模型回复为构造夹具 | 否；真实模型与服务器预试另记 |
| [test_task_authoring.py](test_task_authoring.py) | 材料来源、缺失/冲突、单位、确认失效与操作者 CLI；构造文档夹具 | 否；不证明文档语义理解或真实专家确认 |
| [test_vabench.py](test_vabench.py) | 构造协议 double：冻结、判定、后台作业、漂移、取消/超时、回收完整性、公开导出 | 否；不是 EVAS 运行证明 |
| [test_analog_design_bench.py](test_analog_design_bench.py) | 固定源码哈希、评分结果故障语义、候选快照和私有归档；构造 Podman 进程 double | 否；真实 ngspice 证据另见[运行说明](../../docs/chips/ANALOG_DESIGN_BENCH.md) |
| [test_analog_public.py](test_analog_public.py) / [test_analog_public_cli.py](test_analog_public_cli.py) / [test_analog_session.py](test_analog_session.py) | 受限候选、公开诊断、离线入口及会话动作/冻结边界；构造 Podman 进程 double | 否；真实标准 bundle 证据另见[运行说明](../../docs/chips/ANALOG_DESIGN_BENCH.md) |
| [test_analog_finalize.py](test_analog_finalize.py) / [test_analog_agent.py](test_analog_agent.py) | 冻结候选到独立终评的哈希门槛、真实 MCP 子进程工具边界和构造 Pi 响应 | 否；真实 GLM 单题回合见[验证记录](../../docs/chips/VALIDATION.md)，不由本地测试运行 |
| [test_episode_report.py](test_episode_report.py) | 离线报告、预算/缺失证据、候选修复、归档损坏拒绝和私有输出；构造夹具 | 否；真实历史归档离线验收另见验证记录 |
| [test_experiment.py](test_experiment.py) | 共同单次实验入口的 VABench/Analog 状态、配置冻结、终评和归档恢复；Agent 与服务器回复为构造夹具 | 否；不调用真实模型、SSH 或仿真器 |
| [test_evaluate.py](test_evaluate.py) | 固定批次准备、整批预检、串行启动与收取、断点拒绝重跑、配置冻结、缺失/损坏证据保留、分任务统计；使用构造的单次结果 | 否；不调用真实模型、SSH 或仿真器 |
| [test_runtime_environment.py](test_runtime_environment.py) | 同一任务 session 经 Apollo 原生 Runtime／外部 Runtime＋真实 MCP 的读写、错误反馈、提交契约；仿真反馈为构造夹具 | 否；不代表真实 Pi／GLM 或电路验收 |
| [test_analog_preflight.py](test_analog_preflight.py) | 构造任务和 Podman/HTTPS 边界，检查启动前状态及公开文件篡改 | 否；不发送真实模型请求或仿真 |
| [test_analog_episode.py](test_analog_episode.py) | 私有会话、Pi 轨迹及终评成员的有限打包、去重和离线校验 | 否；实际服务器打包及本机下载核验见[运行说明](../../docs/chips/ANALOG_DESIGN_BENCH.md) |
| [VABench 真实探针](probes/vabench_smoke.py) | 固定 r53 三任务六个正负例，原评分器与 EVAS | 显式部署后运行，不由 CI 自动执行 |
| [test_host_snapshot.py](test_host_snapshot.py) | 真实本地目录/CLI；构造 curl 响应验证报告和参数边界 | 否 |
| [probes/host_snapshot.py](probes/host_snapshot.py) | 可经 SSH stdin 执行的标准库探针 | 仅显式传 `--url` 才做无认证 HTTPS HEAD |
| [test_storage_compare.py](test_storage_compare.py) | 真实本地文件校验、拒绝覆盖及危险/超限归档；两个目录均为本地测试目录 | 否；不模拟成真实 NFS 性能 |
| [probes/storage_compare.py](probes/storage_compare.py) | 对同一归档交替展开、首次/重复哈希校验，保留每轮证据 | 显式部署后读写指定新目录；不由 CI 自动联系服务器 |
| [既有 Harness 测试](../common/execution/test_chips_harness.py) | 进程、GDS、SSH/EMX 构造夹具、恢复和故障注入 | 否；真实商业仿真器未被调用 |
| [ngspice RC 测试](../common/execution/test_chips_ngspice.py) | 参数、解析验收、执行失败、后台归档/重试/清理、下载完整性、阶段计时、标准库包 | 否；PATH 存在 ngspice 时额外运行真实本地 RC |
| [真实 SSH 断线探针](probes/ssh_detached_rc.py) | 显式部署后终止本次 SSH 客户端，真实 ngspice 独立完成/校验/去重 | 是；仅显式调用，不由 pytest 自动执行 |
| [既有 Workflow/MCP 测试](../workflows/test_chips.py) | 配置、真实 stdio MCP、报告 | 否；不调用模型 |
| [CASES.md](CASES.md) | 实验室、网络、存储及 Agent 对照案例 | 文档，不会自动执行 |
| [RECORD_TEMPLATE.md](RECORD_TEMPLATE.md) | 测试范围、预算、状态与证据记录模板 | 文档 |

正式电路任务定义仍放在 [examples/chips/benchmarks](../../examples/chips/benchmarks/README.md)，
不要把程序回归夹具当作电路 benchmark。

## 新平台的协议与隔离检查

| 测试 | 实际覆盖 | 外部依赖与证据边界 |
| --- | --- | --- |
| `test_candidate_bundle.py` | 候选字节冻结、摘要与收取校验 | 本地文件，不运行评分器 |
| `test_current_evas*.py` | 当前源码身份、公开会话、动作去重、预算、失败及冻结 | 构造任务；部分检查使用本地 Docker 或 Codex 沙箱，不调用模型 |
| `test_public_observations.py` | 摘要、波形窗口、完整公开 JSON 分页、版本兼容、完整性和读取预算 | 构造 Docker CLI；实际文件、子进程和离线包，不运行仿真器或模型 |
| `test_native_sandbox.py` | 原生进程的公开写入、私有路径及网络拒绝 | 需要本地 Codex CLI；不发送模型请求 |
| `test_benchmark_spectre.py` | 原 checker 协议、后台作业、归档及 SSH 客户端契约 | 构造 Spectre 与本地传输，不证明服务器或许可证可用 |
| `test_benchmark_replay.py` | 保存候选回放、条件匹配、比较分类及损坏拒绝 | 真实 Docker 检查使用构造 checker，不是电路一致性结论 |
| `test_harbor_profiles.py` | 独立 Agent 与 Model 选择、协议兼容、stock factory、JobConfig CLI 与凭据引用 | 需要固定 Harbor；只使用 fixture 环境，不请求真实模型 |
| `test_harbor_public_gateway.py` | 本地认证 HTTP、公开请求限制、动作完成与 gateway 关闭 | 构造公开会话与本地 HTTP，不运行模型或仿真 |
| `test_harbor_composition.py` | stock Agent 共享电路环境、公开 shell 客户端与候选冻结 | 固定 Harbor；容器检查需要已存在的本地镜像，不自动拉取 |
| `test_harbor_chips_trial.py` | Harbor Trial、单次 Agent、取消、冻结和独立终评 | 需要 Python 3.12+ 与可选 Harbor 依赖；模型及远端终评为构造夹具 |
| `test_harbor_verifier_archive.py` | 终评完成后等待归档、失败及 deadline，不重复提交 | 受控 transport，使用正式 Harbor 配置 |
| `test_harbor_opensource_verifier.py` | 可选开源终评、有效 0/1、坏结果、取消后实际清理 | 离线配置检查；实际 Docker probes 需显式本地镜像，checker 为夹具 |
| `test_harbor_task_bindings.py` | 多题私有配置选择、并行 Trial 隔离、源码与资源漂移 | 本地真实 Harbor 调度，Agent/环境/checker 为夹具 |
| `test_harbor_deployment.py` | 静态预检、gateway 生命周期、期限与进程组清理 | Docker CLI 夹具和真实本地子进程，不联系模型或服务器 |
| `test_harbor_reporting.py` | 计划分母、有效零分、缺失与损坏记录、独立收据核验 | 离线保存证据；模型 usage/cost 缺失不补零 |
| `test_pi_trajectory.py` / `test_harbor_trajectory.py` / `test_atif_sft.py` | 原生 Pi 到 ATIF 的上下文、调用和可见输出；独立评分关联；SFT 选择、脱敏、去重和 split 边界 | Harbor 与 PyArrow；全部为本地夹具，不请求模型或仿真 |
| [test_atif_dataset.py](../learning/off_policy/test_atif_dataset.py) | Parquet 工具结构、完整模板 token、assistant loss mask、截断拒绝及 verl custom dataset hook | CPU tokenizer 夹具；需 Torch/Transformers，hook 检查另需 pinned verl；CI 独立离线 job |

任务和后端要求见[当前 EVAS 会话](../../docs/chips/CURRENT_EVAS_PUBLIC_SESSION.md)、
[独立评测与回放](../../docs/chips/BENCHMARK_EVALUATION.md)和
[Harbor 适配](../../docs/chips/HARBOR.md)。缺少工具或固定镜像时明确记录 skip，
测试不自动拉取镜像。Docker 的测试目录必须在 daemon 可挂载的文件系统内。
macOS 的虚拟机后端可能不共享默认临时目录，此时把 pytest `--basetemp`
指向 worktree 中一个新的、已被忽略的 `.planning/chips/` 子目录。

## 本地回归

在已安装项目依赖的环境中，按改动选择最小相关测试。Chips 集成回归入口：

```bash
source .venv/bin/activate
umask 077
mkdir -p runs/chips/validation/local
python -m pytest -q \
  tests/chips \
  tests/common/execution/test_chips_harness.py \
  tests/common/execution/test_chips_ngspice.py \
  tests/workflows/test_chips.py \
  tests/reasoning/runtime/test_external_bridge.py \
  tests/workflows/resources/test_composition_contract.py \
  --junitxml=runs/chips/validation/local/junit.xml
```

需要 pytest、Chips/MCP 依赖；真实本地 RTL 检查需要 Icarus，真实本地 RC 检查需要 ngspice。
缺依赖的 skip 必须在报告中注明；服务器固定版本验收按 [ngspice 说明](../../docs/chips/NGSPICE.md) 单独留证。
多次运行请更换结果目录，保留相应输入、版本与记录。CI 同样运行这些本地检查并保留 JUnit 报告。

## 复用主机探针

探针只用 Python 3.10+ 标准库；可选 `getfacl` 采集 POSIX ACL，HTTPS 使用 `curl`。
未安装 ACL 命令时明确记录 unavailable，不宣称没有 ACL。它不读取目录内文件、认证文件或环境密钥，
也不修改权限、安装软件、运行仿真或请求模型。

```bash
# 本机：默认不访问网络
python tests/chips/probes/host_snapshot.py --path /path/to/private-run-root

# 已获准服务器：先把 lab-host 和目录替换为自己的配置；结果仅保存到本人私有目录
umask 077
mkdir -p runs/chips/validation/host-check
ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10 lab-host \
  'python3.12 -B - --path /path/to/private-run-root' \
  < tests/chips/probes/host_snapshot.py \
  > runs/chips/validation/host-check/host.json

# 显式检查一个获准 endpoint；不传入凭据、token URL 或业务数据
python tests/chips/probes/host_snapshot.py \
  --url https://api.openai.com/v1/models --timeout 5
```

输出包含基础 OS/Python/逻辑 CPU 信息、指定路径及父目录元数据/ACL/空闲容量，以及显式选择的 HTTPS 测量。
报告含机器路径，默认只在本地/operator 目录留存，不直接提交或上传。
符号链接会同时记录解析后的路径及其祖先；这只是观察，不是抵御并发路径替换的安全隔离机制。
HTTPS 不跟随重定向，不读取 `.curlrc`，不跳过证书校验；拒绝带 userinfo、query 或 fragment 的 URL。
代理及 CA 环境由操作者管理；探针不自动设置或绕过网络策略。

退出码：`0` 表示请求的路径/网络观察已取得，`1` 表示有采集失败，`2` 表示参数错误。
HTTP 401/403 是已观察到的响应，仍标记 `authentication=not_tested`。
顶层永远标记 `acceptance=not_evaluated`：退出码 0 **不表示目录安全、模型可用或 Harness 验收通过**。
ACL 采集失败保存在对应字段，完整的权限验收需要操作者继续核查。
探针不测磁盘吞吐、GPU、商业软件许可证或真实模型；这些按案例单独执行。

## 长期保留什么

可共享的脚本、夹具、规格、模板和修复后的回归测试进入 Git。
原始主机快照、真实配置、完整仿真结果和模型轨迹进入私有归档，引用其摘要与经过审查的结论。
已有的一次性审查文件继续保留为历史证据，不因提炼了新探针就覆盖或删除。

## 存储对照

使用 Python 3.12+、POSIX 系统；先确认两个父目录均获准使用，并为每次测试选择不存在的新目录：

```bash
umask 077
python3.12 tests/chips/probes/storage_compare.py \
  --archive /private/source-slice.tar.gz \
  --nfs-root /private/nfs/probe-001 \
  --local-root /private/local/probe-001 --repeats 3
```

探针只接受普通文件/目录，拒绝链接、路径穿越、重复成员、空归档及超限输入。
压缩包和展开数据分别限制 32 MiB、成员限制 2,000、重复次数限制 1–5。
整个测量使用同一内存归档，以新目录交替展开，分别测量展开与两次内容校验。
不清系统缓存，因此首次校验不是冷缓存测试；展开时间也不是持久化或物理磁盘带宽。
两种存储名称是操作者提供的标签，报告会另记实际文件系统；不能用本地测试目录声称验证 NFS。
600 秒 alarm 是进程预算，不能保证中断 Linux 内核中不可中断的 NFS I/O。
每轮记录负载、耗时和结果，报告保留到两个测试根目录；失败不会清理现场或自动覆盖重跑。
退出 0 只表示这次源码展开和字节校验通过，不证明仿真、模型或生产环境整体验收。
真实测量、适用边界及管理员反馈见 [存储验证记录](../../docs/chips/STORAGE_VALIDATION.md)。

## 后台存储验收

[存储使用说明](../../docs/chips/STORAGE.md) 定义 RC/VABench 的工作根目录和归档契约。
本地测试使用真实进程与文件系统，构造的 RC 数值/上游 VABench 协议不冒充服务器仿真。
`test_chips_ngspice.py` 覆盖归档写入失败后仅重试归档、损坏包拒绝清理、清理/丢失 scratch 后 ID 去重、
危险 tar 成员拒绝、失败仿真也归档和分阶段计时；`test_vabench.py` 覆盖原负例判定及 replay 计时。

真实 SSH 断线探针新增可选参数，默认路径不变：

```bash
python tests/chips/probes/ssh_detached_rc.py \
  --host lab-host --root /absolute/private/chips-root \
  --work-root /approved/local/jobs --archive-root /private/persistent/archives \
  --bundle /private/versioned/chips.pyz --job-id new-archive-disconnect \
  --output runs/chips/validation/new-archive-disconnect
```

启用归档时，重连的首次检查还要求归档已在重连前完成、离线核验通过、同一 ID 只有一次求解。
探针仅终止自身 SSH 客户端；不是物理断网、NFS 故障或服务器重启测试。

## Spectre RC-001

Spectre RC-001 的构造进程、PSF、私有配置与后台归档回归位于
`test_spectre_rc.py`：`python3 -m pytest tests/chips/test_spectre_rc.py -q`。
真实服务器单任务证据及未验证边界见[标准化说明](../../docs/chips/SPECTRE_RC.md)
与[验证记录](../../docs/chips/VALIDATION.md)。这些测试不运行其他领域评测。

## VABench Agent 闭环

`test_vabench_deployment.py` 检查服务器端私有 profile、唯一 run ID、配置生成、
公开预检失败时不发布可启动的 operator，以及无可选依赖时 Chips 操作者 CLI 的加载。
`test_experiment.py` 检查无执行副作用的 `validate` 命令和 Analog plan／operator 字段
误放时的定位提示。真实服务器准备仍需显式运行，参见
[服务器端配置](../../docs/chips/SERVER_LOCAL_DEPLOYMENT.md)。

新增 `test_vabench_session.py` / `test_vabench_agent.py`：公开文件边界、冻结后拒绝修改、
真实独立动作进程、重复请求去重、未知执行阻断、公开归档完整性、模型凭据与子进程环境分离。
网络重试测试使用明确标注的 double，不冒充 SSH；真实集成使用
[Pi 探针](probes/pi_vabench.py)，命令和职责见 [VABench Agent 说明](../../docs/chips/VABENCH_AGENT.md)。

`test_vabench_codex.py` 覆盖本机原生 Codex 入口的模型/Tool 配置、原始事件归档、
未冻结不得终评、按认证方式隔离 Key、登录预检，以及真实 stdio MCP 子进程的工具发现和动作日志。
两个入口的构造回归还检查 1 题 × 1 次的任务清单、实际输入、结果行、失败状态、
服务器任务 ID 核对，以及终评延迟收取后对同一行的更新；配置漂移必须在联系服务器前被拒绝。
Codex CLI 与服务器回复是构造夹具；MCP 进程和本地日志路径是真实的。
运行 `python -m pytest -q tests/chips/test_vabench_codex.py`。
这不等于真实 Codex 模型、SSH、EVAS 或独立终评验收。

探针通过本机 HTTP 服务提供固定模型响应，真实调用 Pi/MCP/SSH/EVAS 和原评分器。
只允许在新私有会话中显式启动；不能将参考答案驱动的结果记为模型成功率。
模型调用上限回归另用 `--budget-only`：核对服务实际收到的请求数，不仅检查停止日志。

`test_native_probe.py` 验证 `probes/native_vabench.py`：它为 Runtime 迁移对照直接组合
Apollo 原生 Runtime、公开 vaBench Environment 和 OpenAI-compatible backend，
不新增 Agent 循环，也不替代统一实验 CLI。探针固定零 HTTP 重试、记录真实请求体与响应，
按请求字节及模型轮数限额；CLI 父进程另限整个 Episode 时长。仅显式传入私有配置、
新的证据目录和 `--key-file` 后才可进行真实调用；终评仍由操作者在确认冻结后另行执行。

Podman 部署入口回归位于 `test_harbor_podman_runner.py`，使用真实本地进程和构造 CLI，
检查私有存储/API 路由、退出清理与错误保留。公开 Podman 路由与 CPU 配额失败分别由
`test_current_evas_public.py`、`test_harbor_deployment.py` 覆盖；这些夹具不认证真实容器隔离。
真实 rootless 主机条件与命令见 [Harbor 部署](../../docs/chips/HARBOR_DEPLOYMENT.md)。
