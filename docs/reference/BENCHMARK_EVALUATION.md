# 冻结候选终评与开源重评

本页定义操作者使用的候选冻结、benchmark 任务包、Spectre 终评、SSH 传输和保存候选重评接口。
Harness 管理执行与证据；vaEVAS benchmark 声明任务、checker、判据、任务集合和后端映射。
这些入口不属于 Agent 的公开工具。

## 输入与任务包

[`candidate_bundle`](../../circuit_harness/execution/evaluation/candidate_bundle.py) 提供：

```python
freeze_candidate(source, destination, files, *, task_id, task_version, reason)
verify_candidate(directory)
```

冻结目录包含 `manifest.json` 和 `files/<声明的相对文件名>`。冻结保留原始字节，不检查语法，也不选择旧候选。
稳定 `candidate_sha256` 包含任务 ID、任务版本和文件清单，排除 `reason`。
验证重新计算文件摘要，拒绝路径穿越、链接、清单差异和超限文件。

任务包是 benchmark 提供的目录，包含 `manifest.json` 和显式文件清单。
[`package_identity`](../../circuit_harness/execution/evaluation/benchmark_spectre.py) 拒绝多余或缺少的 manifest 字段。

| 字段 | 契约 |
| --- | --- |
| `schema_version` | `1` |
| `task_id`, `task_version` | 非空 benchmark 声明；必须与冻结候选一致 |
| `criteria_sha256` | benchmark 声明的 64 位小写 SHA256 判据摘要 |
| `condition_id` | 非空的 benchmark 测试条件 ID；用于比对行为要求、输入与条件 |
| `task_set` | `public` 或 `extension`；表示公开主集或扩展集，不表示 Agent 可见性 |
| `purpose` | `final` 或 `public`；必须与执行请求一致 |
| `entrypoint` | 包内 shell 入口的相对路径，例如 `tests/test.sh` |
| `candidate_file` | 冻结清单中的候选相对路径，例如 `dut.va` |
| `report_path` | checker 输出的相对路径，例如 `verifier/report.json` |
| `files` | 包内全部输入文件，排除 manifest；每项包含 `sha256` 和 `bytes` |
| `feedback_fields` | `final` 必须为空；`public` 显式声明可返回的测量字段 |

平台固定任务包的实际字节摘要。不同后端可以使用不同包，但比较时必须保持候选、任务版本、判据、测试条件和任务集合一致。
平台不从相似文件名推断任务映射，也不替 benchmark 生成判据摘要。
包内文件不能占用冻结候选的 `candidate/` 工作区，输出路径不能覆盖输入。

checker 使用原有 `test.sh` 环境契约：`CANDIDATE` 指向候选文件，`VERIFY_OUTPUT` 指向 report 所在目录。
Spectre 执行还提供 `SPECTRE`。report 的 `candidate_sha256` 必须是该候选文件的 SHA256；
平台结果中的同名字段是完整冻结 bundle 的稳定摘要，两者用途不同。
可选的 `checker_sha256`、`cases_sha256` 必须对应任务包清单中的文件。

评分只消费 checker 的结构化证据。`completed` report 必须包含有限的 `reward`（0–1）和非空 `cases`，
每个 case 必须包含布尔 `passed`。现有 waveform checker 用 `status=graded` 表示完成数值判定；
triangle report 用 `returncode=0`、`timeout=false` 和 `waveform_sha256` 表示完整仿真证据。
`submission_contract_violation` 且 `reward=0` 是明确的任务零分。
其他编译失败、缺波形或仿真超时不会仅凭日志或退出码变成模型零分。
进程完整结束且 report 明确、完整地给出任务判定时，非零 checker 退出码不会清空该分数。
超时、清理失败等未完成执行仍保持未评分。

## 提交与恢复 Spectre 作业

服务器上的私有 profile 是操作者拥有、权限不向组或其他用户开放的普通 JSON 文件，字段如下：

| 字段 | 契约 |
| --- | --- |
| `schema_version` | `1` |
| `shell` | `/bin/sh` 或 `/bin/csh` |
| `setup_scripts` | 最多四个现存绝对文件路径；由所选 shell 加载 |
| `spectre` | 现存且可执行的绝对文件路径 |
| `preflight_script` | 必填的部署预检 shell 脚本绝对路径；普通现存文件 |
| `run_root`, `archive_root` | 已存在、操作者拥有的私有目录；两者不能重叠 |
| `timeout_s` | 有限值，`0 < timeout_s <= 1800` |
| `max_output_bytes` | 整数，1–256 MiB |

提交固定 profile、shell、setup scripts、预检脚本和 Spectre 可执行文件的字节摘要。
文件预检先检查路径、权限和声明范围；随后有限执行部署提供的 `preflight_script`，通过后才启动 checker。
实际部署必须提供能检查本任务所需许可证、PDK 与依赖的 probe；平台不把 `spectre --version` 当作许可验证。
构造测试脚本的 `passed` 不证明真实部署已就绪。
执行不会继承模型凭证；shell setup 与 checker 都必须由操作者信任。
进程时限和输出配额不是服务器访问沙箱。

预检用 `/bin/sh` 执行脚本，并按 profile 加载 setup scripts。
环境提供 `SPECTRE`、固定任务包目录 `TASK_PACKAGE` 和输出文件路径 `PREFLIGHT_OUTPUT`。
脚本在 `PREFLIGHT_OUTPUT` 写入以下结构化报告，所有 checks 必须明确 `passed` 才启动 checker：

```json
{
  "schema_version": 1,
  "checks": [
    {"kind": "license", "name": "required license", "status": "passed", "detail": "deployment probe evidence"},
    {"kind": "dependency", "name": "task dependencies", "status": "passed", "detail": "deployment probe evidence"}
  ]
}
```

上面的内容定义字段形状，不能替代实际 probe 的输出。
报告必须包含至少一个 `license` 和一个 `dependency` check，可另列 `simulator`。
`kind`、`name` 组合不能重复；`name` 非空，`detail` 为字符串。
每项 `status` 只能为 `passed`、`failed` 或 `unknown`。最多 64 项，报告不超过 64 KiB。
明确失败分别保留为 `license_unavailable`、`dependency_unavailable` 或 `simulator_unavailable`；
不明确检查为 `preflight_unknown`，缺失或损坏报告为 `invalid_preflight`，均不记模型零分。
报告、进程日志和实际脚本摘要随作业封存，离线验证重新核对结构化判定。
预检成功要求完整的 passing 报告和退出码零。检查均通过但脚本非零退出为 `invalid_preflight`；明确失败的检查保留 license/dependency/simulator 分类。

预检与 checker 共用同一个总 execution deadline、取消信号和目录输出配额；预检不会获得额外预算。

在执行主机调用：

```bash
python -m circuit_harness.cli submit-benchmark-spectre \
  --candidate saved/frozen --task-package benchmark/final \
  --profile operator/spectre.json --job-id eval-001
python -m circuit_harness.cli job-status operator/jobs/eval-001
python -m circuit_harness.cli verify-job operator/jobs/eval-001
python -m circuit_harness.cli verify-archive operator/archive/eval-001
```

profile 内的根目录使用绝对路径；上面 `job-status` 的目录应指向 profile 的实际 `run_root/eval-001`。
也可以把模块命令换成部署到服务器的 CLI zipapp。
默认 `purpose=final`，原有 RC、gain 和 VABench CLI 默认行为保持不变。

Python 入口为 `jobs.submit_benchmark_spectre(candidate, task_package, profile_path, job_id, *, purpose='final')`。
ack 包含 `job_id`、`state` 和 `directory`。`running` 只表示 worker 已接手；`finished` 包含 `result`。
重复相同 ID 和相同请求只查询原作业；变更输入或运行条件会拒绝该 ID。
`unknown` 表示执行状态不明确，不能换 ID 隐式重启。
工作目录过期后，持久归档注册仍阻止重复执行。归档重试不重跑 checker。

`job-cancel` 是显式服务器取消请求。客户端停止等待或 SSH 断线不会取消 detached worker。
`job-cleanup` 只在归档已独立校验后释放工作文件。

## 通过 SSH 提交与回收

[`RemoteBenchmarkSpectre`](../../circuit_harness/execution/transport/benchmark_remote.py) 复用现有 SSH 传输和相同的服务器 jobs 生命周期。
配置包括 SSH config alias `host`，以及服务器绝对路径 `python`、`bundle`、`profile`、`run_root`、`archive_root`、`upload_root`。
`evidence` 参数是客户端私有证据目录。

```python
transport = RemoteBenchmarkSpectre(config, evidence)
ack = transport.submit(candidate, task_package, "eval-001")
state = transport.query("eval-001")
result = transport.retrieve("eval-001")  # finished 且归档已发布后调用
```

上传通过 `stage-benchmark` 接收显式文件内容，在私有内容摘要目录中验证冻结候选和任务包；不解压调用者的 tar。
服务器 `submit-benchmark-spectre` 再次冻结作业输入。SSH 启用非交互模式和严格 host-key 检查。
丢失回复后恢复同一 ID；查询不提交新候选。`interrupt_wait()` 停止当前客户端命令等待。
现有 artifact download 有独立传输时限，不承诺可由该方法即时中断。
`retrieve()` 下载 receipt 与归档，离线核验全部成员，再核对作业 ID、候选、包内容和私有 profile 路径。
回收材料可能包含隐藏测试或评分细节，只进入操作者证据。

## 保存候选的开源单后端重评

[`replay_candidate`](../../circuit_harness/execution/evaluation/benchmark_replay.py) 消费既有冻结 bundle、benchmark 声明的开源 checker 包和独立输出目录。
不重新生成、修复或选择候选；输出目录存在时拒绝覆盖。
不开 SSH，不探测 Spectre、许可证或模型。可选 Spectre 对照只读取已保存的 job 或 archive。

replay 配置严格包含以下字段：

| 字段 | 契约 |
| --- | --- |
| `schema_version` | `1` |
| `image` | 已安装的 `sha256:<image ID>` 或 `repo@sha256:<digest>`；不允许浮动 tag |
| `task_package_sha256` | benchmark 声明的当前开源任务包实际摘要 |
| `solver_options` | 显式选项对象；传入 `CHIPS_SOLVER_OPTIONS`，非空时 report 必须确认相同对象 |
| `unsupported` | benchmark 声明的不支持分析列表；非空时保留未评分记录且不启动后端 |
| `timeout_s` | 有限值，`0 < timeout_s <= 300` |
| `max_output_bytes` | 整数，1–16 MiB |

容器镜像必须包含 Python 3，以运行独立时限 watchdog。
执行复用 [`run_isolated_docker`](../../circuit_harness/execution/sessions/current_evas_public.py)：
固定 `--pull=never`、无网络、只读容器根、丢弃 capabilities、禁止提权和资源上限。
候选和任务包只读挂载，只有输出目录可写；不挂载 Docker socket 或操作者私有文件。
调用结束清理容器，broker 消失后内部 watchdog 仍使 checker 超时退出。
Docker daemon 必须能访问所声明的挂载目录；缺失挂载不会得到任务成绩。

```bash
python -m circuit_harness.cli replay-benchmark \
  --candidate saved/frozen --task-package benchmark/opensource \
  --config operator/replay.json --output evidence/replay-v2
python -m circuit_harness.cli verify-benchmark-replay evidence/replay-v2
python -m circuit_harness.cli summarize-benchmark-replays evidence/replay-v2
```

若有身份匹配的旧 Spectre 结果，增加 `--spectre-record evidence/spectre-archive`。
平台先核验并复制该记录，不改旧成绩，也不重跑商业后端。
比较拒绝不同候选、任务、版本、判据、测试条件或任务集合。

## 结果与比较分类

执行结果保留 `execution`、`verdict`、`score`、候选与任务身份、判据、测试条件、任务集合和实际包摘要。
`certified=false` 不表示 task score 无效；表示这些接口自身不认证实验或电路能力。

| 状态 | 处理 |
| --- | --- |
| `ok` | 有完整 checker 证据；final 才保留原 `reward` 为 `score` |
| `infrastructure_error` | 工具或 checker 基础设施失败；`score=null` |
| `license_unavailable`, `dependency_unavailable`, `simulator_unavailable` | 部署预检明确失败；不启动 checker，`score=null` |
| `preflight_unknown`, `invalid_preflight` | 部署预检不明确或证据损坏；不启动 checker，`score=null` |
| `timeout`, `cancelled`, `output_limit`, `cleanup_failed` | 有限执行或清理失败；保留现场，`score=null` |
| `invalid_result` | report 缺失、损坏、身份不符或结构不完整；`score=null` |
| `unclassified_failure` | 现有结构不能区分候选错误与环境问题；`score=null` |
| `unsupported_analysis` | 声明范围不能完成该分析；不启动后端，`score=null` |

replay 的 `classification` 为 `match`、`false_accept`、`false_reject`、`infra`、`unevaluable` 或 `not_compared`。
完整通过的开源结果与未完整通过的 Spectre 结果对应 `false_accept`，反向对应 `false_reject`。
分数相同对应 `match`；两者都未完整通过但部分分不同对应 `unevaluable`。
执行故障优先保留 `infra`，其他未评分结果保留 `unevaluable`。
成功的开源结果没有 Spectre 对照时为 `not_compared`，原始 `spectre` 字段为空。
这些名称描述相对既有 Spectre 判定的差异，不把 Spectre 波形自行定义为新答案。

汇总先核验每个 replay receipt，保留失败记录，分别统计 `public` 和 `extension`。
同时报告记录数与不同候选数，拒绝重复引用同一个记录目录；不从重复重评次数推断解题成功率。

## 可见性与验收范围

`purpose=public` 要求独立公有任务包，并只投影显式 `feedback_fields`。
评分、verdict、cases、report、logs、artifacts 等字段不能声明为公开反馈，嵌套私有字段也会拒绝。
旧的宿主执行路径仅供操作者使用，不能作为 Agent 的公开仿真接口。
配置 B 的 Agent 接口使用[隔离 Docker 路径](#isolated-public-spectre-jobs)或[Spectre 进程 namespace](#spectre-process-namespace)。已有本地 Docker 和受控 Harbor Trial 夹具，以及真实 namespace、SSH 和 Spectre 参考探针证据；独立评分的实际模型 Trial 尚未验收。
最终任务包、report 和 archive 不进入公开工具或 Agent 工作区。

本地子进程夹具验证后台执行、同 ID 去重、unknown 不重跑、超时、malformed report 和归档完整性。
实际 Docker 夹具验证只读候选、私有主机文件不可见和网络隔离；构造 checker 不证明 EVAS 算法或商业仿真正确。
测试入口为 `tests/chips/test_benchmark_spectre.py` 和 `tests/chips/test_benchmark_replay.py`。
容器夹具通过 `CHIPS_TEST_DOCKER_IMAGE` 选择已安装的不可变镜像。

目前仍缺 benchmark 正式任务映射、真实 Spectre/许可证/SSH 接入证据，以及开源与 Spectre 的正负控和真实模型候选样本。
不能据夹具宣称所有任务支持、配置 B 验收完成或跨后端判分等价。

## Isolated public Spectre jobs

Agent public requests use `--purpose public --isolated-public`. The server rejects
this request with a host profile. Existing operator public jobs and final profiles
retain their prior default behavior. The Agent adapter cannot register final evaluation
as a public tool. `stage-benchmark --purpose public` validates the public package;
the default purpose remains final.

A private Docker public profile has exactly these fields:

```json
{
  "schema_version": 1,
  "backend": "docker",
  "image": "sha256:<64 lowercase hexadecimal digits>",
  "spectre": "/image-runtime/bin/spectre",
  "preflight_script": "/image-runtime/public-preflight.sh",
  "run_root": "/operator/public/jobs",
  "archive_root": "/operator/public/archive",
  "timeout_s": 120,
  "max_output_bytes": 16777216
}
```

Paths and the image above are placeholders. The image must already exist locally,
contain Python 3, the declared runtime and all required dependencies. The operator
must supply an image containing no private final task or model credentials. The preflight
script may also be a declared file under `/task`; it runs inside the same container
boundary as the public checker. No host setup or license probe executes candidate
input. The structured preflight must establish actual license and dependency checks
before the checker runs. A synthetic passing report proves only the fixture protocol.
No commercially licensed deployment has been validated for this boundary.

Containers have no network, a read-only root, no capabilities, no privilege escalation,
32 PIDs, 512 MiB memory with no extra swap, one CPU, bounded files/output, and a finite
in-container watchdog. Only `/candidate` and `/task` are mounted read-only; only
`/output` and `/preflight` are writable. The checker works in `/task`, with `CANDIDATE`
and `VERIFY_OUTPUT` pointing to these mounts. Scripts needing writable task directories
must use the declared output directory instead. The final package, host profile, model
credentials, host simulator installation and other job directories are never mounted.
Explicit job cancellation reaches both public preflight and checker and removes the
container. Frozen input/profile identity is checked before execution and sealing;
archive validation checks the recorded image and concrete mount/resource constraints.

Network license servers are unsupported under this contract. A deployment requiring
networked licensing needs a separately implemented and verified access policy. There
is no host-network or host-process fallback, and no Boolean declaring a deployment
verified. Supporting this bounded executor does not establish that a particular Spectre
installation can obtain a license with networking disabled.

## Spectre process namespace

`spectre-isolate` lets a trusted host checker launch the candidate's Spectre process
in a Linux Bubblewrap namespace. The checker must prepare a separate condition
directory containing only the candidate, its relative include files and that
condition's netlist. The process sees that directory and explicitly declared
simulator runtime paths; the checker, hidden truth and other task packages stay
outside those mounts. Existing inputs are read-only, while new compiler artifacts,
logs and waveforms can be written in the condition directory. Candidate source is
executed as supplied; this interface imposes no Verilog-A language subset.

```sh
python harness.pyz spectre-isolate --config /private/isolation.json -- \
  -64 tb.scs +log spectre.log -format psfascii -raw psf +mt=1
```

The private configuration is an owned regular file with no group/world access.
Its exact schema is version 1 and requires these fields:

```json
{
  "schema_version": 1,
  "bubblewrap": "/usr/bin/bwrap",
  "spectre": "/opt/spectre/tools/bin/spectre",
  "runtime_readonly_paths": ["/opt/spectre", "/usr", "/lib", "/lib64", "/bin"],
  "network": "disabled",
  "license_env": []
}
```

Runtime grants must include the executable's resolved location, its libraries,
standard models and shell interpreters. Some systems also need a narrow grant
for `/etc/alternatives` or DNS configuration. Operators must exclude private
configuration, home directories, checker packages and secrets from these grants.
The launcher rejects filesystem-root grants, condition/runtime overlap, source
symlinks and configuration inside a mounted condition. Missing dependencies fail
the launch; there is no fallback to an unrestricted process.

`network: "disabled"` creates a network namespace. `"shared_license"` retains the
host network for a network license server and therefore **does not restrict other
network egress**. Only explicitly selected simulator/license environment names are
passed: `CDS_LIC_FILE`, `CDSLMD_LICENSE_FILE`, `LM_LICENSE_FILE`, `CDS_LIC_ONLY`,
`CDS_LIC_QUEUE`, `CDS_AUTO_64BIT`, `LD_LIBRARY_PATH`, `CDS_INST_DIR`,
`CDS_SPECTRE_DIR`. Model credentials and the caller's remaining environment are
removed before Bubblewrap starts. License values remain operator-private.

The launcher replaces itself, preserving the existing job process group for
timeout and cancellation. `-W` grants no condition directory. Other launches write
a configuration-hash and input-hash receipt in the condition directory, before
Spectre starts. This receipt describes launch intent; trusted checkers must still
verify the process result, input identity and output independently.

To opt in to an existing final host profile, set its `spectre` executable to an
operator-owned wrapper which invokes this command using a fixed private config
and pinned bundle. Keep the trusted checker outside the namespace, and retain
its license preflight and actual installation's discipline paths. Existing
profiles are unchanged. The process entry does not authorize mounting a whole task
package.

For public sessions, use a host profile with its normal `shell`, `setup_scripts`,
`spectre`, `preflight_script`, storage roots and resource bounds, plus exactly
`backend: "spectre_namespace"` and `isolation_config: "/private/isolation.json"`.
The profile's Spectre path must match the isolation configuration. The evaluator
generates its own launcher from that validated configuration for both preflight
and verification; the public path accepts no ordinary unisolated host profile.
The profile identity includes the isolation configuration, Bubblewrap executable
and Python interpreter. Namespace public jobs use the same durable job and archive
checks as Docker public jobs. The two backends have different network policies;
the Docker restrictions above do not imply network isolation for `shared_license`.

The public task checker remains trusted host code. It must assemble each child
condition from public assets and frozen candidate inputs only. The optional
[`evas_testbench` declaration](CURRENT_EVAS_PUBLIC_SESSION.md#configuration-b-task-declared-remote-public-spectre)
passes temporary netlist and VA text to that checker without changing the formal
submission inventory. The checker must validate temporary support paths against
its own immutable inputs before launching the isolated child.

Actual Linux validation with Bubblewrap 0.4.1 and Spectre 21.1.0.509.isr12
compiled a relative helper module and installed standard disciplines. A transient
reference and a candidate using `$fopen` against hidden truth and its own source
both completed with zero errors: 13 samples, 0.5V output for 1V input. Both negative
file handles were zero; input files and hidden truth were unchanged. The deployment
used shared license networking. Private raw evidence remains with the operator;
these probes do not establish arbitrary runtime-grant safety or network isolation.
