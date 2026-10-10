# Harbor Agent 与 Model 组合

此入口要求 Python 3.12 或更新版本，以及固定的 Harbor 0.23.0：

```bash
python -m pip install -e '.[harbor]'
```

Harbor 管理 Job、Trial、阶段期限、并发、Agent 安装、模型请求和原生工具。
Harness 提供公开电路会话、候选冻结和独立终评。固定 vaBench 执行协议保留，见[VABench 操作者接口](VABENCH.md)。

第一次运行可以先用 [AnalogBench 快速开始](../../examples/analogbench/README.md)。它复用原生 Harbor 任务与 verifier，
无需配置 EVAS 公开会话。自定义任务的路径选择见[开发者接入指南](../guides/integration.md)。

公开会话任务运行前使用[部署配方与预检](../guides/HARBOR_DEPLOYMENT.md)，多题实验使用[私有任务绑定](HARBOR_TASKS.md)。
公开会话任务运行后可[离线汇总实验结果](../guides/HARBOR_RESULTS.md)，保留失败和未启动项。
原生任务先查看 Harbor 自身的结果与 Agent 日志；当前 Harness 报告依赖冻结候选与独立终评凭据。

## 分别选择 Agent 与 Model

[profiles.example.json](../../examples/chips/harbor/profiles.example.json) 把 Agent 和 Model
放在两个独立目录中。已支持的 Agent 可以独立声明 Harbor 设置；增加一个 Model 只需声明其连接。
扩展新的 Agent 类型时，增加该 Agent 的协议适配即可，不为每个 Model 单独实现。
`compile_agent(catalog, agent_key, model_key, protocol=None)` 产生 stock Harbor `AgentConfig`。
编译器没有模型品牌分支，也不为每个 Agent 与 Model 配对创建实现。

每个 Model 连接包含 `protocol`、原生模型 ID `model`、显式 `base_url` 和环境变量名 `key_env`。
协议描述服务实际接受的请求格式，不能从 Claude、GLM、DeepSeek 或 GPT 品牌推断。
模型可以声明多个协议连接，但每种协议只能声明一次。

以下是固定版本的配置支持范围。它不表示服务已经提供相应协议，也不表示真实模型运行验收通过。

| Harbor Agent | 可选择的连接协议 | 编译后的原生连接设置 |
| --- | --- | --- |
| `codex` | `openai-responses` | `openai/MODEL`，`OPENAI_API_KEY`，`OPENAI_BASE_URL` |
| `claude-code` | `anthropic-messages` | 原生模型 ID，`ANTHROPIC_API_KEY`，`ANTHROPIC_BASE_URL` |
| `pi` | `openai-responses`、`openai-chat-completions`、`anthropic-messages` | 按协议设置连接环境变量和 Pi `model_api`；chat 对应 `openai-completions` |
| `mini-swe-agent` | `openai-chat-completions`、`anthropic-messages` | 对应协议的模型前缀与环境变量，`MSWEA_API_KEY`，原生 `config.model.model_class=litellm` 和 `config.model.model_kwargs.api_base` |

`mini-swe-agent` 的 Responses 连接当前拒绝。
其顶层 `reasoning_effort` 会让固定 Harbor 版本把 `openai/` 模型切换到 Responses，
因此该参数与选择的 chat-completions 连接冲突，编译器也会拒绝。
Codex 在固定 Harbor 版本中会截断带 `/` 的原生模型 ID，因此编译器明确拒绝此组合。
同一个带 `/` 的模型配置仍可供支持它的 Agent 使用。
编译时与电路 Trial 启动时均拒绝会切换认证来源的宿主环境选项，例如
`CODEX_AUTH_JSON_PATH`、`CODEX_FORCE_AUTH_JSON`、`CLAUDE_FORCE_OAUTH` 和 Bedrock 路由。
先清除这些旧选项，再使用 Model 连接声明的凭据。
Agent 目录中的普通设置使用 Harbor 自身的 options schema 校验。
Agent 不能预设模型、端点或凭据；嵌套 native config 中的路由覆盖同样拒绝。
需要 native config 时使用内联对象，以便编译前检查冲突。

编译器先筛选 Agent 支持的协议。没有兼容连接时失败；剩余多个连接时要求显式 `--protocol`。
例如，同一个 GLM 服务同时声明 Messages 和 Responses 时，`codex` 自动选择 Responses，
`pi` 则需要指定协议。只有服务确实提供对应协议时，这些配置才具备运行条件。
未知字段、未知别名、缺失必填项、重复协议和重复 JSON 键均失败。

`key_env` 只允许环境变量名。编译输出保留 `${KEY_ENV}`，编译器不读取密钥。
Harbor 在构造 Agent 时解析该引用。不要把实际密钥填入目录、生成的 JobConfig 或任务文件。
相邻 [profiles.schema.json](../../circuit_harness/harbor/profiles.schema.json)
由固定 Harbor 的 `AgentConfig` 和 Agent options schemas 生成，运行时还检查协议与路由冲突。

## 准备公开任务与私有配置

使用 [示例目录](../../examples/chips/harbor/README.md) 的单步 Harbor task，包含
`instruction.md`、`task.toml` 和 `environment/`。
Agent 环境使用一张带 Python 3 的 Linux 镜像，声明在 `task.toml` 的
`environment.docker_image` 中，或使用一个 Dockerfile。
Harbor 负责 stock Agent 的安装和容器执行；按实际 Agent 要求准备镜像、安装权限及模型网络。

`CircuitDockerEnvironment` 继承 Harbor `DockerEnvironment`，保留其安装、exec、上传和日志接口。
此入口不接受 Windows 镜像、自定义 compose、额外 compose、用户挂载、多步骤任务、resume、
load trajectory 或 simulated user Agent。Harbor 自动使用的 Agent 日志与 artifact 挂载仍然保留，
私有 verifier 日志挂载移除。

公开 task 中不放 `harness.json`。把以下两个配置放在 task 和 Trial 导出目录之外：

- [public-session.example.json](../../examples/chips/harbor/public-session.example.json)
  对应 `PublicSessionConfig`，通过 `environment.kwargs.session_config` 读取。
  它声明公开 task、材料、公开后端、资源路径与动作、仿真预算。
- [final-evaluation.example.json](../../examples/chips/harbor/final-evaluation.example.json)
  对应 `FinalEvaluationConfig`，通过 `verifier.kwargs.config_path` 读取。
  缺省 `backend` 仍为 `remote_spectre`，声明独立终评的 `task_package` 和 `remote`。
  选择 `backend: benchmark_opensource` 时改用
  [final-evaluation-opensource.example.json](../../examples/chips/harbor/final-evaluation-opensource.example.json)，
  只声明 `task_package` 和 `opensource`，不能混入 `remote`。

`opensource` 复用保存候选的 benchmark replay 配置，必须显式声明版本、固定镜像摘要、
任务包摘要、求解选项、不支持分析列表、至多 300 秒的执行期限和至多 16 MiB 的输出限额。
用实际任务包及预先安装的本地镜像替换示例摘要，checker 必须报告实际使用的求解选项。
EVAS 引擎、任务包和评分规则由 benchmark 提供，Harness 不补造判据。
终评配置及 checker 保持在操作者私有目录，不进入公开 task 或 Agent 的日志挂载。

`public_backend` 可以使用已声明的 Docker EVAS、原生沙箱 EVAS 或远端公开 Spectre 后端。
公开工具只返回诊断。独立终评使用匹配 `task_id` 与 `task_version` 的 final package。
远端公开 Spectre 与终评必须使用分开的 package、profile 和存储根目录；同一服务器也不能重叠。
具体后端条件见[当前 EVAS 会话](CURRENT_EVAS_PUBLIC_SESSION.md)和
[独立评测与回放](BENCHMARK_EVALUATION.md)。

Agent 容器内安装 `harness-public`。任务说明引导 Agent 使用自己的 shell 工具调用它：

```bash
harness-public info
printf '%s\n' '{"action_id":"read-001","tool":"evas_read","arguments":{"path":"dut.va"}}' \
  | harness-public action
```

`info` 返回公开 task、预算和工具 schemas。`action` 从 stdin 读取一个 JSON 对象，
包含 `action_id`、`tool` 和 `arguments`。按返回的 schema 调用 `evas_read`、`evas_write`、
`evas_simulate` 和 `evas_submit`；启用实验的 task 还提供 `evas_experiment`。
复用同一动作 ID 时，请求内容必须完全相同。客户端没有自动重试；传输失败可能留下不确定动作，
不能换一个 ID 盲目重复提交。

公开会话与仿真器资源留在 runner 上。Agent 容器只拿到经过认证的公开 HTTP 客户端及其临时 token。
候选写入经过 gateway，容器中的普通文件不是终评提交对象。
Agent 仍使用 stock 原生工具与模型客户端；此设计不声称 Agent 只能调用公开电路工具。

## 声明 Docker 网桥上的 gateway

`environment.kwargs` 必须分别声明 `gateway_bind_host` 和 `gateway_host`。
前者是 runner 的监听地址，后者是 Agent 容器可以访问的主机地址。
端口由每个 Trial 分配，token 按 Trial 生成。将该 HTTP 通道限制在受控的私有 Docker 网桥中。

Colima 常见主机别名为 `host.lima.internal`，Docker Desktop 常见别名为 `host.docker.internal`。
Agent 容器的 DNS 不一定能够解析这些名字；本地 Colima 检查中，显式私有桥 IP 可达，两个别名均无法解析。
这些名字不可互换，也不代表所有 Linux Docker 主机的配置。
Colima 可以用 `colima ssh -- cat /etc/hosts` 查看主机映射，再验证映射 IP 从 Agent 容器可达。
`gateway_host` 使用验证过的地址或 IP，不能只凭宿主机能解析别名就认定容器可达。
让 `gateway_bind_host` 覆盖对应网桥接口。
示例使用 `0.0.0.0` 监听以覆盖 VM 网桥；部署时限制其网络可达范围，
可以绑定专用私有桥地址时优先使用该地址。
容器启动后，环境执行 `harness-public info` 检查连通性。
失败时查看 Trial 的 `public-client-preflight.log`，不要把失败归为模型零分。

## 编译 JobConfig 并交给 Harbor

编辑示例中的镜像、路径、模型协议、ID 和 endpoint。所有 `.invalid` 地址、`*-fixture` 模型、
零值镜像摘要以及 `/operator/` 路径均为占位内容。

```bash
python -m circuit_harness.harbor.profiles \
  --catalog examples/chips/harbor/profiles.example.json \
  --agent pi --model glm --protocol anthropic-messages \
  --job /operator/private/harbor-job.json \
  --output /operator/private/harbor-job.generated.json
harbor run --config /operator/private/harbor-job.generated.json
```

第一条命令只编译配置，要求输出文件尚不存在。
第二条命令才启动 Agent、模型请求及独立终评，使用已经授权的资源和预算。
`harbor run --config` 已对 Harbor 0.23.0 的 CLI help 核对。

[JobConfig 示例](../../examples/chips/harbor/job.example.json) 使用
`CircuitDockerEnvironment` 和 `FrozenCandidateVerifier`。
`compile_job(catalog, agent_key, model_key, job, protocol=None)` 替换模板中的 Agent 选择。
遇到该电路环境时，它把 stock AgentConfig 放入唯一的 `CircuitAgent` 生命周期适配器：
`kwargs.agent_name` 选择 stock Harbor Agent，`kwargs.agent_kwargs` 保留该 Agent 的设置，
顶层 `model_name`、`env`、MCP 和期限声明保留。

`CircuitAgent` 把 setup、run、版本与上下文收集委托给 stock Agent。
它没有新的 Agent loop、模型 transport 或每品牌 Agent 子类。
Agent 正常返回、超时、异常或取消时，适配器关闭 gateway 的动作入口，
等待已接受动作结束，再冻结最后一份完整候选，然后 Harbor 才继续阶段收尾与终评。
终评只读取已冻结的候选。未收妥的公开动作保留恢复证据，不能宣称冻结成功。

参考解入口也可将 `kwargs.agent_name` 设为 `oracle`，继续委托 Harbor 的 stock Oracle。
此时 `kwargs.agent_kwargs.task_dir` 必须是当前任务目录，`agent_timeout_sec` 可指定
原生 solve 执行期限。Harbor 只给顶层 Oracle 注入任务和 Trial 路径；包装器因此从
Harbor 实际提供的 agent 日志目录导出当前 `TrialPaths`，不接受另行指定 `trial_paths`。
setup 再核对任务目录和环境的当前 Trial，失配则不执行参考脚本。
普通模型 Agent 的参数及委托方式不变。

stock Oracle 按原路径上传 solution 并执行 solve，适配器不增加工具循环。
如果 stock Oracle 留下非零 `exit-code.txt`，适配器按参考执行错误冻结候选并保持未评分，
不把脚本失败当成正常参考完成。参考 solve 如需向公开会话提交结果，任务必须提供明确的
提交桥接；这不表示原任务的 test.sh 在容器内执行，终评仍由 `FrozenCandidateVerifier`
独立读取冻结候选。

Harbor 限制 Agent 和 verifier 的独立阶段期限。公开仿真预算由会话管理。
CLI 基础设施失败、终评执行失败、损坏证据和身份不匹配保持无分数；
只有完成的独立 checker 分数与匹配候选身份才能产生 reward。
Harbor 的模型用量与原生轨迹保留其 Agent 实际提供的内容，不据此补造未知成本或工具清单。

实验报告从保存的 `public-session/session.json` 读取实际的 `observation_view_version`，
并在 `actual_conditions.public.public_tools` 记录对应的公开工具列表。旧会话的版本记为 null。
计划中的 `conditions.public_tools` 为 null，因为操作者配置尚不能证明会话实际采用了哪版工具。
同一计划单元若混有不同响应版本，报告不计算合并均分；应分别组织实验条件。

## 检查范围与验收限制

```bash
python -m pytest -q \
  tests/chips/test_harbor_profiles.py \
  tests/chips/test_harbor_public_gateway.py \
  tests/chips/test_harbor_composition.py \
  tests/chips/test_harbor_chips_trial.py
```

配置测试使用实际 Harbor factory 与 fixture 环境变量，不请求模型。
Gateway 测试使用本地 HTTP 和构造的公开会话。
容器检查需要已存在的本地 Docker 镜像与相应主机条件，缺失时跳过；测试不会自动拉取镜像。Colima 等 VM Docker 部署需要把 pytest 的 `--basetemp`
放在 VM 已共享的工作目录内，系统临时目录不一定可挂载。
Harbor Trial 的受控进程检查覆盖生命周期、取消和冻结边界。
这些检查不能证明真实 Claude、GLM、DeepSeek 或 GPT 服务、Agent 安装组合、EVAS 电路语义，
或许可 Spectre 的部署与独立电路验收已经通过。

## 可选的主机原生 Codex 路径

Prepare a Harbor single-step task with `instruction.md`, `task.toml`, and
`environment/harness.json`. Multi-step tasks are rejected. These flags were checked against
`harbor run --help` in version 0.23.0:

```bash
harbor run --path TASKS \
  --agent circuit_harness.harbor.agent:NativeCodexAgent \
  --model MODEL \
  --env circuit_harness.harbor.environment:HarborChipsEnvironment \
  --verifier circuit_harness.harbor.verifier:FrozenCandidateVerifier \
  --n-attempts 1 --n-concurrent 1 --max-retries 0
```

This command starts a model and can submit a remote final job. Configure and authorize those
conditions before running it. `--install-only` prepares the environment without Agent execution
or verification. Do not enable Harbor retries to conceal model or infrastructure failures.

Equivalent custom components in a Harbor JobConfig use these import paths:

```json
{
  "agents": [{
    "import_path": "circuit_harness.harbor.agent:NativeCodexAgent",
    "model_name": "MODEL"
  }],
  "environment": {
    "import_path": "circuit_harness.harbor.environment:HarborChipsEnvironment"
  },
  "verifier": {
    "import_path": "circuit_harness.harbor.verifier:FrozenCandidateVerifier"
  }
}
```

The task's `harness.json` follows
[`HarborChipsConfig`](../../circuit_harness/harbor/config.py) and its adjacent
[JSON Schema](../../circuit_harness/harbor/config.schema.json). Declare:

- `schema_version: 1`, a public `task` with `task_id`, `task_version`, `public_files`,
  `candidate_files`, `feedback_fields` and the public EVAS `manifest`;
- private `materials`, `checkout` and `kernel` paths; `public_backend` defaults to `docker`,
  requiring a Linux ELF kernel and immutable `image` (`sha256:` followed by 64 hexadecimal digits);
  select `native_codex_sandbox` explicitly for the local EVAS worker, set `image: null`, and declare
  absolute `public_codex` and `public_python` paths; a host Mach-O kernel is accepted on this path;
- `executable`, an absolute native Codex binary, version 0.154.0 or newer;
- `auth_file`, one explicit operator credential file; `runtime_readonly_paths`, precise native
  binary/runtime and Python standard-library roots; `allowed_hosts`, exact model endpoint hosts;
- `final_task_package`, an inventoried final benchmark package with matching task identity;
- `final_backend` defaults to `remote_spectre`. Its `final_remote` contains an SSH alias `host`
  and absolute remote `python`, `bundle`, `profile`, `run_root`, `archive_root` and `upload_root` paths;
- for `final_backend: benchmark_opensource`, use `final_opensource` with the fields from the
  private verifier example and omit `final_remote`;
- finite `max_actions`, `max_simulations`, `simulation_timeout_s`, `final_timeout_s` and
  `max_output_bytes`; `reasoning_effort` defaults to `medium`.

Use the packaged native binary and its vendor runtime root, rather than assuming an npm wrapper
can run with minimal filesystem access. The Python executable is resolved before creating the
MCP client; grant its standard-library runtime, not the entire repository or virtualenv packages.
Public workspace and runtime grants must not overlap any private task/session/verifier root.
The Agent executable and public EVAS worker executable are separate declarations. `public_codex`
may name the same already pinned native `executable`, and `public_python` may name the declared
Python runtime. Both must be explicit for the native public backend; no host fallback executes
unisolated. The public worker still runs under its own finite sandbox/process limits.

For a native public worker, the operator fragment is:

```json
{
  "public_backend": "native_codex_sandbox",
  "image": null,
  "public_codex": "/operator/native-runtime/bin/codex",
  "public_python": "/operator/python-runtime/bin/python3",
  "kernel": "/operator/evas-kernel"
}
```

These are placeholders, not a complete runnable task. Public experiment declarations are
Docker-only; a native public backend with `task.experiments` is rejected before preparation.

### 主机原生路径的条件

| Condition | Implementation | Acceptance |
| --- | --- | --- |
| A: local native Codex → local public EVAS MCP → SSH final Spectre | Harbor plugins, explicit Docker or native sandbox public EVAS session, public-only broker, OS sandbox, frozen final transport | Bounded local lifecycle and sandbox checks; real model/SSH/Spectre acceptance pending |
| B: local native Codex → server public Spectre and final Spectre | Shared public session, stable public job recovery, server Docker or Spectre process namespace, independent final verifier | Local Docker and controlled Harbor Trial checks; actual namespace SSH/Spectre reference probes; graded model Trial pending |

There is no switch accepting an arbitrary `isolation_verified` Boolean. Unknown configuration
fields, missing auth/runtime declarations and overlapping private grants fail closed. Do not
label configuration B commercially accepted without its real deployment evidence.

The broker runs outside the Agent sandbox and advertises the session's
[public tools](CURRENT_EVAS_PUBLIC_SESSION.md#compact-observations-and-exact-artifacts),
including bounded observation reads for new sessions. Session budgets, EVAS source, kernel and final task/profile
remain outside the Agent read grants. The entire native process is wrapped in an outer Codex
sandbox with an exact domain allowlist. Native tools remain governed by that boundary; disabling
features does not prove that MCP is the only tool available. Observed native tool calls are
recorded and the full catalog remains unknown until an actual Agent run establishes it.

Only the explicitly supplied auth file is copied into the per-attempt native home. That home
supports Codex cache/lock writes; the auth file has a read-only grant. The same native process
can read its own model credentials. This integration does not claim otherwise. The copied home
is removed on environment cleanup, including when broker cleanup fails; the caller's credential file is never modified.

### 主机原生路径的期限与证据

The Agent launches once, captures stdout and stderr within a combined `max_output_bytes` limit,
and kills and reaps its process group before freezing. Output overflow or an abnormal native CLI exit is an infrastructure
exception and produces no reward. Harbor's Agent deadline can still verify the last complete
candidate; outside Trial cancellation freezes and propagates without starting verification.
A missing candidate remains unscored until the benchmark defines its missing-submission rule.

The verifier has a separate finite deadline. With the default remote Spectre backend, it submits
the frozen candidate under one durable job ID and waits for both job completion and verified
archive publication before retrieval.
Archive publication uses the same verifier deadline; archive failure produces no reward.
Interrupting the local SSH wait does
not cancel or relaunch the server job. Infrastructure failures, malformed reports, candidate/task
identity mismatch and ambiguous checker outcomes never become a model zero. A reward requires
an explicit completed checker score and successful execution.

Harbor retains Trial results and phase timing. Agent logs retain bounded native events/stderr,
`attempt.json`, requested model, version/configuration evidence, freeze identity and observed
tools. A successful process with one complete turn and no malformed events can supply native token usage.
Cancelled or failed processes, malformed evidence, an unfinished later turn, and multiple completed
turns preserve per-turn usage while total usage remains unknown; no last-turn total or price is invented.
The verifier retains evaluation and transport evidence under its private logs directory.
The optional `benchmark_opensource` backend calls `replay_candidate` on the frozen candidate
and seals its local checker evidence in `verifier/replay/`. `verify_replay` checks that receipt
offline. The existing replay result uses `backend: opensource` while its execution identity uses
`benchmark_opensource`. Valid graded pass and fail produce rewards, including a valid zero;
unsupported analyses, checker failures and invalid evidence retain no score.

When Harbor times out or cancels local verification, the verifier signals the replay executor
and waits for its owned process group and Docker cleanup before propagating cancellation.
Repeated cancellation also waits for cleanup. The executor bounds process shutdown and Docker
removal, and retains cleanup status and the sealed unscored receipt in its private directory.
`local-execution.json` records worker completion. A second call cannot overwrite or rerun the
same replay directory. These local semantics do not cancel a durable remote Spectre job.

Run bounded checks without a model:

```bash
python -m pytest -q tests/chips/test_harbor_chips_trial.py
```

These tests use actual Harbor Trial behavior and controlled child processes for cancellation,
process-group cleanup, output limits, candidate freezing and invalid-score rejection. The macOS
production startup check uses the installed native binary, a synthetic public session, real
broker/sandbox and empty fixture credentials for `--version` and file-boundary probes. Both Docker
and native backend session preparation are checked; the native case uses a synthetic Mach-O kernel. It runs
no model, SSH or simulator. Other hosts skip that check. The controlled configuration B Trial
fixtures establish its public-session and final-verifier lifecycle. These fixtures do not certify
a real EVAS workload, Spectre server operation, model behavior, a commercially deployed
configuration B run or cross-condition benchmark comparability.

### Offline Pi trajectory export

Export one completed saved Pi Trial without contacting a model, simulator or server:

```bash
python -m circuit_harness.harbor.trajectory \
  --trial /private/jobs/job/task__trial \
  --output /private/exports/task-trial
```

This requires the pinned Harbor extra and writes Harbor ATIF v1.8
`trajectory.json`. It currently accepts one linear, text-only Pi v3 session per
Trial. Branches, compaction, changing model/tool definitions, unmatched calls,
unfinished sessions and unsupported content fail before publication. Other
Agents need their own native-to-ATIF adapter. The training consumer uses ATIF,
so adding an Agent does not require a new exporter for every model.

This entry requires a Harness public session, a frozen candidate and its collection
receipt. A completed native Harbor Pi Trial without that evidence is not supported.
Native task rewards alone cannot establish the candidate/grade association required
by this exporter. Reading a native Pi session with `read_pi_trajectory` converts
the conversation only; it does not verify that association. Keep native tasks on
their original execution path and retain their results and logs until a supported
export is available. See [task path selection](../guides/integration.md#2-选择任务执行方式).

Native files remain unchanged. ATIF preserves source SHA-256 and line numbers,
ordered system sections, actual tool definitions, native calls and the tool
observations visible to the model, including truncation. A native `bash` call
stays `bash`; related Harness actions are metadata with request/response digests
and explicitly textual call references. The exporter never substitutes a full
backend response for the truncated observation.

System sections are joined with blank lines and labelled `native_reconstruction`.
The record is not an exact provider request snapshot. Provider token IDs and
sampling log-probabilities are unavailable. Requested thinking and the recorded
thinking level remain separate; a zero native cost is treated as unknown.

Task/version, frozen candidate and reward are linked through the same independent
archive checks as offline reports. A matching completed checker result can yield
a valid reward of zero. Invalid evidence fails; absent or unsuccessful grading
with no native reward stays `unscored`. Verifier contents are excluded from the
conversation. Explicit SFT selection can include unscored trajectories, preserving
that status and a null reward without treating them as zero-score examples.

The output directory is private, published atomically and never overwritten.
`--redact-file /private/literals.json` accepts a JSON array of explicit secret
strings for replacement in native text values. A literal in an object key is
rejected because renaming keys could corrupt tool arguments. This is not secret
detection or permission to publish the result. Native context can contain private
paths and task material; raw and derived outputs stay in private run storage.
No environment files, provider configuration or hidden checker files are copied.

For explicit dataset selection, reasoning policy, split checks and the external-trainer dataset
loader, see [ATIF SFT preparation](../../circuit_harness/data/README.md#atif-trajectories-for-sft).


### 主机原生路径的配置 B

Set `public_backend: remote_spectre`, `image: null`, `public_task_package` and
`public_remote`. EVAS `checkout` and `kernel` may be omitted. The public task's
`manifest` is exactly a `condition_id` matching the public package, rather than EVAS
solver settings. Public package identity and feedback declarations are checked before
session preparation. `public_remote` has the same endpoint fields as `final_remote`,
but public and final profiles and storage roots must be separate on the same host.
The public package must be separate from `final_task_package`. These operator records
stay outside the native Agent grants and public tool responses.

The server must use the [isolated public profile](BENCHMARK_EVALUATION.md#isolated-public-spectre-jobs).
It requires all runtime/dependency and actual licensing checks to pass. The `docker`
profile runs with networking disabled. The `spectre_namespace` profile uses the
operator's validated process namespace and may retain host networking for licensing;
that mode does not restrict network egress. Actual namespace reference probes have
run fixed and temporary netlists through SSH and licensed Spectre. Arbitrary PDK or
analysis mappings still require task-owned support. Controlled Trial
fixtures cover one Agent process editing, receiving public feedback, editing again,
freezing and handing the exact final bytes to a separate verifier. Real Docker fixtures
cover input access, cancellation and output limits. Neither fixture establishes a real
native-model or licensed-Spectre trial.
