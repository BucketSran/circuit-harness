# Chips phase 1 validation

> 发布说明：本文保留开源前的版本与实验记录。`demo/chips`、旧提交和历史议题归私有研发档案；
> 当前公开开发以 `main` 为基线，遵循[仓库范围](REPOSITORY_SCOPE.md)中的边界。

Publication note: `lab-server` is an anonymized deployment identifier throughout
this record. Host mappings, site setup scripts and raw evidence remain private.
Historical branch names describe the state on the recorded date; current source
and release boundaries are defined in [repository scope](REPOSITORY_SCOPE.md).

Date: 2026-09-22. A vendor EMX example has now completed through real lab SSH
and the Chips Harness; production circuit-design acceptance remains pending.
The user confirmed that Bio/Robotics are source references only; their domain
runtime tests and dependencies are outside Chips acceptance and CI.

## Measured local checks

Environment: macOS arm64, Python 3.12.13, Icarus Verilog 13.0, gdstk 1.0.0,
MCP SDK 1.30.0 and Ruff 0.15.6.

- Real Icarus: correct design, wrong truth table, malformed RTL and infinite
  simulation. Compiler/simulator actually launched; pass/fail/timeout and VCD
  output checked. These checks validate the local runner, not EMX.
- Real GDS generation: sample JSON written by gdstk and read back as GDS;
  expected cell and downloaded hashes checked.
- Constructed SSH/EMX fixture: real Python subprocesses execute the shipped
  remote zipapp through test SSH/scp executables. This tests protocol, process
  handling and filesystem transfers, **not network SSH authentication or EMX**.
- Fault injection: lost submit reply, interrupted download, persistent network
  failure, bounded retry/deadline, authentication stop, changed resume inputs/GDS,
  missing previously submitted job, corrupt download, EMX nonzero exit, missing
  result, cancellation, process-group cleanup, output limits, missing executable,
  malformed/truncated event records and log-write failure.
- MCP: real stdio discovery and configuration-error feedback; full configured
  pipeline through the model-tool adapter with the constructed lab fixture.
  Existing external-bridge and Workflow composition contracts are included.
- Static checks: Ruff lint/format, Python compilation, schema export, shell
  syntax check for the Cadence example, and Git whitespace checks.

Test command (activate the virtual environment so legacy shell tools find python):

```bash
source .venv/bin/activate
python -m pytest -q \
  tests/common/execution/test_chips_harness.py \
  tests/workflows/test_chips.py \
  tests/reasoning/runtime/test_external_bridge.py \
  tests/workflows/resources/test_composition_contract.py \
  tests/common/execution/tools/test_workspace_tools.py
```

Final local acceptance: **84 passed, 3 skipped** in 55.05 seconds. The skips
are optional existing bridge integration checks; the Chips fixture and real local
GDS/Icarus checks ran. The long-job MCP timeout configuration was then validated
through the existing composition/MCP tests without invoking a real model.
A final SCP connection-error exit-code correction and MCP configuration check
passed 4 targeted tests in 7.40 seconds; the transfer test exercises exit code 1
as well as SSH submission failures using exit code 255.

An earlier broad exploratory run (before the scope clarification) passed 1,631
checks with 164 skips and 44 failures: Bio extras were absent and legacy shell
workspace tools could not find `python` without virtualenv activation. The latter
checks passed after activation. A full-tree collection attempt also lacked
pandas/torch. These broader results are not Chips acceptance gates; no Bio or
Robotics dependencies were installed to pursue them.

## Not measured

- Production PDK/ports, expert task inputs and independent circuit-quality
  acceptance remain unmeasured. The vendor EMX 5.7 example, its logs and Y/S
  outputs were subsequently verified; see the server baseline below.
- Spectre 21.1.0.509.isr12 launched a PDK-free RC attempt but failed license
  checkout with SPECTRE-209 before circuit solving. No PSF or RC result is claimed.
- Real model end-to-end smoke: automatic approval review rejected the proposed
  external Codex run because it could transmit private repository/workspace
  content without specific data-egress authorization. No real model call ran.
  Deterministic MCP tests do not replace this check.
- Circuit-quality scoring, formal equivalence, PPA, model ratings, pricing and
  scheduling improvements: not implemented or claimed in this phase.

## Deployment acceptance

Provide one real layout JSON/converter and a lab-controlled wrapper that accepts
GDS/job-directory arguments, loads EMX's environment, fixes the actual process,
cell/port/frequency options, and returns EMX's exit status. List required outputs
and process/wrapper files in the operator config. Run the documented CLI once,
inspect the result and downloaded logs, then exercise resume on the same job ID.
An EMX exit code of zero remains execution evidence, not circuit acceptance.

## Development SOP and reusable probes (2026-09-22)

Added the [development SOP](DEVELOPMENT_SOP.md), project `chips-dev` Skill,
and [Chips test entry point](../../tests/chips/README.md). Existing module test
locations are preserved. The case catalog separates local regressions, host
observations, real lab/model acceptance and controlled performance comparisons.

The standalone metadata/HTTPS probe has **15 local regression cases**, using real
temporary directories and constructed curl responses. The development session
observed failing tests before implementing ancestor reporting, per-path failure
records, explicit HTTPS observation and rejection of empty curl measurements.
Additional URL/timeout checks were regression additions; not every case has a
separate recorded Red phase.

The aggregate command in the test entry point completed with **77 passed,
3 skipped in 55.81 seconds**. The skips are the existing opt-in live Math/Podman,
Pi/Podman and Claude/Podman bridge checks; none was enabled. An initial sandboxed
attempt had 72 passes, 3 skips and 5 failures because loopback socket binding was
denied. The successful rerun allowed the local MCP test sockets. Both attempts'
JUnit and console logs are retained in private run storage.

Ruff lint and format checks passed across `alphaapollo` and `tests`; Skill
validation passed, local links resolved, workflow YAML parsed and Git whitespace
checks passed. CI now includes `tests/chips` and retains its local JUnit report
for 14 days. The GitHub-hosted workflow subsequently passed for commit 681326ae
([private historical CI run](https://github.com/BucketSran/circuit-harness-private/actions/runs/35715381700)).

The new probe was also executed through actual SSH stdin on the authorized Linux
lab host with Python 3.12.10. Both requested paths and their ACLs were observed;
network probing was disabled and no remote file was installed. Raw host paths
and metadata remain in an ignored private local report, created with mode 0600.
This verifies that probe execution path, not cross-user isolation, storage
performance, model connectivity, EDA licensing or simulation acceptance.

## Real server baseline and clock-skew fix (2026-09-22)

See the [server baseline](BASELINE.md) for the current evidence, exact scope and
remaining gates. Vendor GDS geometry was reconstructed using the existing JSON
converter; layer XOR areas were zero and both labels matched. A direct vendor
run and three Harness job results had identical numerical Y/S data at 1 GHz.

The first Harness control attempt returned unknown even though its remote EMX
job succeeded: NFS mtime lagged the application event time by about 108 seconds.
That same job was recovered with its original worker, without another launch.
The worker now records an initial submission event and bases freshness on
application event timestamps, preserving unknown for missing, stale or future
heartbeats. The new CLI regression failed before the fix and passed afterward.

Affected check command:

    python -m pytest -q -ra tests/common/execution/test_chips_harness.py tests/workflows/test_chips.py

It completed with **29 passed in 53.56 seconds**. Ruff lint/format and Git whitespace
checks passed. Two fresh live Harness jobs after the fix and one completed-job
resume succeeded. Each remote journal records exactly one EMX process launch.
All downloaded artifacts matched their recorded size and SHA-256.

This is a functional reproducibility pilot using one solver/vendor example.
It does not establish long-term reliability, electromagnetic accuracy, production
PDK coverage, live mid-job disconnect/cancellation behavior or a model/Agent loop.
Raw evidence and host-specific operator scripts are retained in ignored private
run storage; proprietary example/process contents are not committed.

## ngspice 47 server-local Harness (2026-09-22)

The [RC Harness](NGSPICE.md) ran on the authorized server with a privately installed
ngspice 47, Python 3.12.10 and the standard-library CLI bundle. No model, PDK,
commercial simulator or other domain benchmark was used.

- Two R=1 kΩ / C=1 nF / V=1 V runs and one R=4.7 kΩ / C=2.2 nF / V=0.8 V run
  all passed the fixed independent analytical AC and transient checks.
- Each had 121 AC samples and 1,012 transient samples. Maximum AC absolute error
  was 3.51e-16 (tolerance 1e-5); maximum normalized transient error was 2.98e-6
  (tolerance 5e-5).
- Server-local CLI wall times were 0.569, 0.619 and 0.469 seconds, including
  tool probing, simulation, logging and grading; these exclude SSH and download.
  This is a three-run functional pilot, not a throughput or reliability benchmark.
- GNU time reported maximum RSS of 24,064 KiB for each CLI run. The installed
  tree occupied 9,712 KiB and the successful build tree 165,072 KiB (`du -sk`).
  The `make -j2` phase took 158.53 seconds and reported 245,620 KiB maximum RSS;
  this is the tool's reported high-water statistic, not a summed/cgroup peak for
  concurrent compilation. No memory hard limit or GPU was used.
- Completed-run resume took 0.168 seconds, reused the same run identity and left
  exactly one ngspice solver launch in the journal.
- A separate real simulation with R deliberately doubled was rejected by the
  original task's oracle: AC error 0.3333 and transient error 0.2500. A corrupted
  copy of a completed run was also rejected by artifact verification.

The initial NFS build and then a local-scratch build with inherited login environment
were stopped during configuration; their logs remain archived. A ten-command probe
measured 0.817 seconds in the inherited environment versus 0.011 seconds in a minimal
environment. The successful build used private local scratch plus a minimal environment,
two make jobs and a HOME installation prefix. This does not isolate an NFS speedup.

The new local regression suite has 15 passing cases plus one optional actual-ngspice
case skipped on the Mac, where ngspice was not installed. Red/Green records exist for
CLI missing-dependency behavior, zero-exit/missing-output rejection, offline bundle
compatibility and downloaded-evidence verification; other cases are regression additions.
The combined affected Chips command completed with **96 passed, 4 skipped in 64.59 s**.
The other three skips are pre-existing opt-in live bridge tests. A preceding sandboxed
attempt had 5 loopback-bind permission failures; enabling those local test sockets fixed
them. Both reports remain in private storage. Ruff lint/format and shell syntax passed.

Local download verification exposed a Linux/macOS libm rounding difference of
1.11e-16 in the derived transient error metric. The verifier now requires strict
artifact hashes, recorded metric-file consistency and the same independently
computed verdict, while allowing only derived error estimates to differ by
1e-12 absolute / 1e-10 relative. Circuit acceptance tolerances are unchanged.
The regression failed before this fix; afterward **19 affected tests passed,
1 skipped**. All three downloaded runs regraded successfully on the Mac. A fourth
real RC run on the updated server bundle passed, including resume with one solver
launch. Old bundles and original results remain available rather than being rewritten.

CI is configured to install the distribution's ngspice and exercise the real local
RC regression; that CI software version differs from the pinned server installation.
At that initial revision, ngspice had only a foreground executor and completed-run
resume. The detached-job follow-up below supersedes that limitation. Memory hard
limits, larger circuits, expert-reviewed production tasks, model MCP integration
and a second member's independent deployment remain open.


## Server-owned RC jobs across SSH disconnects (2026-09-22)

Related design discussion: [Chips Harness proposal #1](https://github.com/BucketSran/circuit-harness-private/issues/1).
The new `submit-rc` entry point starts a detached worker from a per-job standard-library
bundle. The server owns simulation, independent grading and final artifact receipts;
status queries do not advance execution. Stable job IDs prevent duplicate execution.
The foreground CLI remains available with its existing behavior.

Evidence:

- Behavior-first regression failed before the CLI existed, then passed after implementation:
  a real submitting process group was killed while a constructed solver stage was active;
  its detached worker completed and graded the fixture without the submitting process.
- Additional local regressions cover concurrent/repeated submission, changed inputs,
  cancellation, timeout cleanup, missing dependency, unknown-state refusal to restart,
  invalid IDs and artifact corruption. These are process/protocol fixtures, not lab certification.
- Chips/shared regression: **107 passed, 4 skipped** (real local ngspice unavailable;
  three opt-in external agent bridge tests disabled). No other domain runtime tests ran.
- The first live attempt exposed Linux NFS's restriction on an exclusive flock through a
  read-only descriptor. The submitting client failed; its independent worker still finished.
  Status now uses a shared read lock that conflicts with the worker's exclusive lock.
  This failed attempt and its recovered result remain in the private evidence archive.
- After the NFS fix, the dedicated RC regression was rerun: **26 passed, 1 skipped**.
- The live SSH test used an 8-second delay wrapper before invoking the installed real
  ngspice 47. While that stage was unfinished, only this probe's local SSH client was
  terminated. No server connection or polling occurred for the next 12 seconds.
  The first reconnect found the final receipt already written, `execution=ok`,
  `verdict=pass`; resubmitting the same ID preserved exactly **one solver launch**.
- The ordinary `submit-rc.sh` path, with no delay wrapper, also completed a real RC job.
  All three downloaded jobs (including the initially failed client's completed worker)
  passed local artifact checks and independent waveform regrading: **51 artifact receipts**.

Deployed bundle SHA-256:
`cd2907ff89a00a4d8f0c40946e6f8745fb0f32158e488d6274579f9dfe2296db`.
Raw traces, first failure, final receipts, source snapshot and JUnit files are retained
under ignored, private `runs/chips/validation/20260922-detached-jobs/` and the operator's
server archive. The reusable explicit live probe is
[`ssh_detached_rc.py`](../../tests/chips/probes/ssh_detached_rc.py).

This establishes client/session independence on the tested server, not reboot recovery
or a guarantee of circuit success. Actual laptop sleep, physical network loss, server
reboot, OOM, disk failure, session-wide resource cleanup and EMX/Spectre disconnects
were not injected. Cancellation/timeout coverage here is local, not a live EDA claim.

## VABench r53 / EVAS integration (2026-09-22)

Related context: [Harness #1](https://github.com/BucketSran/circuit-harness-private/issues/1)
and [analog extension #2](https://github.com/BucketSran/circuit-harness-private/issues/2).
The new operator CLI pins a task/source/simulator identity, exports the upstream public
surface, freezes submissions, and runs the original profile-bound trusted replay in a
detached server job. Scoring is owned by vaEVAS, not reimplemented in Chips.

- Source: behavioral-veriloga-eval `0685aae05c346e8e60f33ba48e2f64daff54d4f2`;
  immutable `benchmarkv4-r53`, installed `evas-sim==0.8.7`.
- Actual local and lab-server results both matched all six expected outcomes:
  `v4-001` DUT and `v4-1001` bugfix gold passed, their sealed buggy-bundle candidates
  produced `behavior_failure`; `v4-501` reference Testbench passed its original reference
  and mutation scoring, and an explicitly invalid deck produced `compile_failure`.
- The initial local Testbench gold failed with `infrastructure_failure` because the bridge
  explicitly selected the old r52 profile. This was a bridge configuration defect, not a
  bad candidate. A failing regression was added, the environment changed to r53, and all
  six local cases were rerun successfully. Original failures are retained.
- Real SSH-loss test: a private wrapper paused eight seconds at the EVAS `simulate`
  entry point, then invoked unchanged EVAS. Only the local test SSH client was terminated.
  After 20 seconds without any server connection, the first reconnect found completion
  already sealed and the original verdict passed. Same-ID resubmission preserved exactly
  **one simulation launch**. This is client-session loss, not a server-reboot test.
- All **seven downloaded jobs** passed offline hash/receipt checks, including original
  score sidecars and verdict/submission associations. This verification did not rerun EVAS
  on the downloading host. A separately exported real Testbench public surface contained
  five files and no evaluator, evidence, or mutation directories.
- Final affected local checks: **121 passed, 4 skipped** across the selected Chips checks
  and the bridge rerun. The first sandboxed command had five Unix socket permission
  failures; rerunning the bridge with permitted local sockets gave **26 passed, 3 skipped**.
  The four distinct skips are missing local ngspice and three opt-in live agent checks.
  New VABench protocol coverage: **14 passed**, using explicitly constructed upstream API
  doubles for lifecycle/failure handling. Ruff lint/format and focused compilation passed.
  No Bio/Robotics/Math domain evaluation was run.

Server storage: the selected immutable source slice has **795 verified files**, approximately
**11 MiB** unpacked. The separate Python/EVAS runtime uses approximately **204 MiB**; the
Linux wheel cache is approximately **53 MiB**. These are observed disk usage, not peak RAM
or performance benchmarks. A slow full-history extraction was stopped, its partial
private directory preserved, and a recorded dependency slice deployed instead. This was
not a controlled NFS-versus-local-disk speed comparison.

The tested standard-library bundle SHA-256 is
`70c9269377484bcc15c3de8dad24e4118b098fc12ac4ce0543d4fb7b977e37de`.
The downloaded server archive SHA-256 is
`3ad3eaccdbcc9ff0fa5661c9f9e0439daa0e6f0d0766a8ae6b0dfd11e7b70ab2`.
Raw evidence, failed attempts, dependency/source manifests and JUnit reports are in ignored
private `runs/chips/validation/20260922-vabench/` and the operator's server archive.
See [usage](VABENCH.md), [task card](../../examples/chips/benchmarks/vabench/TASK.md),
[real replay probe](../../tests/chips/probes/vabench_smoke.py), and
[SSH probe](../../tests/chips/probes/ssh_detached_vabench.py).

Scope remains development-only frozen-submission replay. No paid model was called,
Pi/Codex experimental conditions were not evaluated, and no cross-task memory was used.
This is not full-release recertification, independent second-member reproduction, Spectre
parity, transistor/PDK validation, or a model-access security sandbox. Spectre's existing
source path was reviewed but was not exercised by this EVAS acceptance.

## Bounded storage comparison and archive validation (2026-09-22)

See [method, measurements and administrator handoff](STORAGE_VALIDATION.md).
The same 795-file source archive was extracted into fresh NFS and local XFS directories
in three alternating pairs, retaining system caches. Median extraction was 15.721 s
versus 0.163 s; first complete hash verification 2.659 s versus 0.057 s; repeated hash
verification 1.681 s versus 0.055 s. All file sets and contents matched.
These are small-file deployment measurements, not disk durability or solver speedups.

Original EVAS scoring of DUT/bugfix/Testbench positive and negative candidates gave all
12 expected outcomes across the two source/job layouts. Their simulator environment
remained the same NFS installation. One actual ngspice RC job on local storage passed.
Source identities matched before execution and all source bytes remained unchanged afterward.

An operator script packed the 13 completed jobs, rejected an injected truncated staging
copy, resumed only the archive copy, and verified identical job hashes afterward.
The downloaded archive's 565 files and all 13 job receipts passed local verification;
an intentionally modified candidate in a copied job was rejected. This is a file-level
fault injection, not a live network/NFS outage or production automatic retry acceptance.

The reusable opt-in storage probe and local I/O/input-boundary tests are retained in
`tests/chips/`. The affected probe suite passed **23 tests**, with no skips; Ruff lint and
format checks passed. Raw inputs, both test phases, host ACL observations, complete
results, the injected corruption and scripts remain in ignored private
`runs/chips/validation/20260922-storage-compare/` and operator server storage.
No shared server setting, previous job or deployed simulator environment was changed.
At that checkpoint, production doctor, storage tier selection and archive retry were unimplemented;
model calls, independent member reproduction and reboot recovery were not tested.

## Detached scratch/archive lifecycle and stage timing (2026-09-22)

The RC/VABench CLI now accepts a separate work root and optional archive root.
A persistent registration precedes launch. The detached worker seals the original
job, packs on the work filesystem, copies one staging archive to persistent storage,
verifies hashes and backend receipts, and publishes an archive receipt. Archive-only
retry never reruns simulation. Explicit cleanup requires fresh verification and keeps
an ID tombstone; persistent registration still prevents relaunch after scratch loss.
Usage, state/exit semantics and measurement boundaries: [STORAGE.md](STORAGE.md).

- Local TDD evidence records failing CLI behavior, then passing archive, retry/cleanup,
  and timing tests. Final affected Chips regression: **140 passed, 4 skipped**.
  Skips: no local ngspice and three opt-in live agent checks. Ruff lint/format, focused
  compilation and diff checks passed. No Bio/Robotics/Math domain runs were requested.
- Real server RC: server-local XFS work root, private NFS archive root, ngspice 47.
  Only the test SSH client was terminated after the wrapper reached an eight-second
  delay before real simulation. After twelve seconds without server connections, the
  first reconnect found **both simulation and archive completed beforehand**, passing
  analytical regrading. Same-ID resubmission preserved **one solver launch**.
- Original EVAS 0.8.7 replay with server-local source/job storage: `v4-001` DUT gold
  passed, DUT negative remained `behavior_failure`, `v4-501` Testbench gold passed.
  All three automatically archived and preserved original score sidecars. The installed
  simulator environment remained on NFS; this was not an environment migration.
- A separate real RC job encountered an intentionally occupied staging-file path.
  Its simulation still passed. Removing that isolated obstruction and dispatching
  `job-archive` completed publication with the same sealed completion hash and exactly
  one solver launch. This is a filesystem fault injection, not a live NFS outage.
  The first test controller stopped after dispatch because it read the previous failed
  status before the new worker updated it. That failed controller report is retained;
  a resumed verifier checked the already-finished archive retry without rerunning simulation.
- On that RC test job, a deliberately truncated archive was rejected by cleanup; after
  restoring and verifying the original bytes, cleanup removed only that job's work files.
  Subsequent same-ID submission returned the archived result without creating another run.
- All **five downloaded archives** passed offline package/member verification and the
  corresponding original backend receipt checks on macOS. Original negative verdicts
  remained negative. The snapshots of stage timing are retained in the archive receipts.

Single-run worker times were approximately 4.240 s / 4.036 s for DUT gold/negative and
17.477 s for Testbench gold. Successful archive-attempt times were 0.056–0.089 s for
these five small packages (approximately 50–57 KiB each), excluding final receipt/status
publication. These are observations with warm caches, not a throughput benchmark,
solver-only times, stable speedup, or guarantees for large waveforms/EMX outputs.
The delayed RC cases intentionally include two/eight seconds of test-wrapper waiting.

Tested bundle SHA-256:
`83fa074e8a6749ab6ae99726cb54e2d7033ea48c346fac6bc7e60ce03817beb8`.
Private scripts, the initial controller failure, resumed report, downloaded archives,
Red/Green records and JUnit report remain in ignored
`runs/chips/validation/20260922-job-storage/` and a separate operator server validation
area. The prior deployed bundle, simulator installations and historical experiments
were not replaced. The new bundle is deployed alongside them for explicit selection.

Automatic disk selection, persistent source-cache management, archive byte-range resume,
retention scheduling, reboot recovery, actual NFS/network failure, large-scale load,
commercial Spectre/EMX adaptation and real model/agent evaluation remain outside this check.

## 2026-09-23：VABench 公开工具与 Pi 协议闭环

实现与复现入口见 [VABench Agent](VABENCH_AGENT.md)。所有动作在私有 `demo/chips` 工作区开发，
未修改 vaEVAS 封存任务、评分代码或 EVAS，也没有运行 Bio/Robotics/Math 领域评测。

- **真实 EVAS，脚本策略**：family 001 的 DUT (`v4-001`)、bugfix (`v4-1001`)、
  Testbench (`v4-501`) 均完成错误语法候选 → 公开失败反馈 → operator 参考候选修复 →
  公开仿真和波形摘要 → 冻结 → 原评分器 `passed` → 最终归档 `verified`。
  这是编译错误恢复/接线验证，不是模型自主找出电路 bug 或全 release 认证。
- **真实 Pi + MCP + SSH，脚本 HTTP 模型服务**：Pi 0.87.0 实际仅看到四个公开工具，
  8 次脚本响应完成上述 DUT 循环；原独立终评通过，公开轨迹和最终包下载后分别验证，候选摘要一致。
  最终候选服务端包 SHA-256：`c24481fdc40f3248ad026b552ad97c3c0e7e535c822f7cea256cc8659efa3ebf`。
- **真实环境问题**：旧 Bubblewrap 不支持 `--clearenv`；适配器改用启动前 `env -i`，
  保留原挂载/网络隔离。公开执行前验证 EVAS 能启动、原源码及私有会话配置不可见、环境无凭据名称。
- **限额负例与修复**：首次在扩展里抛错仍被 Pi 捕获并继续请求，模拟服务实际收到 8 次请求，未通过。
  改为专用 Pi 进程在请求前退出 73 后，同样设置最多一次请求，模拟服务只收到 1 次，第二次被阻止。
  这证明该 Pi 版本下的应用请求限制，不是服务商计费或 HTTP 库内部重试认证。
- **恢复边界**：公开动作独立进程和同 ID 去重有本地真实进程回归，SSH 提交/查询在真实 Pi 流程中运行。
  本轮没有宣称对物理断网、控制机休眠、服务器重启进行了新增故障注入。
- **真实模型接入 smoke**：BigModel 中国区 Coding Plan 的 `glm-5.3-flash` 最小请求返回 HTTP 200；
  真实 Pi 发出两次模型请求，调用一次本地 MCP `vabench_read`，并在第二次请求后正常结束。
  模型总请求 usage 见私有 `glm53flash-smoke/model-budget.jsonl`，实际扣费未核对。
  此 smoke 的工具只返回一条本地固定文件名，没有执行仿真或终评，不能作为 VABench 能力结果。
- **真实模型 VABench 回合**：VPN 恢复后，`glm-5.3-flash` 在新的 `v4-001` 公开会话中完成
  7 次模型请求和 8 次工具调用（5 次读、1 次写、1 次公开仿真、1 次冻结提交）。公开仿真
  `succeeded`；原评分器独立终评 `pass`；公开轨迹和最终评分包下载后均校验通过，
  同一冻结候选 SHA-256 为 `b52fb7b325dfc2758b825d407e8fa3df473910be835e95efdf6f0480d49b48d9`。
  该候选与 operator 参考答案的 SHA-256 不同。报告为 `development_only`、`certified: false`，
  只支持单任务闭环结论；服务商实际扣费仍未核对。
- **接入故障与修复**：首次 CLI 模式把 MCP 子进程模块名误传为 `__main__`，Pi 扩展加载失败，
  在任何模型请求或公开动作前结束。改为明确模块名后，在同一未修改的服务器会话重试成功；
  未收到成功冻结回执时，入口现在停止终评并记录 `unsubmitted`。

私有原始证据在忽略目录 `runs/chips/validation/20260923-vabench-agent/`。
保留最初失败日志、修复复测、三任务报告、Pi 请求/响应、逐动作请求、公开波形、原评分收据及下载归档。
测试控制脚本曾将归档完成状态误写为 `archived`；修正为 `verified` 后按原 ID 恢复检查，没有重复仿真。
`pi-v6-fixture/report.json` 是最终候选包的 Pi 集成报告；`pi-budget-fixture-v2/` 是限额修复证据。
`glm53flash-001-retry/report.json` 是真实模型单任务终评报告，`glm53flash-001-retry/model-budget.jsonl`
记录全部 7 次请求和响应。第二位成员独立复现、扩展任务集及原 campaign 认证仍待完成。

本轮受影响回归最终结果：**189 passed, 5 skipped**（80.61 秒）。首次普通沙箱运行因禁止
绑定本机回环端口有 5 项环境性失败；在允许该端口的环境重跑后全部通过。跳过项为本机未安装 ngspice，
以及未启用的四个既有 live bridge/Pi 条件测试；本轮另用上述专用探针验证了真实 Pi/SSH/EVAS。
Ruff 检查、格式检查、Python 编译检查和 `git diff --check` 均通过。

## 2026-09-23：lab-server 配置集中化检查

[服务器端统一配置](SERVER_LOCAL_DEPLOYMENT.md)新增个人持久 profile、唯一 run ID 的准备命令，
从 profile 生成本地 Pi operator 配置，并先完成 VABench 公开会话与预检。构造的进程回复测试
覆盖非法字段、权限、路径重叠、pin 任务身份、重复 ID 和预检失败。真实 lab-server 上以当前独用
Linux 账号准备了一个 `v4-001` 公开会话：profile 为 0600、run/session 目录为 0700、生成
文件为 0600；预检成功，未启动模型、仿真或终评。这个检查证明配置入口接上了现有公开工具，
不增加新的芯片任务通过率结论。

当前 Pi/Python/Apollo 和 benchmark 源码仍展开在可清理的 scratch。个人持久目录保存了仅本人可读的
配置、pin、运行时/源码恢复包及 SHA-256 校验表；校验和检查与恢复脚本语法检查通过，
**尚未执行清空 scratch 后的恢复演练**。恢复包绑定当前账号与绝对路径，不能直接共享给其他用户。
跨账号隔离、管理员维护的只读共享安装、资源配额及多人并发均未验证。

本次受影响 Chips、workflow、执行层和 Pi 测试为 **161 passed, 2 skipped**（72.29 秒）。
跳过项是本机未安装 ngspice、未启用需凭据的 live Pi 测试。Ruff、格式、编译与 diff 检查通过。

## 2026-09-23：lab-server Spectre 环境修正

管理员指出应加载当前获准安装对应的两个站点环境脚本，而非先前使用的旧 IC618 环境。
在独立 C shell 中只加载这两个脚本，没有启动 alias 末尾的 Virtuoso GUI；
同一份无 PDK RC 网表、相同的 `spectre -64 ... -format psfascii -raw psf +lqtimeout 5 +mt=1`
参数重新运行。Spectre 仍为 21.1.0.509.isr12，本次退出码 0、耗时约 3.32 秒，
日志报告 0 errors、0 warnings，生成 `psf/dcOp.dc` 和 `psf/ac1.ac`。

两套环境的 Spectre 安装路径、`LM_LICENSE_FILE` 与 `CDS_LIC_FILE` 均不同；
因此先前的 SPECTRE-209 不能再解释为“lab-server 没有 Spectre 许可”。
这次单次成功只证明该环境下这个最小 RC 任务可运行，未隔离出旧环境中哪一项配置导致拒签，
也没有完成 VABench 的 Spectre Tool、完整任务集或生产 PDK 验证。
原运行脚本、控制台日志、Spectre 日志和 PSF 保存在本人服务器私有验证目录，不进入 Git。

## 2026-09-23：服务器本地 Spectre RC-001 闭环

按[标准化说明](SPECTRE_RC.md)在 lab-server 独用账号内建立 0600 operator profile、0700 scratch
工作根和 0700 持久归档根，加载管理员指定的两个 csh 环境脚本，仅运行 Spectre 命令行。
任务固定为 1 kΩ、1 nF、1 V 的无 PDK 理想 RC，输入和配置均在服务器；提交命令返回
`running` 后 SSH 会话退出，后台 worker 自主结束。工作目录和归档目录权限均为 0700。

首次真实作业使用服务器默认 Python 3.9，Spectre 已启动但原评分器调用了 3.10 才支持的
`zip(strict=False)`，作业以 `infrastructure_error/not_evaluated` 结束。删除该无行为差异的
参数并保留失败证据后，以新 ID 重跑。第二次作业 `execution=ok`、`verdict=pass`、
`AC=121` 点、瞬态 `1007` 点；AC 最大绝对误差 `6.6718e-7 < 1e-5`，瞬态最大归一化
误差 `3.0233e-6 < 5e-5`。服务器上的 `verify-job` 与 `verify-archive` 均通过，
归档状态为 `verified`。工作阶段约 `2.95 s`；该数值不包含 SSH 或排队开销。
同 ID 再提交仅返回既有终态，`completion.json` 摘要不变。
审查时新增对原始 PSF 的输入激励检查；最终复核包的 SHA-256 为
`69352084da596a855fcae828007044f8151ce153939cfe29d2facee255a34fad`。
用该包在服务器离线重新核验上述真实工作目录与归档，均仍为 `pass`；没有再次启动 Spectre。

构造的本地测试覆盖私有配置与脚本摘要、路径拒绝、PSF 缺失/非有限值、真实进程边界、
许可证失败分类、后台作业、归档和产物篡改拒绝。它们不替代实验室 Spectre 实测。
原始 PSF、日志、配置和归档留在个人服务器私有目录，不上传 Git。尚未测试强制 SSH
中断、服务器重启、Spectre 的取消/超时、多人账号部署、PDK/晶体管或 VABench 的 Spectre Tool。

## 2026-09-24：Analog RLC 模型试点启动条件

在 lab-server 独用账号上，对已创建但尚未执行动作的 100 MHz RLC 会话运行只读
`chips_analog_agent preflight`。Pi 可执行文件、Python MCP 依赖、离线 bundle、
会话公开文件、固定上游源码、Podman 镜像与直接模型 HTTPS HEAD 均返回 `ready`；
`credential` 返回 `missing`，整份报告为 `blocked`。HEAD 只证明地址可达，
`model_auth` 和 `simulator_run` 都明确为 `not_tested`。预检未发送模型请求、
未启动仿真，也未消耗会话动作预算。此结果不构成 Analog 的 Agent 闭环。
随后将已推送的 `a1932d1f` 源码归档及标准离线 bundle 按 SHA-256 核对后部署到新的私有
scratch，另复制到 0700 持久目录并逐文件复核。bundle 摘要为
`ca3578ae7efaac55002213aacf8e2c8a01f451833901de53850b79939a2da9d8`；
以这一版本重新预检仍只有 `credential: missing`，未执行模型或仿真。

本地构造边界回归中，私有密钥文件入口及预检相关 **18 passed**；Pi 原始事件流
捕获、私有 Episode 纳入和受影响外部桥接测试 **36 passed, 3 skipped**。
跳过的是未启用的 live agent 条件测试。原始 Pi JSONL 的写入与 Key 文本替换已做
构造回归，真实 GLM 事件流仍待模型回合产生。部署和评分证据层次见
[Analog Design Bench 说明](ANALOG_DESIGN_BENCH.md)。

## 2026-09-24：VABench 本机原生 Codex 入口

VABench 增加 `codex` 模式：本机显式选择 ChatGPT 登录或 OpenAI API Key，固定 CLI、
模型和回合超时，服务器公开预检后经 stdio MCP/SSH 调四个公开 Tool。原始 JSONL、
规范化结果、动作请求/响应分别保存在本机私有证据目录；未确认提交不会启动独立终评。
认证预检只记录方式和状态，不保存 Key；GLM Key 不进入 Codex 进程环境，MCP 子进程清理父环境后启动。

首个红灯测试在 CLI 不接受 `codex` 模式时失败；原始事件回调与缺少 ChatGPT 登录
也分别先观察失败，再实现。构造 Codex CLI/服务器回复回归和真实 MCP 子进程工具发现、
动作日志测试共 **5 passed**（单测文件）；后者使用构造的 SSH 回复，不证明真实服务器响应。
受影响 VABench/Codex 回归为 **54 passed, 1 skipped**（4.07 秒）。超时原始尾部
另经先失败后通过的测试覆盖；Codex/通用桥接/Chips 组合在可绑定本机回环端口的环境中为
**64 passed, 4 skipped**（7.77 秒）。普通沙箱中的 5 项失败均为回环端口绑定被拒，
扩大权限重跑后通过。相关 Ruff、格式、编译和 diff 检查通过。
本机 `codex-cli 0.154.0` 的 `login status` 返回 ChatGPT 已登录；只读 SSH 探测
`lab-server` 返回 `READY`。这两项只证明本机认证状态和 SSH 可达，不证明模型/Tool
认证或服务器任务可运行。真实 Codex 模型请求、实验室 SSH/EVAS 回合、
独立终评和跨主机轨迹关联尚未运行，不能把这些本地回归记作模型解题成功。

## 2026-09-24：本机原生 Codex + lab-server VABench 真实回合

选择本机 `codex-cli 0.154.0`、ChatGPT 登录、`gpt-5.6-luna` 和显式
`model_reasoning_effort=medium`；任务为 VABench r53 的 `v4-001` DUT，公开与终评
后端均为 lab-server 上固定的 EVAS 0.8.7。首个全新会话在模型请求前被认证预检拦下：
该 Codex CLI 将 `login status` 写到 stderr，而入口当时只检查 stdout。保留该失败证据，
按 TDD 修复并重新建立会话；没有把首轮算作模型尝试或仿真结果。

第二个全新会话经本机 Codex → stdio MCP → SSH → 服务器公开会话完成：
5 次成功读取、1 次候选写入、1 次公开 EVAS 仿真成功、1 次错误路径读取被拒，
随后成功冻结提交，共 9 次 MCP Tool 调用。原始 Codex JSONL 有 27 条记录；
CLI 报告 211255 输入 token（其中 154624 为 cached input）、2032 输出 token，
费用未测量。没有观察到 Codex 内建命令或文件修改动作，但其可用性并未被禁用，
因此此回合不能与 Pi 的强制仅四 Tool 条件当作单变量对照。

独立终评在提交后运行，`report.json` 为 `state=verified`、`execution=ok`、
`verdict=pass`；公开 episode 和终评 job 两份归档均已下载、校验，并确认使用同一
冻结候选。评分权威仍是 `development_only`，不代表原 benchmark 的正式 campaign
成绩，也没有执行 Cadence Spectre 对照。原始事件、工具请求/响应、私有归档和报告保存在
Git 忽略的 `runs/chips/validation/20260924-codex-luna-medium-002/evidence/`。
本轮受影响 Codex/VABench 回归为 **49 passed, 1 skipped**；Ruff、格式和 diff 检查通过。

## 2026-09-24：冻结 `v4-001` 候选的 Spectre 公开波形对照

从上节已校验的 Codex episode 归档只取公开 `visible_test.scs` 和冻结的
`bbpd_ref.va`，分别核对 SHA-256 为 `41cd2e82…94b113c` 和
`3d0747f1…32245584`，再传到 lab-server 本人独用账号的 `0700` 临时目录。
服务器用管理员指定的两个 Cadence 环境脚本在脱离 SSH 的后台运行
Spectre 21.1.0.509.isr12，并把日志、原始 PSF 和退出记录复制到个人持久目录。
实际回合退出码 0，`tran` 覆盖 0–70 ns，3650 个 accepted steps，
Spectre 日志为 0 errors、1 warning。该 warning 是 `VACOMP-2435`：环境中的
`CDS_AHDLCMI_ENABLE` 已不再支持，Spectre 使用默认编译 C 流程；不是许可证失败。
本机下载的日志和 PSF 哈希与服务器持久副本一致。

公开 EVAS CSV 有 3540 个时间点，Spectre PSFASCII 有 3651 个时间点。
用新增的固定信号比较器将 Spectre 线性插值到 EVAS 时间点：`up` 和 `down`
全时段最大绝对差均约 **2.25 mV**，平均差均约 **10.17 µV**。
距输入阈值切换至少 0.2 ns 的 CSV 样本上，五个信号的最大差和逻辑值不一致次数均为 0；
全时段 `up` 有 2 个阈值附近样本的逻辑值不一致，不能称为逐点完全相同。
比较器拒绝意外的 PSF 信号集合，相关构造回归为 **2 passed**。

此对照只覆盖这一个冻结候选与这一份公开测试 deck。原 EVAS 独立终评仍是
`development_only/pass`，没有以 Spectre 结果改写成绩，也未复现完整 VABench campaign、
PDK 或晶体管级验证。原始候选、公开 CSV、PSF、日志与逐信号比较 JSON 留在 Git 忽略的
`runs/chips/validation/20260924-spectre-parity-codex-v4-001/`。

## 2026-09-24：VABench 单次实验账本

Pi 与原生 Codex 的新回合在预检前写入 1 题 × 1 次的任务清单、实际配置、Agent 输入和
一行 `results.jsonl`；`summary.json` 从该行生成。终评未到、未冻结、预检阻断和异常均不
冒充电路失败；完成的 `pass` 与 `fail` 分别记为 true/false。单次收集可更新原行，
不重跑模型。服务端预检新增任务 ID 与会话文件 SHA-256 回报，声明 ID 不符时在模型调用前阻断；
旧服务器未回报时明确保持未验证。后续收集必须匹配原 operator 配置。

行为先行测试分别观察到缺少账本、错误任务仍启动 Agent、错误配置可收取、旧证据被复用和
公开目录可写入等失败，再实现对应边界。本地 Chips 与 Workflow 回归为 **150 passed**；
受影响 Ruff、格式、编译及 diff 检查通过。服务器回复和 Agent 输出使用构造夹具，
没有为这次格式改动再次调用真实模型或 EVAS，也不把已有历史回合补写成新格式。

## 2026-09-24：另一题的双案例实验

按单次实验账本重新运行 VABench r53 的 `v4-002`（Capacitive SAR Feedback DAC），
lab-server 本机 Pi 调用 `glm-5.3-flash`，公开和终评使用固定 EVAS 0.8.7。
首回合 `pi-v4-002-001` 只有 5 次读取，模型最后一条原始响应为
`stopReason=length`，没有候选、公开仿真、提交或终评；旧 Pi 解析器却把终止原因记为
`final`。该失败回合单独归档。修复长度判定并允许每回合指定 thinking 后，
全新 `thinking=off` 回合 `pi-v4-002-off-002` 依次完成 2 次读取、1 次写入、
1 次公开 EVAS 仿真和 1 次冻结提交；6 次模型请求，总墙钟 53.29 秒。
公开仿真无 lint/sim 错误，独立终评复用同一冻结候选并正常执行，最终
`benchmark_status=behavior_failure`、`verdict=fail`。账本为 1 题×1 次、
`completed_attempts=1`、`benchmark_success=false`；终评权威仍为
`development_only`。这验证了实验平台的 Agent→Tool→仿真→终评链路，**没有**
验证该候选通过电路行为要求。

Analog Design Bench 另选 `rlc-broadband-50-to-200-match`，在 lab-server 用固定
Podman RLC 镜像、ngspice 和原版评分脚本分别运行参考解与 starter：
参考解 `graded`、7/7、1.0；starter `graded`、0/1、0.0。两次均为操作者侧
环境正负控，不能算 Agent 解题。对 `rlc-rf-bandpass-100mhz` 还保留了三次
真实 Pi/GLM 诊断回合：每次 3 次模型请求、5 次只读 Tool，最后均因输出长度
耗尽而未写入、仿真或提交；后两次正确报告 `truncated`，终评均未启动。
因此 Analog 的 Agent 自主仿真闭环仍是缺口。

下载后的私有证据位于 Git 忽略的
`runs/chips/validation/20260924-two-case-experiment/`；其中 `record.md` 按任务、
动作、耗时、评分与失败边界给出可审查的文件索引。VABench 完成回合的公开
episode 归档 72 个文件、终评归档 36 个文件均通过离线逐成员校验；首次未提交
回合和 Analog 诊断证据也已按收据校验。两个最小代码改动分别在
`876ecef3` 与 `6cd4d537` 提交并推送；受影响回归 **67 passed, 1 skipped**，
Ruff 和 diff 检查通过。模型费用未测得；私有凭据未进入证据包或 Git。

## 2026-09-24：GLM 输出截断原因与 Analog Agent 对照

对同一 `rlc-rf-bandpass-100mhz` 任务回看前三次真实 Pi/GLM 回合：第三次模型响应
分别在 4096、4096、8192 token 处 `stopReason=length`，其中推理 token 分别为
4091、4095、8191；仅完成 5 次公开读取。最后一次请求体约 8 KB，未碰到
131072 字节的请求上限；整个 episode 已有 3 次模型请求和多轮 Tool 交互，
不是输入上下文装不下或只有一轮 API 请求。安装的 Pi 在 `thinking=off` 时
会发送 `thinking: disabled`，而 [GLM-5.3-Flash 官方说明](https://docs.z.ai/guides/vlm/glm-5.3-flash)
要求保持思考开启。原自定义模型配置也未启用 `reasoning_effort` 传递。

提交 `6e5758d6` 据此拒绝该模型的 `off`，当时默认使用 `low`，让 Pi 把
`low/high/xhigh` 映射为模型支持的 `low/high/max`，并把 Analog 同机模板的
单次输出上限从 4096 调至 8192。行为先行回归和受影响 Pi 检查为
**38 passed, 1 skipped**，Ruff、格式与 diff 检查通过。两个新私有会话实测：

| 回合 | 档位／单次输出上限 | 关键结果 | 独立终评 |
| --- | --- | --- | --- |
| `004` | `low`／8192 | 6 次模型请求；`read×5 → write → simulate → submit`；第三次响应输出 2774、推理 2348 token | 冻结候选与评分输入哈希一致；原 Podman/ngspice 评分器 `graded`、15/15、1.0 |
| `005` | `low`／4096 | 第三次响应输出 3333、推理 3005 token；写入和仿真已发生，但后续耗尽 12 次模型请求与 3 次仿真预算，未提交 | 未启动；不记作 0 分 |

因此 4096 **不是**“首次候选一定生成不了”的硬门槛；把上限增到 8192 也曾
在错误思考配置下完全被推理消耗。支持的低推理档位加合理余量让本次 Agent
真实完成了一题，但不能从一个通过回合推断稳定成功率。单个 Pi episode 内已有
多轮交互；缺少强制“先候选后细化”的阶段检查点仍可能造成长推导或预算耗尽，
需要另设对照验证。

`004` 的模型原始事件、Tool 请求／响应、公开仿真、冻结候选及终评封存在 Git 忽略的
`runs/chips/validation/20260924-two-case-experiment/analog-low-004/`，106 文件 Episode
已在服务器和下载本机后离线核验；`005` 的未提交私有证据包也逐文件核验。
同目录 `record_glm_thinking.md` 给出各次 token、动作和证据索引。密钥未进入归档；
实际模型费用未测量。

## 2026-09-24：实验前显式核对推理档位

真实模型回合不再依赖 Pi、Codex CLI 或模型服务的隐含推理默认值：Analog 和 VABench Pi
要求 operator 中显式填写 `thinking`，VABench 原生 Codex 要求 `reasoning_effort`。
GLM-5.3/5.3-Flash 拒绝 `off`，VABench 的 Pi 自定义模型定义也启用了推理档位映射。
Analog 预检返回 `model_settings`；VABench 服务器端 `prepare` 返回同类字段，
`episode-budget.json` 保存本次模型设置。缺失档位在创建 Analog Agent 证据或
VABench 服务器新 run 之前被拒绝；构造模型夹具仍可省略。

本地受影响回归：Analog **15 passed**，VABench **38 passed**；Ruff、格式、
模板 JSON 与 diff 检查通过。未在此改动后再次请求真实模型或启动实验室仿真，
因此这些回归只证明配置、记录与构造进程边界，不是新的模型成绩。
后续新回合必须在私有 operator/profile 中显式选择档位，不再使用上述默认值。

## 2026-09-24：Analog 与 VABench 共用实验层模型设置

两个任务入口现共用一处模型 ID、Pi `thinking`、Codex `reasoning_effort` 与
GLM-5.3 档位校验；Pi 的 Z.ai 模型描述也由同一模块生成。Analog 预检、
VABench 部署准备和 Pi/Codex `episode-budget.json` 新增同一版本化
`experiment_settings` 结构，包含 Agent、模型、推理参数与预算；旧
`model_settings` 输出保留。任务专有的 SSH／
同机传输、会话与仿真器字段仍留在各自 operator/profile 中。

本地受影响回归 **53 passed**，完整 Chips 测试 **156 passed**；Ruff、格式、
编译与 diff 检查通过。本次是配置与记录层
重构，未重新启动真实模型或实验室仿真；之前的回合证据不会被追溯改写。

## 2026-09-24：VABench 与 Analog 共用单次实验入口

新增 `chips_experiment run/collect/finalize/archive`：私有实验 JSON 引用原任务
operator；共同清单冻结两份配置的 SHA-256，统一写入一行 `results.jsonl` 和
`summary.json`。VABench 可在同一回合收取尚未完成的独立终评；Analog 只在确认
冻结提交后启动原终评，并校验、归档私有 Episode。终评分数 0.0 是有效结果，
Analog 不被强行转换成二元通过率。评分后的归档故障或进程中断可以只恢复归档，
不重跑模型和终评；终评调用超时则保持结果不明，不自动重复评分。

本地 `tests/chips` 加 `tests/workflows/test_chips.py` 回归 **171 passed**；
新增的共同入口测试用构造 Agent 与服务器响应覆盖配置冻结、公开提交、
VABench pass/fail、Analog 0 分、归档恢复、超时与私有 Key 引用。
本轮没有重新调用真实模型、SSH、EVAS、ngspice 或其它实验室仿真器；
因此上述通过数证明的是编排与记录逻辑，而不是新一轮电路设计成绩。

## 2026-09-24：`demo/chips` 仓库范围清理

移除本分支未使用的其他领域示例、展示站、专题文档、AlphaHebe 应用及其专用测试；
保留 `alphaapollo` 共享内核和现有领域适配的公开接口。两个通用 Workflow
测试原来直接读取已移除的 AIME 配置，现移除这些只验证示例存在的断言；
其余运行时行为测试保留。旧的 `main` 全域 CI 配置也从 Chips 分支移除，
Chips CI 增加导入边界、Schema 与编译检查。范围说明见
[仓库保留范围](REPOSITORY_SCOPE.md)。

本机 Chips CI 对应回归 **284 passed, 4 skipped**；通用 Workflow 相关回归
**97 passed**，另一个回环 socket 用例在允许本地端口绑定的环境下 **1 passed**。
Ruff、格式、Schema 快照、编译、Markdown 相对链接及 Workflow YAML 解析通过。
可选 wheel 构建未执行：现有 `.venv` 缺少 `pip`/`setuptools`，没有为这次
目录清理安装新依赖。没有运行 Bio、Robotics、Math 的领域测试，也没有新模型或
实验室仿真调用。

## 2026-09-24：复用共享查看器的离线 Episode 报告

新增 [报告入口](EPISODE_REPORT.md)，复用共享 HTML 渲染，不新建模型/仿真执行循环。
共享查看器移除固定 `p0002` 结论，缺失工具信息显示 unavailable。
Chips 报告关联保存的输入、Agent 事件、模型预算、Tool action ID、候选修改与独立终评；
保留不同任务的评分语义，Analog 连续 reward 不转换为 `benchmark_success`。

行为先行的 Red→Green 覆盖：报告生成、预算耗尽未提交、候选修复关联、冻结字节与终评绑定、
归档重新校验、Analog 仿真失败状态、统一设置读取、HTML 证据跳转及无效执行不发布分数。
另补缺失响应/计时、预检阻塞、损坏 JSON、拒绝覆盖/写入原目录等回归。
相关测试 **44 passed**（报告、共同实验入口、Analog Episode、VABench 归档及共享查看器）；
其中新报告文件 **13 passed**。全仓 Ruff 与格式检查通过，受影响 Python 编译、diff 和文档链接通过。
没有运行其它领域的任务评测。

对四条真实历史证据离线生成报告，并在本次生成时禁止启动子进程；每个被读取来源在生成后再次
核对 SHA-256，全部保持不变。新生成的私有报告位于
`runs/chips/analysis/offline-reports-20260924-final/`，不纳入 Git。

| 历史记录 | 离线结果 | 观测与边界 |
| --- | --- | --- |
| `glm53flash-server-002`，VABench `v4-001` | `success_after_repair` | 9 次模型请求、10 次工具调用；模型区间合计 262.084 秒，工具 7.742 秒；重验公开/终评归档共 142 个成员，候选一致；评分权威仍为 development_only |
| Analog `analog-low-004` | `success` | 6 次模型请求、8 次工具调用；原记录 reward=1、测试 15/15；冻结候选与最终输入字节相同，但此输入是展开目录，报告不宣称重新验证整个归档封印 |
| Analog `analog-agent-attempts/001` | `budget_exhausted_unsubmitted` | 3 次模型请求、5 次工具调用；最后输出 stopReason=length；终评未运行，分数和 benchmark_success 为空 |
| `codex-luna-medium-002`，VABench `v4-001` | `success` | 9 次工具调用；重验公开/终评归档共 122 个成员；模型请求级耗时未记录，不从 Tool 耗时或 outcome 用量推测 |

Pi 修复案例的累计 totalTokens=94,204 与此前人工复盘一致；reasoning 不重复相加。
控制端 Tool 时间与服务器 action 时间分别显示，不能把两者差值直接解释为网络耗时。
这次验收证明离线报告能消费已有真实证据，不是重新验证模型能力、仿真器或实验平台部署。
原始未记录的 provider 请求、CLI 系统上下文、工具发现与内部推理仍无法补回。

## 2026-09-25：Chips Environment 与共享 Runtime 的本地验收

vaBench／Analog 公开 Tool schema 已接入共享 MCP catalog，
`common/environment/chips.py` 将原任务 session 投影到 Apollo Environment。
两任务保留各自输入、仿真反馈、冻结和原评分器规则。公开转移的 `reward=0` 是接口占位，
`success=None`、`evaluation=not_performed`，不能作为 benchmark 成绩。
Pi 的共同启动配方提供显式 `runtime: external`；省略仍为旧的 `direct` 路径。
原始 Pi 事件与标准 `runtime-result.json` 同时保存。原生 Apollo 对照采用
`tests/chips/probes/native_vabench.py`，目前是验证探针，未宣称统一实验 CLI 已支持该 Agent。

本轮行为测试发现并修复四项完整性问题：

- 重建客户端后恢复未决 action ID，未恢复前禁止开始新动作；同 ID 不重启 worker。
- 关闭 Runtime 时中断客户端等待，已接收的独立服务器作业继续运行。
- 外部 CLI 在工具返回后因预算等原因退出时，不再伪造一个没有对应模型消息的最终环境步骤。
- Analog 在冻结或动作预算边界拒绝调用时，将拒绝响应保存到后台请求目录，避免客户端无限等待。

GLM 隔离 Pi 配置新增 `http_retry_limit: 0`，同时关闭 Agent 与 provider 重试。
真实 Pi **0.87.0** 配合本机脚本 HTTP 服务验证两任务的读取、错误、修复、提交、
请求次数／字节预算阻断，以及 `thinking=low` 的实际请求字段。
HTTP 500 fixture 验证 direct／external 均只发一次请求。
原生探针用真实 OpenAI SDK 与构造 HTTP transport 验证相同 thinking 字段、完整请求体字节守卫、
零重试和轮数停止；这些检查均不访问真实模型，也不运行实验室仿真器。

最终代码 `fd524bc4` 的 Chips CI 范围与受影响共享 Runtime 回归为 **522 passed、5 skipped**，
JUnit 保存在私有运行目录。跳过项为本机缺少 ngspice，以及未启用的共享 live Math 工具、
Pi／Claude bridge、真实 Pi 模型测试。Chips 自己的真实 Pi CLI＋本机 HTTP 检查已启用并通过。
Ruff、格式、Schema 导出、Python 编译和 diff 检查通过。早先完整回归发现的 4 个旧配置断言
遗漏了新增 `runtime` 字段，已在独立提交中修正；没有将它们记为跳过。

**服务器验收仍未完成。** 本轮起初可经 SSH 核对私有目录、已有任务 pin 和凭据文件权限，
随后 SSH 在 TCP 建连阶段连续超时；禁用连接复用的独立重试也未恢复。
本轮源码与离线包已在本机准备，但尚未上传部署，真实模型请求、公开仿真和独立终评均为 **0 次**。
这不是模型服务或仿真器运行失败的证据。E4 固定正负控／local–SSH 对照／运行中断线恢复，
以及原定 M1–M4 各三次、共 **12 个服务器 Episode** 均保留待执行；历史成功记录不能替代。
网络恢复后先完成 E4，再按首轮四条件的完整性门槛推进 E5。本机原生 Codex／SSH 的既有
vaBench 路径保留；Analog 的 Codex／SSH 验收仍为独立未完成项。


## 2026-09-25：恢复连接后的真实工具路径验收（E4）

旧源 `37ca3c2d` 与新源 `fd524bc4` 已分别部署并校验源码／离线包摘要；
Pi 固定为 0.87.0，任务 pin、EVAS 0.8.7 与 Analog Podman 镜像保持冻结。
本段是操作者固定候选回放，不是模型自主解题成绩。

| 验收 | 实测结果 |
| --- | --- |
| R1：两任务 × 正负候选 × 新旧路径，8 个有效会话 | 四组公开反馈、候选摘要、冻结输入与原评分器结果等价；波形／测量值要求精确相等 |
| vaBench `v4-001` | 新旧正例均 `passed`；负例均 `behavior_failure` |
| Analog RLC | 新旧正例均 reward=1、15/15；负例均 reward=0、0/15 |
| R2：服务器 local 与本机 SSH，两个新会话 | 同一 vaBench 候选的公开语义结果相同；不据此作延迟比较 |
| R3：关闭 Runtime 后恢复 | 已观察到服务器 worker 运行且尚无响应；关闭客户端后任务独立完成，重建客户端按原 action ID 收取，核对只有一次仿真请求且 worker PID 未变 |

R3 为本次独立 public worker 注入 **15 秒 SIGSTOP／SIGCONT 暂停**，
随后运行真实 EVAS；没有更改共享网络或其它用户进程。该证据验证 Runtime
关闭／客户端中断恢复，不能扩展为服务器重启、硬件故障或任意网络故障验收。

保留全部预试和操作故障：首个候选路径错误发生在任何动作之前；第一对日志比较
因“截断字节数”随耗时文字长度变化而失败，核实上游截断算法后修订比较器，
用全新会话重跑；只忽略列明的 ID、时钟及截断长度元数据，不放宽电路数值或评分。
Analog 归档目录重名发生在终评启动前，换为会话独立目录后从冻结候选继续；
R3 操作者轮询竞态留下的一次预试也保留，未冒充有效断线验收。

包括这些预试，本轮实际 **14 次公开仿真调用、10 次独立终评**；
评分器内部求解次数不包含在这两个调用数里。原始证据留在私有
`runs/chips/validation/20260925-runtime-migration/` 及服务器持久存储，
约 5.0 MB 的收取包和 **421 个成员**均已在本机逐项校验 SHA-256。
本段不修改此前本地测试的通过数。E5 的模型回合另记，E4 不替代其验收。

## 2026-09-25：12 个服务器真实模型回合（E5）

全部 **12 个 Episode 已执行**，保留失败和未提交结果，没有追加回合来补成功。
两任务均为已知开发题，结果不代表正式 benchmark 认证或 Agent 排名。

旧 Pi 路径固定 `37ca3c2d`，原生 Apollo 固定 `fd524bc4`；共享 Runtime＋Pi 首次
使用 `fd524bc4`，发现轨迹缺陷后，剩余 M2 与全部 M4 改用 `1f3e1741`，记为条件 v2。
M2 v1 的失败没有并入 v2；因此 v2 只有两次 vaBench 重复。
Pi 固定 0.87.0；原生 SDK 为 OpenAI 3.17.0。所有 Agent、公开仿真和终评均在 lab-server。
模型请求均为 `glm-5.3-flash`、显式 `reasoning_effort=low`、
`thinking={type:enabled, clear_thinking:false}`，每回合上限为
12 次请求／8192 单次输出 tokens／64000 请求体 bytes／24 个服务器动作／4 次公开仿真。
任务、EVAS／Podman 镜像、原评分器沿用 E4 固定条件，未提高预算或更改提示来重跑失败。

表中动作是**服务器收到的请求**，不包含 Pi 本地 schema 拒绝或原生 Runtime 拒绝的并行调用。
仿真列是实际公开执行次数，拒绝的超预算请求另留在原始轨迹中。

| 回合 | 路径／条件 | 模型请求 | 服务器动作 | 公开仿真 | 提交／独立终评 | 执行完整性 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| M1-1 | 旧 Pi／v1 | 8 | 7 | 1 | 已提交；pass | 模型正常；操作端 UTF-8 读取恢复后终评 |
| M1-2 | 旧 Pi／v1 | 6 | 5 | 1 | 已提交；pass | 完成 |
| M1-3 | 旧 Pi／v1 | 8 | 7 | 2 | 已提交；pass | 完成 |
| M2-1 | 共享 Runtime＋Pi／v1 | 9 | 8 | 2 | 已冻结；另行原定终评 pass | **Runtime 轨迹汇总失败**，保留原失败 |
| M2-2 | 共享 Runtime＋Pi／v2 | 6 | 5 | 1 | 已提交；pass | 完成 |
| M2-3 | 共享 Runtime＋Pi／v2 | 6 | 5 | 1 | 已提交；behavior_failure | 完成；有效的电路失败 |
| M3-1 | 原生 Apollo／v1 | 9 | 8 | 1 | 已提交；pass | 完成 |
| M3-2 | 原生 Apollo／v1 | 9 | 7 | 1 | 已提交；pass | 完成 |
| M3-3 | 原生 Apollo／v1 | 9 | 7 | 1 | 已提交；pass | 完成 |
| M4-1 | 共享 Runtime＋Pi／Analog v2 | 12 | 15 | 4 | 未提交；未终评 | 请求预算耗尽；第 5 次仿真请求被拒绝 |
| M4-2 | 同上 | 12 | 14 | 4 | 未提交；未终评 | 请求预算耗尽 |
| M4-3 | 同上 | 3 | 5 | 0 | 未提交；未终评 | 第 3 次响应以 length 截断 |

合计 **97 次模型请求、93 个服务器动作、19 次公开仿真、9 次独立终评**。
终评为 8 个 pass、1 个 behavior_failure，其中一个 pass 来自 Runtime 失败回合的已冻结候选，
不能计作平台完整验收通过。未提交没有分数，不补零分，也没有由操作者替模型提交。

### 实测发现与修复

M2-1 的模型第一次调用 `vabench_read({})`，Pi 在 MCP 调用前因缺少 `path` 拒绝；
模型自行修正并完成仿真／提交，但 Runtime 错把这次本地拒绝当作一次服务器 step，导致后续轨迹错位。
`1f3e1741` 增加桥接注册／实际 dispatch 证据，仅将可确认的本地参数拒绝标成
`environment_stepped=False`；保留错误返回和模型动作，未经 Environment 执行的其它工具活动仍拒绝。
真实 Pi＋本机脚本 HTTP 夹具先复现失败，再在 vaBench／Analog 两种任务上通过；
最终 Chips 与受影响共享回归 **529 通过、5 条条件跳过**，CI 通过。
跳过项为本机未安装 ngspice，以及未启用的共享 Math／Pi／Claude 真实外部集成；本轮实际 Pi 夹具已运行。

操作端另保留两类恢复：M1-1 的服务器默认 GBK 导致原始 UTF-8 事件后处理失败；
M4-1 将被拒绝的第 5 次仿真请求误计为实际仿真。两者都只恢复既有结果和计数，未重复模型／公开仿真。
M4-1 按时间戳排序是先取得数值但指标严重不达标，随后三次候选仿真遇到奇异矩阵等错误。
M4-3 的原始 usage 为 output=8192、reasoning=8190、stopReason=length；
实际传入 low 不等于这一回合只消耗少量 reasoning tokens，不能把参数名当成输出预算保证。

### 证据与验收范围

`367d7113` 为既有离线报告补充原生 Apollo 的 HTTP 请求／返回、标准 Runtime 回合，
以及按请求 ID 关联的 Pi 实际 provider payload；没有用当前提示补写历史上下文。
报告定向回归 **18 通过**，对应 CI 通过。原生探针只有相对时钟，报告保留时长，
不伪造它与 Tool 的绝对时间交错；未输出的内部思考、服务端权重版本也不作推断。

E5 原始数据位于私有 `runs/chips/validation/20260925-runtime-migration/` 和服务器持久存储，
与源码提交分离。归档排除密钥和临时 CLI 工作区，不跟随 sandbox 符号链接；
M2 原失败、操作端恢复前证据和未提交候选均保留。12,656,467 bytes 的收取包及全部
**1,820 个文件**已在本机校验 SHA-256；17 个符号链接只记录，不跟随收取。
包 SHA-256 为 `c3f625eaefa8943c4c3ea4fdfee12e26d45b1e58e126e0266ff58a8cfeaf1ad0`。
本机离线入口为该私有目录下的 `review-e5/index.html`，可打开全部 12 份报告。
另核对全部 93 个客户端／服务器请求与返回、97 次实际 provider 请求的冻结设置、
有效 Runtime 中的服务器动作 ID，以及 9 份提交与终评候选的归档哈希；均对应一致。

**结论：**E4 新旧工具语义及客户端中断恢复通过；vaBench 的旧 Pi、共享 Runtime＋Pi、
原生 Apollo 均取得真实“仿真→提交→独立终评”证据。M2 的电路失败也证明公开仿真成功不等于终评通过。
**Analog 三回合均未提交，E5 要求 M4 至少一次完整提交闭环的门槛仍未满足。**
当前不宣称整个迁移全部验收通过，不将这些少量开发题回合用于统计优劣比较。
下一组 Analog 实验应先明确模型可见的剩余预算、提交机会及 thinking／输出预算条件；
本轮不追加请求、更换预算或强制提交。原生 Codex＋SSH 的 Analog 验收仍是独立未完成项。

## Analog completion and budget repairs (2026-09-25)

Four independent fixes (`7afaefca`, `94298258`, `382f92c6`, `cc2535f0`) add
operator collection of the last complete candidate on acknowledged termination,
Agent-visible request/action/simulation budgets, explicit stop classifications,
and separate harness/submission/design outcomes in offline reports. Legacy
sessions retain their explicit-submit policy; the original twelve E5 episodes
were neither resumed nor reclassified as successful submissions.

Behavior-first regressions cover pending server actions, immutable collection,
missing candidates, reserved submission capacity, request/byte/output limits,
raw versus harness termination reasons, candidate identity and failed-episode
archiving. Real Pi 0.87.0 with a local scripted HTTP service confirms that the
remaining request count reaches the actual payload and that limit/truncation
does not trigger extra model requests. This fixture does not measure GLM quality.

The affected Chips and shared runtime suite passed **557 tests, with 5 skips**
in 138.79 seconds. Skips are local ngspice absence and disabled optional live
Math/Pi/Claude bridge checks; no unrelated domain experiment ran. Ruff lint and
format, schema export, compilation and whitespace checks passed. CI on
`cc2535f0` passed; the first commit's formatting-only CI failure was corrected
in `94298258`. The new offline report still classifies the unchanged historical
M4-1 as `budget_exhausted_unsubmitted`. Private test evidence is under
`runs/chips/validation/20260925-analog-repair/`.

A new server-side Analog validation (`analog-repair-002`, code `cc2535f0`)
completed through the unified experiment entry with real Pi + GLM-5.3-Flash
(`low`) and the pinned Podman/ngspice evaluator. Limits remained 12 requests,
8192 output tokens per request, 64000 request bytes, 24 actions, four public
simulations and a 1200-second episode deadline; HTTP retries were disabled.
The first new deployment attempt stopped at dependency preflight with **zero
model requests** because its operator selected the system Python. That record
was preserved; the new attempt selected the existing dependency-enabled venv.

Observed result: **6 model requests, 8 server actions, 1 public simulation,
1 explicit Agent submission, independent score 1.0 with 15/15 tests passed**.
Agent plus finalization/archive took 123.57 seconds. All six captured provider
payloads include the decreasing request notice and the declared model, thinking
and output limit. No extension error or automatic HTTP retry was observed.
All eight client/server actions matched; the submitted, frozen and final-input
candidate hashes agree. The episode archive's **116 members** and the downloaded
evidence's **141 files** were hash-verified locally. Raw evidence and the
expandable offline report remain private in the run directory above.

This is one genuine Analog submission/independent-evaluation success under the
new contract, not a retroactive E5 pass or a reliability estimate. Automatic
collection and missing-candidate branches have process/fixture regression
coverage; this live model episode exercised explicit submission. The separate
local-native-Codex/SSH Analog acceptance and broader task coverage remain open.

## Task-authoring D0 pilot (2026-09-25)

Four separate implementation commits (`4b65aa44`, `4e2bba2f`, `5f1b63ef`,
`0a82f48e`) add source-bound drafts/operator confirmation, bounded differential-gain
Spectre execution, server-side public budgets/frozen submissions, and native Codex
image extraction/construction phases. Each was pushed separately; all four CI
runs passed. Usage and safe constructed materials are in the
[task-authoring example](../../examples/chips/task_authoring/README.md).

The current local Chips/bridge suite passed **401 tests, 16 skipped**. Skips cover
12 opt-in real-Pi cases, unavailable local ngspice, and three disabled optional
live Math/Pi/Claude bridge cases. An earlier sandboxed run had 10 failures because
binding localhost sockets was prohibited; the unchanged suite passed with local
socket access. Ruff lint/format, schema export, compilation and whitespace checks
passed. TDD and raw host/model/simulator evidence remain private under
`runs/chips/validation/20260925-task-authoring/`.

The real image micro-probe used native Codex **0.154.0**, **gpt-5.6-luna**, reasoning
**medium**, and correctly read an image-only label. The D0 episode then used two
separate Codex sessions: extraction **28.38 s**, construction **89.06 s**, with
360/840-second limits and no automatic model retries. All **7/7 required fields**
and their source-file associations were correct without editing the model draft,
including mV-to-V/MΩ-to-ohm conversions and an image-only ordered terminal list.
The operator compared the synthetic fixture to its authored values and issued a
`scripted_confirmation`; this is not real expert validation or a generic check of
citation semantics. Text instructions called `system` by the workflow are inlined
into the Codex user prompt, as recorded by the existing adapter.

The Agent first requested a one-node numerator; MCP schema validation rejected it
before server dispatch. It corrected the pair to `["out", "0"]`, ran **one actual
public Spectre simulation**, inspected the feedback, and explicitly submitted the
same candidate. Thus the raw trace contains **three MCP attempts**, while the
server received **two actions** and charged **one simulation**. The candidate used
opposite-phase unit AC sources and `V(out,0)/V(vip,vin)`; its frozen digest is
`feb955592e31770bf4a0374cabafd3e230092e00bd13956aa79f2eb0ef2a3a1a`.
The public tool round trip was 11.01 s, submission 2.82 s; these include SSH/control
work and are not simulator-only timings. Native CLI reported input/output tokens
of 21,067/606 for extraction and 105,576/906 for construction, with cached input
8,960/70,656 and reasoning output 174/277. These are provider-reported usage,
not equal-token budgets or separately instrumented request counts. Startup skill
context remained present, so this is a cooperative development condition, not a
minimal-prompt or adversarially isolated benchmark.

On lab-server, Spectre **21.1.0.509.isr12** used the administrator's existing Cadence
and Mentor initialization scripts, one process/thread, a 60-second job timeout
and 30-second license queue limit. No PDK or shared installation was changed.
A hand-written 10 Hz baseline matched the analytic gain to 8.7e-8 dB. Three
subsequent detached controls passed: a good DUT, a below-spec DUT, and a wrong
single-ended measurement rejected for a 6.02 dB error. Testbench correctness is
scored separately from DUT acceptance.

After explicit submission, four operator reference instances were evaluated
without additional Agent feedback. At the confirmed 20 Hz / 50 dB threshold:

| gain0 / pole (Hz) | Measured gain (dB) | DUT meets spec | Testbench valid |
| --- | --- | --- | --- |
| 800 / 50 | 57.417218 | yes | yes |
| 1500 / 500 | 63.514889 | yes | yes |
| 200 / 100 | 45.850258 | no | yes |
| 500 / 10 | 46.989700 | no | yes |

Maximum gain error was **8.3e-6 dB**, within the fixed 0.2 dB limit; phase checks
also passed the 1-degree limit. A private exported package contains source
snapshots, draft/confirmation, frozen candidate, reference model, CLI bundle,
profile and hashes. Two fresh jobs read that package independently of the Agent
session and produced identical metrics; simulator times were **7.88/7.93 s**.
All **10 downloaded job archives / 290 members** passed local hash and numerical
rechecks. Both dispatched client/server actions match, the seven exported package
files match their manifest, and public/final/replay candidates share the frozen
hash. The private per-run `d0-001/report.html` links prompts, schemas, raw traces,
confirmation, tool responses and final results; it is not yet a generic Episode
report backend. Moving the package to another host still requires an approved
deployment profile and simulator installation; this is not cross-account reproduction.

This is **one development episode**, with a bounded JSON candidate rendered into
a fixed Spectre scaffold, not arbitrary testbench code generation or a statistical
method comparison. The synthetic linear model does not validate transistor/PDK,
supply/common-mode/temperature effects, or signoff. PDF/Word ingestion, real
research documents and expert clarification, sealed T1–T3 materials, the A/B/C
matrix, and integration into the common experiment/report entry remain untested
or unimplemented. The existing output cap covers top-level files, not recursive
disk or memory isolation. No real SSH kill was injected in this D0 run; it used
the existing detached job machinery and its regression coverage. An initial
operator deployment failed before simulation because the csh login shell rejected
multiline quoting; the failure was preserved and deployment succeeded using a
single-line encoded script. This did not restart a model episode or simulator job.

## Fixed-condition batch pilot (2026-09-26)

The batch entry now keeps planned cells in the denominator, reports metric coverage
separately from success on valid results, freezes declared v2 condition pins, and
records lifecycle actions before collection/finalization/archiving. The coordinator
does not replay a cell with an existing output or automatically repeat an
interrupted final scorer. These changes were pushed separately as `2bf87e34`,
`331420ad`, and `b5d2ec99`; the live NFS lock fix is `6ba84b3a`. Their GitHub CI
runs passed. The local affected Chips suite passed **405 tests, 16 skipped** before
the two-line lock fix; its focused NFS regression passed **18 tests** afterward.

On lab-server, one detached Pi 0.87.0 + GLM-5.3-Flash (`thinking=low`) condition used
12 model calls maximum, 8192 output tokens per call, 64,000 request bytes,
1200 seconds per episode, no HTTP retry, 24 public actions and four public
simulations. Six fresh sessions pinned VABench `v4-001` to the existing r53 task
and EVAS 0.8.7, and Analog `rlc-rf-bandpass-100mhz` to its existing public source,
Podman image and ngspice scorer. Cross-episode Memory was disabled. The deployed
source, CLI bundle, Pi wrapper, task, prompt/tool source files, simulator and scorer
pins were checked against each session and all **56 captured provider requests**;
every request used the declared model, `low` reasoning and 8192-token output cap.

| Task / evidence batch | Valid / planned | Independent result | Model calls | Public simulations |
| --- | ---: | --- | --- | --- |
| VABench, `batch-nfsfix` | 3 / 3 task cells | pass, pass, pass | 11, 7, 8 | 2, 1, 1 |
| Analog, `batch-analog` | 3 / 3 | reward 0.0, 1.0, 1.0 | 12, 6, 12 | 4, 1, 4 |

All six Agents explicitly submitted one candidate each. VABench's valid binary
success rate is 3/3 on this small sample; its frozen replay labels the authority
`development_only`, not certified benchmark acceptance. Analog's valid reward
mean is 0.667, median 1.0, min/max 0.0/1.0, sample standard deviation 0.577;
the 0.0 is a valid scored result, not a harness failure. No combined VABench/Analog
score or reliability claim is made from these three runs per task.

The initial six-cell batch stopped before any model call because lab-server's NFS
requires a writable descriptor for an exclusive lock; `6ba84b3a` fixed and tested
that. In the next six-cell manifest, the three VABench cells completed, while its
first Analog cell failed preflight with **zero model requests**: the launcher supplied
an environment Key while that Analog plan also specified `key_file`. The remaining
two cells were `not_run`. That manifest remains unchanged. A fresh, separate
three-cell Analog manifest used only the private Key file and completed all three
episodes. The two manifests' status counts are preserved in the private audit; the
table above describes the six completed episodes across them, not a single perfect
six-cell run. The mixed-batch credential rule is documented in
[UNIFIED_EXPERIMENT.md](UNIFIED_EXPERIMENT.md).

The three VABench final-job archives and public episode archives, plus three Analog
private episode archives, passed offline member-hash verification. Analog archived
candidate digests equal the frozen final-input digests. A compact audit receipt and
credential-scanned **335-file** review package are private under
`runs/chips/validation/20260926-batch-pilot/collected/`; the package SHA-256 is
`e499c6b0d06aa625010e0a74686b67b0986cefd225426c10441a7022d4330015`.
The full server records remain under the account-owned `0700` directory
`~/chips-private/validation/batch-20260926-b5d2ec99/`. Neither trajectories nor
credentials are committed to Git. A larger task set, cross-account reproduction,
different Agent/model comparisons and actual Memory conditions remain untested.

The Analog 0.0 episode was reviewed against its private Tool requests, responses,
candidate hashes and original final log. Its four public simulation calls returned
three `simulation_error` results, then a measurable candidate with **−270.638 dB
at 100 MHz**. The Agent wrote a fifth candidate and submitted it without another
public simulation; its frozen SHA-256 matches the final evaluator input but none
of the four simulated candidates. The original scorer ran (`container_exit=0`)
and failed all 15 checks at the 100 MHz functional gate. The model then reached
its 12-request limit. Thus the observed 0.0 is an independently graded result
for an **untested submitted candidate**, not a conflict between public feedback
and final grading of the same candidate. This case motivates reporting whether
the submitted hash was publicly simulated and how much simulation budget remained;
it does not establish that a particular circuit revision would have passed.

## Single-manifest mixed batch validation (2026-09-26)

The whole-batch preflight change in `f5211bec` was deployed to lab-server and run
against **one fresh v2 manifest containing one VABench and one Analog cell** under
the same Pi 0.87.0 + GLM-5.3-Flash (`thinking=low`) condition. The Analog plan
omitted `key_file`; the detached supervisor loaded the account-owned private
Key into the process environment for both cells. No Key value was copied into
the manifest, command arguments or archive. Both task sessions were newly
created with the previously pinned EVAS and Podman/ngspice environments.

`prepare` recorded `not_run: 2`. The explicit `preflight` returned `ready`
for both cells before a model request: VABench's session identity, Pi launcher,
dependencies and model HTTPS path were ready; Analog additionally checked its
public files, source pin and Podman image. `run` repeated this gate before the
first Agent, then the detached server process completed and reconciled both
cells in the **same manifest**. The final batch summary was `completed: 2`;
both result rows have `final_execution=ok` and `final_validity=valid`.

| Cell | Model calls | Public simulations | Independent result |
| --- | ---: | ---: | --- |
| VABench `v4-001` | 9 | 1 | original EVAS 0.8.7 replay `pass`; `development_only` authority |
| Analog RLC 100 MHz | 9 | 3 | original Podman/ngspice scorer `graded`, 0/15, reward 0.0 |

The Analog Agent hit its **output-token limit** without calling `analog_submit`.
The declared `episode_end` policy froze its last complete candidate, which had
been publicly simulated: its 100 MHz gain was −766.765 dB. The independent
scorer used that same candidate hash and failed the 100 MHz functional gate.
Thus the 0.0 is a valid task outcome, not a harness or archive failure. This
differs from the earlier 0.0 episode above, whose submitted candidate was not
publicly simulated. Neither single sample supports a task success-rate claim.

The VABench final-job archive passed the existing `verify-archive` command;
both private Episode packages passed a second member-by-member SHA-256 audit
(100 VABench and 142 Analog members). A credential-scanned 126-file review
package is preserved privately at
`runs/chips/validation/20260926-mixed-batch/collected/review-evidence.tar.gz`
(SHA-256 `d51d33177d36c70028bd97395b181cfaaddc5d292ec46761d05d38847fe04448`),
with an extracted copy and compact `audit.json` alongside it. The server source
and original records remain under the user's private
`~/chips-private/validation/batch-20260926-f5211bec/` directory. The first
deployment preparation used the system Python without `numpy`, then a copied
Analog operator had an unsupported field; both errors occurred before batch
preparation or model use, and the corrected private configuration was frozen
before the successful run. This validates the single-manifest workflow on this
host and condition, not cross-account reproducibility or Analog Agent success.

## Episode closure and candidate recovery repairs (2026-09-26)

Five independent changes were committed and pushed: candidate history/restore
(`631bc149`), submission and collection facts (`2121e6b6`), output-budget pressure
and measured timing (`53d3af52`), deployment configuration validation (`662cb648`),
and unknown submission execution (`2688bb8c`). Analog now exposes six public
Tools, including `analog_history` and `analog_restore`; snapshots and their public
simulation feedback are archived. Restore consumes an action, requires an exact
saved digest, and works with no simulation budget remaining. It neither ranks
candidates nor exposes hidden grading. Reports separately describe episode
completion, acknowledged Agent submission, automatic collection, termination,
and whether the frozen candidate was successfully publicly simulated. An
unresolved submission stays `unknown`, rather than becoming `not_submitted`.

Pi's explicit output budget is validated centrally for both tasks in the range
1–32768 tokens; the default remains 4096. Reports retain configured limits and
reported usage without treating reasoning-token counts as thinking time or the
output-minus-reasoning subtraction as visible text length. Model-request timing
and measured controller/server simulation timing remain separate. The new
`chips_experiment validate --config ...` entry checks configuration without
model requests, SSH, credential reads or output creation. Workflow exports are
lazy, so Chips configuration CLIs no longer eagerly require Bio dependencies;
existing shared export identities are preserved.

Behavioral changes were exercised through failing-then-passing focused tests.
At `662cb648`, the full prescribed Chips gate returned **443 passed, 16 skipped**;
after the unknown-execution correction, the affected experiment, evaluation and
Episode-report tests returned **65 passed**. Ruff check and format check passed
(627 files already formatted), as did schema export, compilation and diff checks.
All five code commits' GitHub CI runs completed successfully. Conditional skips
cover unavailable local Pi/live integration fixtures, local ngspice and optional
external runtime checks; they do not establish live runtime acceptance.

The two existing mixed-batch episodes above were re-reported offline; fingerprints
confirmed **106 original evidence files were unchanged**. The derived summary
records one Agent submission, one automatic collection, one final termination
and one output-token-limit termination. Both frozen candidates had a successful
public simulation, which does not imply that either circuit meets its spec.
Analog's ninth request reports 8192 output tokens, including 8154 reasoning
tokens, over 237.103 seconds. These observations motivate an explicitly configured
budget comparison; no new model run or final grading was performed in this repair.

A new standard-library Tool bundle was also tested in an isolated private
lab-server validation directory. An operator replayed one previously recorded public
candidate through the pinned Podman/ngspice environment: one public simulation
returned **15 measurements**. After saving a second version distinguished by a
comment, history showed both versions and their feedback; with zero simulations
remaining, restore recovered the first candidate's exact bytes, and submission
froze that same SHA-256. All **35 archive members** passed verification on the
server and again after download. This is an **operator Tool smoke**, with zero
model requests and no independent final evaluation, not evidence that an Agent
will choose or recover a better circuit. The default server Agent deployment was
not replaced. Private reports, test receipts and smoke evidence are retained at
`runs/chips/validation/20260926-episode-fixes/`; raw evidence is not committed.

### Candidate snapshot newline preservation (2026-09-26)

A public-session regression exposed newline conversion during `analog_restore`:
LF passed, but CRLF and CR restored a different candidate SHA-256. Restoration now
decodes the saved UTF-8 bytes without newline conversion, and candidate writes use
binary UTF-8 output. The regression checks exact candidate and frozen bytes,
unchanged digests, linked public feedback, zero remaining simulations and repeated
action IDs, including non-ASCII comments. All three newline cases passed after
the fix; the affected Analog session, CLI, archive, transport, MCP Agent and
finalization checks returned **56 passed**. Changed-file Ruff check and format
check passed. Red, green and regression receipts are private under
`runs/chips/validation/20260926-candidate-bytes/`. These are local constructed
simulator checks; no new model run or lab deployment was performed. The Analog
system prompt was reviewed separately and was not changed in this fix.

### Analog prompt input separation (2026-09-26)

The Analog Pi entry now composes English experiment rules and a separate
Analog public-session protocol. Circuit numerical requirements remain in public
task files, and tool schemas still come from the session. The role requests
evidence-based revisions and concise explanations; the protocol retains current
candidate collection, same-session history/restore, acknowledged submission and
out-of-loop final grading. This is the current RLC entry's prompt, not a global
default for other circuit tasks.

The input-boundary regression first failed in all four model/runtime cases
because the two prompt sections were absent; they passed after the change.
The first sandbox attempt also hit a localhost socket restriction for the
external Runtime; rerunning with local socket permission isolated the intended
failure. Affected Agent, Pi Runtime, Environment, archive and experiment checks
returned **66 passed, 12 skipped**. The skips require an explicitly configured
real Pi CLI/local HTTP fixture. Tests check that the system and task prompt
received by the constructed Pi session match `agent-input.json`, with separate
tool schemas and budgets. Changed-file Ruff check, format check, compilation and
diff checks passed. Receipts are private under
`runs/chips/validation/20260926-prompt-layers/`.
No new provider request, lab deployment or simulator run was performed; these
checks do not establish that the new prompt improves model performance.

### RLC task contracts and shared Agent entry (2026-09-26)

The pinned `rlc-rf-bandpass-100mhz` and `rlc-broadband-50-to-200-match` tasks
now declare their passive subcircuit interfaces and public diagnostics in the
existing task registry. Public sessions, candidate validation, history/restore,
freezing and finalization use the selected task. New schema-v3 sessions pin the
interface hash; legacy v1/v2 sessions remain restricted to the original bandpass
task. Public instructions retain the full upstream text with explicit Harness
execution and completion adapters. The broadband public runner invokes the
pinned upstream finite-Q analyzer and requires all eleven sweep points; missing,
nonfinite or incorrectly indexed/frequency data is a simulation error, not a score.

Both tasks use the same six Tool schemas, English system prompt and Pi launch
path. The experiment, operator and session must agree on task ID before model
launch; `agent-input.json` records that actual ID. Tests cover both runtime
settings and model IDs, source/interface drift, foreign candidate rejection,
task-specific finalization and legacy compatibility. Focused failures preceded
implementation; the affected regression returned **170 passed, 12 skipped**.
The skips require explicitly configured real Pi CLI/local HTTP fixtures.
Changed-file Ruff check, format check, compilation and diff checks passed.

Both example experiment configurations passed the pure configuration-validation
entry after substituting private operator-file paths, with no run output created;
relative documentation links and schema export also passed. The four code
commits' private GitHub CI runs completed successfully.

Both retained upstream task trees match their original fixed SHA-256 values,
and all declared public files exist. A real standard-library bundle created
two sessions from these task trees, read the public instructions, wrote two
versions of a constructed single-resistor candidate per task, restored the first
version by digest, submitted it, and archived and verified both episodes.
These candidates exercise the file protocol; they are not circuit solutions.
The initial probe attempted to write the empty upstream starter and was correctly
rejected for lacking R/L/C elements; a new probe directory retains the corrected
run without overwriting that evidence. The bundle SHA-256 is
`a755bdf7570064f44395117be9c043d1899be48c784934f15dc9fd7acd9cbd05`.

Receipts, original failed probes, source-pin checks and verified archives remain
private under `runs/chips/validation/20260926-rlc-task-reuse/`. Local ngspice and
Podman executables were absent. This change made **zero provider requests,
zero simulator calls and zero final evaluations**, and did not deploy to the
lab. Broadband's new public runner and real Agent loop still need server
acceptance; Analog native Codex/SSH support and OTA public tooling were not added.

### Shared RLC simulator acceptance on lab-server (2026-09-26)

Following explicit authorization to continue live acceptance, source revision
`f0cf97ad` and its unchanged `a755bdf7…` CLI bundle were deployed in a new private
validation directory. Source came from `git archive HEAD alphaapollo`, excluding
unrelated local numerical-validation work; its tar SHA-256 is
`91c5a197d40925fc4fadbc9443d0443d102f04ba8087985a4655f3de839f1915`.
Python, the original task hashes, the existing pinned RLC image and private
credential-file metadata were checked. The default server deployment and prior
sessions were not replaced. Expanded source and simulator outputs use private
scratch; configuration and archives use private persistent storage.

Three actual public simulations completed: bandpass reference returned all
15 scalar measurements; broadband reference and a constructed single 1000-ohm
series-resistor candidate each returned all 11 finite-Q sweep points. Reference
worst Gamma was **0.07316007**, minimum transducer gain **0.8926585**, and center
insertion loss **0.464619081 dB**. The negative candidate had Gamma **0.92** and
gain **0.0256**, agreeing with the simple resistive fixture. The original
broadband final verifier graded the reference **7/7, 1.0** and the constructed
negative **1/7, 0.1**. The empty public starter was rejected before simulation.
These are operator controls; no reference candidate was put in the Agent session.

Two private probe assumptions were corrected without rerunning completed
simulations or grading: `analog-bench` has no `--podman` executable option, and
the legal negative candidate earns structural partial credit rather than zero.
Initial scripts and failed command receipts are retained alongside the corrected
probe. No production-code fix was needed. All five preceding commits' GitHub CI
runs had completed successfully before the live model episode was launched.

The fresh broadband session passed the existing preflight: Pi **0.87.0**, Python
dependencies, task/public hashes, local Podman image and model HTTPS were ready.
The separately launched first Agent condition is external Apollo Runtime plus
Pi/GLM-5.3-Flash, `thinking=low`, zero HTTP retries, 12 model calls, 8192 output
tokens/request, 64000 request bytes, 1200 seconds, 24 actions and 4 public
simulations. Preflight does not certify authentication or Agent success. Its
model result and episode verification are recorded below after completion.
Private evidence is under `runs/chips/validation/20260926-rlc-live/` and the
corresponding independent server validation directory.

### Broadband Pi/GLM episode acceptance (2026-09-26)

The shared RLC code at `f0cf97ad` was exercised with two fresh broadband sessions,
the same pinned source/image, external Apollo Runtime, Pi 0.87.0,
GLM-5.3-Flash, `thinking=low`, zero HTTP retries, 8192 output tokens/request,
1200-second episode limit, 24 actions and four public simulations. References,
hidden tests and final scores stayed operator-only. No prior candidate or memory
was transferred between sessions; no production code or prompt changed to
obtain these results. Supervisors detached from their initiating SSH clients.

| Condition | Model/request budget | Actual model / Tool / public-simulation calls | Collection | Original final scorer |
| --- | --- | --- | --- | --- |
| `broadband-pi-glm-low-001` | 12 requests; 64000 bytes/request | 10 / 11 / 3 | Episode-end collection; no Agent submit; fourth candidate not publicly simulated | 1/7; 0.1 |
| `broadband-pi-glm-low-002` | 16 requests; 131072 bytes/request | 13 / 13 / 4 | Agent restored its first simulated candidate and explicitly submitted it | 6/7; 0.9 |

Condition 001's next request was **65259 bytes**, so the budget bridge stopped it
before contacting the provider. The stop was a request-size limit, not a model
output-length stop or the 12-call limit. Automatic collection correctly retained
that distinction: `agent_submitted=false`, `collection_source=episode_end`,
`final_candidate_publicly_simulated=false`. The fourth candidate was graded as
collected; no earlier, better candidate was substituted after termination.

Condition 002 was planned separately after observing that stop. It performed
three reads, four writes, four simulations, one `analog_restore` and one
`analog_submit`; every Tool completed without failed or unresolved execution.
The submission slot remained available after all four simulations. Its restored,
submitted, frozen and final-scorer input bytes shared SHA-256
`3848d3262854b6db0b5e9c5c3fe8a8ad3027ec0682df7b20970cc8701097d062`.
The original grader's only failed check was nominal worst reflection:
**0.0959463454 > 0.08**. The score is the upstream quantized reward, not a
binary Harness success. `benchmark_success` remains `null`.

The closed-loop Harness acceptance passed for condition 002: task-bound input,
real model requests, public feedback, candidate iteration and restore, explicit
submission, identical frozen/scored bytes, valid independent grading and verified
archive. **The circuit did not pass every criterion.** No model budget or length
stop occurred in condition 002; its largest recorded request was only 46237
bytes. Different fresh trajectories and simultaneous changes to two budget
settings mean these samples do not establish a causal budget improvement.

Agent command wall time was 451.49 / 292.69 seconds for conditions 001 / 002.
Recorded model intervals summed to 435.62 / 274.11 seconds; server public-action
intervals summed to 1.33 / 1.86 seconds. These intervals have different scopes:
model timing includes network, queue and generation; public-action timing
includes Podman/analyzer work and is not isolated ngspice solver time. No latency
or success-rate comparison is claimed from these two samples.

Both sealed Episode archives were downloaded, their package hashes and every
member checked locally, and fresh offline HTML/JSON/CSV reports generated from
the verified extraction. Candidate collection/submission and final-input bytes
also matched independently. The inner archives contain **168 / 193 members**,
with server/local SHA-256 respectively:

- `b1bb50b974011754b71cb45cc7ed766aa7066dda43dca7fe73e909f969606dc1`
- `e5e6f4583ddb707c255a92ec4efb14b3ff625b7fd1ae54d2cf4fa3212b4d5d73`

Outer review packages and their selected raw members were separately verified.
A server-only scan found zero occurrences of the exact configured model Key in
plain or nested selected evidence; the Key value was not downloaded. Reports
retain their own evidence gaps: full provider payloads are unavailable and a
derived report alone does not authenticate an archive; the independent archive
receipts supply that check. Raw evidence, both reports and local verification
receipts remain private under `runs/chips/validation/20260926-rlc-live/`.

After condition 001 had already completed grading and archiving, the private
acceptance probe incorrectly expected a `members` field from the abbreviated
CLI verification response. The failed probe was retained; only its summary was
recovered through the existing library receipt. No Agent, simulation or grader
was replayed. Condition 002 used the correct receipt interface. This and the two
earlier operator-probe corrections required no production-code changes.

The simulator-summary commit `7e7f1879` passed private GitHub Chips CI. This turn
does not add Analog native Codex/SSH, an OTA Agent adapter, cross-account
reproduction, server reboot recovery or a stable circuit-design success rate.


## 2026-09-26 — Chips／Robotics 分支范围清理

按最新用户决定，`demo/chips` 移除 Bio、Math、ALFWorld、Search、Sokoban、
WebShop 的领域实现、配置、注册和专用测试，保留 Chips、Robotics 参考及共享
执行／Agent／Workflow／轨迹／Memory／Learning 契约。通用 `python_execute`
迁到 `common/execution/tools/python.py`；答案提取和有理数投票不再依赖 Math，
评分默认使用 `exact_match`。SFT／DPO 覆盖配置改为通用名称，应用需注册自己的
训练环境池。已删除的领域 API 不提供兼容空壳，详见 [范围说明](REPOSITORY_SCOPE.md)。

- Bio 首次定向验证：139 passed、11 skipped。
- 最终离线回归：2476 passed、31 skipped，162.79 秒。范围包括 Common、
  Workflows、Reasoning、Data Preprocess、Chips、Schema，以及受影响的
  Learning 入口／配置与 rollout 契约。跳过的可选 Torch／verl 检查不能代表训练通过。
- Ruff lint／格式、Schema 快照、Python 编译、安装脚本 Bash 语法与 diff 检查通过。
- Wheel 构建未运行成功：当前虚拟环境没有 pip／setuptools／wheel；不因此安装
  额外依赖或将源码检查当作打包验收。
- 本轮没有模型请求、服务器部署或真实领域仿真。16 个原有 EVAS 未提交文件的
  哈希保持一致，未被纳入本轮提交；本机回归可收集其中的离线测试。

本地文件职责图在忽略的 `.planning/chips/repository-map-20260926/` 重新生成，
不上传 GitHub。参考分支及其历史保持原样。
