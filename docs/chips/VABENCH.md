# VABench / vaEVAS 接入

固定回放使用 vaEVAS-next 的 VABench v4/r53 和 EVAS 0.8.7，执行冻结提交的独立回放。
原项目负责公开任务导出、artifact gate、提交冻结、评分规则及不可变 score sidecar；
Harness 负责版本固定、后台作业、超时/取消、事件日志与产物回收。
固定回放入口是操作者 CLI。新的 Agent 实验按[开发者接入指南](../guides/integration.md)准备 Harbor task，公开会话与独立终评分开绑定。
可信评分器始终不注册为模型可调用的 Tool。

首批验收是 family 001 的 DUT、bugfix、Testbench 三种任务及正负例。
这是 Harness 接入验收，不是模型能力实验，也不是重新认证整个 1,200 任务 release。
EVAS 是电压域行为级 Verilog-A 仿真器；此处的结果不能标成 Spectre 结果。

后台提交可用 `--root` 指定服务器本地工作区、`--archive-root` 指定持久目录。
自动归档、单独重试、校验后清理和阶段计时见 [存储使用说明](STORAGE.md)。

## 环境与固定版本

使用独立 Python 3.11+ 环境安装 `evas-sim==0.8.7`、NumPy、Matplotlib、Pandas。
本次 Linux x86_64 / CPython 3.12 的完整 wheel 版本与哈希见
[依赖锁](../../examples/chips/benchmarks/vabench/requirements-linux-py312.lock)，可用
`pip download --only-binary=:all: --require-hashes -r requirements-linux-py312.lock` 准备离线安装；
跨平台下载时还需显式指定平台、Python 与 ABI，其他平台必须单独安装验收。
Linux 发行 wheel 包含 Rust 动态库，仍需实际检查 `evas --version` 与案例运行，
不能只凭 pip 安装成功就声称求解器可用。保存 wheel SHA-256、安装日志及完整依赖版本。
服务器 CLI zipapp 只使用标准库，仿真器环境独立部署。

参考源码提交：`0685aae05c346e8e60f33ba48e2f64daff54d4f2`。
源码和数据放在本人 0700 根目录，按 [部署规范](DEPLOYMENT.md) 管理。
可以部署完整固定 checkout；如果 NFS 上小文件操作慢，可使用经过哈希记录的依赖子集。
子集须保留原路径及原始字节：根 `runners/`、包内 `runners/scripts/operations` Python 源码、
r53 共享配置与所选完整任务目录。保留完整 MANIFEST/TASK_INDEX 不表示所有任务已经部署。
不得编辑封存 release、金标准、fault 或原评分器来适应 Harness。

在仿真器主机上执行（以下均为占位路径）：

```bash
umask 077
python3.12 chips.pyz pin-vabench \
  --source /private/vaevas-source \
  --python /private/vaevas-env/bin/python \
  --task-id v4-001 --output /private/pins/dut.json
```

每个 pin 固定：任务 ID、manifest、所选任务和共享配置/评分代码摘要、Python/EVAS 启动器摘要、
已安装 EVAS 包含 Rust 库的文件树摘要、四个主要依赖版本。
它不是对所有系统动态库或所有依赖二进制的环境封存；正式跨机器研究仍需完整依赖锁或镜像。
固定源码、pin、候选都不应被被测 Agent 访问或修改。
提交和执行前校验 pin；评分完成后再次校验；漂移会报错，不会默默改用新版本。
新版本显式生成新 pin 文件，命令拒绝覆盖已有 pin。

## 公开任务与提交

```bash
python3.12 chips.pyz export-vabench \
  --pin /private/pins/dut.json --output /private/public-dut
```

导出沿用原 `install_public(..., G2)`：任务说明、契约、公开测试/输入、允许的参考 DUT，
以及独立的 `submission/`。不导出 evaluator、最终评分 sidecar 或操作者 pin。
这里的 G2 只选择公开仿真材料，**没有执行或复现原项目的 G2 Agent 条件**，也未导出 prompt/skills 协议。
Testbench 任务允许访问原契约指定的参考 DUT；要隐藏的是独立 mutation/checker 等评分材料。

候选文件直接放在独立 submission 目录，文件名由原任务契约决定。
`submit-vabench` 将输入复制、校验并冻结到 job，再由原 `submission_artifact_gate` 检查声明范围；
不合格提交保留 `no_submission`，不会绕过原门禁或拿参考答案替换。
原 `snapshot_submission` 与 profile-bound `run_trusted_replay` 生成最终冻结和独立评分凭据。

## 后台回放与回收

```bash
python3.12 chips.pyz submit-vabench \
  --pin /private/pins/dut.json --submission /private/public-dut/submission \
  --root /private/jobs --job-id dut-attempt-001 --timeout 300

python3.12 chips.pyz job-status /private/jobs/dut-attempt-001
python3.12 chips.pyz job-cancel /private/jobs/dut-attempt-001

# 完整下载后，在本机运行；不要求本机安装 EVAS。
python -m circuit_harness.cli verify-job runs/chips/dut-attempt-001
```

提交确认与执行分离，固定 worker 包由服务器独立完成评分和最终清单，SSH 断开不取消。
同 ID/同输入查询已有状态；输入变化拒绝；`unknown` 不自动重新执行。
这也遵守原评分器“已预约的 final replay 不在原目录重跑”的语义。
服务端重启、OOM 或调度器回收不在该进程模型的自动恢复保证内。

默认求解/回放时间预算 300 秒，允许 `(0, 1800]`；版本探针另外最多 30 秒。
原评分器内部还有自身超时。Chips 外层超时/取消属于执行失败，不能计作候选电路不合格。
进程组退出会清理其子进程；没有 cgroup 内存或总目录磁盘硬配额。
执行子进程使用最小环境，显式选择 r53 + EVAS2，不继承模型凭据或旧仿真配置。

| 结果 | 含义 |
| --- | --- |
| `execution=ok, verdict=pass` | 原评分器返回 `passed` |
| `execution=ok, verdict=fail` | 原评分器返回行为/编译/运行失败，或 artifact gate 拒绝提交 |
| `execution=infrastructure_error, verdict=not_evaluated` | 求解器/评分基础设施出错，或无法得到可信结构化结果 |
| `execution=timeout/cancelled, verdict=not_evaluated` | Chips 执行预算耗尽或明确取消 |

评分脚本可能退出 0 但返回失败状态，因此必须读结构化结果。
`verify-job` 校验下载内容的递归哈希、Chips 结果与原 verdict、score sidecar/提交身份的关联；
它**不会在本机重新执行 EVAS**。所有结果保持原 `development_only` 评分权限和 `certified=false`。
哈希提供完整性检查，不是签名，不能抵抗同账号同时篡改内容和收据。

主要证据在 `request.json`、`candidate/`、`worker.pyz`、`completion.json`、
`run/{identity,replay,result}.json`、`run/runtime/evidence/` 及事件/过程日志。
整个 job 含隐藏评分和参考夹具，只归 operator 所有，不能挂载到 Agent 沙箱。
0700 只隔离其他普通账号，不是同账号下 Agent 的安全沙箱；后续模型实验须沿用原项目的隔离执行器。

## 验收与后续接线

- [协议回归](../../tests/chips/test_vabench.py)：构造上游 API double 验证生命周期、漂移、冻结、判定、取消/超时、下载损坏、公开导出。
- [真实无模型探针](../../tests/chips/probes/vabench_smoke.py)：显式调用原封存任务和真实 EVAS，普通 CI 不自动运行。
- [真实 SSH 断线探针](../../tests/chips/probes/ssh_detached_vabench.py)：固定本次部署布局；在实际 EVAS simulate 调用前暂停 8 秒、杀掉本探针 SSH，20 秒无连接后核验完成时间及单次启动。
- [任务登记](../../examples/chips/benchmarks/vabench/TASK.md)：任务、预期判定、适用范围与证据层级。

```bash
# 在已获准的仿真主机上；root 必须是新的目录。包含私有参考答案，不能用于 Agent 输入。
python3.12 vabench_smoke.py prepare --source /private/vaevas-source \
  --sim-python /private/vaevas-env/bin/python --cli /private/chips.pyz \
  --root /private/validation/new-vabench-smoke
# prepare 只提交，完成后检查；未结束时 check 返回非零，保留同一个 job。
python3.12 vabench_smoke.py check --cli /private/chips.pyz \
  --root /private/validation/new-vabench-smoke
```

固定 r53 的公开动作和独立回放按本页协议执行；当前 Harbor 接入需要分别准备
公开材料与 final package，见[任务绑定](HARBOR_TASKS.md)。EVAS0.8.7 的历史结果不能
认证当前 EVAS 源码或任意任务，已观察范围见[验证记录](VALIDATION.md)。
Spectre 使用独立任务包与判据接入，不能把这里的 EVAS 回放成绩改标为 Spectre 成绩。
