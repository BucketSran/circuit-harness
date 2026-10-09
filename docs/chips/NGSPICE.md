# ngspice RC Harness

本后端在安装了 ngspice 的同一台主机上完成任务生成、AC/瞬态仿真和独立验收；
服务器使用标准库 CLI 包，无需安装 Harbor、模型客户端或训练库。

- 输入：带 SI 单位的 RC 参数 JSON；算例为理想 R、C 和 1 V 激励，无 PDK。
- 输出：网表、原始数值、工具版本/二进制哈希、事件、过程日志、指标、结果和产物校验和。
- 验收：AC 复数响应与一阶低通解析式比较；瞬态采用初始电容电压为零的充电响应。
  数据必须有限、坐标严格递增、覆盖指定范围；容差固定在验收器中，不能由被测输出修改。
- 故障：依赖缺失、非零退出、超时/取消、缺失/损坏输出和指标不合格分开报告。
  `execution=ok` 不等于 `verdict=pass`，`certified` 始终为 false。
- 后台作业：`submit-rc` 返回 running/finished 后，服务器独立完成仿真、验收与清单落盘；本机离线不驱动或取消任务。
- 恢复：同一任务 ID 不重复启动；完成结果可以独立校验；状态不明时不自动重跑。
- 预算：单线程构建，编译并行度 2；任务默认 60 秒，输出上限 100 MiB。内存预算是规划值，
  本后端不声称提供 cgroup 内存隔离。

验收条件：真实服务器上的 AC 与瞬态均通过；完成结果 resume 不重复启动求解器；
构造坏输出不能被判成功；异常任务的进程组清理、事件与退出码可检查。
这个 RC 合同不覆盖任意晶体管模型、商业 PDK、Agent 解题或大规模电路性能。

后台提交可用 `--root` 指定服务器本地工作区、`--archive-root` 指定持久目录。
自动归档、单独重试、校验后清理和阶段计时见 [存储使用说明](STORAGE.md)。

## 安装与部署

1. 从 [官方 47 发布目录](https://sourceforge.net/projects/ngspice/files/ng-spice-rework/47/)
   下载 `ngspice-47.tar.gz`，在可联网机器下载后也可离线传入服务器。
   [安装脚本](../../examples/chips/ngspice/install.sh) 固定本次取得的 SHA-256：
   `894e649651f1838a14095e5a5439e7d3aa63e87ede14d283173fda4fcdef675f`。
   这是下载内容的复现指纹，不是声称已验证发布者签名。
2. 在本人持久目录建立 `chips-private`，按 [部署规范](DEPLOYMENT.md) 设置 0700、检查父目录和 ACL。
   不修改系统工具、公共 EDA 或其他人的目录。需 Linux GCC/make/bison/flex、GNU time/timeout。
3. 编译位置建议使用经过核查的本地临时盘：父目录须不可被他人任意替换条目，例如 `/tmp` 的 1777；
   用 `mktemp -d` 创建本人 0700 目录。持久安装前缀仍在 HOME。脚本不覆盖已有构建/安装。

```bash
# 在服务器执行；请先核查存储、权限和可用空间。
umask 077
root="$HOME/chips-private"
mkdir -m 700 "$root"  # 已存在时核查权限，不要递归 chmod HOME
mkdir -p "$root/downloads" "$root/build" "$root/harness" "$root/runs" "$root/archive"
scratch=$(mktemp -d /tmp/chips-ngspice-47.XXXXXXXX)
bash install.sh "$root" "$root/downloads/ngspice-47.tar.gz" "$scratch"
```

安装脚本只在自己的进程中使用最小环境（系统 PATH、C locale），避免继承 EDA 登录环境的编译/库搜索路径。
安装前缀为 `$root/envs/ngspice/47`。采用 `--without-x --with-readline=no --without-fftw3
--disable-openmp CFLAGS=-O2`，保留默认 XSPICE/OSDI/KLU 编译选项；这不等于已验证对应器件模型。
临时构建目录保留 configure/make/install 日志、GNU time 资源记录、版本和二进制 SHA-256；
归档到持久目录后再决定是否清理，临时盘不提供长期留存保证。
编译 `make -j2` 最长 1800 秒，配置/安装步骤没有相同的硬超时；内存没有硬限额。

在项目本机打包，仅包含 Chips CLI 和标准库模块：

```bash
mkdir -p runs/chips/deploy
python -m circuit_harness.cli bundle --output runs/chips/deploy/chips.pyz
scp runs/chips/deploy/chips.pyz examples/chips/ngspice/rc.json \
  examples/chips/ngspice/run-rc.sh examples/chips/ngspice/submit-rc.sh \
  lab-host:chips-private/harness/
```

服务器 zipapp 需要 Python 3.10+，只使用标准库；完整 Python 包安装要求 Python 3.12+。
随包快捷脚本使用 `python3.12`；若部署的版本不同，显式选择对应解释器。
升级时保留旧包与 SHA-256，运行记录绑定实际 Harness 内容。ngspice RC 不使用 GDS/EMX 配置。

## 后台提交、查看与回收（SSH 推荐入口）

```bash
# 任务 ID 由客户端事先确定并保存；连接中断后继续使用这个 ID。
# 脚本采用最小环境，root/jobs 自动建立为本人 0700 目录。
ssh -T lab-host bash chips-private/harness/submit-rc.sh rc-job-001

# 重连后查询；查询本身不推进任务，也不会重新启动求解器。
ssh -T lab-host python3.12 chips-private/harness/chips.pyz job-status \
  chips-private/jobs/rc-job-001

# 只有明确要停止任务时才取消；关闭 SSH 不代表取消。
ssh -T lab-host python3.12 chips-private/harness/chips.pyz job-cancel \
  chips-private/jobs/rc-job-001

# 完成后取回整个目录，在本机检查哈希并从波形重新验收。
scp -r lab-host:chips-private/jobs/rc-job-001 runs/chips/
python -m circuit_harness.cli verify-job runs/chips/rc-job-001
```

完整 CLI 为 `submit-rc --input task.json --root /private/path/jobs --job-id rc-job-001
--ngspice /path/to/ngspice --timeout 60`，在仿真器所在服务器执行。
提交时保存输入、任务/工具身份及每个任务自己的执行包，再创建独立 session 的 worker；
标准输入关闭、输出写服务器文件，不继承 SSH 的管道或终端。运行期间更新主执行包不会替换该任务的执行包。
服务器负责网表生成、仿真、超时/取消后的进程组清理、波形验收和最终产物清单，完全不需要客户端轮询才能推进。

任务目录：

- `request.json`、`worker.pyz`：提交输入、工具/Harness 摘要与固定执行包。
- `worker.json`、`events.jsonl`、`worker.*.log`：服务器 worker 身份与生命周期证据。
- `run/`：网表、ngspice 版本/日志/数值、评分、求解器事件与结果。
- `completion.json`：服务器结束收据，包含结果与证据文件哈希。调试输出 `worker.*.log` 不纳入固定清单。

状态和退出码需要分开理解：

| 状态 | 含义与后续动作 |
| --- | --- |
| `running` | worker 持有执行锁且已写启动收据；服务器正在独立处理 |
| `finished` | 最终收据已落盘；继续看 `result.execution` 和 `result.verdict`，不代表电路通过 |
| `unknown` | 存在任务目录，但缺少活跃 worker/最终收据；保留现场并检查日志，不自动重启 |
| `missing` | 该任务目录不存在；检查原任务 ID、服务器和路径 |

提交/查询命令退出 0 表示找到 running/finished 作业，**不是仿真通过**；unknown/missing 为 1，参数/证据错误为 2。
`verify-job` 只有完整性检查成功、execution=ok 且 verdict 不为 fail 才退出 0。
同一个 ID、相同输入/工具/Harness 的重复提交只返回已有状态；改变输入时拒绝，需显式使用新 ID。
若第一次回执丢失，重连后按原 ID 查询或重复提交；若断在准备与 worker 启动之间，可能返回 unknown，不能假定从未运行。
状态查询使用与 worker 排他锁冲突的共享读锁，兼容 Linux NFS 的锁语义；不依赖 NFS 文件时间判断存活。

本实现处理客户端断网、休眠、SSH 退出或挂断。服务器必须保持运行且存储、求解器等依赖可用。
它不是持久作业调度器：不承诺服务器重启、OOM、管理员终止、会话级资源清理或存储故障后的自动续跑。
若部署要求这些保障，需要接入服务器已有的 Slurm 或持久服务，另行定义任务重试/检查点语义。
时间预算和采样输出上限沿用 RC 执行器；没有 cgroup 内存硬限额。

## 原有前台入口与兼容性

`ngspice-rc`、`run-rc.sh` 和 `--resume` 的调用方式保持不变；前台入口适合交互调试，
仍不承诺 SSH 中断后的完整收尾。`--resume` 只验证并复用完整结果，未完成目录拒绝自动重提。
SSH 长任务使用上面的 `submit-rc` / `submit-rc.sh`。
`status` / `watch` / `report` 对后台任务使用其 `run/` 子目录；`job-status` 查看整个后台作业的状态。
旧记录绑定旧 Harness 内容，复用时选择保留的旧包；离线 `verify-rc` 仍可以校验旧 RC 结果。
原始 JSON 收据与哈希是完整性记录，不是密码学签名，不能抵抗同账号同时重写文件及清单。

## 断线验收复现

[`ssh_detached_rc.py`](../../tests/chips/probes/ssh_detached_rc.py) 是显式 opt-in 的真实服务器测试，
不在普通 pytest/CI 中自动联系实验室。先按本文完成部署，并获准使用对应主机、私有目录和仿真器：

```bash
python tests/chips/probes/ssh_detached_rc.py \
  --host lab-host --root /absolute/private/chips-root \
  --job-id new-disconnect-test --output runs/chips/validation/new-disconnect-test
```

探针创建单独的延迟包装器，在求解阶段等待 8 秒后调用真实 ngspice，不伪造波形；
确认任务未结束后终止本次本机 SSH 客户端，12 秒内完全不连接服务器，
随后验证结果在重连查询之前已经完成、独立评分通过、按同 ID 重提只发生一次求解。
每次使用新 ID，保留失败与成功记录；只终止本探针的本机连接，不修改服务器网络或共享服务。
本探针不能替代服务器重启、物理断网或本机实际休眠测试。

## 任务与开发位置

| 内容 | 位置 |
| --- | --- |
| 任务参数 JSON | [rc.json](../../examples/chips/ngspice/rc.json) |
| 任务规格/独立复现状态 | [RC 任务卡](../../examples/chips/benchmarks/rc/TASK.md) |
| 输入校验、生成网表、执行/恢复 | [ngspice.py](../../circuit_harness/execution/ngspice.py) |
| 后台提交、状态、取消与完成清单 | [jobs.py](../../circuit_harness/execution/jobs.py) |
| 独立解析与验收 | [rc_validation.py](../../circuit_harness/execution/rc_validation.py) |
| 公共命令与离线包 | [chips.py](../../circuit_harness/cli.py)、[bundle.py](../../circuit_harness/execution/bundle.py) |
| 回归与可选真实仿真 | [test_chips_ngspice.py](../../tests/chips/test_ngspice.py) |

评分器只服务此公开 Harness 基线，暂与 Chips 执行模块相邻，避免导入通用 grader 的其他领域依赖；
未来有正式电路任务和评分接口后再迁移。当前新增 Python/CLI 入口，没有注册新的模型 MCP Tool。
CI 使用发行版 ngspice 做回归；服务器使用固定 47，二者的软件来源和版本不能混称同一环境。
