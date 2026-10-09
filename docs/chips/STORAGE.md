# 工作目录、持久归档与阶段计时

本页定义后台作业的存储、归档与清理合同。
`submit-rc` 和 `submit-vabench` 支持在服务器本地 scratch 执行，再把封存证据打包到持久目录。
这里的“本地”指仿真服务器自己的盘，不要求把实验移到操作者电脑。
EMX、前台命令及 Agent 运行目录尚未接入这套归档生命周期。

## 目录和输入

| 参数 / 位置 | 职责 |
| --- | --- |
| `--root` | 作业工作根目录；可以选经过核查的服务器本地盘 |
| `--archive-root` | 可选的持久归档根目录，通常在私有 NFS HOME 下 |
| 固定源码与仿真器路径 | 由任务输入和 pin 指定；不会自动迁移环境、展开完整历史或修改共享安装 |

两个根目录必须互不包含，归当前用户所有且无 group/other 权限。仍需按
[部署规范](DEPLOYMENT.md) 检查父目录、ACL、可用空间、配额和管理员的 scratch 清理策略。
可用空间不是个人配额；`/tmp` 的容量、介质和保留时间也不能从名字推断。
`/tmp` 可能在重启或定期清理时被删除，NFS 持久存储也不等于备份。

未指定 `--archive-root` 时保持原有执行方式，不自动搬运或删除文件。
重试必须使用原任务 ID 和相同的根目录、输入、工具配置；更换条件使用新 ID。

## 示例：在服务器执行

以下是 Bash 示例。先按 [ngspice 部署](NGSPICE.md) 部署版本化 CLI 和仿真器。

```bash
umask 077
chips_root="$HOME/chips-private"
chips_cli="$chips_root/harness/chips.pyz"
chips_work=$(mktemp -d /tmp/chips-work.XXXXXXXX)
chips_archive="$chips_root/archive/jobs"
# 将 chips_work 的实际路径记到自己的持久部署配置，重连时继续使用它。

python3.12 "$chips_cli" submit-rc \
  --input "$chips_root/harness/rc.json" \
  --ngspice "$chips_root/envs/ngspice/47/bin/ngspice" \
  --root "$chips_work/jobs" --archive-root "$chips_archive" --job-id rc-001

python3.12 "$chips_cli" job-status "$chips_work/jobs/rc-001"
python3.12 "$chips_cli" archive-status "$chips_archive/rc-001"
python3.12 "$chips_cli" job-timings "$chips_work/jobs/rc-001"

# 仿真已封存但归档失败时，仅重试打包、复制与校验；后台执行，不重跑仿真。
python3.12 "$chips_cli" job-archive "$chips_work/jobs/rc-001"

# 完整性检查包括包哈希、每个成员、原始结果收据和对应后端的离线核验。
python3.12 "$chips_cli" verify-archive "$chips_archive/rc-001"

# 可选：确认不再需要工作文件时显式清理；内部会再次校验归档和源结果。
python3.12 "$chips_cli" job-cleanup "$chips_work/jobs/rc-001"
```

`submit-vabench --pin task.pin.json --submission candidate/` 同样接受
`--root`、`--archive-root` 和 `--job-id`，仍按原 r53 评分器执行。
源码缓存若移到另一条路径，应重新 pin 并核验源码身份；不能编辑旧 pin 来掩盖改变。

下载时只需复制对应归档目录，包含 `job.tar.gz` 和 `receipt.json`，然后在本机运行
`python -m circuit_harness.cli verify-archive <下载目录>`。
核验无需服务器路径、仿真器或 EVAS 环境，会在系统临时目录中解包并复核；可用 `TMPDIR` 指定核验临时盘。
归档含候选、原始日志和可能的隐藏评分材料，只供操作者保存；不能作为 Agent 公开任务目录。

## 完成、归档与清理是不同状态

```text
持久登记 job ID → scratch 内独立 worker → 仿真/评分 → completion.json
                                                 ↓
                                 本地打包 → NFS .partial → 校验 → 发布归档收据
                                                 ↓
                                      默认保留工作文件；可显式清理
```

- `job-status.state=finished` 表示执行已结束，仍需查看 `result.execution` 和 `result.verdict`。
  归档有独立的 `pending/running/failed/verified` 状态；归档失败不会改写仿真结果。
- 打包只收录 `completion.json` 及其列出的证据；变化中的外层 worker 调试日志和归档状态不属于封存结果。
  NFS 上先写 `.partial`、flush/fsync，比较整包哈希，再逐个验证成员和后端收据，最后发布包及 `receipt.json`。
  只有最终收据存在才表示完成发布；孤立的 tar 或 `.partial` 不算成功。
- 自动归档和 `job-archive` 都在服务器独立 worker 中运行，客户端退出不负责推进后续步骤。
  重试重新打包和复制，不提供按字节续传；不会再次运行求解器。状态查询不会触发重试。
- 启动前持久登记 ID。工作区丢失但归档存在时，重复提交返回已有终态；若没有归档则返回 `unknown`，
  不偷偷创建第二次实验。此保障依赖保留并继续使用原归档根目录。
- `job-cleanup` 只清理该作业内的文件，保留 request 和小型清理标记；不会删除源码缓存、环境、其他任务或归档。
  工作区仍活动、证据损坏或未完成归档时拒绝清理。清理中断后可再次执行同一命令。
- `archive-status` 是最近一次落盘状态，不是进程活性或重新校验的证明。异常终止可能留下 `running`；
  检查原工作区和 worker 日志后可显式重试归档。已发布但后来损坏的归档会拒绝核验，不自动覆盖它。
- `verify-archive` 退出 0 表示证据完整；失败电路也可以有完整归档。它不会把负例改成通过。
  `job-archive` 退出 0 只表示已派发或已清理，不是归档完成；配置/证据错误退出 2。

无自动定时删除、服务器重启续算或存储故障下的持久调度保证。NFS hard mount 的内核 I/O 可能长时间阻塞，
不能把仿真超时当作归档硬超时。需要管理员确认个人 scratch 和存储运维策略。

## 计时如何解释

`job-timings` 可读取工作目录或归档目录；旧记录缺少测量时返回 `not_measured`，不填零。
当前收据以兼容的附加字段保存计时，原结果判定与默认 CLI 行为保持不变。

| 字段 | 测量范围 |
| --- | --- |
| `job_s.validate_request` | 校验任务、工具和固定执行包 |
| `job_s.run_backend` | 后端整个调用，包含下方各后端步骤 |
| `job_s.verify_result` / `seal_artifacts` | 结果复核 / 汇集并计算证据哈希 |
| `job_s.elapsed` | worker 启动后的处理，不包含最终 completion 发布和归档 |
| `backend_s.prepare_input` / `parse_and_grade` | RC 网表生成 / 数值解析与评分 |
| `backend_s.process:version` / `process:ngspice` | 版本查询 / 仿真进程，包括进程管理开销 |
| `backend_s.preflight` / `postflight` | VABench 输入与 pin 核查 / 输出核查 |
| `backend_s.process:replay` | 原 EVAS 仿真与评分的整体 replay，暂未拆分内部耗时 |
| `archive.timings_s` | 源校验、打包、复制与 fsync、目标校验、发布，以及本次尝试总耗时 |

父阶段和子阶段存在包含关系，不能相加为总耗时。计时包含实际 I/O 和调度等待，不是纯 CPU 或求解器耗时。
归档 `attempt_elapsed` 截止于发布包，未包含最终收据/状态写入；不是严格的提交到归档端到端时间。
源码解包/缓存准备仍由 [存储探针](STORAGE_VALIDATION.md) 单独测量；本次未自动迁移 Python 环境，
也没有测量模型请求、网络下载、许可证排队、批量并发或每次实验的峰值内存。
