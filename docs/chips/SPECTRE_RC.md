# Spectre RC-001：服务器本地闭环

这个后端只接受 `RC-001` 的电阻、电容和电压数值，生成固定的无 PDK 网表，在同一台服务器用 Spectre 跑 AC 与瞬态，读取原始 PSF，再交给已有的 `ideal_rc_v1` 解析解评分器。它不启动 Virtuoso，也不替换 VABench 的 EVAS 评分或声称覆盖晶体管/工艺任务。

## 一次性配置

在**每位实验者自己的账号**下，将[配置模板](../../examples/chips/cadence/spectre-rc.profile.example.json)复制到服务器私有目录，改成实际绝对路径并设为 `0600`。`setup_scripts` 按顺序列出管理员要求的 csh 环境脚本；只接受文件路径，不接受任意命令。`spectre` 指向可执行文件或厂商提供的启动符号链接。两个根目录必须事先建立为本人拥有的 `0700`，且互不包含：`run_root` 放快速临时盘，`archive_root` 放持久盘。两者的父目录也应核对权限。配置只保存在服务器，不提交到 Git。

示例任务 JSON：

```json
{"resistance_ohm": 1000, "capacitance_f": 1e-9, "voltage_v": 1}
```

本地在 `main` 构建标准库 zipapp，再送到自己的服务器私有目录：

```bash
python3 -m alphaapollo.workflows.chips bundle --output chips-agent.pyz
scp chips-agent.pyz YOUR_HOST:YOUR_PRIVATE_DEPLOY_DIR/chips-agent.pyz
```

在服务器上用本机 Python 3.9 或更新版本执行；配置、任务和程序均留在服务器：

```bash
python3 chips-agent.pyz submit-spectre-rc \
  --input /private/config/rc001.json \
  --profile /private/config/spectre-rc.profile.json \
  --job-id rc001-example-001
python3 chips-agent.pyz job-status /private/scratch/chips-spectre-jobs/rc001-example-001
python3 chips-agent.pyz verify-job /private/scratch/chips-spectre-jobs/rc001-example-001
python3 chips-agent.pyz verify-archive /private/persistent/chips-spectre-archives/rc001-example-001
```

提交只等后台 worker 接管；SSH 退出后服务器继续执行。`job-status` 只观察，不自动重跑。相同 ID 和相同配置重复提交返回原作业；任务、脚本、Spectre 或 Harness 变动后不得复用 ID。运行中取消用 `job-cancel`，持久归档确认后可用 `job-cleanup` 显式释放临时盘。临时盘可能在重启后清空；持久归档是需要长期保留的证据。

## 结果与失败语义

工作目录包含 `rc.scs`、`run.csh`、`spectre.log`、原始 `psf/`、归一化 `ac.dat`/`transient.dat`、`metrics.json`、事件和摘要；归档记录文件哈希。`verify-job` 和 `verify-archive` 不需要 Spectre 许可证，会重新从原始 PSF 生成表格、验证哈希并调用独立评分器。解析器还核对 `in` 是否对应固定电压源：AC 为 1 V，瞬态为任务给定电压。AC 必须有 121 个正确频点；瞬态必须覆盖 10τ 且足够密集。AC 最大绝对误差阈值为 `1e-5`，瞬态最大归一化误差阈值为 `5e-5`。`pass` 只表示这个理想 RC 任务通过。

进程超时、取消、输出缺失或许可证 `SPECTRE-209` 都标为 `not_evaluated`，不会混作电路失败。许可证错误归类为 `infrastructure_error`。配置或工作环境在提交后变化，会拒绝运行并保留诊断证据。服务器宕机/进程被外部杀死后的不完整作业不会自动重启；须先人工检查再换 ID。此处不是 PDK signoff、VABench Spectre Tool，也没有做多人共享账号的权限验收。

真实主机、管理员脚本路径、许可证变量及原始仿真日志属于私有部署信息，不放入本仓库。个人验收见[验证记录](VALIDATION.md)；构造夹具见[测试](../../tests/chips/test_spectre_rc.py)。
