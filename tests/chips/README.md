# Circuit Harness 测试入口

按受影响边界选择测试。开发遵循 [SOP](../../docs/chips/DEVELOPMENT_SOP.md)；真实实验使用 [CASES](CASES.md) 和[记录模板](RECORD_TEMPLATE.md)，不能以本地夹具代替实验室证据。

## 执行与报告

- `test_simulator.py`、`test_ngspice.py`、`test_spectre*.py` 检查进程、GDS、作业、取消、SSH fixture 和离线包。Icarus/ngspice 已安装时额外执行本地仿真。
- `test_vabench*.py`、`test_analog*.py` 保留固定任务、公开会话、冻结、终评和归档边界。它们不调用真实模型或实验室服务器。
- `test_task_authoring.py` 检查草稿来源、确认、单位与操作者 CLI。
- `test_episode_report.py` 检查旧轨迹离线读取、HTML 转义、缺失证据与归档校验。
- [test_standalone.py](../test_standalone.py) 阻止 Apollo/verl 导入，检查新 CLI 与 Harbor 插件。
- [wheel_smoke.py](../wheel_smoke.py) 在隔离环境安装 wheel，在仓库之外检查 CLI、资源和纯标准库 zipapp。
- `tests/test_analog_example.py` 和 `tests/test_evas_example.py` 检查入门任务的准备、配置与证据读取边界。真实 ngspice 与 EVAS 控制实验另见 [Analog 入门](../../docs/chips/ANALOG_QUICKSTART.md)和 [EVAS 入门](../../docs/chips/EVAS_QUICKSTART.md)，这些回归不发送模型请求。

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
| [test_atif_dataset.py](../test_atif_dataset.py) | Parquet 工具结构、完整模板 token、assistant loss mask、截断拒绝及独立 Dataset 加载 | CPU tokenizer 夹具；需 Torch/Transformers，阻止导入旧训练器；CI 独立离线 job |

任务和后端要求见[当前 EVAS 会话](../../docs/chips/CURRENT_EVAS_PUBLIC_SESSION.md)、
[独立评测与回放](../../docs/chips/BENCHMARK_EVALUATION.md)和
[Harbor 适配](../../docs/chips/HARBOR.md)。缺少工具或固定镜像时明确记录 skip，
测试不自动拉取镜像。Docker 的测试目录必须在 daemon 可挂载的文件系统内。
macOS 的虚拟机后端可能不共享默认临时目录，此时把 pytest `--basetemp`
指向 worktree 中一个新的、已被忽略的 `.planning/chips/` 子目录。

## 本地检查

```bash
python -m pip install -e '.[dev,harbor,chips,sft]'
ruff check circuit_harness tests
ruff format --check circuit_harness tests
HF_HUB_OFFLINE=1 python -m pytest -q
python -m circuit_harness._schema_export
python -m compileall -q circuit_harness tests
python -m pip wheel --no-deps . --wheel-dir dist
python tests/wheel_smoke.py dist/circuit_harness-0.1.0-py3-none-any.whl
```

缺少可选软件或指定镜像时报告 skip。容器测试不自动拉取镜像；macOS Docker 的临时目录须在 daemon 可挂载范围，必要时指定新的 ignored `--basetemp`。CI 的 CPU SFT job 独立安装 Torch/Transformers，无 verl 子模块。

`probes/host_snapshot.py`、`storage_compare.py`、`vabench_smoke.py` 和 SSH 断线探针只在明确的主机、存储与预算授权下调用。保存原始证据到私有运行目录。探针存在不代表对应实验已运行。

Apollo 通用 runtime、旧 Agent/批次入口和训练器的测试随功能退役；电路协议回归保留。具体范围见[迁移说明](../../docs/chips/MIGRATION.md)。历史运行记录见 [VALIDATION](../../docs/chips/VALIDATION.md)。
