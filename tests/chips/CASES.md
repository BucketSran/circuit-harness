# Circuit Harness 验收案例

本表定义需要验证的边界，不表示本次已经执行。按[测试入口](README.md)选择检查，
真实实验按[记录模板](RECORD_TEMPLATE.md)填写版本、资源、证据与限制。

| ID | 验证目标 | 当前检查入口 | 证据边界 |
| --- | --- | --- | --- |
| CHIPS-PROBE | 主机与网络探针失败保留证据 | `test_host_snapshot.py`、`probes/host_snapshot.py` | 本地夹具不证明真实 endpoint 或认证可用 |
| CHIPS-TRANSPORT | 有界重试、相同作业恢复与产物完整性 | `test_simulator.py` | 构造 SSH/EMX，不认证实际网络与许可证 |
| CHIPS-RC | RC 仿真、独立解析解与恢复 | `test_ngspice.py`、`ngspice-rc`、`verify-rc` | 本地回归和可选真实 ngspice；服务器条件另记 |
| CHIPS-ACCESS | Agent/普通账号读取与写入边界 | `test_native_sandbox.py`、`test_harbor_public_gateway.py`；实际不同 UID 探针 | 同 UID 或元数据检查不代替跨账号访问验收 |
| CHIPS-NET | 模型 endpoint 的连接、认证和协议 | 无认证探针、获准的模型请求、实际 Agent Trial | 403、HTTP200或能访问其他网站都不单独证明协议可用 |
| CHIPS-EDA | 真实仿真、解析与独立电路判定 | EMX、Spectre RC 和 benchmark 对照 | 参考控制、Agent 成绩与任意PDK验收分别报告 |
| CHIPS-RECOVERY | 独立后台完成、断线查询、取消/超时 | `test_ngspice.py`、`test_spectre_rc.py`；获准 SSH 故障探针 | 本地进程回归不认证所有远端取消条件 |
| CHIPS-STORAGE | 固定输入下的存储与归档 | `probes/storage_compare.py`、`test_ngspice.py`、`test_vabench.py` | 历史测量见[存储记录](../../docs/validation/STORAGE_VALIDATION.md)，不外推求解器性能 |
| CHIPS-ARCHIVE | 后台归档、仅归档重试与校验后清理 | `test_ngspice.py`、`test_vabench.py` | 操作者协议覆盖不代表自动选盘或保留期调度 |
| CHIPS-VABENCH-AGENT | 固定公开动作、冻结与终评 | `test_vabench_session.py`；当前 Agent 接入按 Harbor task 验收 | 固定操作者协议不等于开箱即用的 Agent 任务 |
| CHIPS-EPISODE-REPORT | 保存记录、候选与归档的离线读取 | `test_episode_report.py` | 不执行模型或重新评分；只支持既定来源布局 |
| CHIPS-HARBOR | Agent/Model 配置、Trial 生命周期与部署 | `test_harbor_profiles.py`、`test_harbor_composition.py`、`test_harbor_deployment.py` | 协议、容器与模型/许可证验收分别判断 |
| CHIPS-FINAL | 冻结候选与独立评分关联 | `test_harbor_verifier_archive.py`、`test_harbor_opensource_verifier.py`、`test_harbor_reporting.py` | checker 夹具不认证真实电路或商业后端 |
| CHIPS-TRAJECTORY | 完整轨迹、评分关联与数据筛选 | `test_pi_trajectory.py`、`test_harbor_trajectory.py`、`test_atif_sft.py` | 目前正式 Pi 导出依赖公开会话冻结证据 |
| CHIPS-PLACEMENT | 不同部署的真实可用性与比较 | 同任务、工具、模型、预算，分别运行 | 原生 RLC 已有单次服务器成功；部署差异的因果比较仍待执行 |

## 实际执行要求

执行前固定主机、账号、私有目录、允许的数据、软件与工艺身份、资源/费用预算、
停止条件和恢复方式。复用已有用户授权；模板和技能不会自动授权新模型调用或实验。

独立 checker 判断设计成绩。分别记录执行成功、设计不达标、不可评、平台失败及
Agent 是否主动提交。计划中的失败与未启动项保留，不能只统计成功样本。
比较方案前固定条件与重复次数；证据不足时不报告因果结论或外推分位数。

已有实测范围见[验证记录](../../docs/validation/VALIDATION.md)，后续条件见
[后续工作](../../docs/development/NEXT_WORK.md)。历史成功不自动认证当前版本或新任务。

## 增加案例

将实际故障的最小复现加入相关模块测试，再登记其覆盖的验收边界。
只增加真实需要的 fixture、脚本与规格。
