# Chips 验收案例目录

本表定义要验证什么；“已有测试”不等于本次已执行。每次运行依据 [模板](RECORD_TEMPLATE.md) 记录结果。

| ID | 验证目标 | 入口/方法 | 当前覆盖 |
| --- | --- | --- | --- |
| CHIPS-PROBE | 只读采集；默认无网络；缺路径/网络失败保留证据；不把 403 当认证成功 | `test_host_snapshot.py`、`probes/host_snapshot.py` | 可执行本地回归；curl 响应为构造夹具 |
| CHIPS-TRANSPORT | 非法输入不提交；重试有界；断线恢复不重复启动；产物完整性 | `tests/common/execution/test_chips_harness.py` | 已有构造 SSH/EMX 回归；不能证明真实网络/EMX |
| CHIPS-RC | 理想 RC 的真实 ngspice AC/瞬态、独立解析验收、完成恢复、产物哈希 | `tests/common/execution/test_chips_ngspice.py`、`ngspice-rc` / `verify-rc` | 本地构造进程回归及可选真实 ngspice；服务器实测结果单列 |
| CHIPS-ACCESS | 不同普通账号的读取/写入边界、同组访问、父目录及 ACL | 临时无秘密文件，获准的第二测试账号独立尝试 | 元数据探针已具备；跨账号执行步骤待具体环境提供，不模拟成通过 |
| CHIPS-NET | 实际 endpoint 的 DNS/TCP/TLS/HTTP；再分别验证认证与模型协议 | 无认证探针 → 获准的最小模型请求 → 一次工具调用 | 无认证探针已具备；真实模型验收未接入自动入口 |
| CHIPS-EDA | 确定性输入，经真实转换、EMX/Spectre、解析和独立评分 | 专家确认样例、版本、端口/单位、参考指标和容差 | EMX 厂商样例数值一致性通过；Spectre 的无 PDK RC-001 真实仿真及解析解评分通过；专家/PDK 任务待接入 |
| CHIPS-RECOVERY | 实际长作业断开客户端后可查；取消/超时回收；重试不重复启动 | 对隔离作业中断客户端/注入通信故障，核对 job/PID/结果 | ngspice 真实 SSH 断线通过；Spectre 后台提交后 SSH 退出、终态查询与同 ID 去重通过；Spectre 强制断线、取消/超时待验收 |
| CHIPS-STORAGE | 本地文件系统与 NFS 对同一真实负载的影响 | `probes/storage_compare.py`：固定有界源码包、交替运行、记录缓存与负载 | 795 文件源码包三轮真实对照已完成；见存储验证记录，不能外推为求解器或物理磁盘性能 |
| CHIPS-ARCHIVE | 后台 scratch 执行后独立归档与可控清理 | `test_chips_ngspice.py`、`test_vabench.py`；真实 SSH 探针可指定工作/归档根目录 | 自动归档、仅归档重试、清理门禁、持久 ID 和阶段计时已实现；范围及证据见 [存储说明](../../docs/chips/STORAGE.md) 和验证记录 |
| CHIPS-VABENCH-AGENT | 公开候选修复、冻结、隔离终评、关联归档与 Pi 请求限额 | `test_vabench_session.py`；旧 Agent runner 已退役，见迁移说明 | 三类任务脚本闭环与真实 Pi/SSH/EVAS 协议测试通过；模型响应为脚本 fixture，旧 Agent 证据按原版本解释；新 Harbor task 需独立验收 |
| CHIPS-EPISODE-REPORT | 已有轨迹离线重建成功、修复和预算停止；未知数据不造零；候选/归档不一致拒绝 | `test_episode_report.py`、`tests/workflows/test_visualize.py`；四条真实历史证据离线核对 | 已实现，见[报告说明](../../docs/chips/EPISODE_REPORT.md)和验证记录；不调用新模型或仿真 |
| CHIPS-PLACEMENT | 本机 Agent + 服务器 Tools vs 全服务器的耗时、稳定性和成本 | 固定任务、工具、模型、预算，分别运行，核对结果有效性 | 待两条路径具备同等功能后执行 |

2026-09-22 真实验证进展见 [服务器基线](../../docs/chips/BASELINE.md)：EMX 厂商例子已通过真实 SSH/Harness；
完成任务的 resume 和下载哈希已验证；NFS mtime 偏差导致的状态误判已补回归并修复。
2026-09-23 管理员指定的环境脚本已解除无 PDK RC 的 Spectre 许可证阻塞；
服务器本地 RC-001 作业、PSF 解析、独立评分和持久归档通过，见[验证记录](../../docs/chips/VALIDATION.md)。
EMX/Spectre 的强制运行中断线、取消/超时与真实工艺任务仍待验收。
ngspice 的真实 SSH 断线、重连前自主结束、同 ID 去重及 RC 独立评分已通过，
后台取消/超时另有本地真实进程与构造波形回归，详见 [验证记录](../../docs/chips/VALIDATION.md)。

## 实验室案例必须补齐的条件

执行前填清实际主机、账号/工作目录、允许的数据、工具/工艺身份、资源/费用预算、停止条件和恢复方式。
复用已有用户授权；模板或 Skill 不会授予登录、付费调用、数据外发或操作其他账号的权限。
元数据权限检查不能替代不同 UID 的实际访问测试；第二账号未提供时明确记为 blocked/not_run。
同样不测试或承诺对 root/存储管理员不可见。

EDA 参考结果由独立的数值/结构判定验证，不由被测 Agent 自评。
记录成功、设计不达标、执行失败和不可评四种不同事实；比较时保存全部预定任务的结果。
性能阈值和重复次数在预试后固定；记录配对任务、中位数/波动、峰值内存和所需传输量。
超出样本支持范围的分位数或因果结论标为证据不足。

## 增加案例

出现新故障时，先将最小复现加入相关模块测试，再在这里登记它支持的验收能力。
新案例只新增其实际需要的 fixture、脚本或规格，不为空的“将来测试”创建目录树。
