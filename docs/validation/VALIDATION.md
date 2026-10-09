# Circuit Harness 验证记录

本页汇总已取得的证据及其边界。每项结果只适用于记录的软件、任务与部署条件，
不认证任意机器、Agent、模型、PDK 或许可证。完整原始记录留在私有运行存储。
当前入口见[文档导航](../README.md)，检查方法见[测试入口](../../tests/chips/README.md)。

## 公开包与本地检查

2026-10-09 实现版本 `4409953` 的完整离线回归为 **625 passed、24 skipped**。
使用固定 Analog 来源与可挂载临时目录；跳过可选依赖或环境项不代表对应能力通过。
随后 `46e7e8e` 与 `3f862c3` 仅修改文档，PR #2 的两项 CI 均通过，合并版本为 `6e3ecde`。

该交付还完成新旧入口兼容检查、仓库外 wheel 安装、schema 资源与标准库 zipapp
检查。具体代码边界由测试定义；这些结果不证明真实模型或商业许可证可用。

## AnalogBench 原题与操作者控制

示例及来源固定方式见 [AnalogBench](../../examples/analogbench/README.md)。
任务题面、公开 bench 与原 checker 保留，各任务分别固定源码与 runtime。

| 条件 | 已观察结果 | 能证明什么 |
| --- | --- | --- |
| 本机 RLC 原 checker 与 Harbor Oracle/Nop | 参考15/15，starter0/15；独立 verifier | 环境、参考与失败控制，不是 Agent 成绩 |
| 本机 SKY130 OTA 原 checker 与 Harbor Oracle/Nop | 参考7/7，空 starter0/7 | starter 在仿真前被合法性拒绝，不能证明性能测量已执行 |
| Linux rootless Podman、ngspice-46、固定 SKY130 任务 | 参考7/7；四个器件L由1.0改为0.7µm的合法负例4/7 | 原增益测量与性能判据执行；CMRR/PSRR按原规则blocked |

每次分别记录来源、镜像和候选摘要。B 的评分隔离按原任务实现判断，不等同于
Harness 私有冻结会话。上述条件未调用真实模型。

## 原生 Harbor 的 Pi + GLM 自主 RLC 实验

2026-10-09，在 Linux 服务器运行 `rlc-rf-bandpass-100mhz`，使用 Harbor 0.23.0、
Pi 0.87.0、`glm-5.3-flash` 与 ngspice。Agent 预算1800秒，单次尝试，无自动 Trial 重试。
任务固定于 `fb0ec30463d005d3e463caf4e48ab9a26008e869`；原英文题面与 checker 保留。

- Trial 自然结束，无异常；原独立禁网 verifier **15/15、reward=1.0**。
- Agent 耗时约17分52秒，完整 Trial 约20分39秒。
- 59个逻辑模型请求均记录实际发送 `reasoning_effort=low`；这不证明供应商内部算力分配。
- 使用私有模型元数据适配、预装 Pi 镜像和受控 Podman 运行配置。通用公开配置尚未覆盖这些适配。
- CPU controller 不可用，明确未强制 CPU 配额。清理确认完成。
- 原生轨迹和 checker 结果均保存；会话转换得到61步，但尚未经过正式候选/评分关联导出。

这是一道题的一次自主解题证据，不是总体成功率、跨 Agent 比较或速度提升实验。
本次不使用 EVAS、Spectre、PDK 或商业许可证。

## EVAS 与独立 Spectre

两种 EVAS 路径分别记录，不能混用版本或评分含义：

| 条件 | 已观察结果 | 限制 |
| --- | --- | --- |
| 本机固定 EVAS0.14、VA07 原八例 replay | 参考8/8；wrong-speed六例graded fail、两例backend error，整题null/unevaluable | 开发版 EVAS 评分，不代替正式 Spectre |
| Linux 服务器 Pi + GLM、公开 EVAS、600秒 Agent预算 | 原 Trial 超时，三版候选触发 EVAS 支持范围诊断，未完成测量与显式提交 | 原 Trial 保留failed、score=null |
| 同一冻结候选事后独立 Spectre 恢复评分 | 原八例8/8 | 单独恢复记录，不改写原 Trial 成功状态 |
| 无模型参考控制 | 公开EVAS、Python测量、显式提交与独立Spectre8/8 | 证明工具与 verifier 链路，不证明 Agent 自主完成 |

版本、运行条件与实际失败见 [EVAS 示例](../../examples/evas-va07/README.md)。
EVAS 语义和算法由 vaEVAS 改进，Harness 负责支持范围诊断和证据关联。

## 保留后端的早期实测

以下结果来自2026-09的固定实现与部署条件，证明当时的具体执行协议，
不能直接当作当前 Harbor Agent Trial 的验收。

| 后端或协议 | 已观察范围 | 仍不能据此声称 |
| --- | --- | --- |
| EMX5.7 厂商电感样例 | JSON重建GDS几何一致，真实SSH运行，Y/S对照一致，恢复与下载哈希校验 | 专家定义的电感/Q判据、任意PDK、真实Agent完成 |
| ngspice47 RC | AC/瞬态解析解验收，后台独立完成，SSH断线后查询与同ID去重 | 任意晶体管任务、通用作业服务或求解器性能排名 |
| Spectre RC-001 | 指定授权环境下仿真、PSF解析、解析解评分与归档通过 | 任意商业任务、所有许可证模式或强制取消验收 |
| VABench r53 / EVAS0.8.7 | family001三类任务的正负例回放与固定源码身份 | 整个任务集或当前EVAS版本的验收 |
| scratch与归档 | 固定输入下归档完整性、单独重试、校验后清理 | 自动选盘、保留期调度或磁盘性能承诺 |

操作合同分别见 [EMX 示例](../../examples/chips/emx/README.md)、[ngspice](../reference/NGSPICE.md)、
[Spectre RC](../reference/SPECTRE_RC.md)、[VABench](../reference/VABENCH.md)和[存储记录](STORAGE_VALIDATION.md)。
旧实现的详细运行记录可以在 Git 历史中查阅，不再作为当前操作指南。

## 尚未验收

按[后续工作](../development/NEXT_WORK.md)补齐配置化服务器部署、原生任务的统一结果与正式轨迹导出、
更多任务和重复尝试、其他 Agent/模型组合，以及实际训练器和目标 tokenizer 的训练效果。
新增记录明确区分本地夹具、真实仿真、参考控制、Agent Trial 与独立成绩。
