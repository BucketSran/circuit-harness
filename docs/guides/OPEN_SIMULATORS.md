# 接入仿真器

按任务需要选择后端，并记录软件版本、模型文件、分析条件与执行位置。
Harness 提供执行与证据接口；仿真器负责求解，benchmark 负责判据。
安装成功、公开仿真成功和最终电路达标分别验收。

## 当前接口

| 后端 | 入口 | 适用范围与前置条件 |
| --- | --- | --- |
| ngspice | [RC 执行](../reference/NGSPICE.md)、[AnalogBench 示例](../../examples/analogbench/README.md) | 无 PDK 的 RLC，或任务固定的兼容器件模型；使用者准备运行环境 |
| EVAS | [公开会话](../reference/CURRENT_EVAS_PUBLIC_SESSION.md)、[VA07 重放](../../examples/evas-va07/README.md) | 固定源码与 kernel 的电压行为模型；支持范围由对应引擎版本决定 |
| Spectre | [RC](../reference/SPECTRE_RC.md)、[冻结终评](../reference/BENCHMARK_EVALUATION.md) | 已授权安装、兼容 PDK 与许可证；使用者提供私有 profile 和访问配置 |
| EMX | [JSON/GDS 示例](../../examples/chips/emx/README.md) | 固定转换器、几何、端口、工艺与 wrapper；执行成功不等于电磁设计达标 |
| Icarus | [执行模块](../../circuit_harness/execution/README.md) | RTL smoke 与进程/波形检查，不是模拟电路或 PDK 验收 |

解题工具与最终评分后端可以不同。例如 EVAS 提供公开诊断，Spectre 对最终提交评分。
使用开源或商业后端都要固定评分条件，不能未经校准就把两个后端的成绩混合比较。
商业软件和 PDK 不随适配代码提供，发布范围见[仓库边界](../development/REPOSITORY_SCOPE.md#publication-boundary)。

## 增加一个后端

1. 选择真实任务，明确候选格式、单位、模型语法与分析类型。
2. 固定输入、软件与模型版本，用参考候选确认求解和输出解析。
3. 补充合法不达标候选、非法输入、超时和依赖缺失的处理。
4. 在 `circuit_harness/execution/` 接入实际输入输出，复用进程、作业、transport 与归档模块。
5. 若 Agent 需要受控访问，接入对应公开会话；最终评分使用任务声明的独立 checker。
6. 按[测试入口](../../tests/chips/README.md)验证协议，再按实际部署完成实验验收。

当前没有任意 `SimulatorAdapter` 的自动发现机制。根据真实后端需求扩展接口；
EVAS 的行为模型接口、SPICE 晶体管网表和 EMX 的 GDS 输入分别有自己的合同。
不支持的分析明确报告，不静默换求解器或把依赖故障记成模型零分。

执行框架的能力不意味着每种模型、PDK、平台和许可证模式都已验证。
已完成的条件见[验证记录](../validation/VALIDATION.md)，部署选择见[接入指南](integration.md)。
