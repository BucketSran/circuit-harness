# Analog Design Bench：三题接入卡

状态：三道题的原版评分脚本已在 lab-server 用上游固定基础镜像环境完成参考解与公开 starter 正反控。两道 RLC 题另在服务器原生 ngspice 47 上复现。`rlc-rf-bandpass-100mhz` 已有真实 Pi＋GLM 提交、独立终评闭环（15/15）；宽带题也已完成自主公开仿真、恢复、提交、终评及服务器／本机归档核验（6/7，0.9），电路最坏反射系数仍未达标。OTA 没有 Agent 解题证据，三题均尚无第二位成员复现证据。详见[验证记录](../../../../docs/chips/VALIDATION.md)。

| task_id | 上游源码提交 | 本次验证 |
| --- | --- | --- |
| `rlc-rf-bandpass-100mhz` | [`fb0ec304`](https://github.com/Arcadia-1/analog-design-bench/tree/fb0ec30463d005d3e463caf4e48ab9a26008e869/tasks/rlc-rf-bandpass-100mhz) | 原参考电路 15/15，starter 0/15 |
| `rlc-broadband-50-to-200-match` | [`fb0ec304`](https://github.com/Arcadia-1/analog-design-bench/tree/fb0ec30463d005d3e463caf4e48ab9a26008e869/tasks/rlc-broadband-50-to-200-match) | 原参考电路 7/7，starter 0/1 |
| `sky130-ota-5t-gain40-pm60-noise50uv-pvt` | [`c23f124d`](https://github.com/Arcadia-1/analog-design-bench/tree/c23f124de1e461655d2e02ce6cfae2654ccea0d3/tasks/sky130-ota-5t-gain40-pm60-noise50uv-pvt) | 原参考电路 7/7，starter 0/7 |

两道 RLC 题不属于当前 50 题版本，不能把三题结果合并成当前榜单成绩。上游任务、参考解、验证器和镜像均由 Analog Design Bench 提供；本仓库只保存固定版本、调用与结果解释，不复制题目内容。其软件和 benchmark 内容分别受上游 [Apache-2.0 / CC BY-NC 4.0 声明](https://github.com/Arcadia-1/analog-design-bench/blob/main/LICENSE)约束。

候选提交物均为 `circuit.spi`。Agent 只能看到该题公开 instruction、starter、自己的候选和公开仿真反馈；`solution/`、`tests/` 及最终评分留在操作者侧。`analog-bench` 是操作者侧评分入口，Agent 使用独立的公开仿真／提交 Tool。完整执行说明和证据级别见[运行说明](../../../../docs/chips/ANALOG_DESIGN_BENCH.md)。

两道 RLC 题现在由 `task_id` 选择，共用六个公开 Tool、会话与 Pi 入口。
宽带题使用原公开分析脚本的有限 Q、11 点扫频反馈；终评保留原评分器。
用 [100 MHz 实验模板](experiment.example.json) 或 [宽带实验模板](experiment.broadband.example.json)，
并让 operator 与新会话使用相同的任务 ID。新入口已通过本地构造回归和真实上游文件的
离线会话／归档检查；2026-09-26 宽带公开仿真在 lab-server 的固定镜像正负控也已通过。
自主 Agent 闭环与这些操作者控制分开验收。宽带真实模型首回合预算停止后收取，得分 0.1；
第二个新会话主动恢复并提交已仿真的候选，得分 0.9，均未达到全部电路指标。
OTA 尚未接入该 RLC Agent 协议。
