# VABench r53 — family 001 接入验收

来源：用户维护的 vaEVAS-next / behavioral-veriloga-eval，固定源码
`0685aae05c346e8e60f33ba48e2f64daff54d4f2`，封存 `benchmarkv4-r53`，仿真器 `evas-sim==0.8.7`。
完整 release 声明 400 个 family、1,200 个任务；这里验收一个 family 的三种任务。
原封存内容独立部署，不复制进 Chips Git；访问者需自行取得授权的数据和源码。

| 任务 | 目标 | 正例 | 负例与预期 |
| --- | --- | --- | --- |
| `v4-001` DUT | 实现 bang-bang phase detector 行为模型 | 原 evaluator/solution 通过 | 原 bugfix buggy_bundle → behavior_failure |
| `v4-1001` bugfix | 修复相同电路的故障实现 | 原 evaluator/solution 通过 | 原 public/buggy_bundle → behavior_failure |
| `v4-501` Testbench | 编写能验证参考 DUT、识别规定 mutation 的测试台 | 原 evaluator/reference_tb.scs 通过 | 显式构造非法测试台 → compile_failure |

参考答案与负例只用于 **operator 无模型接入验收**；不作为给 Agent 的 prompt、tool response 或 memory。
模型可见材料通过原 exporter 单独导出；最终测试、mutation/checker 和 score sidecar 保持独立。
原 Testbench 评分含参考 DUT 与五个负向 mutation；沿用原安全检查、信号要求及判定，无自设替代容差。

接口、命令与目录：[VABench 使用说明](../../../../docs/chips/VABENCH.md)。
执行/固定：[vabench.py](../../../../circuit_harness/execution/vabench.py)；
原接口桥接：[vabench_worker.py](../../../../circuit_harness/execution/vabench_worker.py)；
验收：[vabench_smoke.py](../../../../tests/chips/probes/vabench_smoke.py)。

状态：本机与实验室服务器的真实 EVAS 六个正负例均符合预期；服务器仿真阶段断开 SSH 后完成、同 ID 不重跑。
下载完整性验收与版本证据见 [验证记录](../../../../docs/chips/VALIDATION.md)。
已在单题 `v4-001` 分别完成服务器 Pi＋GLM 与本机 Codex＋GPT 的真实模型、公开仿真和独立终评；
见[双路径示例](README.md)及[验证记录](../../../../docs/chips/VALIDATION.md)。
未做第二位成员独立复现、全量 release 重认证或 Spectre 等价性验证。
该行为级切片不能外推晶体管/PDK、版图、EMX 或全部模拟 IC 设计能力。
