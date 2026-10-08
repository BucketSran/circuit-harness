# Chips 任务集与数据集入口

这是电路 benchmark 的定义和登记入口。首次使用先运行 [AnalogBench 快速开始](../../../docs/chips/ANALOG_QUICKSTART.md)，EVAS 开发版评测见 [VA07 重放示例](../../../docs/chips/EVAS_QUICKSTART.md)。

[RC-001](rc/TASK.md) 是最小 Harness 基线；
已接入 [VABench r53](vabench/TASK.md) 的固定任务回放与原评分器；
新增 [Analog Design Bench 三题任务卡](analog_design_bench/TASK.md)：两道历史 RLC 和当前 OTA 已在 lab-server 用原评分脚本完成正反控；
Analog Design Bench 已有一题 Pi/GLM 的 development 单回合与原评分器终评，尚无稳定表现或 Codex 条件验收；
目前没有跨任务集的通用加载器。
先复制 [TASK_TEMPLATE.md](TASK_TEMPLATE.md)，在本目录的 `<任务族>/TASK.md` 中填写。
新的 Agent 实验使用 [Harbor 配置](../../../docs/chips/HARBOR.md)与[开发者接入指南](../../../docs/chips/INTEGRATION.md)；任务定义和评分器仍由 benchmark 维护。

## 每个任务族保存什么

- `TASK.md`：任务说明、输入/可调参数、预算、有效性规则、指标与容差、专家确认状态。
- `fixtures/`：允许随仓库共享的小输入、精简结果，以及用于判定逻辑的正确/错误样例。
- `sources.md`：大数据或受限资料的来源、版本、校验摘要、访问和准备说明；不直接提交其内容。
- 实现数据准备与评测后，再登记真实命令、构建 manifest 和对应代码路径。

这些是目录约定，没有隐含的自动发现机制。VABench 通过显式 pin 和 task ID 选择，不隐式扫描任务目录。
可执行逻辑分别归属 data_preprocess、execution、grader；任务目录作为统一入口引用它们，
不要在多个 examples 中复制同一份数据/评分逻辑。运行产物写到 `runs/` 或配置的外部位置。

## 任务进入评测的条件

1. 输入及可修改范围明确，代码能够验证不合法输入。
2. 有专家认可的参考运行，结果记录了软件、工艺/配置版本与输入摘要。
3. 运行成功、结果有效、设计达标各有判定；工具故障不会计成设计错误。
4. 至少有用于验证判定区分力的成功、设计失败或不可评样例。
5. 明确数据来源、版本、开发/测试划分与可复现条件；未知项不以默认值掩盖。
6. 第二位成员可按说明复现后，才标记为已复现。

先做一个案例闭环，随后扩展到复现、诊断、受约束优化等任务族。
首次工具、Agent、memory 实验共享冻结的任务版本和最终评测规则。
最终评测答案/标签不传给模型；目录名叫 private 不构成隔离，运行器必须控制实际可见内容。

任务登记随首个真实贡献开始，使用“规格草稿 / 工具已复现 / 第二位成员已复现 / Agent 已验收”
等有证据支撑的状态。没有运行证据时，不写“已跑通”。
