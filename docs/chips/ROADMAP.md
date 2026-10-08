# Circuit Harness 方向

目标是完整的电路实验平台：以相同任务、预算和评分条件比较 Agent、模型与工具，并为复现实验和训练保存可核查的数据。Harbor 拥有 Agent/Job/Trial，Harness 拥有电路执行与证据，任务方拥有题目和判据。

已实现的代码入口见 [README](../../README.md)，历史实测见 [VALIDATION](VALIDATION.md)。本地回归不替代模型、许可证或任务成功率验收。

后续优先补齐真实多任务和重复实验、调用方的新包迁移，以及各部署组合的独立验收。更完整的轨迹训练还需要训练器侧验证与学习效果对照。经验检索或 Memory 作为后续实验变量，通过现有 Agent/Harbor 接口扩展，不恢复 Apollo runtime。

固定 VABench r53 保留作复现后端；当前 EVAS 随 vaEVAS 发展，单独记录版本与源码摘要。二者的结果不能不加区分地合并统计。公开 Spectre、更多商业仿真器和许可证模式由实际需要推动，不作为当前公开 EVAS 主线的前置。
