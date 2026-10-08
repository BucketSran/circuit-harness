# 开源仿真器接入与 Harness 优化建议

2026-09-22。基于服务器只读巡检、已有 EMX 运行记录，以及用户的《ChipAgents：芯片设计闭环与 RSI 学习地图》。
本文保留最初巡检时的选型依据。随后已落实 [ngspice RC Harness](NGSPICE.md)：固定 47 安装、
服务器本地执行、AC/瞬态独立验收与下载校验；实测见 [验证记录](VALIDATION.md)。其他候选与优化仍为建议。

## 安装状态与当前瓶颈

最初巡检在获准节点的 PATH、RPM 包记录、环境模块查询，以及常见系统目录中未找到 ngspice。
本人 HOME 和 Cadence 目录的文件名扫描达到 12 秒上限；部分系统子目录无读取权限。
结论是“未发现可直接使用的 ngspice”，不是穷尽全盘后证明没有安装。
容器 CLI 存在，但容器运行权限、已有镜像及镜像内软件未验证。
实际路径、完整命令、超时和权限错误保存在操作者私有记录，不进入 Git。

复核两个既有 EMX 任务的首次运行事件，不包含后续 resume：

| 指标 | 任务 02 | 任务 03 |
| --- | --- | --- |
| 整条 CLI 耗时 | 16.33 s | 16.86 s |
| 远端 EMX 进程耗时 | 0.81 s | 0.87 s |
| SSH/scp 子进程数 | 15 | 15 |
| 其中下载次数 | 8 | 8 |
| 下载子进程累计耗时 | 8.46 s | 9.36 s |

15 次调用包括准备目录 1 次、上传 3 次、状态查询 2 次、提交 1 次、下载 8 次。
该行为与 [emx.py](../../alphaapollo/common/execution/chips/emx.py) 的逐文件传输一致。
这些时长包含进程启动、连接、远端工作等，不能称为纯网络延迟，也不构成不同部署方案的速度对比。
现阶段优先验证减少往返的收益，比假设换求解器一定更快更有依据。

## 哪些阶段可以使用开源工具

| 任务 | 可接入组件 | 在我们平台中的用途与边界 |
| --- | --- | --- |
| 无 PDK RC/RLC，兼容模型的模拟电路 | [ngspice](https://ngspice.sourceforge.io/ngspice-control-language-tutorial.html) | 批处理、参数扫描、日常回归；先用解析解验收，再增加开放器件模型 |
| 更大电路的并行求解 | [Xyce](https://xyce.sandia.gov/about-xyce/) | 后续候选；并行能力不等于小电路更快，需在相同有效模型下测量 |
| 可兼容的 Verilog-A 紧凑器件模型 | [OpenVAF/OSDI + ngspice](https://ngspice.sourceforge.io/osdi.html) | 编译并加载器件模型；逐项验证模型语法、参数与分析类型，不承诺原有 Spectre 工程直接兼容 |
| 电磁场、无源结构与 S 参数 | [openEMS](https://docs.openems.de/en/latest/intro.html) | 开源 3D FDTD 对照；需几何、材料叠层、端口、边界与网格适配，不是直接替换 EMX 命令 |
| EMX 输出分析 | [scikit-rf](https://scikit-rf.readthedocs.io/en/latest/tutorials/Networks.html) | 读取 Touchstone 和网络参数，作为解析器候选；电感/Q 等指标仍需明确端口、终端条件和公式 |
| 未来 RTL 功能调试 | [wave-mcp](https://github.com/Tencent/wave-mcp/blob/main/README.en.md) + 独立 RTL 仿真 | wave-mcp 查询 RTL/FST 证据，本身不运行仿真；模拟 PSF/频响需另做适配 |
| 未来数字综合与物理实现 | [OpenROAD-flow-scripts](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts) | RTL→GDS 的独立任务分支；不承担当前模拟电路或 EMX 后端的职责 |

电路网表求解和版图电磁提取是不同任务。ngspice 不直接消费 GDS 来替代 EMX。
商业 PDK 的模型格式、加密、器件支持和许可需逐项确认；开源工具可完成独立基线，但不能自动替代原工艺验收。
“开源”表示环境和代码可获取；运行速度、收敛性和精度均需测量。

## 哪些现有项目值得复用

**CACE 最贴近当前模拟电路评测需求。** 它把指标、单位、限值、条件和 testbench 写入 datasheet，
自动表征并输出汇总；[5T OTA 教程](https://cace.readthedocs.io/en/latest/tutorials/ota_5t.html)
展示了 ngspice、增益/UGF/相位裕度及角点/温度组合。
先借鉴其 [datasheet 结构](https://cace.readthedocs.io/en/latest/reference/datasheet_format.html)，
再用一个固定版本的例子评估是否作为可选表征后端。它不自动提供 Apollo 的 SSH 作业协议或 Agent 轨迹。
其当前文档将自定义结果后处理支持限定在 ngspice，不能据此假设已有 Spectre/EMX 原生适配。
来源：[后处理文档](https://cace.readthedocs.io/en/latest/tutorials/custom_scripts.html)。本次未安装或执行 CACE。

[IIC-OSIC-TOOLS](https://github.com/iic-jku/IIC-OSIC-TOOLS) 提供包含开源 EDA 工具与开放 PDK 的容器环境，
可以解决协作者环境差异；需单独确认服务器容器权限、存储、镜像获取及许可边界。
固定镜像 digest、平台架构和 PDK 版本；先完成独立小型 ngspice 基线，再判断是否值得引入完整镜像。
若服务器无法下载，可准备已校验摘要、适配架构的离线安装包，在本人环境部署；本轮未执行部署。

用户笔记中的 poRTLe/VerilogEval 主要用于学习数字任务的 runner、agent 接口和评测统计。
本次 poRTLe 文档链接获取失败，仅保留笔记中的学习定位，不声称重新核查或成功复现了其实现。

## 从 ChipAgents 学什么

[快慢验证循环](https://chipagents.ai/blogs/ai-agent-driven-timing-closure) 的可借鉴点是：
快速反馈帮助筛选候选，可信的完整流程负责确认。以下是我们对模拟/RF 的适配建议，不是其公开源码实现。

```mermaid
flowchart LR
    A[任务规格与候选参数] --> B[输入和能力检查]
    B --> C[快速反馈：解析模型或经校准的简化电路]
    C --> D[候选筛选]
    D --> E[完整任务：EMX / Spectre / 已验收开源后端]
    E --> F[独立指标检查与约束判定]
    F --> G[带来源的结果与经验]
    G --> A
```

同一 ngspice 后端既可承担完整的小电路验收，也可运行简化模型；“快/慢”和“开源/商业”不是同一维度。
先定义快速模型的适用范围与误差，再用保留工况检查筛选是否会漏掉好候选。
[行为模型文章](https://chipagents.ai/blogs/ai-created-behavioral-models) 也强调参考仿真、误差要求和迭代校准；
我们可先做等效 RLC 或响应拟合，保留全仿真复核，不能拿拟合数据上的误差当成泛化证据。

wave-mcp 提供按信号、时间及结构查询证据的具体参考。模拟/RF 可对应为按频段查 S/Y、按时间窗口查波形、
比较两次仿真的指标及差异；每个返回值关联 run ID、单位、端口、原始产物和解析器版本。
这能同时减少传输和 Agent 上下文体积，证据查询与最终评分保持独立。

经验库先保存“任务条件—修改—观测—是否接受—证据”，优先用不可变记录与检索。
工具、模型或策略更新后，在未参与改进的任务上验证；同一电路调好几轮不等于 Agent 自身持续进化。

## 按现有代码安排优化

| 优先级 | 具体改动 | 落地位置 | 验收方式 |
| --- | --- | --- | --- |
| P0 | 单一预检入口：后端/模型支持、版本、目录、许可证状态、就绪原因 | 扩展 tests/chips/probes 与 workflows/chips.py；上线时提炼运行时代码 | 缺软件、缺模型、许可证故障分别返回明确状态；基础检查不隐式启动商业仿真 |
| P0 | 减少逐文件下载；worker 内容缓存；随后再比较连接复用 | common/execution/chips/emx.py、ssh_worker.py | 与现有结果一致，保留逐文件哈希、部分下载恢复、同 job 幂等；统计往返和端到端耗时 |
| P0 | 新增 ngspice 无 PDK 基线，复用进程与事件基础 | common/execution/chips/ 下的新 adapter；examples/chips/benchmarks/；tests/chips/ | CLI 真实执行 RC，独立检查 AC/阶跃响应及失败语义 |
| P1 | 转换、仿真、解析全部在服务器运行，默认返回指标和证据引用 | 现有 execution/chips 模块与部署配置；复用 process.py/journal.py | 本机与服务器路径共用任务定义，同输入结果一致；比较上传/下载量与延迟 |
| P1 | 任务规格、解析器与纯评分器 | examples/chips/benchmarks/；拟新增 common/grader/chips/ | 明确运行失败、结果无效、设计不达标；单位、缺值和端口错误不能默认为通过 |
| P1 | 故障恢复与资源排队 | execution/chips 与 tests/chips 验收入口 | 活动作业断线、取消、超时及许可证阻塞；不重复启动，不无限重试，不超分配 |
| P2 | 有适用域的快速模型、候选缓存、工具/Agent 对照和经验检索 | grader、Tools 与 Workflow 各自负责 | 固定任务/总预算，保留测试集；记录缓存命中来源和模型版本 |

上述目录均相对仓库根目录。现有 EmxConfig 与 worker 绑定 GDS/EMX，不能仅替换命令便称为通用后端。
先用第二个具体后端找出真正共享的部分，再提炼接口；保留现有 emx_simulate 的兼容性。
不要让 Agent 每次自行拼接安装路径、工艺、命令和评分脚本。初期只暴露少量经过验收的运行和查询 Tools，
后端及分析模式由任务规格明确选定；不支持的请求应明确拒绝，不静默换求解器。

结果缓存需包含输入依赖、工具/模型/工艺版本、求解参数和随机种子；评分规则变化可重新评分，
但不能伪装成新仿真。远端完整归档仍需哈希、恢复和保留策略，按需下载不表示丢弃原始证据。

## 最小的下一轮验证

1. 在用户独立环境固定 ngspice 版本，用 R=1 kΩ、C=1 nF 的 RC 网络作为不依赖 PDK 的公共小任务。
2. 分别检查 AC 传递函数和阶跃响应的解析参考，固定激励、初始条件、采样点及绝对/相对容差。
   R/C 与输入可由任务参数改变，评分器独立实现，不从被测输出反推参考值。
3. 沿用已跑通的 EMX 厂商例子，比较现状与减少传输往返后的执行；先验证恢复与哈希，再比较性能。
4. 许可证可用后，把同一 RC 物理任务接到 Spectre，用各自合法网表表达并对同一解析参考评分。
5. 随后选 CACE 的一个开放 OTA 案例，锁定工具、PDK、testbench 与指标；作为第二级真实电路基线。

上述计划中的 ngspice RC 安装与后端现已完成；CACE、Spectre 和 EMX 传输优化对照尚未完成。
三次 RC 小任务的实测耗时不构成不同求解器或部署方案的性能比较。
