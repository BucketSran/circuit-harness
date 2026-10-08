# circuit-harness

Circuit Harness 是面向开发者的电路智能体实验框架。你可以接入自己的 benchmark，通过仿真器适配代码连接电路工具，并组合不同的 Agent 与模型开展实验。框架提供任务接入、仿真执行、独立评测和轨迹处理；Harbor 负责 Agent 安装、Job 与 Trial 调度。

研究者与外部开发者使用同一套代码。任务、仿真器、PDK、模型连接和部署资源按实验配置提供，真实凭据与服务器配置留在本地或私有存储。

```text
选择 benchmark + Agent + Model + 运行环境
  → Agent 阅读任务、修改电路、调用公开仿真工具
  → 收取候选 → 按任务声明的 verifier 评分
  → 查看成绩、失败原因、轨迹和执行证据
```

## 跑第一个实验

从 [AnalogBench 快速开始](examples/analogbench/README.md)运行一个 100 MHz 无源滤波器任务。它只需要 R、L、C 与 ngspice，不需要 PDK 或商业许可证。先用参考解检查运行环境，再提供模型连接运行 Agent。示例引用固定版本的外部任务，保留原题面和评分规则。

要求 Python 3.12+ 与可用的 Docker。Harbor 固定为 0.23.0。

```bash
python -m pip install -e '.[harbor]'
python -m circuit_harness.benchmarks.analogbench --help
```

| 示例 | 仿真与评测 | 用途 |
| --- | --- | --- |
| [A：AnalogBench RLC](examples/analogbench/README.md) | ngspice，原任务 verifier，无 PDK | 首次准备任务并运行 Harbor |
| [B：AnalogBench SKY130 OTA](examples/analogbench/README.md) | ngspice、开源 SKY130 模型，原任务 verifier | 接入需要工艺模型的晶体管电路 |
| [EVAS：vaBench VA07](examples/evas-va07/README.md) | 固定 EVAS 源码，benchmark 自有八例 checker | 重放保存候选，查看开发版后端的评分和不可评原因 |

EVAS 示例是开发版评测路径，其结果不能代替任务声明的正式 Spectre 成绩。每个示例分别说明已验证的运行层级和限制，参考解通过也不表示某个 Agent 已解出该题。

## 接入你的实验

按[开发者接入指南](docs/guides/integration.md)准备任务、公开工具和 verifier，再选择 Agent、模型及部署方式。

- **Benchmark** 提供题面、候选接口、评测条件和 checker。可以引用 AnalogBench、接入 vaBench，或编写自己的 Harbor 任务。
- **仿真器** 按任务需要选择。已有 ngspice、EVAS、Spectre 等具体执行接口；新后端需要适配其输入、输出与错误语义。不同仿真器的分析能力和模型语法并不通用。
- **Agent 与 Model** 分别配置。已有 Harbor 配置入口支持 Pi、Codex、Claude Code、mini-swe-agent，具体组合取决于模型服务提供的协议。
- **运行环境** 可以在本机或服务器。用户提供实际工具安装、工艺模型、模型凭据以及商业仿真器所需的许可证。

解题时使用的仿真器与最终 verifier 可以不同。例如用 EVAS 提供公开调试反馈，再由独立 Spectre verifier 评分。正式比较需要固定任务版本、评分器与运行条件。

| 需要 | 入口 |
| --- | --- |
| 配置 Agent 与模型 | [Harbor 配置](docs/chips/HARBOR.md) |
| 将公开工具与私有评分材料分开 | [公开会话](docs/chips/CURRENT_EVAS_PUBLIC_SESSION.md)、[任务绑定](docs/chips/HARBOR_TASKS.md) |
| 本地、服务器及许可证环境 | [部署步骤](docs/chips/HARBOR_DEPLOYMENT.md)、[独立终评](docs/chips/BENCHMARK_EVALUATION.md) |
| 查看结果与准备训练数据 | [结果报告](docs/chips/HARBOR_RESULTS.md)、[ATIF 数据](circuit_harness/data/README.md) |
| 直接调用仿真器或重放候选 | [执行模块](circuit_harness/execution/README.md)、[其他示例](examples/chips/README.md) |

## 公共代码与私有配置

仓库保存适配代码、示例准备脚本、安全模板和合成测试。外部任务、PDK、真实服务器配置、许可证、模型密钥与运行产物分别保存在操作者的私有目录。`.gitignore` 覆盖 `private/`、`runs/`、`.env` 和 `*.local.json` 等常见位置；运行时还需要控制 Agent 能读取的文件，Git 忽略规则不能代替隔离。

框架代码采用 [Apache License 2.0](LICENSE)，来源声明见 [Notice.txt](Notice.txt)。外部 benchmark 内容、仿真器和 PDK 适用各自的许可证。提供商业仿真器适配代码不包含该软件、工艺库或使用许可。来源和发布规则见[发布边界](docs/chips/REPOSITORY_SCOPE.md#publication-boundary)。

## 参与开发

Python 包名是 `circuit_harness`，命令行入口是 `circuit-harness`。benchmark 准备与候选重放在 `circuit_harness/benchmarks/`，执行代码在 `circuit_harness/execution/`，Harbor 插件在 `circuit_harness/harbor/`，训练数据准备在 `circuit_harness/data/`。训练器由使用者另行选择。

默认分支为 `main`。开发约定见 [AGENTS.md](AGENTS.md) 和[开发 SOP](docs/chips/DEVELOPMENT_SOP.md)，检查入口见[测试说明](tests/chips/README.md)。
