# Chips 第一阶段：执行基础与实验室接入

日期：2026-09-22。状态：代码、本地验收和真实 EMX 厂商样例已通过；生产工艺及电路验收待完成。
详细证据与未执行项目见 [VALIDATION.md](VALIDATION.md)。
平台目标、Pi 优先实验及经验库方向见 [ROADMAP.md](ROADMAP.md)；
专家与其 Agent 的具体修改位置见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 当前确定的链路

用户确认的首个目标是：**JSON → GDS → SSH 上传 → 实验室 EMX 命令行 → 下载日志与结果**。
JSON 格式、生成脚本、EMX 工艺文件和端口设置目前没有现成实例，因此这些均为部署配置，不能靠猜测填入。
提供一个真实生成 GDS 的最小 polygon/label 示例；该版图不代表符合某个 PDK。
Icarus 仅用于本地进程管理、失败分类和波形产物的自测。

现有 vaEVAs 的 labctl/Cadence 连接模式已整理为
[Cadence 示例](../../examples/chips/cadence/README.md)。其用途是网表/Verilog-A 的 Spectre 仿真，
不能替代 EMX。没有复制实际账号、主机地址、PDK、私有评测数据或凭据。

## 已实现

| 能力 | 实现与可观察证据 |
| --- | --- |
| 执行关联 | MCP call ID、本地运行目录、固定远端 job ID、运行 attempt ID、阶段与每次 SSH/scp 尝试 |
| 运行监控 | 追加式 events.jsonl、阶段开始/结束、进程心跳、剩余时间、重试等待、CLI status/watch |
| 事后检查 | HTML 时间线；stdout/stderr；配置、GDS、远端命令与工艺/包装脚本哈希；下载清单 |
| 输入与产物 | 1 MiB JSON 限额；实际 converter；阶段检查点；逐文件字节数与 SHA-256 校验 |
| SSH 恢复 | 有限次数的指数退避加抖动；单命令/总预算；认证/主机密钥错误停止；不修改 SSH 安全设置 |
| 避免重复仿真 | 提交前持久化 job ID 和意图；远端 receipt/锁保证相同 ID 不重启；重连查询同一作业 |
| 断点恢复 | 相同输入、配置与 worker/可识别 converter 文件哈希；复用 GDS 和完成作业；重试未完成下载 |
| 生命周期 | 所属进程组清理；取消；远端超时；远端取消请求与终止确认分开记录 |
| 证据异常 | 未完成的尾事件明确标记并保留备份；坏完整记录报错；磁盘写失败不伪装成功 |
| Agent 入口 | 复用外部 runtime/MCP，新增一个 emx_simulate 工具；服务器和命令不能由模型参数覆盖 |

EMX 执行完成仅表示 `execution=ok`；`verdict=not_evaluated`、`certified=false`。
尚无可信的独立电路指标/设计验收器。未知费用、供应商真实 HTTP 次数不填零。
Agent 侧已有工具调用/返回轨迹，Chips 记录工具请求收到的 call ID。
MCP 服务会生成独立调用 ID；Pi 原生 toolCallId 并非自动原样透传。
模型调用、MCP 调用、run/job 与产物的跨层关联仍需真实运行验收。
本阶段没有改写公共 trajectory schema，也没有将 Chips 事件直接接入已有通用 HTML。

## 执行与恢复边界

- SSH 使用既有 OpenSSH 配置、BatchMode 和严格主机密钥检查；通过 argv 和引用后的远端命令调用。
- 每个作业使用独立目录。远端 worker 只依赖 Python 3.10+ 标准库，不要求实验室安装 AlphaApollo。
- SSH 丢失回复时，有限重试相同幂等请求；持久化记录不完整或远端状态过期时返回 unknown。
- 曾提交的作业消失后不自动重新提交。EMX 非零退出、缺失预期结果不会按网络错误重跑。
- 下载以 partial 文件落盘，校验后才发布；恢复不覆盖旧的 transport 尝试日志。
- 本地取消不等于远端终止。清理最多追加 5 秒查询/请求预算，无法确认时需要人工检查 job ID。
- 管理的是所属 POSIX 进程组；这是资源控制，不是安全沙箱。主动脱离进程组的工具需要调度器/容器。
- 输出限额通过轮询文件大小执行，不能当作硬磁盘配额。完整依赖环境/PDK 闭包仍需实验室部署管理。
- 模型 SDK 的网络重试仍由既有后端负责，本实现观测的是 SSH/scp 物理尝试，没有新增叠加的模型重试层。

## 代码归属与入口

- `common/execution/chips/`：进程生命周期、事件记录、GDS/SSH 编排、远端作业协议、本地 RTL smoke。
- `common/execution/tools/chips.py`：唯一模型工具与结构化反馈。
- `reasoning/runtime/external/bridge/mcp_server.py`：沿用工具桥的路由接线。
- `workflows/resources.py`、`workflows/_resources/runtime.py`：领域授权与部署环境变量转发。
- `workflows/chips.py`：直接运行、恢复、监控、报告 CLI。
- `examples/chips/`：布局、转换器、部署模板、Workflow 和 Cadence 连接参考。

操作命令见 [examples/chips/README.md](../../examples/chips/README.md)。

## 分支与协作

开发基于 `main @ 264a00cba2048bbf06153ea32f4b0271d07cdaeb`。
开源前在私有 `demo/chips` 上开发，其历史现保存在 `circuit-harness-private`。
当前公开仓库 `BucketSran/circuit-harness` 的集成分支为 `main`。
以下是 2026-09-22 核对过的上游参考提交。对应远端分支现已删除，所需历史留在私有备份；
这张表记录来源，不是当前可 checkout 的远端分支清单。

| 分支 | 上游提交 | 用途 |
| --- | --- | --- |
| `main` | `264a00cba2048bbf06153ea32f4b0271d07cdaeb` | 通用框架基线 |
| `demo/robotics` | `df5cac375132a3281c455a1497378d2147b9a271` | Robotics 领域接入参考 |
| `feat/553-math-tool-ablation` | `12d42e4d705517f34db681ac444760a37d31f905` | 数学工具、调用证据与消融参考 |
| `feat/556-math-benchmark-evaluation` | `8fbfad9274e521409ffa1dc1ef8285466202130f` | 数学基准评测与结果报告参考 |

当时两个数学分支各有独立提交，未合并为一个分支。当前保留范围与公开源码准备方式见
[仓库范围](REPOSITORY_SCOPE.md)。参考领域的测试不属于 Chips 验收；
迁移上游变更时按实际共享依赖选择检查，不自动启动其他领域评测。

公开贡献通过 `BucketSran/circuit-harness` 的 PR 向 `main` 集成，不向原 AlphaApollo 上游推送。
原始许可与来源声明随源码保留；旧研发历史继续私有。具体工作区和交付约定见[开发 SOP](DEVELOPMENT_SOP.md)。

## 实验室验收还需补齐

厂商电感样例已完成 JSON 重建、真实 SSH/EMX、日志下载及哈希校验，见 [服务器基线](BASELINE.md)。
以下条件面向专家定义的正式电路任务：

1. 真实任务的 JSON 输入及可信 GDS 转换器；当前通用 converter 已验证厂商样例的几何一致性。
2. 任务对应的 EMX wrapper、工艺文件、cell/端口/频率选项和预期结果文件。
3. 在真实 SSH/许可证环境执行该任务，保留版本、日志、产物校验与独立指标判定。
4. 在获得所需外部模型数据出口授权后，完成真实模型调用该工具的端到端 smoke。
5. 将部署前检查、任务/环境版本与结果判定规范化，让第二位成员在自己的机器独立复现。

独立电路结果判定属于首个真实任务验收。任务集、Tools 对照、Pi/Codex 对照及 Memory
按路线图逐步建设；多模型自动路由放在有稳定评测数据之后。
