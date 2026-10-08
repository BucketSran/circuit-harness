# Chips 单次实验的调用链与开跑约定

本文用于**每次新实验开始前**确定运行方式、服务依赖和证据位置。它是操作约定，
不是新的可执行配置格式；实际参数仍使用各任务已有的 operator/profile 模板。
实验结论及原始证据按 [测试记录模板](../../tests/chips/RECORD_TEMPLATE.md) 保存，
私有目录和数据访问遵循 [部署规范](DEPLOYMENT.md)。本约定对应
[Chips 平台共建讨论 #1](https://github.com/BucketSran/circuit-harness-private/issues/1)。

## Harbor 实验入口

新任务使用 [Harbor](HARBOR.md) 的 Job、Trial 和 Agent 实现。Agent 与 Model 独立选择，
runner 所在主机另行配置；服务器 Pi/GLM 与本机 runner 使用相同的集成入口。
公开会话连接任务指定的 EVAS；冻结候选由私有 verifier 选择已声明的 Spectre 或开源 checker。
最终评分、隐藏测试和凭据不作为公开工具返回。

开跑前记录实际 Agent/Model、工具、预算、公开后端与终评后端；按本页后续约定分别保存证据。
原生 Codex 的可见工具限制仍适用。配置可用、受控协议测试和真实电路实验分别验收，
不将旧 Apollo 服务器同机实验等同于新 Harbor 部署已验收。

## 既有 Apollo 两条调用链

以下部署表、预算扩展和 operator 模板描述保留的 VABench/Analog 入口。
这些入口继续兼容，其专用凭据和预算参数不自动成为 Harbor 插件的通用设置。

```text
服务器同机：任务/配置 → lab-server 上的 Agent → 本机 MCP Tool → 服务器任务会话
                                                       ↔ 仿真器 → 公开反馈

本机控制：任务/配置 → 本机 Agent → 本机 MCP Tool → SSH → 同一类服务器任务会话
                                                      ↔ 仿真器 → 公开反馈

两条路径的后段：确认冻结候选 → 操作者侧独立终评 → 私有归档与校验
```

Agent 决定何时读题、修改候选和调用公开 Tool；服务器任务会话负责限制输入、
执行仿真、保存动作与结果。最终评分器不作为 Agent Tool，隐藏答案也不进入 Agent 工作区。
这里的 MCP 是随 Agent 回合启动的本机子进程接口；现有路径不要求在服务器开放常驻 MCP 端口。

| 运行条件 | 模型请求与凭据 | Tool 传输 | 断开本机 SSH 后 | 当前证据 |
| --- | --- | --- | --- | --- |
| lab-server Pi＋GLM | lab-server 上的 Pi；私有 Key 留在该账号 | `local` | `detach` 回合与已提交作业继续；服务器进程被杀不自动续接模型会话 | [VABench 单题实测](VABENCH_AGENT.md)；[Analog 单题自主仿真与终评实测](ANALOG_DESIGN_BENCH.md) |
| 本机 Pi＋GLM | 本机 Pi；Key 留在本机 | `ssh` | 已提交服务器动作继续；本机 Agent 的后续决策暂停 | [VABench 单题实测](VABENCH_AGENT.md)；Analog SSH 模式尚未完成同等级预检/真模型验收 |
| 本机 Pi＋GPT | 本机 Pi；私有 `openai-codex` OAuth 留在本机 | 同一四个公开 MCP Tool 经 SSH | 已提交服务器动作继续；本机 Agent 的后续决策暂停 | 入口与构造回归完成；Pi OAuth 尚未配置，真实模型回合待验收 |
| 本机原生 Codex＋OpenAI 模型 | 本机 Codex CLI；本机 ChatGPT 登录或 API Key | 同一公开 MCP Tool 经 SSH | 已提交服务器动作继续；本机 Agent 的后续决策暂停 | `v4-001` 单题真实模型＋SSH/EVAS/终评已通过；仍为 development 条件，见 [验证记录](VALIDATION.md) |

本机模式不把模型请求经 SSH 转发给 lab-server；服务器只收到任务允许的 Tool 请求。
如果服务器模型服务不可达，可另建 `run_id` 改用本机模式，不能在同一实验中静默切换条件。
两种 Agent 位置下，仿真、公开会话、冻结和终评仍在服务器。

现成配置入口按任务选择：VABench 使用[本机＋SSH 模板](../../examples/chips/benchmarks/vabench/agent.config.example.json)
或[服务器同机模板](../../examples/chips/benchmarks/vabench/agent.server-local.config.example.json)，
后者还可由[服务器私有 profile](../../examples/chips/benchmarks/vabench/server-deployment.profile.example.json)
准备新会话；Analog RLC 目前有[服务器同机模板](../../examples/chips/benchmarks/analog_design_bench/agent.server-local.config.example.json)。
这些 JSON 仍是任务运行入口，但其中的模型设置属于同一层实验条件，由
`chips_experiment_settings` 统一校验并写成相同的 `experiment_settings` 记录；
服务器会话、仿真器和传输路径仍由任务配置负责。新增的[单次实验入口](UNIFIED_EXPERIMENT.md)
用同一种实验 JSON 选择 VABench 或 Analog，登记共同的运行清单与结果；
它引用各自的 operator 配置，不把一份 VABench operator 直接拿给 Analog 使用。
本机原生 Codex 使用
[VABench Codex 模板](../../examples/chips/benchmarks/vabench/agent.codex.config.example.json)：
固定认证方式、Codex CLI 路径、模型 ID、推理档位和回合超时，预检本机认证及服务器公开会话。
Pi＋GPT 使用独立的 [Pi Codex 模板](../../examples/chips/benchmarks/vabench/agent.pi-codex.config.example.json)。
Agent／模型对照的固定条件与当前样本解释见[对照协议](VABENCH_COMPARISON.md)。

实验层的 `experiment_settings` 使用同一结构：`agent`、`model`、
`reasoning: {parameter, value}`、`budgets` 和 `schema_version`。
Pi 的 `parameter` 是 `thinking`，原生 Codex 是 `reasoning_effort`；
预算包括模型请求数、请求字节数、单次输出 token 数和回合超时。
原生 Codex 当前没有由此 Harness 强制的前三项上限，记录为 `null`，
不能把它们当成零或已执行的限制。真实模型回合必须显式选择推理档位，
GLM-5.3/5.3-Flash 只允许 `low`、`high`、`xhigh`。
这份结构用于预检与回合审计；原有 `model_settings` 输出及
operator/profile 字段保持兼容。比较新实验时读取 `experiment_settings`。

Pi 的 `max_output_tokens` 是**每次模型请求**的输出预算，可能包含供应商计入的推理 token，
与整个 Episode 的 `max_model_calls` 分开。两个任务入口共用 1–32768 的 Harness 配置范围，
默认仍为 4096；该范围不保证指定供应商／模型接受对应上限。实际支持情况由该条件预试确认。
对截断问题可在新 run ID 下只调整一个变量，例如 8192 与 16384，保持任务、工具版本、
模型、thinking、请求数和仿真数相同；保留全部失败与收取状态，不能把加大预算当作已解决。
不自动提高上限、重试付费请求或切换 thinking。单次 `length` 停止与请求次数耗尽分别报告。

Pi 启动时把模型预算写入 system prompt，每次请求前再注入剩余请求数；
`agent-input.json` 保存静态输入，`model-budget.jsonl` 的 `budget_context` 保存动态提示。
新建的 Analog／VABench 会话在 Tool 回复中提供剩余动作和仿真次数，
最后一个动作位置只允许显式提交。仿真额度耗尽后仍可提交已有候选。
旧会话保持原预算策略；这些限制不会自动增加，也不表示原生 Codex 已支持同样的请求计数。

`pi-outcome.json` 保留底层 `termination_reason`，另记 `harness_termination_reason`：
`model_request_limit`、`request_bytes_limit`、`output_token_limit`、`deadline`、
`runtime_error` 或正常 `final`。预算退出须同时有退出码和扩展停止记录；
不能仅凭退出码推断额度耗尽。截断或预算结束不会自动追加模型请求。
Analog 的 operator 收取使用这项明确原因，未知执行状态仍先恢复原动作。

## 开跑前固定什么

先在私有的 [测试记录](../../tests/chips/RECORD_TEMPLATE.md) 写明以下内容，再选择对应任务的配置模板：

1. **任务与结果**：任务 ID、源码/任务卡摘要、公开材料、允许修改的候选、独立终评规则、
   正负控和本次通过条件。个人探索任务没有既定评分器时，明确记为人工审核或 `not_evaluated`。
2. **Agent 与模型**：Pi 或原生 Codex、CLI/扩展版本、实际模型 ID、provider 与登录方式、
   显式选定的思考模式／推理档位、Agent 运行主机、提示版本、实际可见 Tool 集合。
   没有可见选择器不等于模型会自动选择档位；从本次 operator/profile 确认计划设置，
   运行后再用原始模型事件核对执行情况。
   原生 Codex 自带工具的边界须另行核对；
   不能仅凭同一模型名称就把 Pi/Codex 结果视为单因素对照。
3. **服务器执行**：SSH 别名或 `local`、固定 bundle/任务会话路径、仿真器及版本、
   Podman/Bubblewrap/实验室后端、工艺与许可证条件、scratch 和持久归档位置。
   任务决定后端，不让 Agent 在同一任务中自行更换评分环境。
4. **预算与停止**：模型请求/输出、公开仿真次数、每次作业与整个回合超时、
   资源及费用边界、失败/超时/网络中断后的停止与恢复规则。实际账单未知时不要标成零费用。
5. **轨迹与数据**：本机或服务器的 Agent 证据目录、服务器动作/仿真目录、归档位置、
   原始材料能否发往该模型服务，以及哪些结果只给操作者查看。

配置来源分开：版本化仓库保存任务、Tool 和无凭据模板；每位操作者的私有配置保存主机路径与
本次参数；密钥留在 Agent 所在机器的私有凭据位置。多厂商 Key 可放在同一 `0700` 的
`secrets/` 目录，但每厂商独立 `0600` 文件、每次只向选中的模型进程提供对应凭据。
服务器 GLM 的现有路径是 `~/chips-private/secrets/glm.key`；当前 Analog Pi 入口仍固定为
GLM 配方及 `CHIPS_MODEL_KEY`，多厂商选择器尚未实现。Codex 的 ChatGPT 登录缓存
属于另一种凭据，不应复制进服务器 Key 目录或实验归档。
Codex 配置用 `auth_kind=chatgpt` 时检查本机登录，不向其进程传 `OPENAI_API_KEY`；
用 `auth_kind=api_key` 时必须由本机环境提供该变量。两者都不向 MCP 子进程传模型 Key。

## Runtime 与任务会话的边界

[ChipsEnvironment](../../alphaapollo/common/environment/chips.py) 可将现有 vaBench／Analog
transport 接到 Apollo 的 `environment_factory`。同一个环境同时支持
`AlphaApolloAgentRuntime` 的模型调用和 `ExternalAgentRuntime` 的 MCP 调用；工具说明直接来自
原任务 session。任务 session 仍负责候选约束、仿真次数、动作去重及冻结，环境不运行终评。
只有 submit 的服务端确认才表示 `submitted`；普通最终文本不会产生提交。环境的中性
`reward=0` 配合 `evaluation=not_performed`，不表示电路评分为零，`success` 保持未知。

vaBench 和 Analog 的 Pi operator 配置可显式填写 `"runtime": "external"` 来使用共享 Runtime；
省略时仍为 `direct`，保留原 CLI 会话路径作为对照。两者使用同一 Pi 启动配方、模型设置和预算扩展。
GLM 的隔离配置可指定 `"http_retry_limit": 0`，同时关闭 Pi Agent 与 provider 自动重试；
省略时保留 CLI 默认值。对照实验必须记录该值，不能把逻辑请求数当作 HTTP 尝试数。
`test_pi_runtime.py` 的可选真实 Pi 检查用本机 HTTP 500 服务确认单次请求，不访问真实模型。
`pi-events.jsonl` 保留原始 CLI 事件，`pi-outcome.json` 保留已有投影；共享 Runtime 另存
`runtime-result.json`，包含模型／工具与 Environment 的关联。桥接令牌文件不进入归档。
这一接口已有本地契约及真实 Pi／本地 HTTP fixture 验证，尚不表示真实 Pi／GLM 对照已经验收。
原生 Codex 仍使用已有 SSH 路径，不能绕过共享 Runtime 对其原生工具可见性的限制。

## 预检、运行与证据

1. **静态预检**：在真实模型实验的私有配置中显式填写 `thinking`（Pi）或
   `reasoning_effort`（原生 Codex）；逐项核对模型 ID、档位、单次输出上限、模型请求数、
   仿真次数和回合超时。更换任一设置使用新的 run ID，不能与原回合合并统计。
   然后确认本次 `run_id` 未使用，任务/代码/镜像或执行器版本匹配，
   私有目录权限、空间、SSH 或同机路径、仿真器可用。只检查实际所选主机的模型 HTTPS 和
   凭据状态；无认证的 HEAD/端口探测不能证明模型调用成功。
2. **公开工具正负控**：在新会话中用固定候选直接调用 Tool，分别确认有效反馈和明确失败；
   这证明工具链，不算 Agent 成绩。商业仿真器还须单独确认许可证和获准数据。
3. **Agent 回合**：使用新的会话、job ID 和证据目录。保存模型事件、Tool 请求/响应及稳定动作 ID、
   候选摘要、仿真日志和时间；SSH 重连查询同一动作，状态不明时不启动第二次仿真。
4. **终评与归档**：只对确认冻结的候选启动独立终评；核对终评输入与冻结候选摘要。
   将成功、设计失败、基础设施失败、未提交分别记录。归档通过离线哈希校验后才考虑清理 scratch。

同机模式的 Agent 轨迹在服务器；本机模式的 Agent 轨迹先在本机、动作与仿真原始记录在服务器。
归档时应以 `run_id`、动作 ID、候选摘要和逐文件哈希关联两端证据，保留原始事件与规范化摘要。
现有 VABench `collect` 可以把服务器公开与终评包下载到操作者证据目录；Analog Episode 可选择
Agent 证据打包，但跨主机上传、统一索引和完整性自动验收尚未成为通用入口。
在实现前应把两端证据位置和校验结果明确记录，不能宣称自动形成了一份完整 Episode。
轨迹是私有证据，不自动作为下次 Agent 的 Memory；分享前检查提示、路径、日志和凭据泄露。

需要针对每种新组合依次证明：配置可用、公开工具可运行、真实模型能自主调用 Tool、
终评与两端轨迹可核对。已跑通一种位置、Agent 或任务，不替另一种组合提供验收结论。
