# 任务构建：来源与操作者确认

这是 Chips 任务构建的初始入口，关联平台共建讨论 #1。当前支持检查结构化草案及记录确认；
不表示已实现通用 PDF/图片理解、任意 testbench 执行或正式 benchmark 发布。

草案 JSON 包含 `schema_version: 1`、`task_id`、`sources` 和 `fields`。
`sources` 保存相对材料根目录的 `path` 和文件 `sha256`；不允许链接、路径穿越或重复来源。
每个字段包含 `value`、`unit`、`status`、`evidence`：

- `unit` 使用操作者要求的规范单位；无物理单位的字段使用 `null`。
- `status` 为 `explicit`、`inferred`、`missing` 或 `conflict`。
- 每项证据包含 `source`、从 1 开始的 `page`、描述图/表/文字区域的 `region`。
- 推断值必须有 `note`，交给操作者判断；缺失/冲突字段不能被直接确认。

操作者另行提供 `requirements.json`，例如 `{"supply": "V", "frequency": "Hz"}`。
这些必需字段与单位由任务定义，不允许 Agent 通过删除要求来消除待确认问题。
系统检查引用结构及材料哈希；引用的语义、页码和电路条件仍需人工/独立检查。

```bash
python -m alphaapollo.workflows.chips_task_authoring inspect \
  --draft /private/draft.json --materials /private/materials \
  --requirements /private/requirements.json

python -m alphaapollo.workflows.chips_task_authoring confirm \
  --draft /private/draft.json --materials /private/materials \
  --requirements /private/requirements.json \
  --reviewer reviewer-name --kind human --output /private/confirmation.json

python -m alphaapollo.workflows.chips_task_authoring verify \
  --draft /private/draft.json --materials /private/materials \
  --requirements /private/requirements.json --confirmation /private/confirmation.json
```

确认文件绑定整个草案、来源摘要及操作者要求，使用 0600 创建且拒绝覆盖。
修改后必须重新确认；模拟实验使用 `--kind scripted_confirmation`，不能冒充真实专家审核。
确认是操作者侧协议，不是密码学签名或用户身份认证，不能抵抗同账号恶意重写草案与收据。
该命令不暴露为 Agent 可调用的自批准工具。后续执行器须显式调用 `require_confirmation` 才能落实执行门槛。

`inspect` 在待澄清时返回 1；`verify` 在材料变化、确认缺失或失效时返回 1。
当前通过只表示确认契约有效，不表示物理条件正确或电路达标。

实现：[草案与确认](../../../alphaapollo/common/execution/chips/task_authoring.py)、
[操作者 CLI](../../../alphaapollo/workflows/chips_task_authoring.py)。
本地检查：`python -m pytest -q tests/chips/test_task_authoring.py`，不调用模型或服务器。

## 有界 Spectre 构建试验

`opamp_gain/reference.va` 是自行构造的线性单极点放大器；`manual.scs` 是操作者手写基线。
它们不使用 PDK，不代表晶体管级设计或流片验证。当前候选只描述五端口顺序、两个 AC
激励及测量分子/分母节点；固定框架由 `spectre_testbench.prepare_testbench` 渲染。
这是先验证数据与执行契约的受限试验，不是任意网表生成能力的验证。

`python -m alphaapollo.workflows.chips submit-spectre-gain --input REQUEST.json
--profile PRIVATE_PROFILE.json --job-id NEW_ID` 接受操作者准备的请求：
`draft`、`confirmation`、`materials` 目录、`candidate`、`reference`（gain0/pole_hz）及
`model` 文件路径。候选不能设置环境脚本、命令或模型代码。Profile 沿用 Spectre RC
字段，但 `task_id` 为 `OPAMP-GAIN-001`；需由操作者选择私有工作/归档目录和已授权的环境。
这些机器配置不应提交到 Git。

执行复用现有 detached job、超时、取消和归档机制。`job-status`、`verify-job` 也适用；
完整归档包含材料快照、确认凭据、候选网表、模型、原始 PSF、日志、阶段耗时及评分。
后台进程可以独立于提交 SSH 会话工作，但不是支持服务器重启恢复的调度器。
现有输出限额检查顶层文件，不能称为递归磁盘或内存硬隔离。

评分分别回答两件事：测量是否与模型的解析增益/相位一致；在测量有效时，DUT 是否达标。
正确识别不达标的 DUT 仍是有效测试台；零激励、缺失或非有限 PSF 不会得到成功分数。
离线校验重读原始复数 PSF、重新计算指标，并核对快照和哈希。这是可审计协议，
不防同一账号的恶意篡改，不等价于签名或可信执行环境。

服务器公开会话由操作者通过 `gain-session --input REQUEST.json --profile PROFILE.json
--session NEW_PRIVATE_DIR --max-simulations 3` 创建。Agent 只能请求 `gain_simulate` 和
`gain_submit`；确认与独立终评不在其工具列表中。服务器锁定请求 ID、保存仿真计数，
同一 ID 只能对应一个请求。通信状态不明时必须恢复原请求，不能另起仿真；提交后禁止继续修改。
创建会话会固定部署/源文件哈希，不会在 Agent 运行中自动接受修改后的材料。

## 本机 Codex 两阶段预试

`d0-notes.md` 与 `d0-interface.png` 是构造材料。数值在文字中，端口顺序只在图片中。
本机使用已有 Apollo `CodexSession`，无需改 Pi、Codex 或模型 SDK。私有配置至少包含：

```json
{
  "task_id": "gain-d0",
  "materials": "/absolute/materials",
  "notes": "d0-notes.md",
  "image": "d0-interface.png",
  "codex_cli": "/absolute/codex",
  "model": "gpt-5.6-luna",
  "reasoning_effort": "medium"
}
```

1. `python -m alphaapollo.workflows.chips_task_authoring_agent extract --config CONFIG.json
   --evidence NEW_EXTRACT_DIR`：图片通过 Codex `--image` 传入，输出有来源的草案，不自动确认。
2. 操作者核对字段、来源和单位，再使用上面的 `confirm` 入口。真实材料应由研究者确认；
   构造夹具自动核对只能标记 `scripted_confirmation`，不能称为专家确认。
3. 用该草案/确认在服务器创建 `gain-session`。在本机配置增加 `draft`、`confirmation`
   文件路径和 `remote`（`host` SSH 别名、`python`、固定 `bundle`、`session` 绝对路径）。
4. `python -m alphaapollo.workflows.chips_task_authoring_agent build --config CONFIG.json
   --evidence NEW_BUILD_DIR`：MCP 经 SSH 访问服务器会话；真实 Spectre 始终在服务器运行。
5. 收到成功提交后，操作者下载服务器 `submission.json`，核对候选哈希，再用相同候选和
   不同的操作者参考模型参数做独立终评。不得用公开反馈代替独立终评。

两个模型阶段分别限制 360/840 秒，不自动重跑。当前只提供开发条件的工程预试，
尚未提供 A/B/C 比较矩阵或任意文档解析器。原生 Codex 工具仍可用；“仅用两工具”是任务指令，
不能当作强隔离。`system` 文本由现有适配器拼入用户输入，不是供应商独立 system 消息。

本机证据包含完整输入、实际 argv、图片路径与哈希、原始 Codex JSONL、可见事件/usage、
阶段时间和每次 MCP 请求/响应。服务器保存仿真动作、冻结候选和作业归档。凭据使用已有
本机 Codex 认证，不写入任务配置；不要把含真实主机路径、材料或轨迹的输出提交到 Git。
