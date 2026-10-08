# 服务器同机部署与个人实验

服务器同机 Pi / MCP / EVAS 闭环已有 `v4-001` 真实模型和独立终评记录，见[验证记录](VALIDATION.md)。
本文使用占位路径说明部署步骤；实际账号、安装路径、恢复脚本和机器配置保存在操作者私有目录。
这个入口把**实验参数的唯一来源**放在服务器的个人私有目录：
[profile 模板](../../examples/chips/benchmarks/vabench/server-deployment.profile.example.json)。
准备命令从它生成每次运行的 `operator.json` 和公开会话；不读取模型 Key，也不调用模型。
详细 Tool、终评和计时契约见 [VABENCH_AGENT.md](VABENCH_AGENT.md)。

Chips 的配置／批次／部署／离线报告 CLI 使用标准库即可加载帮助和校验配置；
`alphaapollo.workflows` 的公共共享 API 在实际访问时再导入，避免为 Chips 操作者入口
提前加载 Bio 等领域依赖。真实 Pi/MCP、Apollo Runtime 和 HTML 渲染仍需要各自运行依赖，
本改动不替代固定 Python/Node 环境的准备及预检。
首次部署先选择明确的 Python 绝对路径，再用单次实验 `validate` 检查私有配置：

```bash
/absolute/runtime/bin/python -m alphaapollo.workflows.chips_experiment validate \
  --config /absolute/private/experiment.json
```

不要在 Analog operator 中放置 `archive_root` 或 `final_output`；它们属于实验 plan。
配置校验不会展开 benchmark 或创建 session，也不证明依赖、镜像、模型网络和许可证已就绪；
混合批次继续使用统一的 `prepare → preflight → run → reconcile → report` 入口。

| 位置 | 内容 | 生命周期 |
| --- | --- | --- |
| 本人持久私有目录（例如 `~/chips-private/config/`） | `deployment.profile.json`、任务 pin、工具版本和恢复说明 | 重启后保留；文件 0600，目录 0700 |
| 本人 scratch（例如受控的 `/tmp/<user>/`） | 展开的 Pi/Python/Apollo 运行时、公开会话、候选、每次运行的 operator/evidence | 可被清理；每次使用前检查 |
| 本人持久归档目录 | 原评分器包、公开轨迹包与收据 | 归档校验后保留；不能交给模型读取 |
| 模型凭据 | 交互式输入并仅注入 Pi 进程环境 | 不放进 profile、命令参数、Git 或归档 |

profile 由当前 Linux 用户持有且不可被其他普通用户读取。`run_root`、`job_root` 和
`archive_root` 必须是该用户已有的 0700 目录；scratch 和归档不能互相包含。
`bundle`、`python` 和 `pi_cli` 可以指向未来由管理员维护的只读共享安装，
但候选、运行目录、归档和 Key 始终按用户分开。部署前由操作者确认服务器账号是否独用；
共用 UID 的程序不能靠普通文件权限相互隔离。

在目标服务器上放置本分支源码或安装包后，把模板复制到自己的持久私有目录、填写真实绝对路径，
确认所用 Python 能导入 `alphaapollo`，再准备一个新的 run ID：

```bash
umask 077
/private/pi-venv/bin/python -m alphaapollo.workflows.chips_vabench_deployment prepare \
  --profile "$HOME/chips-private/config/deployment.profile.json" \
  --run-id trial-001
```

真实模型实验须在 profile 显式填写 `thinking`，不继承 Pi 或模型服务的默认值；
GLM-5.3/5.3-Flash 可选 `low`、`high`、`xhigh`，不能使用 `off`。已有私人 profile
也须补上该字段后才能准备新实验。程序验证 profile、路径、pin 任务身份及公开预检；
成功时输出生成的 `config`、`evidence` 路径和 `model_settings`，启动前核对模型、
思考档位、单次输出及总请求预算。重复 run ID 会拒绝覆盖。失败时保留新 run 目录供检查，不伪装成已准备成功。
`deployment.json` 记录 profile、pin、bundle 的 SHA-256；它是追溯信息，不代表
整个 Python/Node 环境或外部模型服务也被哈希锁定。

随后在**服务器交互终端**读取 Key，再使用输出的两个绝对路径启动：

```bash
read -r -s -p 'Model key: ' CHIPS_MODEL_KEY
printf '\n'
export CHIPS_MODEL_KEY
/private/pi-venv/bin/python -m alphaapollo.workflows.chips_vabench_agent detach \
  --config /private/scratch/experiments/trial-001/operator.json \
  --evidence /private/scratch/experiments/trial-001/evidence
unset CHIPS_MODEL_KEY
```

`detach` 后本机断网或 SSH 退出不影响服务器上的 Agent 和已提交作业。
可读 `evidence/report.json`、模型/工具计时和归档收据；服务器重启、进程 OOM、
scratch 清理**不会**自动续跑完整模型会话。不要把自动恢复同 ID 的公开动作当成模型 checkpoint。
应在准备下一次实验前检查 scratch 中的固定源码、Pi、Python 环境和 pin 指向的路径。
如需恢复 scratch，在持久私有目录保存获准备份的运行时、benchmark 源码、SHA-256 清单和恢复说明。
恢复前先校验包，检查目标路径不会覆盖现有工作，再用新 run ID 做公开预检；
恢复不能续接旧模型会话。实际恢复需在各自部署环境演练，不能从文件校验通过推断恢复成功。

## Spectre RC 环境

无 PDK `RC-001` 的 Spectre 路径使用另一份同机 operator profile：固定 csh 环境脚本、
Spectre 启动路径、私有 scratch 作业根和持久归档根。它与 Pi/VABench profile 相互独立，
不用模型 Key，也不执行 Virtuoso GUI。配置模板、提交/验证命令及错误语义见
[Spectre RC 标准化说明](SPECTRE_RC.md)。个人配置留在服务器 0600 文件中，
原始 PSF、仿真日志和归档留在本人 0700 目录中。

## 将来提供给其他服务器用户

个人配置、scratch 和归档目录保持 0700，**不作为共享安装目录**。
如需多人使用，请管理员先确定允许共享的安装路径、软件许可证、配额、清理策略和组/ACL；
再将审过的 Apollo 代码、Pi/Python 运行时和无受限内容的模板放在只读共享位置。
每位用户用独立 Linux UID，在自己的持久目录保存 profile/pin，在自己的 scratch 创建 run，
独立提供模型凭据与归档路径。共享安装不能包含他人的 Key、实验轨迹、隐藏评分材料、
受限 PDK 或实验室许可证文件。不要通过放宽当前个人目录权限来实现共享。

多人验收还需至少验证：各用户对共享运行时仅有读/执行权限；不能读取或修改彼此的
profile、候选、归档和进程环境；并发 run ID、磁盘/CPU/许可证限制与清理策略互不干扰。
当前只验证了独用账号上的同机闭环，尚未完成跨账号隔离或管理员级共享部署。
