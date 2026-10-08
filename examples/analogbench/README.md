# AnalogBench 入门：先 RLC，再 SKY130

这两个例子使用 Analog Design Bench 原版题目和 checker。
A 是无 PDK 的 100 MHz 被动 RLC 带通滤波器；B 是使用开放 SKY130 模型的五管 OTA。
Harbor 负责 Agent、Job、Trial、Docker 工作区和终评；Harness 只校验固定题目源码并编译 Agent/Model 配置。
A/B 沿用 stock Harbor 生命周期，并用轻量 verifier 核对原 checker 的执行证据。
B 的历史 task 另需补齐原 checker 期望的上传路径，
并在评分阶段启用 Harbor 禁网。两者都不要求 EVAS 网关或 Spectre 许可证。

这条原生任务路径不同于 Harness 的私有冻结候选会话。A 使用上游独立 verifier 容器；
B 使用上游原有 shared verifier，Agent 结束后才上传 tests，并在同一容器中禁网评分。
它不提供独立私有 verifier、不可变候选提交或防恶意 Agent 篡改的安全级别。
需要这些隔离时使用 [Harbor 私有会话集成](../../docs/chips/HARBOR.md)，不要把本例当作同等级保证。

## 准备软件与原题

需要 Python 3.12+、Docker daemon、curl 和 tar。在仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[harbor]'
export ANALOG_CACHE="$PWD/.planning/chips/analog-cache"
bash examples/analogbench/fetch-sources.sh "$ANALOG_CACHE"
export RLC_SOURCE="$ANALOG_CACHE/analog-design-bench-fb0ec30463d005d3e463caf4e48ab9a26008e869"
export SKY130_SOURCE="$ANALOG_CACHE/analog-design-bench-c23f124de1e461655d2e02ce6cfae2654ccea0d3"
export ANALOG_JOBS="$PWD/.planning/chips/analog-jobs"
```

缓存包含参考解与私有评分脚本，留在操作者侧。不要把缓存加入 Git，或挂入 Agent 工作区。
原 task 的 environment Dockerfile 只复制公开 starter；Harbor 在评分阶段提供 `tests/`。
Oracle 控制需要 `solution/`，真实 Agent 不能获得它。
`prepare` 对整个原 task 目录校验 SHA-256，修改任何文件都会拒绝准备。
运行期间也应保持外部缓存只读，不要修改已经准备的原题。

三题登记与直接公开会话的既有证据见 [TASKS.md](TASKS.md)。本指南只运行 A/B 原生 Harbor 路径，
固定 task tree 身份如下：

| 示例 | task_id | 源码提交 | task tree SHA-256 |
| --- | --- | --- | --- |
| A | `rlc-rf-bandpass-100mhz` | `fb0ec30463d005d3e463caf4e48ab9a26008e869` | `5ff88ba55bd7108fba3f37a4d257dd47eb04ee5c0771102229c2704c0f728310` |
| B | `sky130-ota-5t-gain40-pm60-noise50uv-pvt` | `c23f124de1e461655d2e02ce6cfae2654ccea0d3` | `55e6145bf287bf0dd8ab18fe969ae85ebc5f6d3bee0d361dd5df0092ccf3a00f` |

预先拉取固定基础镜像：

```bash
docker pull ghcr.io/arcadia-1/circuit-bench-sky130-ngspice@sha256:bd5c425675eb99fc1a2c3bca10b63a871c457613767e2c6984d6c207b3160500
docker pull ghcr.io/arcadia-1/analog-arena-sky130-tools@sha256:668783dd8cbcd5ebce5264785803ce73a9466e79dbe4c98cdf4566348a33437a
```

A 的基础镜像名称含 SKY130，但这道 RLC 题只使用理想被动元件与 ngspice，不加载 PDK。
B 的镜像已经包含 `/opt/sky130/continuous/sky130.lib.spice` 和上游电路合法性检查器，
不要求另外购买或安装商业 PDK。首次 Harbor 启动还可能构建任务和网络控制镜像。

## A：验证无 PDK 路径

先运行无需模型和 API key 的 Oracle。它把上游参考解送给上游 checker，检查安装和评分路径。

```bash
python -m circuit_harness.benchmarks.analogbench \
  --task rlc-rf-bandpass-100mhz --source-root "$RLC_SOURCE" \
  --output "$ANALOG_CACHE/a-oracle" --jobs-dir "$ANALOG_JOBS" --oracle --cpus 2
harbor run --config "$ANALOG_CACHE/a-oracle/job.json"
```

`--cpus 2` 适用于只分配两个 CPU 的 Docker Desktop。原题声明四个 CPU；
宿主资源充足时可去掉该参数，保持上游默认值。每次重跑使用新的 `--output` 名字。
配置默认只有一次 trial、一次 attempt，失败不自动重试。
`job.json` 是 Harbor 原生配置，`provenance.json` 记录任务提交、摘要、固定基础镜像和实际生成的覆盖项。
准备器不改变 task 文件或镜像，不覆盖上游 timeout。CPU 只在显式指定 `--cpus` 时覆盖。
Agent profile 自己声明的超时仍会由配置编译器传递。

A 使用 `OriginalAnalogVerifier` 的证据检查，继续在原题的独立 verifier 容器中运行。
它不向 Agent 工作区上传评分文件，也不重设上游评分网络。
原脚本会在 checker 崩溃而没有结果时用 trap 写出 `0/15`，所以非空候选还必须有原 checker
的十五项 CTRF 报告。没有报告是基础设施错误，原 starter 的完整 `0/15` 报告仍是有效零分。
空候选沿用原脚本的失败评分规则。

参考解应得到 `reward=1.0`、`tests_passed=15`、`tests_total=15`。
这证明本次参考解通过该固定原题的 checker。它不证明 Agent 已解题。

## B：验证开放 SKY130 路径

A 完成后运行 B，使用相同准备入口和 Harbor 生命周期：

```bash
python -m circuit_harness.benchmarks.analogbench \
  --task sky130-ota-5t-gain40-pm60-noise50uv-pvt --source-root "$SKY130_SOURCE" \
  --output "$ANALOG_CACHE/b-oracle" --jobs-dir "$ANALOG_JOBS" --oracle --cpus 2
harbor run --config "$ANALOG_CACHE/b-oracle/job.json"
```

历史 B 的 `test.sh` 使用 `/app/analog_arena_tests/verify.py`，而 stock Harbor 上传到 `/tests`。
准备器为 B 指定 `OriginalAnalogVerifier` 与 `AnalogDockerEnvironment`，在评分阶段补齐路径，
检查 ngspice、SKY130 模型、合法性 checker 与 `verify.py`，然后执行原 `test.sh`。
合法候选必须产生原 checker 的七项 CTRF 报告；缺报告视为基础设施错误。
starter 只有空 subcircuit，原合法性 checker 在仿真前拒绝它，仍按原规则记为有效的 `0/7`。
这里的七是评分总项数，不表示已经执行七项电路测量并全部失败。
评分目录的 `network-policy.json` 记录 Harbor 已执行禁网切换。

参考解应得到 `reward=1.0`、`tests_passed=7`、`tests_total=7`。
模型、工艺角和指标由原题定义，Harness 不改题、不替换 checker、不放宽评分。

## 选择 Agent 与 Model

把 [现有独立 profiles 示例](../chips/harbor/profiles.example.json) 复制到私有目录，
将所选 model 的 `model`、`base_url` 和 `key_env` 改为实际服务配置。
示例中的 `.invalid` 地址不能连接服务，`*-fixture` 也不是真实模型。
Agent 与 Model 各选一个别名，不要求为每个组合复制任务或 runner。
例如选择 Pi 与支持 chat-completions 的 GLM 服务：

```bash
python -m circuit_harness.benchmarks.analogbench \
  --task rlc-rf-bandpass-100mhz --source-root "$RLC_SOURCE" \
  --output "$ANALOG_CACHE/a-pi-glm" --jobs-dir "$ANALOG_JOBS" \
  --catalog /absolute/private/profiles.json --agent pi --model glm \
  --protocol openai-chat-completions --cpus 2
harbor run --config "$ANALOG_CACHE/a-pi-glm/job.json"
```

运行前，在运行 Harbor 的同一个 shell 设置 catalog `key_env` 指定的变量。
准备阶段不会读 key，生成文件保留 `${VARIABLE_NAME}` 引用。
调用 `harbor run` 会安装并启动所选 Agent，可能产生模型费用。
需要预算上限时，在 Agent profile 中设置 Harbor 支持的超时和该 Agent 的预算选项。
B 只需替换 `--task`、`--source-root` 和输出名，配置编译方法相同。

A 原题 Agent 和独立 verifier 的基线为 `no-network`。
准备器把所选模型 hostname 加入 `AgentConfig.extra_allowed_hosts`，只开放 Agent 阶段的模型连接，
不放入 `environment.extra_allowed_hosts`，所以不会向评分容器开放同一域名。
B 历史 task 的 Agent 网络默认为 `public`，本例保留这一行为，评分阶段显式切换为 `no-network`。
Agent 及其调用的 ngspice 处于同一个工作区，解题过程没有单独的仿真禁网边界。
安装阶段的网络由 Harbor 管理。模型服务若跳转到其他域名，在私有 JobConfig 的
`agents[0].extra_allowed_hosts` 中显式列入所需域名，并核查实际权限。
协议兼容性由现有编译器检查；不兼容或多协议歧义会在准备阶段报错。

## 看结果与定位失败

```bash
harbor view "$ANALOG_JOBS"
```

查看具体 trial 的 `result.json`、`exception.txt` 和 `verifier/` 日志。
Oracle 的通过结果、starter 负控、真实模型自主解题是三种不同证据。
有异常而没有有效 reward 的 trial 是基础设施失败，不能当作电路零分。
Docker 报 CPU 范围错误时核对 daemon 的 CPU 数量；镜像存在但报 content digest 缺失时，
重新拉取相同 digest 并确认容器能启动，不要替换为浮动 tag。
保存原始轨迹、候选和日志到私有目录，公开报告只引用安全的版本与结论。

## 来源和许可边界

源码来自 [Analog Design Bench](https://github.com/Arcadia-1/analog-design-bench)。
这两个历史提交的根目录没有 `LICENSE`。
[当前上游 LICENSE](https://github.com/Arcadia-1/analog-design-bench/blob/main/LICENSE)
分别声明软件使用 Apache-2.0，benchmark 内容使用 CC BY-NC 4.0。
不能把当前声明当作历史版本中已存在的许可文件，也不能推断本仓库许可证覆盖上游题目、
参考解、checker、镜像或 PDK。使用或分发前按上游及 SKY130 各自声明核对范围。
本仓库只发布接入代码与固定身份，不 vendoring 第三方 benchmark 内容。

## 本次验证范围

2026-10-09 在本机 Docker 上使用上述固定源码与镜像进行验证，Docker 分配两个 CPU。
A 的 Harbor Oracle 为 15/15，Harbor Nop 保留 starter 为 0/15；
原 checker 直接执行的参考解为 15/15，starter 为 0/15。
B 的 Harbor Oracle 为 7/7，原 checker 直接执行的参考解为 7/7，starter 为 0/7。
B 的 Harbor Nop Agent 保持空 subcircuit starter，仿真前合法性检查拒绝，记为 0/7；未执行七项电路测量。
这些都是本次控制运行，不引用历史模型成绩。当前没有运行付费模型，也没有验收真实 Agent 自主解题。
最初 B 原生 Harbor 运行因为 checker 路径缺失，由上游 shell trap 写出零分；
本例适配修复该路径并拒绝合法候选缺少 checker 报告的情况。
准备边界回归涵盖源码漂移、stock job、密钥引用、缺 checker 和缺执行报告。A 的回归经生成配置与真实 Harbor Verifier 工厂，
模拟 checker exit=2 但只有 trap 零分的独立 verifier 环境，确认返回基础设施错误。
外部原题回归需设置 `ANALOG_EXAMPLE_SOURCES`，默认不下载第三方数据：

```bash
export ANALOG_EXAMPLE_SOURCES="$RLC_SOURCE:$SKY130_SOURCE"
python -m pytest -q tests/test_analog_example.py
```
