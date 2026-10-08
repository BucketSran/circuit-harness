# Harbor runner 部署与预检

本机和服务器使用相同的 Harbor Job、CircuitAgent、公开会话环境与冻结候选终评入口。
runner 位置由操作者选择，Agent 和 Model 由现有 profile 编译器分别选择。
[本机模板](../../examples/chips/harbor/deployments/local.job.example.json)与
[服务器模板](../../examples/chips/harbor/deployments/server.job.example.json)中的路径是占位符，
不是可直接运行的部署。服务器可选择 Pi + GLM，不需要安装 Codex。

## 准备条件

在实际 runner 上准备 Python 3.12+、已安装的项目依赖和固定 `harbor==0.23.0`。
Harbor run 的 Agent 安装由 Harbor 管理。预检不会安装 Agent，也不会发送模型请求。
选择 Docker 时，准备本地 Unix socket daemon、Compose 和已有 Linux Agent 镜像；
选择 Podman 时，使用下面的私有 runner 入口。
镜像必须包含 `sh`、Python 3 和 root 用户。使用本地 `sha256:` 镜像 ID，
把[环境表](../../examples/chips/harbor/deployments/task.environment.example.toml)合入导出的 `task.toml`。
有界检查拒绝 Dockerfile、Compose 扩展和额外网路策略，避免构建、拉取或修改策略。

导出的公开 task、私有 public-session 配置、final-evaluation 配置和 Harbor jobs 目录分别放置。
私有配置及其资源必须在所有公开 task 与 jobs 导出目录之外。
public-session 的容器 EVAS checkout、Linux kernel 和 materials 是本机路径；
final-evaluation 的连接声明按[终评契约](HARBOR.md)配置。预检不会联系终评主机或评分。
模型连接留在[profile](../../examples/chips/harbor/profiles.example.json)中，凭据使用环境变量名引用，
运行时由操作者的凭据系统提供。勿把真实路径、密钥或主机配置提交进公开 task 或 Git。

本机 Docker Desktop 常用 `gateway_host=host.docker.internal`；Linux 服务器需要显式填写
容器可访问的 runner 私有 IP 或既有 DNS 名。`gateway_bind_host` 是 runner 监听地址。
仅允许获准容器访问该监听地址；预检不会自动改防火墙或网络。
服务器和本机必须分别取得自己的环境检查与实验验收证据。

## 编译与静态预检

以下命令均在选定 runner 的项目根目录运行。先把模板复制到自己的私有目录，
替换所有占位路径、网桥地址、任务身份和镜像 ID，再编译。预检不自动复制模板或覆盖输出。

```bash
umask 077
python -m circuit_harness.harbor.profiles \
  --catalog /operator/private/profiles.json \
  --agent pi --model glm --protocol openai-chat-completions \
  --job /operator/private/runner-job.json \
  --output /operator/private/harbor-job.generated.json
python -m circuit_harness.harbor.deployment \
  --job /operator/private/harbor-job.generated.json \
  > /operator/private/preflight-static.json
```

默认检查配置结构、字段、任务和私有配置位置，以及本地声明路径是否存在。
多任务使用[共享 task_bindings](HARBOR_TASKS.md)，同一次预检解析所有显式本地任务；
共享解析器读取私有配置、包 manifest 身份元数据与资源内容摘要，并核对编译时 pins。
多个任务不得使用 scalar session_config/config_path，也不支持在预检中解析远程 dataset。
拒绝重复 JSON key、未知 Harbor 字段、未知 adapter 选项、未知公共任务字段与冲突。
静态绑定身份检查可能使用只读 Git 元数据命令；不会启动容器环境、Agent、模型、
SSH、公开仿真或终评。公共 manifest 的电路语义和
源/kernel 身份仍需现有会话启动校验与实际任务验收。

## 显式容器网桥检查

```bash
python -m circuit_harness.harbor.deployment \
  --job /operator/private/harbor-job.generated.json \
  --check-environment --timeout-s 60 --cleanup-timeout-s 30 \
  > /operator/private/preflight-environment.json
```

该检查读取 Docker context 的本地 endpoint 元数据，只允许 Unix socket；
拒绝 SSH/TCP daemon 路由，以免 Docker CLI 隐式联系服务器。随后用有界
`docker image inspect` 确认 Agent 与公共 EVAS 镜像已在本机，
随后调用现有 `CircuitDockerEnvironment.start(force_build=False)`。该入口准备一次
公开会话和 gateway，并在 Agent 容器内执行现有 `harness-public info`。
它不会安装或启动 Agent，不调用动作工具或仿真，也不会创建终评器。
显式环境检查支持 Docker 和 Podman 公开后端；native 或 remote 公开后端标记
`not_checked/unsupported_public_backend`，不能据此声称网桥验证通过。

每个任务的镜像检查与环境启动各有独立的 `timeout-s` 预算，清理有独立的 `cleanup-timeout-s`。
Harbor 同步 daemon-info 检查最多额外 10 秒，每次取消清理先等待进程组 TERM 最多 1 秒，再等待 KILL 后退出最多 5 秒。
预检专用环境按固定 Harbor 0.23 构造 Compose 命令，仅增加独立进程组，
使容器 CLI 与其 Compose plugin 在取消时一同终止；独立 image inspect 也遵循同一规则。
多任务依次检查，各任务分别记录 environment/cleanup 与 task_id。
数值必须有限且在 0 到 300 秒之间。异常与期限结束后仍尝试 `stop(delete=True)`，
关闭本次 gateway、公开会话与容器。清理成功后删除本次 `jobs_dir/preflight-*`；
清理失败或超时会保留该目录供操作者恢复。预算依赖 Harbor 异步操作响应取消，
不能保证中断内核阻塞。报告不会输出异常里的私有内容。

## 使用 rootless Podman

Podman 与 Agent/Model 独立选择。把 Job 的 `environment.import_path` 改为
`circuit_harness.harbor.podman_environment:CircuitPodmanEnvironment`，
公开会话配置改为 `public_backend: "podman"`。已有 Docker 配置和默认值继续有效。
容器到宿主的 `gateway_host` 可使用实测可达的 `host.containers.internal`。

私有 [runtime 模板](../../examples/chips/harbor/deployments/podman.runtime.example.json)
声明 Podman 可执行文件、`root`、`runroot` 和 `control_dir`。这三个目录必须分离，
属于当前用户且权限为 0700；`control_dir` 每次运行必须是新目录，其路径需足够短以创建 Unix socket。
把存储放在适合 Podman 的本地文件系统上，避免直接使用 NFS home。
`ignore_chown_errors` 默认关闭；只有单 UID 映射的部署经过验收后才显式启用。
同一 store 的两个本入口进程不能并发运行。

镜像的 `USER` 必须能被用户命名空间映射。单 UID/GID 的服务器可准备仅增加
`USER 0:0` 的派生镜像；这里的 root 映射宿主普通用户。入口不会自动修改镜像用户。
镜像也需预装所选 Agent 的系统依赖。例如 Harbor 的 Pi 安装需要 `curl`；
单 UID 部署中的 apt 用户切换可能失败，应在准备镜像时安装这些依赖，再交给 Harbor 安装 Agent。
镜像、Linux kernel 和节点架构必须匹配。用所选引擎的 `image inspect` 获取本地 ID，
不要把 Docker Image ID 直接当作导入后的 Podman ID。离线转移另保存镜像包 SHA256
及 manifest/config 对应关系。

公开执行 `public_cpu_limit` 默认为 `1`，可设置 `(0, 1]` 的硬配额。
只有显式 `null` 才不要求 CPU 硬配额；会话与每次执行的 `backend.json` 保存这个条件。
离线报告记录实际配额，并拒绝将与计划配额不符的尝试计入评分统计；旧会话缺少该字段时按原默认值 `1` 读取。
Harbor Agent 容器的 CPU 配额由 `task.toml` 的 `environment.cpus` 和 Job 的
`environment.override_cpus` 独立声明。无 CPU controller 的部署需省略这两个 Agent 配额，
同时显式设置公开会话的 `public_cpu_limit: null`。需要硬配额时，先解决主机委派条件。
入口不会因 Podman 拒绝配额而删掉参数重试。公开执行仍禁网，限制为 512 MiB、32 PID、
只读根与输入、输出配额和动作期限。

```bash
python -m circuit_harness.harbor.podman_runner \
  --config /operator/private/podman-runtime.json -- \
  python -m circuit_harness.harbor.deployment \
  --job /operator/private/harbor-job.generated.json \
  --check-environment --timeout-s 120 --cleanup-timeout-s 45
```

入口启动一个临时用户 API 服务，把 `DOCKER_HOST` 指向私有 Unix socket，并为子命令提供
固定存储参数的 `podman` 包装器。Python 子进程启用 UTF-8，避免宿主 locale 改变任务文本解码。
入口不修改系统服务或全局 PATH，也不提供管理员权限。上例显式增加环境启动预算，
包括公开材料复制和 gateway 初始化；按实际部署记录所用预算与超时结果。
Podman 公开后端预检会先用真实公开容器限制运行 Python 就绪命令，再启动电路环境检查
`harness-public info`。前者不调用 EVAS，最多使用 10 秒执行预算，另有最多约 50 秒容器回收预算。
失败时保留 `jobs_dir/preflight-*/public-runtime` 证据，不能视为真实仿真验收。

运行完整 Job 时为 runtime 配置选择另一个新 `control_dir`，把上面 `--` 后的命令替换为
`harbor run --config /operator/private/harbor-job.generated.json`。Harbor 继续负责 Agent 安装、
执行、传输和 Trial；Harness 复用同一公开 gateway、冻结候选与独立 verifier。

退出及 SIGINT/SIGTERM 后，入口关闭子进程组，核查并回收本次标签的公开容器及登记的
Harbor Compose 项目，然后停止 API、删除 socket。Linux 上按精确项目名、store 路径和
当前 UID 匹配延迟退出的 conmon exec monitor，并通过 pidfd 回收。
它不清空其他项目、镜像缓存或整个 store。`control_dir/runtime.json` 保存退出码与清理结果，
`api.log` 保存服务输出；回收不完整时入口返回非零并保留材料。
SIGKILL、主机断电或内核阻塞仍需操作者根据保留证据恢复。

## 读取报告与运行

每项包含 `name`、`status`、`code`。状态只能是 `passed`、`failed`、`not_checked`。
`acceptance` 永远是 `not_evaluated`。退出 0 表示请求的检查未失败；退出 1 表示检查失败，
退出 2 表示 CLI 参数错误。检查结果不等于完整平台验收。

常见失败码包括 `configuration_invalid`、`missing_local_resource`、`dependency_unavailable`、
`existing_image_only`、`local_docker_required`、`local_image_unavailable`、`image_inspection_timeout`、
`local_podman_required`、`public_runtime_unavailable`、`environment_unavailable`、`environment_timeout`、`cleanup_failed` 和 `cleanup_timeout`。
`model_auth`、`commercial_license`、`final_evaluation` 均为 `not_checked/outside_preflight`。
无模型请求的预检不能证明模型鉴权、商业许可证或终评可用。

检查通过并获得实际实验授权后，沿用 Harbor 入口：

```bash
harbor run --config /operator/private/harbor-job.generated.json
```

Harbor 将结果保留在 JobConfig 的 `jobs_dir/job_name` 下。读取 job 的 `result.json`
与各 trial 的 `result.json`、`public-session`、`verifier` 记录，按任务身份与候选摘要关联证据。
基础设施失败看 `exception_info`；缺失评分不能当成零分。运行和结果契约见
[Harbor 适配说明](HARBOR.md)与[独立评测](BENCHMARK_EVALUATION.md)。

本次代码回归验证的是本地文件、外部 Docker 命令夹具和环境生命周期协议。
真实本机容器网桥、服务器 Pi + GLM、商业许可证及独立电路终评按实际部署单独验收；
旧 Pi + GLM 流程的成功记录不自动认证这条 Harbor 部署路径。
