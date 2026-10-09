# Harbor 多任务绑定

Harness 复用 Harbor 0.23 的任务与 Trial 调度。操作者为每个实际本地任务目录声明公开会话和私有终评配置，两种配置都留在私有 runner 目录。单题继续使用 `session_config` 与 `config_path`；多题使用同一份 `task_bindings`，两种方式互斥。

私有 manifest 的格式见 [示例](../../examples/chips/harbor/task-bindings.example.json) 和 [schema](../../circuit_harness/harbor/task_bindings.schema.json)。`task_path` 必须是绝对路径。选择按解析后的完整目录路径进行，两个不同目录可以有相同 basename。重复路径或重复 `task_id/task_version`、缺失计划项、未计划项与身份不一致都会拒绝。`task_id/task_version` 必须与公开会话和终评 package manifest 一致。

在 Job 的 `environment.kwargs.task_bindings` 与 `verifier.kwargs.task_bindings` 指定同一份 manifest。保持 [Job 示例](../../examples/chips/harbor/job.example.json) 的环境与 verifier import_path、gateway 设置，在 `tasks` 列出实际本地任务，并移除两个 scalar 配置参数。用既有 profiles `compile_job` 或 CLI 编译 Job。编译检查不启动模型、容器或仿真，但会读取并哈希本地任务、配置和资源文件，允许调用 Git 读取源码身份。新映射模式只支持显式本地任务；远程 Git、dataset 与动态下载会在编译前拒绝。Harbor 仍拥有 agents、attempts 与并发控制。

编译结果把 manifest 摘要和每项输入收据写入环境及 verifier kwargs。Trial 按实际任务路径选择并重新检查字节身份，拒绝编译后任务源码、公开材料、kernel、配置或终评包变化。EVAS checkout 的身份直接使用公开会话既有 `snapshot_repository` 和 `EVAS_SOURCE` 取得的源码范围及文件摘要，不包括 Git 元数据、范围外运行产物和被 Git 忽略的缓存。checkout 必须是含 HEAD 的 Git 根目录。公开 Codex/Python 链接按实际解析目标的路径身份与文件字节固定；kernel 和任务材料继续遵守既有普通文件约束。每项 Trial 的 `task-binding.json` 只含任务身份和摘要，不包含配置路径、checker 内容或凭据。环境与 verifier 的收据必须一致。该机制检测启动边界的漂移；操作者仍须冻结私有材料并禁止运行期间并发修改。

Manifest、配置以及其声明的本地资源均须在所有 task 和 Trial/export 根目录外。环境不会上传整份 manifest 或私有配置目录。只允许原有 Harbor agent 日志和公开 artifacts mount；独立 verifier 日志不会挂载给 Agent。

`support="unsupported"` 的条目必须包含 `reason`，且不能提供可执行配置。该条目保留在计划中，编译会报告任务及原因并阻止启动整个 Job，不能静默删去它或改变分母。先由 benchmark 作者明确支持范围和绑定材料，再提交可运行计划。Harness 不推断 EVAS 支持语义，不修改题目或 checker。

`tests/chips/test_harbor_task_bindings.py` 通过两项不同任务的真实 Harbor Trial 并发执行检查路由、候选与私有 checker 隔离，并覆盖错误绑定和字节漂移。Agent、环境执行和 checker 是受控本地夹具，不调用模型、Docker、SSH 或商业仿真器。实际 vaEVAS 任务目录的只读接入审计与真实电路执行分别记录；VA07 既有结果只能复用于匹配的源码、版本和执行条件，不能推广到全部七题。
