# 公开电路任务说明模板

此文件中的任务目标和电路约束需要由任务所有者填写。它本身不是 benchmark。

在 shell 中执行 `harness-public info`，读取公开 task、剩余预算与可用工具 schema。
候选文件通过 `evas_read` 和 `evas_write` 读取、修改，通过 `evas_simulate` 获得公开诊断。
不要把容器中普通路径下的文件当作终评候选。

调用工具时，将 JSON 请求从 stdin 传给 `harness-public action`，例如：

```bash
printf '%s\n' '{"action_id":"read-001","tool":"evas_read","arguments":{"path":"dut.va"}}' \
  | harness-public action
```

每个新动作使用新的 `action_id`。重复查询同一动作时保留原 ID 和完全相同的请求内容。
如果客户端提示动作结果不确定，不要用新 ID 盲目重试。

完成后按 `info` 返回的 schema 调用 `evas_submit`，冻结最后一份完整候选。
公开工具不提供独立终评分数。
