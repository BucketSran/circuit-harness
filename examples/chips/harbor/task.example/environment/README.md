此目录只保留 Harbor 公开任务的环境材料。示例 `task.toml` 使用一张预构建镜像，
需要替换占位摘要，并确认镜像为 Linux、包含 Python 3，满足所选 stock Agent 的安装条件。

私有会话和终评配置放在 task 与 Trial 导出目录之外，不能复制进镜像或此目录。
此入口不接受自定义 compose 或用户挂载。
