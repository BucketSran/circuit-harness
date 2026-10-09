# 安装

使用 Python 3.12 或更新版本，在独立虚拟环境运行：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[harbor,data]'
circuit-harness --help
```

基础包提供操作者 CLI 和电路执行代码；Harbor 为可选依赖，完整实验平台固定使用 Harbor 0.23.0。

| extra | 用途 |
| --- | --- |
| `harbor` | Job/Trial、Agent、容器环境与 ATIF 模型 |
| `chips` | EMX 示例的 gdstk 转换器 |
| `data` | ATIF 导出的 Parquet I/O |
| `sft` | CPU/GPU PyTorch Dataset 与本地 tokenizer 验证，不含训练器 |
| `dev` | pytest、Ruff、Parquet 与测试使用的 MCP 客户端 |

开发安装 `.[dev,harbor,chips,sft]`，检查步骤见 [测试入口](tests/chips/README.md)。只安装所需 extras；直接 CLI 和服务器 zipapp 不加载 Harbor 或训练库。实验还需要对应的 EVAS checkout、镜像、仿真器、模型凭据与私有任务配置，见[开发者接入指南](docs/guides/integration.md)。全部文档见[文档导航](docs/README.md)。
