# EVAS benchmark 保存候选入门

这个例子连接 [vaEVAS 公开仓库](https://github.com/BucketSran/vaEVAS)的
[VA07 积分三角波修复题](https://github.com/BucketSran/vaEVAS/tree/f8b624f8887f5d55ce726373cff2a856c1487b79/benchmark/tasks/va07-triangle-repair)。
固定源码 commit 为 `f8b624f8887f5d55ce726373cff2a856c1487b79`。
任务是开发原型，原 benchmark 集合标签为 `extension`，不是正式公开主集。

任务要求模型在上下限间连续积分、按方向换向并正确计数。独立 checker 来自 benchmark，
Harness 负责冻结候选、隔离执行和证据。原八例、两档源设置、1 µV 波形误差和 200 ns
换向误差保留。EVAS 的固定映射为 `vabstol=1e-8`、`reltol=0`，
`iabstol` 和 `traponly` 没有对应设置。两个预定观察网格来自独立解析根。
这些设置不表示两种求解器的误差保证相同。

本入口重新验收一份保存的提交，不调用模型。默认候选复制 benchmark 参考解，
用于验证接入路径。修改 `candidate/dut.va` 后可以验收自己的提交。
模型生成和完整 Harbor Agent Trial 是后续工作；本例没有生成可直接开始 Agent 的 Harbor 配置。

## 准备

前置条件为 Python 3.12+、安装本仓库 `pip install -e .`、Docker 和磁盘空间。
Linux kernel 必须在 Linux 中构建，macOS Cargo 二进制不能直接进入 Docker。
所有工作目录和输出目录必须是新目录，建议放在本地忽略的 `.planning/chips/`。

```sh
git clone https://github.com/BucketSran/vaEVAS.git .planning/chips/vaEVAS
python -m circuit_harness.benchmarks.evas_va07 prepare \
  --evas-checkout .planning/chips/vaEVAS \
  --workspace .planning/chips/evas-quickstart
```

`prepare` 从 Git object 导出固定 commit 的原始字节，忽略 checkout 的当前分支和本地修改。
它调用 benchmark 原 `build_triangle_evas_replay.py`，输出原八例 checker 包、
题面、参考候选与逐文件来源摘要。

构建 EVAS runtime 需要在线下载 Rust/Cargo 依赖。先取得实际 base RepoDigest，再构建：

```sh
docker pull rust:1.90-bookworm
docker pull python:3.12.12-slim-bookworm
EVAS_RUST_BASE=$(docker image inspect rust:1.90-bookworm --format '{{index .RepoDigests 0}}')
EVAS_PYTHON_BASE=$(docker image inspect python:3.12.12-slim-bookworm --format '{{index .RepoDigests 0}}')
docker build \
  --build-arg RUST_BASE="$EVAS_RUST_BASE" \
  --build-arg PYTHON_BASE="$EVAS_PYTHON_BASE" \
  -f examples/evas-va07/Dockerfile \
  -t evas-quickstart-runtime .planning/chips/evas-quickstart
EVAS_IMAGE=$(docker image inspect evas-quickstart-runtime --format '{{.Id}}')
```

保存构建输出及这两个实际 base digest。Dockerfile 在公开源码上执行 `cargo build --locked --release`，
写入 commit、EVAS Python 文件和 kernel 摘要。镜像是本地构建产物，没有预发布镜像下载承诺。
运行前 Harness 核对实际 Python/kernel 字节与构建记录及准备源码。
内核摘要标识产物，不能单凭 commit 字符串证明构建链可信。

## 运行与结果

```sh
python -m circuit_harness.benchmarks.evas_va07 run \
  --workspace .planning/chips/evas-quickstart --image "$EVAS_IMAGE" \
  --output .planning/chips/evas-reference-result
python -m circuit_harness.benchmarks.evas_va07 result \
  --output .planning/chips/evas-reference-result
```

运行以本地不可变 image ID 固定镜像，容器无网络。每例运行原网格和补充网格，
外层总预算 300 秒，每轮 benchmark 原预算为 15 秒。八条记录始终保留。
只有全部可评时原 checker 才产生 0 或 1 分，后端拒绝、样本不足或未决仍为 `null`。
`unevaluable` 不等于零分。参考解在某版本通过不能证明任意候选都可验收。
历史 wrong-speed 在 stop=3 s 的端点拒绝属于已知限制；不得改 stop 或删条件得到分数。

输出包含：

- `inputs.json`，公开 source commit、源码摘要、候选 bundle 摘要、checker 包摘要、runtime image/kernel 身份。
- `candidate/`，本次冻结的完整候选字节和 manifest。
- `replay/receipt.json`，可由 `result` 重新核验的独立执行收据。
- `replay/` 中的原 benchmark 报告、八条件记录、两个网格波形与日志。

`result` 只读取已保存证据。没有实际 Spectre archive 对照时分类为 `not_compared` 或
`unevaluable`，不能报告新 Spectre 一致性结论。它不执行许可证、SSH 或模型探针。
raw 波形、日志和模型提交留在本地运行存储，不提交到 Git。

原任务和判据说明见 [benchmark replay 合同](https://github.com/BucketSran/vaEVAS/blob/f8b624f8887f5d55ce726373cff2a856c1487b79/benchmark/checkers/triangle_evas_replay.md)。

## 已执行的本地检查

2026-10-09 使用上述公开 commit 的 EVAS 0.14.0，在本机 ARM64 Docker 中实际构建并执行。
参考解原八例均为 `graded/pass`，benchmark 原 score 为 1，分类为 `not_compared`。
同镜像将参考解的积分速度乘以 0.5，保留 benchmark 历史 wrong-speed 变体：
constant 两档发生 baseline 后端拒绝，其余六例为 `graded/fail`。
整题 score 为 `null`，分类为 `unevaluable`，八条条件均保留。

本次 kernel SHA-256 为
`cfc802ead7ee22d010f5425e8e8e6ca563a6afba84423b00439f1fae0d4ee010`，
本地 image ID 为
`sha256:8d0eeb0955e966858fc48927ced81cf17d48f11f152aacec77d2d697d170f68b`。
这些标识记录本次构建，不是可下载镜像承诺。参考候选字节摘要为
`27e015ced23d7486bec8b1888b81294fd863f565d751652e24dc4a9bb1035eae`。
本次没有模型生成、Spectre、SSH 或任意候选支持认证。
