# circuit-harness 仓库范围

本分支维护 Chips 实验平台，保留 Robotics 作为环境、工具、仿真后端和
Memory 设计的代码参照。Chips 的应用入口集中在 `examples/chips/`、
`docs/chips/`、`tests/chips/` 和 `alphaapollo` 中的 Chips 模块。

## 保留

- Chips 的任务、公开 Tools、SSH／服务器仿真、冻结终评、批次和离线轨迹报告。
- Robotics 的 Environment、ToolSpecs、LIBERO／RoboCasa 后端、数据准备、
  Memory 投影和对应契约测试。它仍是参考适配，未在本轮运行真实机器人仿真。
- 共享的 Generation、原生／外部 Agent Runtime（Pi、Codex、Claude Code）、
  MCP 桥、通用文件／Shell／Python 工具、沙箱、工作区、轨迹、Workflow、
  Verification、Memory 和 Learning 基础。通用 `python_execute` 的实现位于
  `common/execution/tools/python.py`，不再依赖 Math 目录。
- 通用数据准备与 `exact_match` 评分；`environment_success` 用于环境终局。
  通用声明式答案分组只把纯数值当作精确有理数，带单位的答案保持文本语义。

## 移除

Bio（包括 PerturbHD／STATE）、Math（AIME、圆堆积、MathMemory、Regina、
CGAL、MATLAB、Sage 和 Lean4）、ALFWorld、Search、Sokoban、WebShop 的
专用实现、注册项、配置、依赖选项和专用测试。Sokoban 训练启动脚本及其配置
一同移除；训练器和通用回合／池接口保留，应用需显式注册自己的池工厂。

这是本分支有意的 API 收缩。旧配置中的 `grading`、Math 工具预设、
`integer_answer`／`math_expression` 评分器、`lean4` Verifier 和 Math Memory
不能继续使用。不会为这些已移除功能提供空壳兼容实现。

旧参考分支已从远端删除，完整研发历史和所需材料保存在私有 Git 备份中。
旧 `demo/chips` 的可达历史仍有已删除的代码与旧领域资产，留在 `circuit-harness-private`。
本公开仓库的 `main` 从独立的源码快照开始，不包含那段历史。
同步上游时，先确认共享能力的依赖，再迁移必要变更，不自动恢复已移除的领域应用。

## 验收边界

清理以共享启动、工具发现／调用、资源生命周期、Robotics 契约和 Chips
离线回归验证。CI 不访问真实模型或实验室许可证；真实 EDA、SSH 与模型实验
仍以 [验证记录](VALIDATION.md) 的逐项证据为准。运行产物和本机文件职责图
保存在 Git 忽略目录，不随源码发布。

<a id="publication-boundary"></a>

## 公开源码与私有材料

`circuit-harness` 是仓库名称，Python distribution、导入和 CLI 继续使用 `alphaapollo`。
发布不改变 API、任务协议或仿真器选择。公开仓库的首次提交是审查后的源码快照；
完整研发历史、旧 Issue/PR 和 Actions 记录留在私有的 `circuit-harness-private`。
指向该仓库的历史链接需要原有权限，不能当作本公开仓库的同号议题。

| 内容 | 发布规则 |
| --- | --- |
| Harness 执行、会话、Agent/Model 配置、Harbor、轨迹导出和通用 verifier 源码 | 随源码发布，保留来源和适用许可证 |
| Simulator adapter、无 PDK 合成电路、协议回归和脱敏配置模板 | 可随源码发布；真实许可证、安装包和工艺材料由操作者另行提供 |
| 文档和验证结论 | 保留可复现条件、版本、结果与限制；匿名化主机和个人路径，移除实际账号/部署细节 |
| API key、登录 token、SSH 私钥、Cookie、许可证文件和实际 operator/profile | 不提交到 Git，包括私有仓库；使用操作者私有配置和凭据存储 |
| 原始提示、模型轨迹、SFT 数据、完整波形/PSF、仿真日志、候选归档和恢复包 | 默认留在私有运行存储；选作公开数据时独立审核和发布 |
| PDK、厂商模型/样例、商业软件、外部 benchmark、隐藏测试和参考答案 | 先核对具体来源和再分发条件，不能因 Harness 的许可证自动纳入 |
| `.planning/`、Git bundle、旧研究资产和机器状态 | 留在私有备份，不进入公开快照、Release、镜像或 CI 附件 |

通用 verifier 代码可以公开。Agent 在运行时不能读取私有评分输入，是执行隔离要求；
具体隐藏用例、阈值和参考设计仍由 benchmark 的发布政策决定。
vaEVAS 引擎与 benchmark 内容归其所属项目，Harness 只发布自己的适配代码和版本引用。

测试材料按内容和来源判断，不按扩展名一律排除。构造的 `.va`、`.scs` 和 `.jsonl` 可以是
安全夹具；从真实运行裁剪的协议样本需替换身份、时间、路径与敏感文本，并记录处理范围。
例如 [外部 Agent 流夹具](../../tests/reasoning/runtime/fixtures/README.md)保留事件结构和解析断言。
公开文档中的 `lab-server` 是匿名部署标识，不是可连接主机；真实映射留在私有配置。

来源检查包括根目录 [LICENSE](../../LICENSE)、[Notice.txt](../../Notice.txt)、
[Pi 的 MIT 文本](../../third_party/pi/LICENSE)及 [第三方说明](../../third_party/README.md)。
保留 Robotics 参考适配时，一并核对其 [数据来源](../../alphaapollo/data_preprocess/robotics/UPSTREAM.md)
和派生任务清单；项目代码的许可证不自动覆盖每个数据集或厂商资产。

## 准备可审查的源码快照

先完成文件清理和受影响的本地检查，再选择已审查的具体提交：

```bash
umask 077
mkdir -p .planning/chips/publication
git archive --format=tar --prefix=circuit-harness/ \
  --output=.planning/chips/publication/source-candidate.tar REVIEWED_COMMIT
```

将 `REVIEWED_COMMIT` 替换为具体 commit SHA。这个命令只导出该提交的跟踪文件，
不包含 `.git`、提交历史、未跟踪或 ignored 文件；它不是敏感信息过滤器。
检查展开后的文件清单、文件摘要、秘密扫描结果、二进制内容和第三方归属，保留候选包摘要。
运行所需的外部 EVAS/benchmark/商业工具不打入包；Git submodule 的源码也不由 `git archive`
递归导出，另记录其 URL 和精确提交，按 [第三方说明](../../third_party/README.md)准备。

快照审查通过后，才能将它用于单独的公开源码仓库或分发包。不要把保留旧历史的私有研发仓库转公开：
那会带出仍可达的历史，且 GitHub 上的 Issue/PR、附件和 Actions 日志还需单独审查。
删除旧分支、增加 `.gitignore` 或一次秘密扫描无匹配，都不能替代这一步。
后续引入私有开发成果时仍须审查实际文件，只迁移源码，不合并私有 Git 历史。
发布新数据、恢复旧资产或改变私有仓库可见性需要对应范围的明确授权。
