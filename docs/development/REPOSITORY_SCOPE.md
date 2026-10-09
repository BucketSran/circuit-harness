# circuit-harness 仓库范围

本仓库维护独立的电路实验平台。模块职责与使用入口见 [README](../../README.md)和[文档导航](../README.md)。

仓库包含 benchmark 适配、电路执行、公开会话、独立 verifier、Harbor 集成、离线报告与 ATIF 数据接口。Harbor 负责实验调度与 Agent 执行；训练器由使用者提供。固定 VABench r53 协议和兼容 Episode 读取能力继续维护。

清理后的源码仍需本地回归和安装包检查；这些检查不能认证真实模型、EDA 许可证或新任务成绩。历史实测条件见 [VALIDATION](../validation/VALIDATION.md)。

<a id="publication-boundary"></a>

## 公开源码与私有材料

Python distribution 是 `circuit-harness`，导入包是 `circuit_harness`。
安装方式见[安装指南](../../INSTALL.md)，任务评分规则由 benchmark 声明。公开仓库的首次提交是审查后的源码快照；
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
例如 [Pi 轨迹回归](../../tests/chips/test_pi_trajectory.py)保留事件结构和解析断言。
公开文档中的 `lab-server` 是匿名部署标识，不是可连接主机；真实映射留在私有配置。

来源检查包括根目录 [LICENSE](../../LICENSE)、[Notice.txt](../../Notice.txt) 和[第三方说明](../../third_party/README.md)。项目代码的许可证不自动覆盖依赖、外部数据集或厂商资产。

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
运行所需的外部 EVAS/benchmark/商业工具不打入包；另行记录实际 URL、精确版本与使用权限。

快照审查通过后，才能将它用于单独的公开源码仓库或分发包。不要把保留旧历史的私有研发仓库转公开：
那会带出仍可达的历史，且 GitHub 上的 Issue/PR、附件和 Actions 日志还需单独审查。
删除旧分支、增加 `.gitignore` 或一次秘密扫描无匹配，都不能替代这一步。
后续引入私有开发成果时仍须审查实际文件，只迁移源码，不合并私有 Git 历史。
发布新数据、恢复旧资产或改变私有仓库可见性需要对应范围的明确授权。
