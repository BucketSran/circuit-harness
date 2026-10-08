# Pretrain 使用指南(runbook)

> 怎么用 AlphaApollo 的 pretrain 能力:从语料到 RL 的完整流程,带可复制命令。
> 配套:设计见 `docs/design/pretrain-extension.md`。
> 核心 5 步:**语料 → 预处理(mmap)→ 多卡 from-scratch 训练 → 转 HF → 喂 RL(verl)**。
> **单环境即可**:`alphaapollo-unify`(torch 2.11 + verl + vLLM + TE + mcore0.18 + bridge)已实测覆盖全链路(2026-08-25:预训练 3 iters → 转换 338 张量/eos 自动解析 → SFT/GRPO,见各节)。HF checkpoint 仍是 pretrain 与 RL 两阶段之间唯一的产物接口。`alphaapollo-pretrain`(torch 2.12)为可选的独立快轨,非必需。

---

## 前置:激活 pretrain env

```bash
conda activate alphaapollo-unify    # 单环境;或 alphaapollo-pretrain(可选 torch 2.12 快轨)
# activate.d hook 已自动设好 LD_LIBRARY_PATH(nccl/cudnn)+ PYTHONPATH(Megatron clone)
```

环境变量(下 HF 数据时):
```bash
export HF_HOME=/tmp/hf_cache HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN="$(cat ~/.hf_token)"        # gated 数据需要
export HF_HUB_DISABLE_XET=1                  # hf-mirror 下 XET 数据需要
```

---

## 场景 A:from-scratch 预训练(真数据,完整流程)

### 1. 准备语料 → jsonl

每行一个 `{"text": <一段文本>}`(代码/文档皆可)。例:把 the-stack-smol 的 `{"content":code}` 重写成 `{"text":code}`。

### 2. 预处理 → Megatron mmap(Qwen tokenizer)

```bash
MG=${MEGATRON_LM_BRIDGE:-$(pwd)/Megatron-LM-bridge}
QDIR=/tmp/hf_cache/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987  # Qwen tokenizer 目录
python "$MG/tools/preprocess_data.py" \
  --input /tmp/code_corpus.jsonl --output-prefix /tmp/code_mmap \
  --tokenizer-type HuggingFaceTokenizer --tokenizer-model "$QDIR" \
  --append-eod --workers 8
# 产出 /tmp/code_mmap_text_document.{idx,bin}
```

### 3. 多卡 from-scratch 训练

```bash
cd ${REPO_ROOT:-$(pwd)}
CUDA_VISIBLE_DEVICES=1,7 torchrun --nproc_per_node 2 --nnodes 1 --master_port 29562 \
  -m alphaapollo.learning.off_policy.pretrain.main_pretrain \
  --num-layers 24 --hidden-size 896 --num-attention-heads 14 --num-query-groups 2 --ffn-hidden-size 4864 \
  --seq-length 2048 --max-position-embeddings 32768 \
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
  --micro-batch-size 4 --global-batch-size 64 \
  --train-iters 2000 --eval-iters 5 --eval-interval 200 \
  --tokenizer-type HuggingFaceTokenizer --tokenizer-model "$QDIR" \
  --data-path /tmp/code_mmap_text_document --split 980,10,10 --vocab-size 151936 \
  --swiglu --normalization RMSNorm --position-embedding-type rope --rotary-base 1000000 \
  --use-rotary-position-embeddings --group-query-attention --add-qkv-bias --disable-bias-linear \
  --transformer-impl transformer_engine --attention-backend flash --bf16 --distributed-backend nccl \
  --no-persist-layer-norm --no-gradient-accumulation-fusion \
  --lr 1.0e-4 --lr-decay-style constant --weight-decay 0.1 --clip-grad 1.0 --initial-loss-scale 65536 \
  --log-interval 20 --exit-interval 2000 --save-interval 1000 --save /tmp/code_pretrain_ckpt
```

**关键参数说明:**
- **架构**:改 `--num-layers/--hidden-size/--num-attention-heads/--num-query-groups/--ffn-hidden-size` 换模型尺寸;`--add-qkv-bias --disable-bias-linear` 对齐 Qwen2.5 布局(仅 q/k/v 带 bias,mcore 默认的 add_bias_linear 会训出 o_proj/MLP bias,HF Qwen2 没有槽位、导出只能丢弃);`--rotary-base 1000000` 是 Qwen2.5 的 rope theta。
- **并行**:`--tensor/pipeline-model-parallel-size` 配 TP/PP,余下 GPU 为 DP。0.5B 用 DP 即可(TP=PP=1);大模型再上 TP/PP。
- **数据**:`--data-path <mmap前缀>_text_document`;⚠️ **`--eval-interval` 不能是 0**(Megatron 内部 `train_iters // eval_interval` 会除零)。
- **加速**:TE 内核(`--transformer-impl transformer_engine`)+ flash attention(`--attention-backend flash`,需 flash-attn 已装)。
- **换自定义架构**:加 `--arch custom:<module>:<Class>` 或 `--arch <注册名>`(见场景 C)。

### 4. mcore checkpoint → HuggingFace

```bash
cd ${REPO_ROOT:-$(pwd)}
# 先拷 tokenizer(转换器只写 config.json + model.safetensors;且会从输出目录的
# tokenizer_config.json 解析 eos_token_id 写进 config——Qwen 系真实文件没有
# eos_token_id 键,转换器会经 eos_token 字符串 + added_tokens_decoder 反查)。
# 没有它,HF-rollout 生成永不早停(转换器会对缺失打 WARNING;也可显式传
# eos_token_id=<训练 eod>,NullTokenizer 冒烟即用此参数)。
mkdir -p /tmp/code_hf
cp "$QDIR"/tokenizer.json "$QDIR"/tokenizer_config.json "$QDIR"/vocab.json "$QDIR"/merges.txt /tmp/code_hf/
python -c "
from alphaapollo.learning.off_policy.pretrain.checkpoint_bridge import export_mcore_to_hf
export_mcore_to_hf('/tmp/code_pretrain_ckpt', '/tmp/code_hf',
    num_layers=24, hidden_size=896, num_attention_heads=14, num_query_groups=2,
    ffn_hidden_size=4864, vocab_size=151936)   # vocab 会自动从 ckpt 的 embedding 实际 shape 推断
"
```

### 5. 喂 RL(verl SFT / RL)

```bash
conda activate alphaapollo-unify   # 同一个环境
CUDA_VISIBLE_DEVICES=6 torchrun --standalone --nnodes 1 --nproc_per_node 1 \
  -m alphaapollo.learning.off_policy.main_off_policy \
    data.train_files=<你的SFT数据.parquet> \
    model.path=/tmp/code_hf \
    +model.override_config.attn_implementation=sdpa \
    data.train_batch_size=2 data.micro_batch_size_per_gpu=2 \
    trainer.total_epochs=1 \
    trainer.project_name=... trainer.experiment_name=...
```

SFT parquet 需要 `messages` 列(list of `{role, content}`,见 `alphaapollo/learning/off_policy/configs/sft_trainer.yaml`)。RL 的 `model.path` 指向预训练产出的 HF 目录；本分支已移除专用环境启动脚本，应用需显式注册环境池。保留的 HF-rollout 入口是 `main_on_policy_hf`，见下方对应段落。

---

## 场景 B:快速冒烟(无需语料,验证流水线)

```bash
conda activate alphaapollo-unify    # 或 alphaapollo-pretrain(可选快轨)
CUDA_VISIBLE_DEVICES=1 bash examples/pretrain/run_pretrain_qwen_smoke.sh   # mock data, ~3 iter
```

---

## 场景 C:自定义架构

1. 复制骨架:
   ```bash
   cp alphaapollo/learning/off_policy/pretrain/archs/example_custom.py alphaapollo/learning/off_policy/pretrain/archs/user_arch.py
   ```
2. 编辑 `user_arch.py` 里的类(它继承 `BaseModelBuilder`):把 `build(self, args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None)` 方法体里的 layer spec 换成你的自定义 spec / 自定义 attention/MLP 模块(返回一个 Megatron `GPTModel`),并把类名 / `@ARCHS.register("...")` 名字改成你的。`build` 方法签名不能改 —— adapter 按这个调用。
3. 训练时二选一:
   - `--arch custom:alphaapollo.learning.off_policy.pretrain.archs.myarch:MyArch`(动态导入该类)
   - 或直接 `--arch myarch`(若你在 `user_arch.py` 里用 `@ARCHS.register("myarch")` 注册了它)

   (其余参数同场景 A 第 3 步)。多卡 from-scratch 无需改其他代码。

⚠️ 自定义架构导出 HF:decoder-only 架构(Llama/Qwen2/Mistral 风格)可直接用通用转换器——
```bash
python -c "from alphaapollo.learning.off_policy.pretrain.converters import get_converter as g; \
  g('decoder').export('<mcore_ckpt>','<hf_out>', hf_model_type='llama', \
  num_layers=..,hidden_size=..,num_attention_heads=..,num_query_groups=..,ffn_hidden_size=..)"
```
(vocab / 是否有 qkv-bias / 是否 tied 均**自动从 ckpt 推断**;`hf_model_type` 可选 `llama`/`qwen2`/`mistral`)。**任意其他架构**(bert 扩散、MoE、双塔、真正自定义结构)用**映射驱动转换器**——给一份 mcore→HF 映射 JSON + HF config 即可,无需写 Python:
```bash
python -m alphaapollo.learning.off_policy.pretrain.converters.mapping \
  --mcore <mcore_ckpt> --out <hf_out> \
  --mapping <map.json> --config <hf_config.json>   # 模板见 converters/examples/
```
映射格式:`{num_layers, rules:[{from,to,layers?,split_dim?,split_sizes?,optional?,fill_zeros?}]}`——支持重命名 / 逐层 `{i}` 展开 / 按 sizes 拆分(QKV、gate-up)/ 可选跳过 / 零填充。详见 `converters/mapping.py` 顶部文档与 `converters/examples/qwen2_mapping.json`。

---

## 场景 D:扩散式语言模型(MDLM/LLaDA)

从零训练一个**扩散式**语言模型(双向 encoder;按扩散时间步把随机位置打成 `[MASK]`,模型预测原 token,损失 = masked 位置的加权交叉熵,吸收扩散权重 `1/t`（MDLM/LLaDA x0-prediction ELBO，线性 schedule）)。底座用 Megatron `BertModel`(双向;`lm_labels` 传入时内部算 per-token CE)。

```bash
CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node 1 --master_port 29595 \
  -m alphaapollo.learning.off_policy.pretrain.main_pretrain \
  --arch bert --forward masked_diffusion \
  --tokenizer-type HuggingFaceTokenizer --tokenizer-model "$QDIR" --diff-mask-id <mask-id> \
  --num-layers 12 --hidden-size 768 --num-attention-heads 12 --ffn-hidden-size 3072 \
  --seq-length 512 --max-position-embeddings 512 \
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
  --micro-batch-size 4 --global-batch-size 16 \
  --train-iters 2000 --eval-iters 5 --eval-interval 200 \
  --data-path <mmap> --vocab-size <vocab> \
  --position-embedding-type rope \
  --transformer-impl transformer_engine --attention-backend flash --bf16 --distributed-backend nccl \
  --no-persist-layer-norm --no-gradient-accumulation-fusion \
  --lr 3.0e-4 --lr-decay-style constant --weight-decay 0.1 --clip-grad 1.0 --initial-loss-scale 65536 \
  --log-interval 20 --exit-interval 2000 --save-interval 1000 --save /tmp/diff_pretrain_ckpt
```

**关键说明:**
- **架构** `--arch bert`:双向 encoder,**不要**加 `--group-query-attention`/`--add-qkv-bias`(BERT 是 MHA)。RoPE/SwiGLU 经 config 可开(RMSNorm 仅 TE spec 生效)。
- **forward** `--forward masked_diffusion`:每步 `t~U(--diff-t-min, --diff-t-max)`,按概率 `t` 把 token 打成 mask id,吸收扩散权重 `1/t`（MDLM/LLaDA x0-prediction ELBO，线性 schedule）。
- **mask token** `--diff-mask-id <id>`:corruption 用的 `[MASK]` token id(必须在词表内)。mock 可随便给一个(如 `vocab-2`);真 HF 模型需确保有 mask token(Qwen 默认没有,需自行加一个并 resize embedding)。
- **范围**:本场景只做**预训练**。采样/推理(从全 `[MASK]` 迭代去噪)、`BertModel→HF` 转换器是后续工作。

---

## 场景 E:任意自定义架构 → verl 强化学习(含非-transformer,如 conv/MLP-Mixer)

适用:vLLM 不认的架构(非标准 transformer 家族)。verl 0.9 的 rollout 是 server-based,本框架提供一个**模型无关**的 HF-generate server 适配(`alphaapollo/learning/on_policy/{hf_rollout_worker,hf_task_runner,main_on_policy_hf}.py`),让任意能 HF 加载的架构做 GRPO。已用无 attention 的 conv 架构(CausalMLP)端到端验证通过(2 步 GRPO,exit 0)。

**前置**:模型已 pretrain + 转 HF(trust_remote_code,`modeling_xxx.py` + config `auto_map`)。

**RL 命令**(verl env,Ray 管 GPU;`rollout.name=hf`、`n=8`、`use_remove_padding=false`、`engine_kwargs` 等已由 `ppo_trainer_hf` overlay 默认):
```bash
CUDA_VISIBLE_DEVICES=6,7 \
python -m alphaapollo.learning.on_policy.main_on_policy_hf \
  actor_rollout_ref.model.path=<hf_dir> \
  actor_rollout_ref.model.trust_remote_code=true \
  +actor_rollout_ref.model.override_config.attn_implementation=eager \
  reward.custom_reward_function.path=<reward.py> reward.custom_reward_function.name=compute_score \
  data.train_files=<prompts.parquet> data.val_files=<同> \
  trainer.total_training_steps=2 trainer.total_epochs=2 trainer.n_gpus_per_node=2
```

**关键 flag**:
- `rollout.name=hf`(overlay 默认):用 HF `model.generate`(独立 `HFServerActor` server),**绕过 vLLM**(vLLM 不认非-transformer 架构)。
- `use_remove_padding=false`(overlay 默认):engine 把序列还原成 padded `[bs,seq]+attention_mask` 再喂模型(verl 默认的 nested/jagged packed 路径对非-attention 架构会跨样本混)。
- `enable_gradient_checkpointing=false`:自定义模型通常不支持 grad checkpoint(小模型不需要)。
- `attn_implementation=eager`:无 attention 的架构用 eager(sdpa/flash 要求模型有 attention 实现)。
- `total_epochs` 要 ≥ 预期步数对应的 epoch 数:verl 在 dataloader 耗尽(每 epoch 步数 = 样本数/train_batch_size)时正常收尾,不会为凑 `total_training_steps` 续跑。
- **多卡(≥2 GPU)无需手动设 `NCCL_P2P_DISABLE=1`**:入口对 >1 GPU 自动开启(verl 全局 PG init 在 NCCL 2.27 的 P2P transport probe 上会卡死,关 P2P 走 SHM 绕过;单卡不涉及)。

**加新自定义架构**(只需 HF 端,适配层不动):
1. 写 `modeling_xxx.py`:`forward(input_ids, attention_mask, position_ids, labels=None, use_cache=False) → CausalLMOutputWithPast(logits)`,**用 `attention_mask` 屏蔽 padding**(非-attention 架构如 conv 否则把 pad token 卷进真实 token)。config 加占位 `num_attention_heads`/`num_key_value_heads` + `text_config` property 返回 self(verl 的 MFU/flops 探测会读)。
2. `config.json` 的 `auto_map` 指向 `modeling_xxx.{Config,ForCausalLM}`。
3. 用上面命令(`actor_rollout_ref.model.path` 指向你的 HF 目录)。

**机制简述**:verl 0.9 `RayPPOTrainer` → `LLMServerClient` → `server.generate.remote()`(Ray actor)。`HFReplica` 起一个独立 `HFServerActor`(普通 `ray.remote`,`generate` Ray 可见),用确定性 named-actor 名字;`HFServerAdapter`(= worker.self.rollout)用 `ray.get_actor(name)` 找到它,每 step 把 actor 权重 Ray-RPC 推过去(`load_weights`)。生成各样本带 padding 独立(不混样本);log-prob/actor 走 padded(`use_remove_padding=false`)。

**限制**:HF generate 慢(无 KV cache,每步重算)——适合小模型/smoke;weight 每步 Ray push(小模型 OK)。生产规模 RL 建议用 vLLM 支持的标准 transformer 架构(见场景 A/myarch)。

---

## 暂不可用:continued-pretrain(加载已有 HF 权重续训)

`--load-hf <HF目录>` 已在 `model_provider` 接线(调 `megatron.bridge.AutoBridge.from_hf_pretrained` + `load_hf_weights`),但卡在 bridge↔mcore0.18 的 API 漂移(`load_hf_weights` 报 `'NoneType' object has no attribute 'megatron_module'`),**当前跑不通**。

- 想用已有 Qwen 权重:目前直接走 RL env 的 verl SFT(它本身支持加载 HF)。
- 待修:bridge/0.18 兼容性,或改用 Megatron 原生 `tools/checkpoint` 转换器把 HF→mcore 再走 `--load`。

---

## 速查:常用 flag

| 用途 | flag |
|---|---|
| 架构(默认/自定义) | `--arch gpt` / `--arch custom:<module>:<Class>` 或 `--arch <注册名>` |
| 扩散式语言模型(MDLM) | `--arch bert --forward masked_diffusion --diff-mask-id <id>` |
| 加速 | `--transformer-impl transformer_engine --attention-backend flash` |
| 并行 | `--tensor-model-parallel-size` / `--pipeline-model-parallel-size`(余为 DP) |
| 真/mock 数据 | `--data-path <mmap>` / `--mock-data` |
| 续训(加载 HF,暂不可用) | `--load-hf <HF目录>` |
| 加载已有 mcore ckpt | `--load <mcore_ckpt_dir>` |
| tokenizer | `--tokenizer-type HuggingFaceTokenizer --tokenizer-model <Qwen目录>` |

---

## 已验证的参考运行(2026-07-14)

from-scratch 代码预训练(the-stack-smol,2 GPU DP=2,200 iter):loss 7.23→5.81,val PPL 2113→921;→ HF(494M params)→ verl SFT(loss 7.57→4.63)。详见工作总结。
