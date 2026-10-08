"""HuggingFace trust_remote_code model mirroring the Megatron CausalMLPModel EXACTLY
(same modules, same names) so the mcore->HF converter is a near-identity rename.

This is NOT a transformer: causal depthwise-conv token mixing + GLU channel mixing, residual
pre-norm blocks. generate() recomputes the full sequence each step (no KV cache) — correct,
just slow; fine for short RL rollouts.
"""

import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


class CausalMLPConfig(PretrainedConfig):
    model_type = "causal_mlp"

    def __init__(
        self,
        vocab_size=8192,
        hidden_size=512,
        num_hidden_layers=6,
        intermediate_size=1024,
        conv_kernel=8,
        layer_norm_eps=1e-5,
        tie_word_embeddings=False,
        num_attention_heads=8,
        num_key_value_heads=8,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.conv_kernel = conv_kernel
        self.layer_norm_eps = layer_norm_eps
        # Placeholders verl's flops_counter / MFU probe reads (this arch has no attention, so
        # MFU will be inaccurate — fine, it's only a metric). Custom configs must satisfy these.
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

    @property
    def text_config(self):
        # verl's multimodal flops branch reads config.text_config.*; pure-text models return self.
        return self


class CausalDepthwiseConv(nn.Module):
    def __init__(self, hidden, kernel):
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv1d(hidden, hidden, kernel_size=kernel, groups=hidden, bias=True)

    def forward(self, x):  # [b, s, h]
        xt = x.transpose(1, 2)
        xt = F.pad(xt, (self.kernel - 1, 0))  # causal left-pad
        return self.conv(xt).transpose(1, 2)


class GLUMLP(nn.Module):
    def __init__(self, hidden, ffn):
        super().__init__()
        self.fc1 = nn.Linear(hidden, 2 * ffn, bias=False)
        self.fc2 = nn.Linear(ffn, hidden, bias=False)

    def forward(self, x):
        g, v = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(g) * v)


class CausalMLPLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.conv = CausalDepthwiseConv(cfg.hidden_size, cfg.conv_kernel)
        self.norm2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.mlp = GLUMLP(cfg.hidden_size, cfg.intermediate_size)

    def forward(self, x, attention_mask=None):
        x = x + self.conv(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        # Re-zero padding positions AFTER the layer. The causal Conv1d has a bias, so even with pad
        # zeroed at the layer input, conv(norm(pad)) == bias makes pad positions non-zero again;
        # a later layer's causal conv would then fold that non-zero pad into neighbouring real
        # tokens -> batched (left-padded) generation differs from unbatched. Re-masking each layer
        # keeps pad at 0 at every conv input, so a real token's conv window sees only real tokens
        # (pad neighbours contribute 0) == unbatched (causal left-pad is also 0).
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(x.dtype)
        return x


class CausalMLPForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = CausalMLPConfig
    _no_split_modules = ["CausalMLPLayer"]
    supports_gradient_checkpointing = False

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [CausalMLPLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        position_ids=None,
        **kwargs,
    ):
        h = self.embed(input_ids)
        if attention_mask is not None:
            # Zero out padding positions so the causal conv doesn't fold pad-token embeddings
            # into neighbouring real tokens. verl passes a full [bs, seq] mask when
            # use_remove_padding=false; without it, conv-based arches mix samples.
            h = h * attention_mask.unsqueeze(-1).to(h.dtype)
        for layer in self.layers:
            h = layer(h, attention_mask)
        h = self.final_norm(h)
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, **kwargs
    ):
        # Propagate attention_mask into forward during generation. Dropping it here (the prior
        # version returned only input_ids) meant generate() forwarded with attention_mask=None, so
        # the pad-zeroing in forward never ran -> padded batch generation leaked across samples.
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def _reorder_cache(self, past, beam_idx):
        return past


# Register with the Auto* system so trust_remote_code loading (and vLLM's
# TransformersForCausalLM fallback, which uses AutoModel.from_config) can resolve us.
#
# Why config.json's `architectures: ["TransformersForCausalLM"]` is NOT CausalMLPForCausalLM:
# this is intentional, not a mismatch with auto_map. The path we actually use
# (HF-rollout) loads via AutoModelForCausalLM(trust_remote_code=True), which follows auto_map and
# IGNORES `architectures` — so the value is irrelevant there. But a loader that consults
# `architectures` directly (e.g. vLLM's generic TransformersForCausalLM wrapper, which internally
# does AutoModel.from_config) needs the GENERIC name to land on the registered CausalMLPForCausalLM
# via the AutoModel.register below. Setting it to CausalMLPForCausalLM would break that fallback.
try:
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

    AutoConfig.register("causal_mlp", CausalMLPConfig)
    AutoModel.register("causal_mlp", CausalMLPForCausalLM)
    AutoModelForCausalLM.register("causal_mlp", CausalMLPForCausalLM)
except Exception:
    pass
