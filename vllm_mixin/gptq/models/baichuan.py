# coding=utf-8
# author=Qainle Wang
# Adapted from
# https://huggingface.co/baichuan-inc/Baichuan2-13B-Base/blob/main/modeling_baichuan.py
# Copyright (c) Baichuan Intelligent Technology.
# LICENSE: https://huggingface.co/baichuan-inc/Baichuan2-13B-Base/blob/main/Baichuan2%20%E6%A8%A1%E5%9E%8B%E7%A4%BE%E5%8C%BA%E8%AE%B8%E5%8F%AF%E5%8D%8F%E8%AE%AE.pdf
"""Inference-only QWen model compatible with HuggingFace weights.

The input of the model is flattened to a 1D tensor of tokens. The model uses
InputMetadata to extract the original 2D shape of the input.
"""


import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers.activations import ACT2FN

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.attention import (
    PagedAttentionWithRoPE,
    PagedAttentionWithALiBi
)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.weight_utils import (
    convert_pyslice_to_tensor,
    hf_model_weights_iterator
)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size,
)
from vllm.sequence import SamplerOutput
from vllm.transformers_utils.configs.baichuan import BaiChuanConfig

KVCache = Tuple[torch.Tensor, torch.Tensor]


def _get_alibi_slopes(total_num_heads: int) -> torch.Tensor:
    closest_power_of_2 = 2 ** math.floor(math.log2(total_num_heads)) # 32
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), # 2 ^ (-8 / 32)
        dtype=torch.float32,
    )
    powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32) # 1, 2, 3, 4, 5, ..., 32
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != total_num_heads:
        extra_base = torch.tensor(
            2**(-(2**-(math.log2(2 * closest_power_of_2) - 3))), # 2 ^ (-8 / 64)
            dtype=torch.float32,
        )
        num_remaining_heads = min(closest_power_of_2,
                                  total_num_heads - closest_power_of_2) # min(32, 8) = 8
        extra_powers = torch.arange(start=1,
                                    end=1 + 2 * num_remaining_heads, # 1, 3, 5, ..., 13, 15
                                    step=2,
                                    dtype=torch.int32)
        slopes = torch.cat(
            [slopes, torch.pow(extra_base, extra_powers)], dim=0)
    return slopes


class BaichuanMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()
        self.gate_proj = torch.nn.Linear(
            hidden_size, intermediate_size, bias=False
        )
        self.down_proj = torch.nn.Linear(
            intermediate_size, hidden_size, bias=False
        )
        self.up_proj = torch.nn.Linear(
            hidden_size, intermediate_size, bias=False
        )
        self.act_fn = ACT2FN[hidden_act]
        
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    

class BaichuanAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        position_embedding: str,
        rope_theta: float = 10000,
        max_position_embeddings: int = 8192,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // self.num_heads
        self.postion_embedding = position_embedding
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        
        self.W_pack = torch.nn.Linear(
            self.hidden_size, 3 * self.hidden_size, bias=False
        )
        self.o_proj = torch.nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        
        # Create the alibi slopes.
        if self.postion_embedding == "ALIBI":
            alibi_slopes = _get_alibi_slopes(self.num_heads).tolist()
            scaling = self.head_dim ** -0.5
            self.attn = PagedAttentionWithALiBi(
                self.num_heads, self.head_dim,
                scaling, alibi_slopes
            )
        else:
            self.scaling = self.head_dim ** -0.5
            self.attn = PagedAttentionWithRoPE(
                self.num_heads,
                self.head_dim,
                self.scaling,
                rotary_dim=self.head_dim,
                base=self.rope_theta,
                max_position=self.max_position_embeddings
            )
            
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        qkv = self.W_pack(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        k_cache, v_cache = kv_cache
        if self.postion_embedding == "ALIBI":
            attn_output = self.attn(
                q, k, v, k_cache, v_cache, 
                input_metadata, cache_event
            )
        else:
            attn_output = self.attn(
                positions, q, k, v, k_cache, v_cache,
                input_metadata, cache_event
            )

        output = self.o_proj(attn_output)
        return output
    
    
class BaichuanLayer(torch.nn.Module):
    def __init__(self, config: BaiChuanConfig, position_embedding: str):
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        max_position_embeddings = getattr(
            config, "max_position_embeddings", 8192
        )
        self.self_attn = BaichuanAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            position_embedding=position_embedding,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
        )
        self.mlp = BaichuanMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
            cache_event=cache_event,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
    

class BaichuanModel(nn.Module):
    def __init__(self, config: BaiChuanConfig, position_embedding: str):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = nn.Embedding(
            self.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList([
            BaichuanLayer(config, position_embedding)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for i in range(len(self.layers)):
            if cache_events is None:
                cache_event = None
            else:
                cache_event = cache_events[i]
            layer = self.layers[i]
            hidden_states = layer(
                positions,
                hidden_states,
                kv_caches[i],
                input_metadata,
                cache_event,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states
    
    
class BaichuanBaseForCausalLM(nn.Module):
    layer_type: str = "BaichuanLayer"
    lm_head_name: str = "lm_head"
    outside_layer_modules: List[str] = ["model.embed_tokens", "model.norm"]
    
    def __init__(self, config, position_embedding: str):
        super().__init__()
        self.config = config
        self.model = BaichuanModel(config, position_embedding)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
        self.sampler = Sampler(config.vocab_size)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> SamplerOutput:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                input_metadata, cache_events)
        next_tokens = self.sampler(self.lm_head.weight, hidden_states,
                                input_metadata)
        return next_tokens
        
    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        load_format: str = "auto",
        revision: Optional[str] = None,
    ):
        state_dict = self.state_dict()

        for name, loaded_weight in hf_model_weights_iterator(
            model_name_or_path, cache_dir, load_format, revision
        ):
            if "rotary_emb.inv_freq" in name:
                continue
            
            loaded_weight = convert_pyslice_to_tensor(loaded_weight)
            param = state_dict[name]
            param.data.copy_(loaded_weight)
            
            
class BaichuanForCausalLM(BaichuanBaseForCausalLM):  # baichuan 13b
    def __init__(self, config):
        super().__init__(config, "ALIBI")


class BaiChuanForCausalLM(BaichuanBaseForCausalLM):  # baichuan 7b
    def __init__(self, config):
        super().__init__(config, "ROPE")