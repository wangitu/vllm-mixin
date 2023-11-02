# coding=utf-8
# author=Qainle Wang
# Adapted from
# https://huggingface.co/Qwen/Qwen-7B/blob/main/modeling_qwen.py
# Copyright (c) Alibaba Cloud.
# LICENSE: https://huggingface.co/Qwen/Qwen-7B/blob/main/LICENSE
"""Inference-only QWen model compatible with HuggingFace weights.

The input of the model is flattened to a 1D tensor of tokens. The model uses
InputMetadata to extract the original 2D shape of the input.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.attention import PagedAttentionWithRoPE
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.weight_utils import (
    convert_pyslice_to_tensor,
    hf_model_weights_iterator
)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size,
)
from vllm.sequence import SamplerOutput
from vllm.transformers_utils.configs.qwen import QWenConfig

KVCache = Tuple[torch.Tensor, torch.Tensor]


class QWenMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
    ):
        super().__init__()
        self.w1 = nn.Linear(
            hidden_size, intermediate_size, bias=False
        )
        self.w2 = nn.Linear(
            hidden_size, intermediate_size, bias=False
        )
        self.c_proj = nn.Linear(
            intermediate_size, hidden_size, bias=False
        )
    
    def forward(self, x):
        a1 = self.w1(x)
        a2 = self.w2(x)
        x = a1 * F.silu(a2)
        output = self.c_proj(x)
        return output
    
    
class QWenAttention(nn.Module):
    def __init__(self,
        hidden_size: int,
        num_heads: int,
        max_position_embeddings: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // self.num_heads
        
        self.c_attn = nn.Linear(
            hidden_size, 3 * self.num_heads * self.head_dim
        )
        self.c_proj = nn.Linear(
            self.num_heads * self.head_dim, hidden_size, bias=False
        )
        self.scaling = self.head_dim ** -0.5
        self.attn = PagedAttentionWithRoPE(
            self.num_heads,
            self.head_dim,
            self.scaling,
            rotary_dim=self.head_dim,
            base=rope_theta,
            max_position=max_position_embeddings,
            rope_scaling=rope_scaling
        )
        
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)

        k_cache, v_cache = kv_cache
        attn_output = self.attn(
            positions, q, k, v, k_cache, v_cache,
            input_metadata, cache_event
        )

        output = self.c_proj(attn_output)
        return output
    
    
class QWenBlock(nn.Module):
    def __init__(self, config: QWenConfig):
        super().__init__()
        self.ln_1 = RMSNorm(
            config.hidden_size, eps=config.layer_norm_epsilon
        )

        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        self.attn = QWenAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling
        )

        self.ln_2 = RMSNorm(
            config.hidden_size, eps=config.layer_norm_epsilon
        )
        self.mlp = QWenMLP(
            config.hidden_size, config.intermediate_size // 2
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
        hidden_states = self.ln_1(hidden_states)
        hidden_states = self.attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
            cache_event=cache_event,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
    

class QWenModel(nn.Module):
    def __init__(self, config: QWenConfig):
        super().__init__()
        
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
        assert tensor_model_parallel_world_size == 1, "Tensor model parallel is not compatible with GPTQ, please set `tensor_parallel_size = 1`"
        
        self.config = config
        self.vocab_size = config.vocab_size

        self.wte = nn.Embedding(
            self.vocab_size,
            config.hidden_size,
        )
        self.h = nn.ModuleList(
            [QWenBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.ln_f = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> torch.Tensor:
        hidden_states = self.wte(input_ids)
        for i in range(len(self.h)):
            if cache_events is None:
                cache_event = None
            else:
                cache_event = cache_events[i]
            layer = self.h[i]
            hidden_states = layer(
                positions,
                hidden_states,
                kv_caches[i],
                input_metadata,
                cache_event,
            )
        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class QWenLMHeadModel(nn.Module):
    
    layer_type: str = "QWenBlock"
    lm_head_name: str = "lm_head"
    outside_layer_modules: List[str] = ["transformer.wte", "transformer.wpe", "transformer.ln_f", "transformer.visual"]
    
    def __init__(self, config: QWenConfig):
        super().__init__()
        self.config = config
        self.transformer = QWenModel(config)
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
        hidden_states = self.transformer(
            input_ids, positions, kv_caches,
            input_metadata, cache_events
        )
        next_tokens = self.sampler(
            self.lm_head.weight, hidden_states,
            input_metadata
        )
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
