import sys
import os

from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size,
)


def _verify_parallel():
    tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
    assert tensor_model_parallel_world_size == 1, "Tensor model parallel is not compatible with GPTQ, please set `tensor_parallel_size = 1`"


class GPTQForCausalLM:
    def __init__(self):
        _verify_parallel()