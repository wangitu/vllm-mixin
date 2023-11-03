import sys
import os
import json
from typing import Optional, Union, Dict

from vllm.model_executor.model_loader import _MODEL_REGISTRY
from vllm.worker import worker

from .models.qwen import QWenLMHeadModel
from .models.baichuan import BaichuanForCausalLM, BaiChuanForCausalLM
from .quantization_utils import get_model, bool_to_env


_old_modules = []
_old_get_model = worker.get_model


def enable_gptq_support(
    force_download: bool = False,
    resume_download: bool = False,
    proxies:  Optional[Dict[str, str]] = None,
    local_files_only: bool = False,
    use_auth_token: Optional[Union[bool, str]] = None,
    subfolder: str = "",
    _commit_hash: Optional[str] = None,
    disable_exllama: bool = False,
    disable_exllamav2: bool = True,
    use_triton: bool = False,
    use_cuda_fp16: bool = True,
    device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = None,
    max_memory: Optional[Dict] = None
):
    _old_modules.append(("QWenLMHeadModel", _MODEL_REGISTRY.get("QWenLMHeadModel")))
    _old_modules.append(("BaichuanForCausalLM", _MODEL_REGISTRY.get("BaichuanForCausalLM")))
    _old_modules.append(("BaiChuanForCausalLM", _MODEL_REGISTRY.get("BaiChuanForCausalLM")))
    _MODEL_REGISTRY["QWenLMHeadModel"] = QWenLMHeadModel
    _MODEL_REGISTRY["BaichuanForCausalLM"] = BaichuanForCausalLM
    _MODEL_REGISTRY["BaiChuanForCausalLM"] = BaiChuanForCausalLM
    
    worker.get_model = get_model
    
    bool_to_env("force_download", force_download)
    bool_to_env("resume_download", resume_download)
    if proxies is not None:
        if isinstance(proxies, Dict):
            proxies = json.dumps(proxies)
        os.environ["proxies"] = proxies
    bool_to_env("local_files_only", local_files_only)
    if use_auth_token is not None:
        if isinstance(use_auth_token, bool):
            bool_to_env("use_auth_token", use_auth_token)
        else:
            os.environ["use_auth_token"] = use_auth_token
    os.environ["subfolder"] = subfolder
    if _commit_hash is not None:
        os.environ["_commit_hash"] = _commit_hash
    
    bool_to_env("disable_exllama", disable_exllama)
    bool_to_env("disable_exllamav2", disable_exllamav2)
    bool_to_env("use_triton", use_triton)
    bool_to_env("use_cuda_fp16", use_cuda_fp16)
    
    if device_map is not None:
        if isinstance(device_map, Dict):
            device_map = json.dumps(device_map)
        os.environ["device_map"] = device_map
    if max_memory is not None and isinstance(max_memory, Dict):
        os.environ["max_memory"] = json.dumps(max_memory)
    

def disable_gptq_support():
    for name, attr in _old_modules:
        _MODEL_REGISTRY[name] = attr
    
    worker.get_model = _old_get_model
