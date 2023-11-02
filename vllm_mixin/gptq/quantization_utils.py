import sys
import os
import json

import torch
import torch.nn as nn
import accelerate
from transformers.utils.hub import (cached_file)
from transformers.utils.generic import ContextManagers

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.weight_utils import prepare_hf_model_weights
from vllm.model_executor.model_loader import (
    _set_default_torch_dtype,
    _get_model_architecture
)
from auto_gptq import BaseQuantizeConfig
from auto_gptq.modeling import BaseGPTQForCausalLM
from auto_gptq.modeling._utils import (
    find_layers, make_quant,
    make_sure_no_tensor_in_meta_device,
    get_module_by_name_suffix,
    autogptq_post_init
)

logger = init_logger(__name__)


def bool_to_env(flag: str, value: bool):
    os.environ[flag] = str(int(value))

def bool_from_env(flag: str, default: bool):
    return int(os.environ.get(flag, int(default))) == 1

def cached_kwargs_from_env(model_config: ModelConfig):
    proxies = os.environ.get("proxies")
    if proxies is not None:
        try:
            proxies = json.loads(proxies)
        except json.decoder.JSONDecodeError as e:
            pass
    use_auth_token = os.environ.get("use_auth_token")
    if use_auth_token is not None:
        if use_auth_token == '0' or use_auth_token == '1':
            use_auth_token = bool(use_auth_token)
        
    cached_file_kwargs = {
        "cache_dir": model_config.download_dir,
        "force_download": bool_from_env("force_download", False),
        "resume_download": bool_from_env("resume_download", False),
        "proxies": proxies,
        "local_files_only": bool_from_env("local_files_only", False),
        "use_auth_token": use_auth_token,
        "revision": model_config.revision,
        "subfolder": os.environ.get("subfolder", ""),
        "_raise_exceptions_for_missing_entries": False,
        "_commit_hash": os.environ.get("_commit_hash"),
    }
    return cached_file_kwargs

def extension_kwargs_from_env(model_config: ModelConfig):
    extension_kwargs = {
        "disable_exllama": bool_from_env("disable_exllama", False),
        "disable_exllamav2": bool_from_env("disable_exllamav2", True),
        "use_triton": bool_from_env("use_triton", False),
        "use_cuda_fp16": bool_from_env("use_cuda_fp16", True)
    }
    return extension_kwargs

def dispatch_kwargs_from_env(model_config: ModelConfig):
    device_map = os.environ.get("device_map")
    if device_map is not None:
        try:
            device_map = json.loads(device_map)
        except json.decoder.JSONDecodeError as e:
            pass
    max_memory = os.environ.get("max_memory")
    if max_memory is not None:
        max_memory = json.loads(max_memory)
        
    dispatch_kwargs = {
        "device_map": device_map,
        "max_memory": max_memory
    }
    return dispatch_kwargs


def post_load_checkpoint(model: nn.Module):
    from vllm.model_executor.layers.attention import (PagedAttention, PagedAttentionWithRoPE, PagedAttentionWithALiBi)
    
    device = None
    for name, module in model.named_modules():
        if not isinstance(module, PagedAttention):
            for param in module.parameters():
                if param.is_cuda:
                    device = param.device
                    break
            if device is not None:
                break
    assert device is not None
    
    def set_tensor_to_device(module: nn.Module):
        if isinstance(module, (PagedAttention, PagedAttentionWithRoPE, PagedAttentionWithALiBi)):
            if isinstance(module, PagedAttention):
                module.head_mapping = module.head_mapping.to(device)
            if isinstance(module, PagedAttentionWithRoPE):
                module.rotary_emb.cos_sin_cache = module.rotary_emb.cos_sin_cache.to(device)
            else:
                # TODO: PagedAttentionWithALiBi
                pass
            torch.cuda.empty_cache()
            return
        
        for name, sub_module in module.named_children():
            set_tensor_to_device(sub_module)
    
    set_tensor_to_device(model)


# We have to rewrite `simple_dispatch_model`:
# 1. avoid early return if `"" in device_map` to add `AlignDevicesHook` to module
# 2. `io_same_device=False` for efficiency since input_metadata is always on `cuda:0`
def simple_dispatch_model(model, device_map):
    from accelerate.hooks import add_hook_to_module, AlignDevicesHook

    if "" in device_map:
        # d = device_map[""]
        # model = model.to(torch.device(d))
        # model.hf_device_map = device_map
        # return model
        pass
        
    tied_params = accelerate.utils.modeling.find_tied_parameters(model)
    if set(device_map.values()) == {"cpu"} or set(device_map.values()) == {"cpu", "disk"}:
        main_device = "cpu"
    else:
        main_device = [d for d in device_map.values() if d not in ["cpu", "disk"]][0]

    cpu_offload_group = [(n, d) for n, d in device_map.items() if d == "cpu"]
    prev_hook = None
    for idx, (n, d) in enumerate(cpu_offload_group):
        m = get_module_by_name_suffix(model, n)
        _, prev_hook = accelerate.cpu_offload_with_hook(m, execution_device=main_device, prev_module_hook=prev_hook)
    # set first cpu offload module's prev_module_hook to the last cpu offload module's hook
    if len(cpu_offload_group) > 1:
        get_module_by_name_suffix(model, cpu_offload_group[0][0])._hf_hook.prev_module_hook = prev_hook

    for n, d in device_map.items():
        m = get_module_by_name_suffix(model, n)
        if d != "cpu":
            d = torch.device(d)
            hook = AlignDevicesHook(d, io_same_device=False, place_submodules=True)
            add_hook_to_module(m, hook)
    accelerate.utils.modeling.retie_parameters(model, tied_params)
    model.hf_device_map = device_map

    return model


def get_model(model_config: ModelConfig) -> nn.Module:
    config = model_config.hf_config
    model_class = _get_model_architecture(config)
    
    # == step1: prepare configs and file names == #
    load_format = model_config.load_format
    use_safetensors = False
    fall_back_to_pt = False
    if load_format == "auto":
        use_safetensors = True
        fall_back_to_pt = True
    elif load_format == "safetensors":
        use_safetensors = True
    elif load_format == "pt":
        pass
    else:
        raise ValueError(f"Unsupported load_format for GPTQ: {load_format}")
    
    cached_file_kwargs = cached_kwargs_from_env(model_config)
    model_name_or_path = model_config.model
    quantize_config = BaseQuantizeConfig.from_pretrained(model_name_or_path, **cached_file_kwargs)
    quantize_config.model_name_or_path = model_name_or_path
    
    if quantize_config.model_file_base_name:
        possible_model_basenames = [quantize_config.model_file_base_name]
    else:
        possible_model_basenames = [f"gptq_model-{quantize_config.bits}bit-{quantize_config.group_size}g", "model"]
    
    # `prepare_hf_model_weights` may bring undesired hf folder or hf weights files, so just utilize it to parse `use_safetensors`
    _, _, use_safetensors = prepare_hf_model_weights(
        model_name_or_path,
        cache_dir=model_config.download_dir,
        use_safetensors=use_safetensors,
        fall_back_to_pt=fall_back_to_pt,
        revision=model_config.revision
    )
    
    if use_safetensors:
        extensions = [".safetensors"]
    else:
        extensions = [".bin", ".pt"]
    
    model_name_or_path = str(model_name_or_path)
    is_local = os.path.isdir(model_name_or_path)
    
    resolved_archive_file = None
    true_model_basename = None
    searched_files = []
    if is_local:
        for ext in extensions:
            for possible_model_basename in possible_model_basenames:
                model_save_name = os.path.join(model_name_or_path, possible_model_basename)
                searched_files.append(possible_model_basename + ext)
                if os.path.isfile(model_save_name + ext):
                    resolved_archive_file = model_save_name + ext
                    true_model_basename = possible_model_basename
                    break
    else:  # remote
        for ext in extensions:
            for possible_model_basename in possible_model_basenames:
                resolved_archive_file = cached_file(model_name_or_path, possible_model_basename + ext, **cached_file_kwargs)
                searched_files.append(possible_model_basename + ext)
                if resolved_archive_file is not None:
                    true_model_basename = possible_model_basename
                    break
    
    quantize_config.model_file_base_name = true_model_basename
    
    if resolved_archive_file is None:
        raise FileNotFoundError(f"Could not find a model in {model_name_or_path} with a name in {', '.join(searched_files)}. Please specify the argument model_basename to use a custom file name.")
            
    model_save_name = resolved_archive_file
    
    extension_kwargs = extension_kwargs_from_env(model_config)
    
    # == step2: convert model to gptq-model (replace Linear with QuantLinear) == #
    init_contexts = [
        _set_default_torch_dtype(model_config.dtype),
        accelerate.init_empty_weights(include_buffers=False)
    ]
    with ContextManagers(init_contexts):
        model = model_class(model_config.hf_config)
        
        layers = find_layers(model)
        ignore_layers = [model_class.lm_head_name] + model_class.outside_layer_modules
        for name in list(layers.keys()):
            if any([name.startswith(ignore_layer) for ignore_layer in ignore_layers]):
                logger.info(f"{name} not been quantized, will be ignored when make_quant.")
                del layers[name]
                
        make_quant(
            model,
            layers,
            quantize_config.bits,
            quantize_config.group_size,
            use_triton=extension_kwargs["use_triton"],
            disable_exllama=extension_kwargs["disable_exllama"],
            disable_exllamav2=extension_kwargs["disable_exllamav2"],
            use_cuda_fp16=extension_kwargs["use_cuda_fp16"],
            desc_act=quantize_config.desc_act,
            trainable=False # Exllama kernel does not support training
        )
        if hasattr(model, "tie_weights"):
            model.tie_weights()
            
    # == step3: load checkpoint and dispatch == #
    dispatch_kwargs = dispatch_kwargs_from_env(model_config)
    device_map, max_memory = dispatch_kwargs["device_map"], dispatch_kwargs["max_memory"]
    
    if isinstance(device_map, str) and device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
        raise ValueError(
            "If passing a string for `device_map`, please choose 'auto', 'balanced', 'balanced_low_0' or "
            "'sequential'."
        )
    if isinstance(device_map, dict):
        max_memory = None
    else:
        if not device_map and not max_memory:
            device_map = "auto"
        if not isinstance(device_map, dict) and device_map != "sequential":
            max_memory = accelerate.utils.get_balanced_memory(
                model=model,
                max_memory=max_memory,
                no_split_module_classes=[model_class.layer_type],
                low_zero=(device_map == "balanced_low_0")
            )
    if not isinstance(device_map, dict):
        device_map = accelerate.infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=[model_class.layer_type]
        )
        
    make_sure_no_tensor_in_meta_device(model, extension_kwargs["use_triton"], quantize_config.desc_act, quantize_config.group_size, bits=quantize_config.bits)
    
    accelerate.utils.modeling.load_checkpoint_in_model(
        model,
        checkpoint=model_save_name,
        device_map=device_map,
        offload_state_dict=True,
        offload_buffers=True
    )
    post_load_checkpoint(model)
    model = simple_dispatch_model(model, device_map)
    
    # RuntimeError: The temp_state buffer is too small in the exllama backend. Please call the exllama_set_max_input_length function to increase the buffer size. Example:
    # from auto_gptq import exllama_set_max_input_length
    # model = exllama_set_max_input_length(model, 4096)
    model = autogptq_post_init(model, use_act_order=quantize_config.desc_act, max_input_length=model_config.max_model_len)
    model.eval()
    
    # == step4: make model compatible with peft
    BaseGPTQForCausalLM.make_sure_compatible_with_peft(
        model, extension_kwargs["use_triton"], quantize_config.desc_act, quantize_config.group_size, bits=quantize_config.bits
    )
    
    torch.cuda.empty_cache()
    
    return model
