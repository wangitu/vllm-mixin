<h1 align="center">VLLM-Mixin</h1>
<p align="center" width="100%">
<a ><img src="assets/logo.png" alt="VLLM-Mixin" style="width: 40%; margin: auto;"></a>
</p>
<h3 align="center">An easy mixin with just a single line of code to seamlessly integrate powerful functionalities into <a href="https://github.com/vllm-project/vllm">VLLM</a>.</h3>
<h4 align="center">
    <p>
        <b>VLLM-Minxin</b> |
        <a href="https://github.com/vllm-project/vllm/blob/main/README.md">VLLM-README.md</a>
    </p>
</h4>

---

*Latest News* 🔥
- [2023/11] We initially released VLLM-Mixin.

---

## Installation

### Prerequisites
Before installation, ensure that [vllm](https://github.com/vllm-project/vllm) is installed.

### Install from source
Clone the source code:
```Bash
git clone https://github.com/wangitu/vllm-mixin.git && cd vllm-mixin
```
Then, install from source as an editable package:
```Bash
pip install -v -e .
```

## Major mixins

### auto_gptq
To mixin auto_gptq, ensure that **auto_gptq** (Refer to either [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) or [unpadded-AutoGPTQ](https://github.com/wangitu/unpadded-AutoGPTQ)) is installed. Once auto_gptq is activated:

1. Unleash the full potential of auto_gptq through just a single line of code: `enable_gptq_support()`. This seamless integration with VLLM allows you to **freely configure the quantized model via arguments specification** while reserving VLLM's state-of-the-art serving throughput. 
2. Flexible model loading strategy. You could specify `device_map` and `max_memory` to **achieve highly customized model parallelism** regardless of VLLM's "first GPU first" placement strategy.

<details>
  <summary>Advanced usages (click to expand)</summary>

```Python
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
    """
    Arguments group:
    
    force_download, resume_download, proxies, local_files_only, local_files_only,
    use_auth_token, subfolder, _commit_hash
    
    These arguments are used to determine the model loading strategy, such as loading from a local or remote repository.
    Same with auto_gptq.from_quantized.
    
    
    Arguments group:
    disable_exllama, disable_exllamav2, use_triton, use_cuda_fp16
    
    These arguments are used to configure cuda kernel to further speedup inference.
    
    
    Arguments group:
    device_map, max_memory
    
    These arguments are used to customize model parallelism.
    """
```

</details>

### smart_server
With smart_server, you can **effortlessly build a lightning-fast server**, powered by the VLLM engine. We provide:

1. A load balancer to **smartly dispatch user requests to different slave servers**.
2. Serving which is capable of **responding to batch requests in a streaming manner** and unleashing the full efficiency of the VLLM.

[Usages](#smartserver)

### todo


## Quick start

### auto_gptq (see examples/test_gptq.py)
Below is an example of activating auto_gptq in VLLM for lower memory usage and faster inference:
```Python
import sys
import os
import fire

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from vllm_mixin.gptq import enable_gptq_support


texts = [
    "### Instruction:\n迪士尼有哪些知名的动画形象？\n\n### Response:\n",
    "### Instruction:\n什么是做多？什么是做空？\n\n### Response:\n"
]


def main(
    quantized_model_dir: str,
    custom_text: str = None
):
    # one line to activate auto_gptq support
    enable_gptq_support(device_map={'': 0})
    
    tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir, use_fast=False, trust_remote_code=True)
  
    model = LLM(quantized_model_dir, trust_remote_code=True, seed=42)
    
    global texts
    if custom_text is not None and isinstance(custom_text, str):
        texts = [custom_text]
        
    sampling_params = SamplingParams(
        temperature=1.0, top_k=50, top_p=0.8, max_tokens=2048, stop_token_ids=[tokenizer.eos_token_id]
    )
    
    outputs = model.generate(texts, sampling_params=sampling_params)
    for text, output in zip(texts, outputs):
        print("prompt:", text.split("### Instruction:\n")[-1].split("### Response:\n")[0].strip())
        print("response:", output.outputs[0].text, "\n" + "=" * 20 + "\n")


if __name__ == '__main__':
    fire.Fire(main)
```

### <a id="smartserver"></a>smart_server

To experience smart and lightning-fast serving, follow the steps below:

1. Start the load balancer: `python -m vllm_mixin.smart_server.load_balancer`
2. Initiate several serving individually:
    ```Bash
    CUDA_VISIBLE_DEVICES=0 python -m vllm_mixin.smart_server.api_server --model <your model> --host 127.0.0.1 --port 8000
    CUDA_VISIBLE_DEVICES=1 python -m vllm_mixin.smart_server.api_server --model <your model> --host 127.0.0.1 --port 8001
    ...
    ```
    In the script above, use `host` to specify the server's host, `port` to designate the port. The remaining parameters (such as `model`) are consistent with VLLM's `EngineArgs`.
3. **(Optional)** Test the server with: `python test_client.py` (see examples/test_client.py)

### todo
