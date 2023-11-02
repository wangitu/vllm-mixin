import sys
import os
import fire

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from vllm_mixin.gptq import enable_gptq_support


texts = [
    "<|im_start|>### Instruction:\n你是谁？\n\n### Response:\n",
    "<|im_start|>### Instruction:\n什么是做多？什么是做空？\n\n### Response:\n"
]


def main(
    quantized_model_dir: str,
    custom_text: str = None
):
    enable_gptq_support(device_map={'': 0})
    
    tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token = '<|endoftext|>'
    tokenizer.padding_side = 'left'
    
    model = LLM(quantized_model_dir, trust_remote_code=True, seed=42)
    
    global texts
    if custom_text is not None and isinstance(custom_text, str):
        texts = [custom_text]
        
    sampling_params = SamplingParams(
        temperature=1.0, top_k=50, top_p=0.8, max_tokens=1024, stop_token_ids=[tokenizer.eos_token_id]
    )
    
    outputs = model.generate(texts, sampling_params=sampling_params)
    for text, output in zip(texts, outputs):
        print("prompt:", text.split("### Instruction:\n")[-1].split("### Response:\n")[0].strip())
        print("response:", output.outputs[0].text, "\n" + "=" * 20 + "\n")


if __name__ == '__main__':
    fire.Fire(main)
    