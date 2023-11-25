import sys
import os
import time
import json
import yaml
import atexit
import argparse
from typing import AsyncGenerator

import asyncio
import aiohttp
import requests
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
import uvicorn

from vllm.engine.arg_utils import EngineArgs, AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

from .model_utils import get_stop, get_dialogue_indicators, get_prompt


TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_REPORT_HEARTBEAT = 2  # seconds.
SLAVE_API = "/generate"

app = FastAPI()
engine = None


@atexit.register
def post_termination():
    load_balancer_addr = get_load_balancer_addr()
    local_addr = get_local_addr()
    num_requests = -1
    
    requests.post(load_balancer_addr, json={
        "api_url": local_addr,
        "num_requests": num_requests 
    })
    

@app.post(SLAVE_API)
async def batch_stream_generate(request: Request) -> Response:
    request_dict = await request.json()
    batch = request_dict.pop("batch", None)
    if batch is None:
        return Response("`batch` field must be a list of string!")
    
    histories = request_dict.pop("histories", [])
    if not histories:
        histories = [[] for _ in range(len(batch))]
    
    user_indicator, ai_indicator = get_dialogue_indicators(args)
    for i in range(len(batch)):
        batch[i] = get_prompt(
            batch[i], histories[i], user_indicator=user_indicator, ai_indicator=ai_indicator, sep=get_stop(args)
        )[0]
        
    req_ids = request_dict.pop("req_ids", None)
    if req_ids is None:
        for i in range(len(batch)):
            req_ids.append(random_uuid())
    
    decoding_args = request_dict["decoding_args"]
    sampling_params = [
        SamplingParams(**decoding_arg, stop=get_stop(args)) for decoding_arg in decoding_args
    ]
    if len(sampling_params) < len(batch):
        for _ in range(len(batch) - len(sampling_params)):
            sampling_params.append(
                SamplingParams(temperature=1.0, top_k=50, top_p=0.8, max_tokens=2048, stop=get_stop(args))
            )
    sampling_params = sampling_params[:len(batch)]
         
    streams = []
    for prompt, params, req_id in zip(batch, sampling_params, req_ids):
        arrival_time = time.monotonic()
        # add_request 其实是同步的
        stream = await engine.add_request(
            req_id, prompt, params, arrival_time=arrival_time
        )
        streams.append(stream)
        
    async def stream_results() -> AsyncGenerator[bytes, None]:
        rets = [{} for _ in range(len(streams))]
        finished = [False] * len(streams)
        while True:
            for i, stream in enumerate(streams):
                if finished[i]:
                    continue
                try:
                    request_output = await anext(stream)
                except StopAsyncIteration:
                    if not rets[i]:
                        rets[i]["req_id"] = "unaccessible"
                    finished[i] = True
                    rets[i]["finished"] = True
                    continue
                if request_output.finished:
                    finished[i] = True
                req_id = request_output.request_id
                prompt = request_output.prompt
                text_output = request_output.outputs[0].text
                rets[i] = {
                    "req_id": req_id, "prompt": prompt, "text": text_output, "finished": finished[i]
                }
            yield (json.dumps(rets) + "\0").encode("utf-8")
            
            if all(finished):
                break
            
    return StreamingResponse(stream_results())


def get_local_addr():
    host = "127.0.0.1" if args.host is None else args.host
    return f"http://{host}:{args.port}{SLAVE_API}"

def get_load_balancer_addr():
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config/config.yaml")
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    load_balancer = config["LoadBalancer"]
    host, port = load_balancer["host"], load_balancer["port"]
    api = load_balancer["heartbeatApi"]
    return f"http://{host}:{port}{api}"


async def report_heartbeat():
    load_balancer_addr = get_load_balancer_addr()
    local_addr = get_local_addr()

    async with aiohttp.ClientSession(headers={"User-Agent": "Test report heartbeat"}) as session:
        while True:
            await asyncio.sleep(TIMEOUT_TO_REPORT_HEARTBEAT)
            
            num_requests = len(engine._request_tracker._request_streams) + \
                        engine._request_tracker._new_requests.qsize() - \
                        engine._request_tracker._finished_requests.qsize()
            try:
                await session.post(
                    url=load_balancer_addr,
                    json={
                        "api_url": local_addr,
                        "num_requests": num_requests 
                    }
                )
            except:
                pass
            

# 启动时运行心跳任务
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(report_heartbeat())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
        
    args.trust_remote_code = True
    
    if '4bit' in args.model.lower():
        from vllm_mixin.gptq import enable_gptq_support
        
        enable_gptq_support(device_map={'': 0})

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE
    )
