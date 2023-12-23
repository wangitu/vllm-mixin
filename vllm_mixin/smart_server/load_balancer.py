import sys
import os
import time
import json
import yaml
import uuid
import random
import logging
from typing import List, Tuple, Optional

import requests
from fastapi import FastAPI, Request, Depends
from fastapi.responses import Response, StreamingResponse
import uvicorn
import asyncio


app = FastAPI()
host = None
port = None
heartbeat_api = None
api_urls = {} # 从节点的 api: num_request

concurrency = 50  # 并发queries数量
wait_period = 1  # 等待时间

queries_queue = asyncio.Queue()
sentinel = object()


class QueryProcessor:
    @staticmethod
    async def process(queries: List[Tuple]):
        batch, histories, decoding_args, result_queues = zip(*queries)
        
        id2req = {str(uuid.uuid4().hex): i for i in range(len(batch))}
        pload = {
            "batch": batch,
            "histories": histories,
            "decoding_args": decoding_args,
            "req_ids": list(id2req.keys())
        }
        
        min_num_requests = min(api_urls.values())
        min_api_urls = [api_url for api_url, num_requests in api_urls.items() if num_requests == min_num_requests]
        api_url = min_api_urls[random.randrange(0, len(min_api_urls))]
        
        logging.info("available api uris:\n" + '\t'.join(map(lambda x: f"{x[0]}: {x[1]}", api_urls.items())))
        logging.info(f"dispatch to: {api_url}\n")
        
        async with aiohttp.ClientSession() as session:
            response = session.post(api_url, json=pload)
            async for chunk in response.content.iter_chunked(8192):
                if chunk:
                    data = json.loads(chunk.decode("utf-8").rstrip('\x00'))
                    for ret in data:
                        result_queue = result_queues[id2req[ret["req_id"]]]
                        await result_queue.put(ret["text"])
                        if ret["finished"]:
                            await result_queue.put(sentinel)
                        await asyncio.sleep(0)


async def decoding_arguments(temperature: float=1.0, top_k: int=50, top_p:float=0.8, max_tokens: int=2048):
    return {
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "max_tokens": max_tokens
    }


@app.get("/generate")
async def query(query: str, history: str=None, decoding_arg: dict=Depends(decoding_arguments)):
    if history is None:
        history = []
    else:
        try:
            history = json.loads(history)
        except json.decoder.JSONDecodeError as e:
            return Response("`history` must be a JSON string of List[(query, answer)]", status_code=400)
        if not isinstance(history, list) or \
            (len(history) > 0 and \
                (not isinstance(history[0], (list, tuple)) or \
                    len(history[0]) < 2 or not isinstance(history[0][0], str))):
            return Response("`history` must be a JSON string of List[(query, answer)]", status_code=400)
    
    async def stream_result_queue(result_queue):
        while True:
            result = await result_queue.get()
            if result is sentinel:
                break
            yield (json.dumps(result) + '\0').encode('utf-8')

    result_queue = asyncio.Queue()
    await queries_queue.put((query, history, decoding_arg, result_queue))
    return StreamingResponse(stream_result_queue(result_queue))


@app.get("/test")
async def test(shuffle: bool=False):
    question_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data/questions.txt")
    with open(question_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    questions = [line.strip() for line in lines]
    
    if shuffle:
        random.shuffle(questions)
    
    first_question = questions[0]
    first_queue = asyncio.Queue()
    other_queues = []
    # `put_nowait` to avoid yielding execution
    queries_queue.put_nowait((first_question, [], {"max_tokens": 256}, first_queue))
    for question in questions[1:]:
        result_queue = asyncio.Queue()
        queries_queue.put_nowait((question, [], {"max_tokens": 256}, result_queue))
        other_queues.append(result_queue)
        
    await asyncio.sleep(0)
    
    start = time.time()
    
    first_result = None
    while True:
        result = await first_queue.get()
        if result is sentinel:
            break
        first_result = result
    for queue in other_queues:
        while (result := await queue.get()) is not sentinel:
            continue
    
    print(f"first question:\n{first_question}\n")
    print(f"first result:\n{first_result}")
    
    return f"{len(questions)} questions took {time.time() - start} seconds"


async def process_queries():
    while True:
        await asyncio.sleep(wait_period)
        all_queries = []
        while not queries_queue.empty():
            all_queries.append(await queries_queue.get())
        for i in range(0, len(all_queries), concurrency):
            chunk = all_queries[i:i + concurrency]
            asyncio.create_task(QueryProcessor.process(chunk))


# 启动时开始初始化 logging, 运行 process_queries 任务, 监听心跳
@app.on_event("startup")
async def startup_event():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
        datefmt='%m-%d %H:%M'
    )
    
    asyncio.create_task(process_queries())
    
    
    @app.post(heartbeat_api)
    async def collect_heartbeat(heartbeat: Request):
        heartbeat_dict = await heartbeat.json()
        slave_url = heartbeat_dict.pop("api_url", None)
        if slave_url is None:
            return
        num_request = heartbeat_dict.pop("num_requests", 0)
        try:
            num_request = int(num_request)
        except ValueError:
            # bad report, temporarily set unavailable
            num_request = float('inf')
        
        api_urls[slave_url] = num_request
        # terminate connection, del it
        if num_request < 0:
            del api_urls[slave_url]


if __name__ == "__main__":
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config/config.yaml")
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    load_balancer = config["LoadBalancer"]
    host, port = load_balancer["host"], load_balancer["port"]
    heartbeat_api = load_balancer["heartbeatApi"]
    
    uvicorn.run(app, host=host, port=port)
    
