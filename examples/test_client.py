"""Example Python client for vllm.entrypoints.api_server"""

import argparse
import json
import urllib.parse
from typing import Iterable, List

import requests


def clear_line(n: int = 1) -> None:
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)


def get_stream_http_request(
    prompt: str,
    api_url: str,
    temperature: float=1.0,
    top_k: int=50,
    top_p: float=0.8,
    max_tokens: int=2048
) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    history = [("中国的首都是哪里？", "中国的首都是北京。")]
    history = urllib.parse.quote(json.dumps(history))
    api = f"{api_url}?query={prompt}&history={history}&temperature={temperature}&top_k={top_k}&top_p={top_p}&max_tokens={max_tokens}"
    print(f"test api: {api}")
    response = requests.get(api, headers=headers, stream=True)
    return response


def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    for chunk in response.iter_lines(
        chunk_size=8192,
        decode_unicode=False,
        delimiter=b"\0"
    ):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data
            yield output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--prompt", type=str, default="那俄罗斯的呢？回答得详细一点。")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_tokens", type=int, default=2048)
    args = parser.parse_args()
    
    prompt = args.prompt
    api_url = f"http://{args.host}:{args.port}/generate"

    print(f"Prompt: {prompt!r}\n", flush=True)
    response = get_stream_http_request(prompt, api_url, args.temperature, args.top_k, args.top_p, args.max_tokens)

    num_printed_lines = 0
    for h in get_streaming_response(response):
        clear_line(num_printed_lines)
        num_printed_lines = 0
        for i, line in enumerate(h):
            num_printed_lines += 1
        #     print(f"Beam candidate {i}: {line!r}", flush=True)
        print(h)
    