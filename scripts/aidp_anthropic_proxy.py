#!/usr/bin/env python3
"""Anthropic Messages -> ByteDance AIDP crawl proxy for Claude Code."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from typing import Any
from urllib.parse import urlencode

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

import featherless_anthropic_proxy as anthropic_proxy


MAX_TOKENS_CAP = int(os.environ.get("AIDP_MAX_TOKENS_CAP", "32768"))

app = FastAPI()


def aidp_model_name(model: str | None) -> str:
    return model or os.environ.get("AIDP_MODEL", "glm-4.7")


def openai_request_from_anthropic(payload: dict[str, Any]) -> dict[str, Any]:
    max_tokens = payload.get("max_tokens") or MAX_TOKENS_CAP
    try:
        max_tokens = min(int(max_tokens), MAX_TOKENS_CAP)
    except Exception:
        max_tokens = MAX_TOKENS_CAP

    request: dict[str, Any] = {
        "model": aidp_model_name(payload.get("model")),
        "messages": anthropic_proxy.anthropic_messages_to_openai(payload),
        "max_tokens": max_tokens,
        "stream": False,
    }
    if "temperature" in payload:
        request["temperature"] = payload["temperature"]
    if "top_p" in payload:
        request["top_p"] = payload["top_p"]
    tools = anthropic_proxy.anthropic_tools_to_openai(payload)
    if tools:
        request["tools"] = tools
    return request


def call_aidp(payload: dict[str, Any]) -> dict[str, Any]:
    api_key = os.environ.get("AIDP_AK")
    if not api_key:
        raise HTTPException(status_code=500, detail="AIDP_AK is not set")

    base_url = os.environ.get(
        "AIDP_CRAWL_URL",
        "https://aidp.bytedance.net/api/modelhub/online/v2/crawl",
    )
    url = f"{base_url}?{urlencode({'ak': api_key})}"
    logid = f"claude-code-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"
    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json", "X-TT-LOGID": logid},
            json=payload,
            timeout=int(os.environ.get("AIDP_REQUEST_TIMEOUT", "900")),
        )
    except requests.RequestException as exc:
        print(f"upstream request failed: {type(exc).__name__}", file=sys.stderr, flush=True)
        raise HTTPException(status_code=502, detail=f"upstream request failed: {type(exc).__name__}") from exc
    if response.status_code >= 400:
        print(
            f"upstream returned {response.status_code} logid={logid}: {response.text[:2000]}",
            file=sys.stderr,
            flush=True,
        )
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.json()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request) -> dict[str, int]:
    payload = await request.json()
    text = json.dumps(payload.get("messages") or "", ensure_ascii=False)
    return {"input_tokens": max(1, len(text) // 4)}


@app.post("/v1/messages")
async def messages(request: Request):
    payload = await request.json()
    openai_payload = openai_request_from_anthropic(payload)
    data = call_aidp(openai_payload)
    message = anthropic_proxy.openai_response_to_anthropic(data, model=openai_payload["model"])
    if payload.get("stream"):
        return StreamingResponse(
            anthropic_proxy.stream_anthropic_message(message),
            media_type="text/event-stream",
        )
    return JSONResponse(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8878)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level=os.environ.get("PROXY_LOG_LEVEL", "info"))


if __name__ == "__main__":
    main()
