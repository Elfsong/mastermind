#!/usr/bin/env python3
"""Small Anthropic Messages -> OpenAI chat/completions proxy for Claude Code.

Claude Code speaks Anthropic's /v1/messages API. Featherless exposes an
OpenAI-compatible /v1/chat/completions API. This proxy translates the subset of
Anthropic messages and tool-use payloads that Claude Code needs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from typing import Any

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse


# Featherless GLM-5.1 advertises a large context window, but direct probes showed
# stable responses up to roughly 120k prompt tokens and 504s near the 180k range.
MODEL_CONTEXT_WINDOW = int(os.environ.get("FEATHERLESS_CONTEXT_WINDOW", "120000"))
MODEL_MAX_OUTPUT_TOKENS = int(os.environ.get("FEATHERLESS_MAX_OUTPUT_TOKENS", "8192"))
MAX_TOKENS_CAP = int(os.environ.get("FEATHERLESS_MAX_TOKENS_CAP", str(MODEL_MAX_OUTPUT_TOKENS)))


app = FastAPI()


def model_metadata(model: str | None = None) -> dict[str, Any]:
    model_id = featherless_model_name(model)
    return {
        "id": model_id,
        "type": "model",
        "display_name": model_id,
        "created_at": "2026-06-23T00:00:00Z",
        "contextWindow": MODEL_CONTEXT_WINDOW,
        "maxOutputTokens": MODEL_MAX_OUTPUT_TOKENS,
    }


def text_from_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text") or ""))
                elif block.get("type") == "tool_result":
                    parts.append(text_from_content(block.get("content")))
                else:
                    parts.append(json.dumps(block, ensure_ascii=False))
        return "\n".join(part for part in parts if part)
    return json.dumps(content, ensure_ascii=False)


def anthropic_messages_to_openai(payload: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    system = payload.get("system")
    if system:
        messages.append({"role": "system", "content": text_from_content(system)})

    for message in payload.get("messages") or []:
        role = message.get("role")
        content = message.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            messages.append({"role": role, "content": text_from_content(content)})
            continue

        if role == "assistant":
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    text_parts.append(str(block))
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    text_parts.append(str(block.get("text") or ""))
                elif block_type == "tool_use":
                    tool_calls.append(
                        {
                            "id": str(block.get("id") or f"toolu_{uuid.uuid4().hex}"),
                            "type": "function",
                            "function": {
                                "name": str(block.get("name") or ""),
                                "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                            },
                        }
                    )
            out: dict[str, Any] = {
                "role": "assistant",
                "content": "\n".join(part for part in text_parts if part) or None,
            }
            if tool_calls:
                out["tool_calls"] = tool_calls
            messages.append(out)
            continue

        pending_text: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                pending_text.append(str(block))
                continue
            block_type = block.get("type")
            if block_type == "text":
                pending_text.append(str(block.get("text") or ""))
            elif block_type == "tool_result":
                if pending_text:
                    messages.append({"role": "user", "content": "\n".join(pending_text)})
                    pending_text = []
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(block.get("tool_use_id") or ""),
                        "content": text_from_content(block.get("content")),
                    }
                )
            else:
                pending_text.append(json.dumps(block, ensure_ascii=False))
        if pending_text:
            messages.append({"role": role or "user", "content": "\n".join(pending_text)})
    return messages


def anthropic_tools_to_openai(payload: dict[str, Any]) -> list[dict[str, Any]] | None:
    tools = payload.get("tools")
    if not isinstance(tools, list) or not tools:
        return None
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description") or "",
                    "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
                },
            }
        )
    return converted or None


def featherless_model_name(model: str | None) -> str:
    model = model or os.environ.get("FEATHERLESS_MODEL", "zai-org/GLM-5.1")
    if model.startswith("featherless_ai/"):
        return model.removeprefix("featherless_ai/")
    return model


def openai_request_from_anthropic(payload: dict[str, Any]) -> dict[str, Any]:
    max_tokens = payload.get("max_tokens") or MAX_TOKENS_CAP
    try:
        max_tokens = min(int(max_tokens), MAX_TOKENS_CAP)
    except Exception:
        max_tokens = MAX_TOKENS_CAP

    request: dict[str, Any] = {
        "model": featherless_model_name(payload.get("model")),
        "messages": anthropic_messages_to_openai(payload),
        "max_tokens": max_tokens,
    }
    if "temperature" in payload:
        request["temperature"] = payload["temperature"]
    if "top_p" in payload:
        request["top_p"] = payload["top_p"]
    tools = anthropic_tools_to_openai(payload)
    if tools:
        request["tools"] = tools
    return request


def call_featherless(payload: dict[str, Any]) -> dict[str, Any]:
    api_key = os.environ.get("FEATHERLESS_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="FEATHERLESS_API_KEY is not set")
    base_url = os.environ.get("FEATHERLESS_BASE_URL", "https://api.featherless.ai/v1").rstrip("/")
    url = f"{base_url}/chat/completions"
    try:
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=int(os.environ.get("FEATHERLESS_REQUEST_TIMEOUT", "900")),
        )
    except requests.RequestException as exc:
        print(f"upstream request failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise HTTPException(status_code=502, detail=f"upstream request failed: {exc}") from exc
    if response.status_code >= 400:
        print(
            f"upstream returned {response.status_code}: {response.text[:2000]}",
            file=sys.stderr,
            flush=True,
        )
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.json()


def openai_response_to_anthropic(data: dict[str, Any], model: str) -> dict[str, Any]:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content_blocks: list[dict[str, Any]] = []
    if message.get("content"):
        content_blocks.append({"type": "text", "text": str(message.get("content"))})
    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        raw_args = function.get("arguments") or "{}"
        try:
            args = json.loads(raw_args)
        except Exception:
            args = {"arguments": raw_args}
        content_blocks.append(
            {
                "type": "tool_use",
                "id": str(tool_call.get("id") or f"toolu_{uuid.uuid4().hex}"),
                "name": str(function.get("name") or ""),
                "input": args,
            }
        )
    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    usage = data.get("usage") or {}
    stop_reason = "tool_use" if any(block.get("type") == "tool_use" for block in content_blocks) else "end_turn"
    finish_reason = choice.get("finish_reason")
    if finish_reason == "length":
        stop_reason = "max_tokens"

    return {
        "id": data.get("id") or f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "model_info": {
            "contextWindow": MODEL_CONTEXT_WINDOW,
            "maxOutputTokens": MODEL_MAX_OUTPUT_TOKENS,
        },
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
        },
    }


def sse_event(event: str, payload: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


def stream_anthropic_message(message: dict[str, Any]):
    start = {**message, "content": [], "stop_reason": None, "stop_sequence": None}
    yield sse_event("message_start", {"type": "message_start", "message": start})
    for index, block in enumerate(message["content"]):
        if block["type"] == "text":
            yield sse_event(
                "content_block_start",
                {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}},
            )
            text = block.get("text") or ""
            if text:
                yield sse_event(
                    "content_block_delta",
                    {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": text}},
                )
        elif block["type"] == "tool_use":
            yield sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": block["id"],
                        "name": block["name"],
                        "input": {},
                    },
                },
            )
            yield sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(block.get("input") or {}, ensure_ascii=False),
                    },
                },
            )
        yield sse_event("content_block_stop", {"type": "content_block_stop", "index": index})
    yield sse_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
            "usage": {"output_tokens": message["usage"]["output_tokens"]},
        },
    )
    yield sse_event("message_stop", {"type": "message_stop"})


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "contextWindow": MODEL_CONTEXT_WINDOW,
        "maxOutputTokens": MODEL_MAX_OUTPUT_TOKENS,
        "maxTokensCap": MAX_TOKENS_CAP,
    }


@app.get("/v1/models")
def models() -> dict[str, Any]:
    model = os.environ.get("FEATHERLESS_MODEL", "zai-org/GLM-5.1")
    metadata = model_metadata(model)
    return {
        "data": [metadata],
        "has_more": False,
        "first_id": metadata["id"],
        "last_id": metadata["id"],
    }


@app.get("/v1/models/{model_id:path}")
def model(model_id: str) -> dict[str, Any]:
    return model_metadata(model_id)


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request) -> dict[str, int]:
    payload = await request.json()
    text = json.dumps(payload.get("messages") or "", ensure_ascii=False)
    return {"input_tokens": max(1, len(text) // 4)}


@app.post("/v1/messages")
async def messages(request: Request):
    payload = await request.json()
    openai_payload = openai_request_from_anthropic(payload)
    data = call_featherless(openai_payload)
    message = openai_response_to_anthropic(data, model=openai_payload["model"])
    if payload.get("stream"):
        return StreamingResponse(stream_anthropic_message(message), media_type="text/event-stream")
    return JSONResponse(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8877)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level=os.environ.get("PROXY_LOG_LEVEL", "info"))


if __name__ == "__main__":
    main()
