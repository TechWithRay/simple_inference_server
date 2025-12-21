from __future__ import annotations

import json
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import api
from app.dependencies import get_model_registry
from app.models.openai_proxy import OpenAIChatProxyModel

HTTP_OK = 200
EXTRA_FIELD_VALUE = 123


class DummyRegistry:
    def __init__(self, models: dict[str, object]) -> None:
        self._models = models

    def get(self, name: str) -> object:
        if name not in self._models:
            raise KeyError
        return self._models[name]

    def list_models(self) -> list[str]:
        return list(self._models.keys())


def create_app(models: dict[str, object]) -> FastAPI:
    app = FastAPI()
    app.include_router(api.router)
    app.dependency_overrides[get_model_registry] = lambda: DummyRegistry(models)
    return app


def test_proxy_chat_passthrough_accepts_tool_role_and_preserves_response_fields() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content.decode("utf-8"))
        seen["body"] = body
        return httpx.Response(
            status_code=200,
            headers={"content-type": "application/json"},
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": body["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "extra_field": 123,
            },
        )

    transport = httpx.MockTransport(handler)
    proxy = OpenAIChatProxyModel(
        "gpt-4o-mini",
        config={"upstream_base_url": "https://upstream.test"},
        transport=transport,
    )

    client = TestClient(create_app({"proxy-chat": proxy}))
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "proxy-chat",
            "messages": [
                {"role": "user", "content": "hi"},
                # Tool role is not accepted by local schema; proxy must pass through.
                {"role": "tool", "content": "result"},
            ],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}},
        },
    )
    assert resp.status_code == HTTP_OK
    payload = resp.json()
    assert payload["extra_field"] == EXTRA_FIELD_VALUE

    # The proxy model id sent upstream should be the configured upstream model id.
    assert seen["body"]["model"] == "gpt-4o-mini"
    # Unknown fields must survive (transparent passthrough).
    assert "tools" in seen["body"]


def test_proxy_chat_stream_passthrough() -> None:
    sse = b'data: {"id":"1"}\n\ndata: [DONE]\n\n'

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            content=sse,
        )

    transport = httpx.MockTransport(handler)
    proxy = OpenAIChatProxyModel(
        "gpt-4o-mini",
        config={"upstream_base_url": "https://upstream.test"},
        transport=transport,
    )
    client = TestClient(create_app({"proxy-chat": proxy}))
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "proxy-chat", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as resp:
        assert resp.status_code == HTTP_OK
        body = b"".join(resp.iter_bytes())
        assert b"data: [DONE]" in body
