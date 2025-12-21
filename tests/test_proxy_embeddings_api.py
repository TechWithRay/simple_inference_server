from __future__ import annotations

import json
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import api
from app.dependencies import get_model_registry
from app.models.openai_proxy import OpenAIEmbeddingProxyModel

HTTP_OK = 200


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


def test_proxy_embeddings_passthrough_preserves_payload_and_response_fields() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/embeddings"
        body = json.loads(request.content.decode("utf-8"))
        seen["body"] = body
        return httpx.Response(
            status_code=200,
            headers={"content-type": "application/json"},
            json={
                "object": "list",
                "model": body["model"],
                "data": [{"object": "embedding", "index": 0, "embedding": [0.0, 1.0]}],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
                "extra_field": "kept",
            },
        )

    transport = httpx.MockTransport(handler)
    proxy = OpenAIEmbeddingProxyModel(
        "text-embedding-3-small",
        config={"upstream_base_url": "https://upstream.test"},
        transport=transport,
    )

    client = TestClient(create_app({"proxy-embed": proxy}))
    resp = client.post(
        "/v1/embeddings",
        json={
            "model": "proxy-embed",
            "input": "hello",
            "encoding_format": "float",
            "some_vendor_field": {"a": 1},
        },
    )
    assert resp.status_code == HTTP_OK
    payload = resp.json()
    assert payload["extra_field"] == "kept"

    # Upstream should receive the actual proxy upstream model id.
    assert seen["body"]["model"] == "text-embedding-3-small"
    assert "some_vendor_field" in seen["body"]
