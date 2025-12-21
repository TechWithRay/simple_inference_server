from __future__ import annotations

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import api
from app.dependencies import get_model_registry
from app.models.openai_proxy import OpenAIChatProxyModel

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


class LocalDummyModel:
    dim = 2


def test_v1_models_sets_owned_by_for_proxy_models() -> None:
    transport = httpx.MockTransport(lambda _req: httpx.Response(200, json={}))
    proxy = OpenAIChatProxyModel(
        "gpt-4o-mini",
        config={"upstream_base_url": "https://upstream.test"},
        transport=transport,
    )
    client = TestClient(create_app({"local": LocalDummyModel(), "proxy-chat": proxy}))
    resp = client.get("/v1/models")
    assert resp.status_code == HTTP_OK
    data = {item["id"]: item for item in resp.json()["data"]}
    assert data["local"]["owned_by"] == "local"
    assert data["proxy-chat"]["owned_by"] == "openai"
