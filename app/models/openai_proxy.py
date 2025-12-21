from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, cast

import httpx

from app.config import settings
from app.models.upstream_proxy import _UpstreamProxyBase, build_upstream_config


def _coerce_headers(value: object) -> Mapping[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("upstream_headers must be a mapping")
    out: dict[str, str] = {}
    for k, v in value.items():
        if k is None or v is None:
            continue
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError("upstream_headers keys/values must be strings")
        if k and v:
            out[k] = v
    return out


def _resolve_api_key(cfg: Mapping[str, Any], default_key: str) -> str | None:
    # Prefer env indirection so keys don't end up in YAML.
    env_name = cfg.get("upstream_api_key_env")
    if isinstance(env_name, str) and env_name:
        return os.getenv(env_name) or None

    key = cfg.get("upstream_api_key")
    if isinstance(key, str) and key:
        return key

    return default_key or None


def _resolve_timeout_sec(cfg: Mapping[str, Any], default_timeout: float) -> float:
    value = cfg.get("upstream_timeout_sec")
    if value is None:
        return float(default_timeout)
    return float(value)


class OpenAIChatProxyModel(_UpstreamProxyBase):
    """Proxy a chat model to an OpenAI-compatible upstream (e.g. OpenAI)."""

    def __init__(  # noqa: PLR0913 - explicit args for clarity
        self,
        hf_repo_id: str,
        device: str = "auto",
        config: dict[str, Any] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        cfg = config or {}
        base_url = str(
            cfg.get("upstream_base_url") or getattr(settings, "openai_base_url", "https://api.openai.com/v1")
        )
        timeout_sec = _resolve_timeout_sec(cfg, getattr(settings, "openai_proxy_timeout_sec", 60.0))
        api_key = _resolve_api_key(cfg, getattr(settings, "openai_api_key", ""))
        extra_headers = _coerce_headers(cfg.get("upstream_headers"))

        upstream = build_upstream_config(
            provider="openai",
            base_url=base_url,
            api_key=api_key,
            timeout_sec=timeout_sec,
            extra_headers=extra_headers,
        )
        super().__init__(
            model_id=hf_repo_id,
            capabilities=["chat-completion"],
            owned_by="openai",
            upstream=upstream,
            transport=transport,
        )
        self.device = cast(Any, device)  # unused by proxy path, but helps logs


class OpenAIEmbeddingProxyModel(_UpstreamProxyBase):
    """Proxy an embeddings model to an OpenAI-compatible upstream (e.g. OpenAI)."""

    def __init__(
        self,
        hf_repo_id: str,
        device: str = "auto",
        config: dict[str, Any] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        cfg = config or {}
        base_url = str(
            cfg.get("upstream_base_url") or getattr(settings, "openai_base_url", "https://api.openai.com/v1")
        )
        timeout_sec = _resolve_timeout_sec(cfg, getattr(settings, "openai_proxy_timeout_sec", 60.0))
        api_key = _resolve_api_key(cfg, getattr(settings, "openai_api_key", ""))
        extra_headers = _coerce_headers(cfg.get("upstream_headers"))

        upstream = build_upstream_config(
            provider="openai",
            base_url=base_url,
            api_key=api_key,
            timeout_sec=timeout_sec,
            extra_headers=extra_headers,
        )
        super().__init__(
            model_id=hf_repo_id,
            capabilities=["text-embedding"],
            owned_by="openai",
            upstream=upstream,
            transport=transport,
        )
        self.device = cast(Any, device)
