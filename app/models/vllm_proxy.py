from __future__ import annotations

from typing import Any, cast

import httpx

from app.config import settings
from app.models.openai_proxy import _coerce_headers, _resolve_api_key, _resolve_timeout_sec
from app.models.upstream_proxy import _UpstreamProxyBase, build_upstream_config


class VLLMChatProxyModel(_UpstreamProxyBase):
    """Proxy a chat model to a vLLM OpenAI-compatible server."""

    def __init__(  # noqa: PLR0913 - explicit args for clarity
        self,
        hf_repo_id: str,
        device: str = "auto",
        config: dict[str, Any] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        cfg = config or {}

        base_url = str(cfg.get("upstream_base_url") or getattr(settings, "vllm_base_url", ""))
        if not base_url:
            raise ValueError("vLLM proxy requires upstream_base_url (or VLLM_BASE_URL env)")

        timeout_sec = _resolve_timeout_sec(cfg, getattr(settings, "vllm_proxy_timeout_sec", 60.0))

        # vLLM often runs without auth, but allow configuring one.
        api_key = _resolve_api_key(cfg, getattr(settings, "vllm_api_key", ""))

        # Allow arbitrary extra headers (e.g. routing headers on an ingress).
        extra_headers = _coerce_headers(cfg.get("upstream_headers"))

        upstream = build_upstream_config(
            provider="vllm",
            base_url=base_url,
            api_key=api_key,
            timeout_sec=timeout_sec,
            extra_headers=extra_headers,
        )
        super().__init__(
            model_id=hf_repo_id,
            capabilities=["chat-completion"],
            owned_by="vllm",
            upstream=upstream,
            transport=transport,
        )
        self.device = cast(Any, device)


class VLLMEmbeddingProxyModel(_UpstreamProxyBase):
    """Proxy an embeddings model to a vLLM OpenAI-compatible server."""

    def __init__(
        self,
        hf_repo_id: str,
        device: str = "auto",
        config: dict[str, Any] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        cfg = config or {}

        base_url = str(cfg.get("upstream_base_url") or getattr(settings, "vllm_base_url", ""))
        if not base_url:
            raise ValueError("vLLM proxy requires upstream_base_url (or VLLM_BASE_URL env)")

        timeout_sec = _resolve_timeout_sec(cfg, getattr(settings, "vllm_proxy_timeout_sec", 60.0))
        api_key = _resolve_api_key(cfg, getattr(settings, "vllm_api_key", ""))
        extra_headers = _coerce_headers(cfg.get("upstream_headers"))

        upstream = build_upstream_config(
            provider="vllm",
            base_url=base_url,
            api_key=api_key,
            timeout_sec=timeout_sec,
            extra_headers=extra_headers,
        )
        super().__init__(
            model_id=hf_repo_id,
            capabilities=["text-embedding"],
            owned_by="vllm",
            upstream=upstream,
            transport=transport,
        )
        self.device = cast(Any, device)
