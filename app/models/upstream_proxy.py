from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import Request
from fastapi.responses import Response, StreamingResponse


def _normalize_api_root(base_url: str) -> str:
    """Normalize an upstream base URL to an OpenAI-compatible API root.

    Accepts either:
      - https://host/v1
      - https://host

    Returns an API root that ends with `/v1`.
    """

    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        return root
    return f"{root}/v1"


def _build_upstream_headers(
    *,
    inbound_headers: Mapping[str, str],
    api_key: str | None,
    extra_headers: Mapping[str, str] | None = None,
) -> dict[str, str]:
    headers: dict[str, str] = {}

    # Prefer server-side configured key; fall back to inbound Authorization if present.
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        auth = inbound_headers.get("authorization") or inbound_headers.get("Authorization")
        if auth:
            headers["Authorization"] = auth

    # Forward OpenAI metadata headers if present.
    for key in ("OpenAI-Organization", "OpenAI-Project", "x-request-id", "X-Request-Id"):
        value = inbound_headers.get(key) or inbound_headers.get(key.lower())
        if value and key not in headers:
            headers[key] = value

    if extra_headers:
        headers.update({k: v for k, v in extra_headers.items() if v})

    # Ensure JSON content type for non-stream requests; StreamingResponse will set its own.
    headers.setdefault("Content-Type", "application/json")
    return headers


@dataclass(frozen=True, slots=True)
class UpstreamConfig:
    provider: str
    api_root: str
    api_key: str | None
    timeout_sec: float
    extra_headers: Mapping[str, str] | None = None


class _UpstreamProxyBase:
    """Base for upstream proxy models.

    These models are *not* local inference handlers. Routes detect them and
    forward raw OpenAI-compatible payloads upstream.
    """

    is_proxy: bool = True
    supports_structured_outputs: bool = True

    def __init__(
        self,
        *,
        model_id: str,
        capabilities: list[str],
        owned_by: str,
        upstream: UpstreamConfig,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.name = model_id
        self.capabilities = capabilities
        self.owned_by = owned_by
        self.upstream = upstream
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._client_lock: asyncio.Lock | None = None

    async def aclose(self) -> None:
        if self._client is None:
            return
        await self._client.aclose()
        self._client = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client

        if self._client_lock is None:
            self._client_lock = asyncio.Lock()

        async with self._client_lock:
            if self._client is None:
                self._client = httpx.AsyncClient(timeout=self.upstream.timeout_sec, transport=self._transport)
        return self._client

    async def proxy_chat_completions(self, request: Request, payload: dict[str, Any]) -> Response:
        client = await self._ensure_client()
        return await _proxy_openai_endpoint(
            request=request,
            client=client,
            upstream=self.upstream,
            path="/chat/completions",
            payload=payload,
            stream=bool(payload.get("stream")),
            upstream_model_id=self.name,
        )

    async def proxy_embeddings(self, request: Request, payload: dict[str, Any]) -> Response:
        # Embeddings are non-streaming in OpenAI API.
        client = await self._ensure_client()
        return await _proxy_openai_endpoint(
            request=request,
            client=client,
            upstream=self.upstream,
            path="/embeddings",
            payload=payload,
            stream=False,
            upstream_model_id=self.name,
        )


async def _proxy_openai_endpoint(
    *,
    request: Request,
    client: httpx.AsyncClient,
    upstream: UpstreamConfig,
    path: str,
    payload: dict[str, Any],
    stream: bool,
    upstream_model_id: str,
) -> Response:
    api_root = upstream.api_root
    url = f"{api_root}{path}"

    # Always force upstream model id (the local model name may differ).
    forwarded = dict(payload)
    forwarded["model"] = upstream_model_id

    headers = _build_upstream_headers(
        inbound_headers=request.headers,
        api_key=upstream.api_key,
        extra_headers=upstream.extra_headers,
    )

    # If upstream expects SSE, keep the Accept header permissive.
    if stream:
        headers["Accept"] = "text/event-stream"

    # Non-streaming: simple passthrough with status/body.
    if not stream:
        resp = await client.post(url, json=forwarded, headers=headers)
        content = await resp.aread()
        media_type = resp.headers.get("content-type", "application/json")
        return Response(content=content, status_code=resp.status_code, media_type=media_type)

    # Streaming: keep the upstream connection open and yield bytes as they arrive.
    cm = client.stream("POST", url, json=forwarded, headers=headers)
    upstream_resp = await cm.__aenter__()

    if upstream_resp.status_code >= 400:
        try:
            content = await upstream_resp.aread()
        finally:
            await cm.__aexit__(None, None, None)
        media_type = upstream_resp.headers.get("content-type", "application/json")
        return Response(content=content, status_code=upstream_resp.status_code, media_type=media_type)

    media_type = upstream_resp.headers.get("content-type", "text/event-stream")

    async def _iter() -> Any:
        try:
            async for chunk in upstream_resp.aiter_bytes():
                # Best-effort: stop forwarding once the client disconnects.
                if await request.is_disconnected():
                    break
                yield chunk
        finally:
            await cm.__aexit__(None, None, None)

    return StreamingResponse(_iter(), status_code=upstream_resp.status_code, media_type=media_type)


def build_upstream_config(  # noqa: PLR0913 - explicit knobs for clarity
    *,
    provider: str,
    base_url: str,
    api_key: str | None,
    timeout_sec: float,
    extra_headers: Mapping[str, str] | None = None,
) -> UpstreamConfig:
    # Validate early to avoid confusing runtime errors later.
    api_root = _normalize_api_root(base_url)
    if not api_root.startswith(("http://", "https://")):
        raise ValueError(f"Upstream base_url must be http(s), got: {base_url!r}")
    if timeout_sec <= 0:
        raise ValueError("Upstream timeout_sec must be positive")
    return UpstreamConfig(
        provider=provider,
        api_root=api_root,
        api_key=api_key or None,
        timeout_sec=float(timeout_sec),
        extra_headers=extra_headers,
    )
