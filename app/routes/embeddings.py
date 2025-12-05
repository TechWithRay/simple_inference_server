from __future__ import annotations

import asyncio
import os
import threading
import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.batching import EmbeddingBatchQueueTimeoutError
from app.concurrency.limiter import (
    EMBEDDING_QUEUE_TIMEOUT_SEC,
    QUEUE_TIMEOUT_SEC,
    QueueFullError,
    QueueTimeoutError,
    ShuttingDownError,
    embedding_limiter,
    reset_queue_label,
    set_queue_label,
)
from app.dependencies import get_model_registry
from app.models.registry import ModelRegistry
from app.monitoring.metrics import observe_latency, record_request
from app.routes.common import (
    _ClientDisconnectedError,
    _RequestCancelledError,
    _run_work_with_client_cancel,
    _WorkTimeoutError,
)
from app.threadpool import get_embedding_count_executor, get_embedding_executor

router = APIRouter()


class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]
    encoding_format: str | None = Field(default="float", description="Only 'float' is supported")
    user: str | None = Field(default=None, description="OpenAI compatibility placeholder")


class EmbeddingObject(BaseModel):
    object: str = "embedding"
    index: int
    embedding: list[float]


class Usage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: int | None = None


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingObject]
    model: str
    usage: Usage


def _normalize_embedding_texts(req: EmbeddingRequest) -> list[str]:
    if req.encoding_format not in (None, "float"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only 'float' encoding_format is supported",
        )

    texts = [req.input] if isinstance(req.input, str) else list(req.input)
    max_batch = int(os.getenv("MAX_BATCH_SIZE", "32"))
    if len(texts) > max_batch:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch too large; max {max_batch} items",
        )

    max_text_chars = int(os.getenv("MAX_TEXT_CHARS", "20000"))
    for idx, t in enumerate(texts):
        if len(t) > max_text_chars:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Input at index {idx} exceeds max length {max_text_chars} chars",
            )

    return texts


async def _build_embedding_usage(model: Any, texts: list[str]) -> Usage:
    disable_usage_tokens = os.getenv("EMBEDDING_USAGE_DISABLE_TOKEN_COUNT", "0") != "0"
    if disable_usage_tokens:
        prompt_tokens = 0
    else:
        try:
            loop = asyncio.get_running_loop()
            prompt_tokens = await loop.run_in_executor(
                get_embedding_count_executor(),
                lambda: model.count_tokens(texts),
            )
        except Exception:
            prompt_tokens = 0

    return Usage(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens, completion_tokens=None)


async def _run_embedding_generation(  # noqa: PLR0913 - explicit kwargs for clarity
    *,
    registry: ModelRegistry,
    model_name: str,
    texts: list[str],
    request: Request,
    cancel_event: threading.Event,
    timeout: float,
) -> Any:
    try:
        model = registry.get(model_name)
    except KeyError as exc:
        record_request(model_name, "404")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_name} not found",
        ) from exc

    try:
        batcher = getattr(request.app.state, "batching_service", None)
        loop = asyncio.get_running_loop()
        if batcher is not None and getattr(batcher, "enabled", False):
            work_task = asyncio.ensure_future(
                batcher.enqueue(model_name, texts, cancel_event=cancel_event)
            )
        else:
            executor = get_embedding_executor()
            work_task = asyncio.ensure_future(
                loop.run_in_executor(
                    executor,
                    lambda: model.embed(texts, cancel_event=cancel_event),
                )
            )
        return await _run_work_with_client_cancel(
            request=request,
            work_task=work_task,
            cancel_event=cancel_event,
            timeout=timeout,
        )
    except _WorkTimeoutError as exc:
        cancel_event.set()
        record_request(model_name, "504")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Embedding generation timed out",
        ) from exc
    except _RequestCancelledError as exc:
        cancel_event.set()
        record_request(model_name, "499")
        raise HTTPException(
            status_code=status.HTTP_499_CLIENT_CLOSED_REQUEST,
            detail="Request cancelled",
        ) from exc
    except _ClientDisconnectedError as exc:
        cancel_event.set()
        record_request(model_name, "499")
        raise HTTPException(
            status_code=status.HTTP_499_CLIENT_CLOSED_REQUEST,
            detail="Client disconnected",
        ) from exc
    except EmbeddingBatchQueueTimeoutError as exc:
        record_request(model_name, "429")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Embedding batch queue wait exceeded",
            headers={"Retry-After": str(int(QUEUE_TIMEOUT_SEC))},
        ) from exc
    except Exception as exc:  # pragma: no cover - unexpected runtime failure
        record_request(model_name, "500")
        __import__("logging").getLogger(__name__).exception(
            "embedding_failed",
            extra={
                "model": model_name,
                "batch_size": len(texts),
                "device": getattr(registry.get(model_name), "device", None)
                if model_name in registry.list_models()
                else None,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Embedding generation failed",
        ) from exc


@router.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(  # noqa: PLR0912
    req: EmbeddingRequest,
    registry: Annotated[ModelRegistry, Depends(get_model_registry)],
    request: Request,
) -> EmbeddingResponse:
    texts = _normalize_embedding_texts(req)
    start = time.perf_counter()
    embed_timeout = float(os.getenv("EMBEDDING_GENERATE_TIMEOUT_SEC", "60"))
    cancel_event = threading.Event()
    label_token = set_queue_label(req.model or "embedding")
    limiter_cm = _get_embedding_limiter()
    limiter_ctx = limiter_cm() if callable(limiter_cm) else limiter_cm
    try:
        async with limiter_ctx:
            vectors = await _run_embedding_generation(
                registry=registry,
                model_name=req.model,
                texts=texts,
                request=request,
                cancel_event=cancel_event,
                timeout=embed_timeout,
            )
    except QueueFullError as exc:
        record_request(req.model, "429")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Embedding request queue full",
            headers={"Retry-After": str(int(EMBEDDING_QUEUE_TIMEOUT_SEC))},
        ) from exc
    except ShuttingDownError as exc:
        record_request(req.model, "503")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is shutting down",
        ) from exc
    except QueueTimeoutError as exc:
        record_request(req.model, "429")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Timed out waiting for embedding worker",
            headers={"Retry-After": str(int(EMBEDDING_QUEUE_TIMEOUT_SEC))},
        ) from exc
    finally:
        reset_queue_label(label_token)

    latency = time.perf_counter() - start
    observe_latency(req.model, latency)
    record_request(req.model, "200")
    __import__("logging").getLogger(__name__).info(
        "embedding_request",
        extra={
            "model": req.model,
            "latency_ms": round(latency * 1000, 2),
            "batch_size": len(texts),
            "status": 200,
        },
    )

    data = [
        EmbeddingObject(index=i, embedding=vec.tolist()) for i, vec in enumerate(vectors)
    ]
    usage_model = registry.get(req.model)
    usage = await _build_embedding_usage(usage_model, texts)
    return EmbeddingResponse(data=data, model=req.model, usage=usage)
def _get_embedding_limiter() -> Any:
    try:
        from app import api as api_module  # noqa: PLC0415 - local import to avoid circular import

        return getattr(api_module, "embedding_limiter", embedding_limiter)
    except Exception:
        return embedding_limiter
