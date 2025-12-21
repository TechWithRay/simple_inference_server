from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from app.dependencies import get_model_registry
from app.state import WarmupStatus

router = APIRouter()


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "local"
    embedding_dimensions: int | None = None


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


class WarmupDetails(BaseModel):
    required: bool
    completed: bool
    failures: list[str] | None = None
    ok_models: list[str] | None = None
    capabilities: dict[str, dict[str, bool]] | None = None


class QueueDepth(BaseModel):
    model: str
    size: int
    max_size: int | None = None


class HealthResponse(BaseModel):
    status: str
    models: list[str] | None = None
    warmup_failures: list[str] | None = None
    warmup: WarmupDetails | None = None
    chat_batch_queues: list[QueueDepth] | None = None
    embedding_batch_queues: list[QueueDepth] | None = None
    runtime_config: dict[str, Any] | None = None


@router.get("/v1/models", response_model=ModelsResponse)
async def list_models(registry: Annotated[Any, Depends(get_model_registry)]) -> ModelsResponse:
    models: list[ModelInfo] = []
    for name in registry.list_models():
        model = registry.get(name)
        dim = getattr(model, "dim", None)
        owned_by = getattr(model, "owned_by", "local")
        models.append(ModelInfo(id=name, owned_by=owned_by, embedding_dimensions=dim))
    return ModelsResponse(data=models)


def _resolve_warmup_status(request: Request | None) -> WarmupStatus:
    if request is None:
        return WarmupStatus()

    status_obj = getattr(request.app.state, "warmup_status", None)
    if isinstance(status_obj, WarmupStatus):
        return status_obj

    return WarmupStatus()


@router.get("/health", response_model=HealthResponse)
async def health(
    request: Request,
    registry: Annotated[Any, Depends(get_model_registry, use_cache=False)] = None,
) -> HealthResponse | Any:
    if registry is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model registry not initialized")
    try:
        models = registry.list_models()
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Registry unavailable") from exc

    warmup_status = _resolve_warmup_status(request)
    warmup_failures = list(warmup_status.failures)
    warmup_details = WarmupDetails(
        required=warmup_status.required,
        completed=warmup_status.completed,
        failures=warmup_failures or None,
        ok_models=warmup_status.ok_models or None,
        capabilities=warmup_status.capabilities or None,
    )

    chat_queue_depths: list[QueueDepth] | None = None
    embed_queue_depths: list[QueueDepth] | None = None

    chat_batcher = getattr(request.app.state, "chat_batching_service", None)
    if chat_batcher and getattr(chat_batcher, "queue_stats", None):
        chat_queue_depths = [
            QueueDepth(model=name, size=size, max_size=max_size)
            for name, (size, max_size) in chat_batcher.queue_stats().items()
        ]

    embed_batcher = getattr(request.app.state, "batching_service", None)
    if embed_batcher and getattr(embed_batcher, "queue_stats", None):
        embed_queue_depths = [
            QueueDepth(model=name, size=size, max_size=max_size)
            for name, (size, max_size) in embed_batcher.queue_stats().items()
        ]

    runtime_cfg: dict[str, Any] | None = getattr(request.app.state, "runtime_config", None)

    health_status = "ok"
    http_status = status.HTTP_200_OK
    if warmup_status.required and (not warmup_status.completed or warmup_failures):
        health_status = "unhealthy"
        http_status = status.HTTP_503_SERVICE_UNAVAILABLE

    response = HealthResponse(
        status=health_status,
        models=models,
        warmup_failures=warmup_failures or None,
        warmup=warmup_details,
        chat_batch_queues=chat_queue_depths,
        embedding_batch_queues=embed_queue_depths,
        runtime_config=runtime_cfg,
    )

    if http_status != status.HTTP_200_OK:
        from fastapi.responses import JSONResponse  # noqa: PLC0415 - conditional import

        return JSONResponse(status_code=http_status, content=response.model_dump())

    return response
