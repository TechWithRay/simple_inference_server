from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import time
from collections.abc import Sequence
from typing import Annotated, Any, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.chat_batching import ChatBatchQueueFullError, ChatBatchQueueTimeoutError, get_count_executor
from app.concurrency.limiter import (
    CHAT_QUEUE_TIMEOUT_SEC,
    QUEUE_TIMEOUT_SEC,
    QueueFullError,
    QueueTimeoutError,
    ShuttingDownError,
    chat_limiter,
    reset_queue_label,
    set_queue_label,
)
from app.config import settings
from app.dependencies import get_model_registry
from app.models.base import ChatGeneration
from app.models.registry import ModelRegistry
from app.monitoring.metrics import (
    observe_chat_latency,
    record_chat_request,
)
from app.routes.common import (
    _ClientDisconnectedError,
    _RequestCancelledError,
    _run_work_with_client_cancel,
    _WorkTimeoutError,
)
from app.threadpool import get_chat_executor

logger = logging.getLogger(__name__)
router = APIRouter()


class ImageURL(BaseModel):
    url: str
    detail: Literal["auto", "low", "high"] | None = Field(default=None)


class ChatContentPart(BaseModel):
    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: ImageURL | None = None

    @staticmethod
    def _assert_valid(part: ChatContentPart) -> ChatContentPart:
        if part.type == "text" and part.text is None:
            raise ValueError("text content part requires 'text'")
        if part.type == "image_url" and (part.image_url is None or not part.image_url.url):
            raise ValueError("image_url content part requires 'image_url.url'")
        return part

    def model_post_init(self, __context: object) -> None:  # noqa: D401
        self._assert_valid(self)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str | list[ChatContentPart]


class ChatCompletionMessage(BaseModel):
    role: Literal["assistant"]
    content: str


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: str = "stop"


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=None, description="Max new tokens to generate")
    temperature: float | None = None
    top_p: float | None = None
    n: int = 1
    stream: bool = False
    stop: str | list[str] | None = None
    user: str | None = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: int | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


def _contains_image_content(messages: Sequence[dict[str, Any]]) -> bool:
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


def _normalize_stop(stop: str | list[str] | None) -> list[str] | None:
    if stop is None:
        return None
    if isinstance(stop, str):
        return [stop]
    return [s for s in stop if s]


def _resolve_generation_params(
    req: ChatCompletionRequest,
    model: Any,
) -> tuple[int, float, float]:
    defaults = getattr(model, "generation_defaults", {}) or {}
    max_tokens_default = defaults.get("max_tokens") or settings.max_new_tokens
    temperature_default = defaults.get("temperature", 0.7)
    top_p_default = defaults.get("top_p", 0.9)

    max_tokens = req.max_tokens or max_tokens_default
    temperature = req.temperature if req.temperature is not None else temperature_default
    top_p = req.top_p if req.top_p is not None else top_p_default

    return max_tokens, temperature, top_p


def _build_generation_kwargs(  # noqa: PLR0913
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: list[str] | None,
    cancel_event: threading.Event | None,
    accepts_cancel: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stop": stop,
    }
    if accepts_cancel and cancel_event is not None:
        kwargs["cancel_event"] = cancel_event
    return kwargs


async def _prepare_chat_request(
    model: Any,
    messages: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, int]:
    loop = asyncio.get_running_loop()
    prepare_timeout = settings.chat_prepare_timeout_sec
    count_executor = get_count_executor(
        use_chat_executor=settings.chat_count_use_chat_executor
    )
    if hasattr(model, "prepare_inputs"):
        try:
            prepared, prompt_tokens = await asyncio.wait_for(
                loop.run_in_executor(
                    count_executor,
                    lambda: model.prepare_inputs(messages, add_generation_prompt=True),
                ),
                timeout=prepare_timeout,
            )
            return prepared, prompt_tokens
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "chat_prepare_inputs_failed",
                extra={"model": getattr(model, "name", "unknown"), "error": str(exc)},
            )

    prompt_tokens = await asyncio.wait_for(
        loop.run_in_executor(count_executor, lambda: model.count_tokens(messages)),
        timeout=prepare_timeout,
    )
    return None, int(prompt_tokens)


async def _run_chat_generation(  # noqa: PLR0915
    *,
    req: ChatCompletionRequest,
    registry: ModelRegistry,
    request: Request,
    raw_messages: list[dict[str, Any]],
    has_images: bool,
) -> tuple[ChatGeneration, int, int]:

    model = _resolve_chat_model_and_caps(registry, req.model, has_images=has_images)
    max_tokens, temperature, top_p = _resolve_generation_params(req, model)

    if max_tokens <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="max_tokens must be positive",
        )

    loop = asyncio.get_running_loop()
    executor = get_chat_executor()
    max_prompt_tokens = settings.chat_max_prompt_tokens
    prepared_inputs, prompt_tokens = await _prepare_chat_request(model, raw_messages)
    if prompt_tokens > max_prompt_tokens:
        record_chat_request(req.model, "400")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prompt too long; max {max_prompt_tokens} tokens",
        )
    cancel_event = threading.Event()
    gen_timeout = settings.chat_generate_timeout_sec
    generate_accepts_cancel = "cancel_event" in inspect.signature(model.generate).parameters
    generate_prepared_accepts_cancel = hasattr(model, "generate_prepared") and "cancel_event" in inspect.signature(model.generate_prepared).parameters
    batcher = getattr(request.app.state, "chat_batching_service", None)
    stop = _normalize_stop(req.stop)

    async def _run_generation() -> ChatGeneration:
        if prepared_inputs is not None and hasattr(model, "generate_prepared"):
            kwargs = _build_generation_kwargs(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                cancel_event=cancel_event,
                accepts_cancel=generate_prepared_accepts_cancel,
            )
            return await loop.run_in_executor(
                executor,
                lambda: model.generate_prepared(prepared_inputs, **kwargs),
            )

        kwargs = _build_generation_kwargs(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            cancel_event=cancel_event,
            accepts_cancel=generate_accepts_cancel,
        )
        return await loop.run_in_executor(
            executor,
            lambda: model.generate(raw_messages, **kwargs),
        )

    async def _run_batched_or_fallback() -> ChatGeneration:
        if (
            batcher is not None
            and getattr(batcher, "is_supported", lambda _m: False)(req.model)
            and not has_images
        ):
            try:
                return await batcher.enqueue(
                    req.model,
                    raw_messages,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                    prompt_tokens=prompt_tokens,
                    prepared_inputs=prepared_inputs,
                    cancel_event=cancel_event,
                )
            except ValueError as exc:
                record_chat_request(req.model, "400")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(exc),
                ) from exc
            except ChatBatchQueueFullError as exc:
                record_chat_request(req.model, "429")
                logger.info(
                    "chat_batch_queue_full", extra={"model": req.model, "status": 429}
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Chat batch queue full",
                    headers={"Retry-After": str(int(QUEUE_TIMEOUT_SEC))},
                ) from exc
            except ChatBatchQueueTimeoutError as exc:
                record_chat_request(req.model, "429")
                logger.info(
                    "chat_batch_queue_timeout", extra={"model": req.model, "status": 429}
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Chat batch queue wait exceeded",
                    headers={"Retry-After": str(int(QUEUE_TIMEOUT_SEC))},
                ) from exc
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.warning(
                    "chat_batcher_failed_falling_back",
                    extra={"model": req.model, "error": str(exc)},
                )
        return await _run_generation()

    work_task = asyncio.ensure_future(_run_batched_or_fallback())
    try:
        generation = await _run_work_with_client_cancel(
            request=request,
            work_task=work_task,
            cancel_event=cancel_event,
            timeout=gen_timeout,
        )
    except _WorkTimeoutError as exc:
        cancel_event.set()
        record_chat_request(req.model, "504")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Chat generation timed out",
        ) from exc
    except _RequestCancelledError as exc:
        cancel_event.set()
        record_chat_request(req.model, "499")
        raise HTTPException(
            status_code=status.HTTP_499_CLIENT_CLOSED_REQUEST,
            detail="Request cancelled",
        ) from exc
    except _ClientDisconnectedError as exc:
        cancel_event.set()
        record_chat_request(req.model, "499")
        raise HTTPException(
            status_code=status.HTTP_499_CLIENT_CLOSED_REQUEST,
            detail="Client disconnected",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - unexpected runtime failure
        cancel_event.set()
        record_chat_request(req.model, "500")
        logger.exception(
            "chat_generation_failed",
            extra={"model": req.model, "max_tokens": max_tokens, "status": 500},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chat generation failed",
        ) from exc

    return generation, int(prompt_tokens), int(max_tokens)


def _resolve_chat_model_and_caps(
    registry: ModelRegistry,
    model_name: str,
    *,
    has_images: bool,
) -> Any:
    try:
        model = registry.get(model_name)
    except KeyError as exc:
        record_chat_request(model_name, "404")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_name} not found",
        ) from exc

    capabilities = getattr(model, "capabilities", [])
    if "chat-completion" not in capabilities:
        record_chat_request(model_name, "400")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model {model_name} does not support chat/completions",
        )
    if has_images and "vision" not in capabilities:
        record_chat_request(model_name, "400")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model {model_name} does not support image inputs",
        )

    return model


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completions(  # noqa: PLR0912
    req: ChatCompletionRequest,
    registry: Annotated[ModelRegistry, Depends(get_model_registry)],
    _request: Request,
) -> ChatCompletionResponse:
    if req.stream:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Streaming responses are not supported yet",
        )
    if req.n != 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only n=1 is supported",
        )

    start = time.perf_counter()
    label_token = set_queue_label(req.model or "chat")
    try:
        async with chat_limiter():
            raw_messages = [msg.model_dump(mode="python") for msg in req.messages]
            has_images = _contains_image_content(raw_messages)
            generation, prompt_tokens, max_tokens = await _run_chat_generation(
                req=req,
                registry=registry,
                request=_request,
                raw_messages=raw_messages,
                has_images=has_images,
            )
    except QueueFullError as exc:
        record_chat_request(req.model, "429")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Chat request queue full",
            headers={"Retry-After": str(int(CHAT_QUEUE_TIMEOUT_SEC))},
        ) from exc
    except ShuttingDownError as exc:
        record_chat_request(req.model, "503")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is shutting down",
        ) from exc
    except QueueTimeoutError as exc:
        record_chat_request(req.model, "429")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Timed out waiting for chat worker",
            headers={"Retry-After": str(int(CHAT_QUEUE_TIMEOUT_SEC))},
        ) from exc
    finally:
        reset_queue_label(label_token)

    latency = time.perf_counter() - start
    observe_chat_latency(req.model, latency)
    record_chat_request(req.model, "200")
    logger.info(
        "chat_request",
        extra={
            "model": req.model,
            "latency_ms": round(latency * 1000, 2),
            "status": 200,
            "max_tokens": max_tokens,
        },
    )

    completion_tokens = generation.completion_tokens or 0
    prompt_tokens = (
        generation.prompt_tokens
        if generation.prompt_tokens is not None
        else prompt_tokens
    )
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    choice = ChatCompletionChoice(
        index=0,
        message=ChatCompletionMessage(role="assistant", content=generation.text),
        finish_reason=generation.finish_reason or "stop",
    )
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid4().hex}",
        created=int(time.time()),
        model=req.model,
        choices=[choice],
        usage=usage,
    )

