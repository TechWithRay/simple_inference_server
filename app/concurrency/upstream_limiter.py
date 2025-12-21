import asyncio
import contextlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from app.concurrency.limiter import QueueFullError, QueueTimeoutError, ShuttingDownError
from app.config import settings

# Separate shutdown coordination from local limiters.
_state = {"accepting": True}

# OpenAI proxy limiter state
OPENAI_MAX_CONCURRENT = settings.effective_openai_proxy_max_concurrent
OPENAI_MAX_QUEUE_SIZE = settings.effective_openai_proxy_max_queue_size
OPENAI_QUEUE_TIMEOUT_SEC = settings.effective_openai_proxy_queue_timeout_sec

_openai_semaphore: asyncio.Semaphore = asyncio.Semaphore(OPENAI_MAX_CONCURRENT)
_openai_queue: asyncio.Queue[int] = asyncio.Queue(OPENAI_MAX_QUEUE_SIZE)
_openai_in_flight = {"count": 0}
_openai_lock = asyncio.Lock()

# vLLM proxy limiter state
VLLM_MAX_CONCURRENT = settings.effective_vllm_proxy_max_concurrent
VLLM_MAX_QUEUE_SIZE = settings.effective_vllm_proxy_max_queue_size
VLLM_QUEUE_TIMEOUT_SEC = settings.effective_vllm_proxy_queue_timeout_sec

_vllm_semaphore: asyncio.Semaphore = asyncio.Semaphore(VLLM_MAX_CONCURRENT)
_vllm_queue: asyncio.Queue[int] = asyncio.Queue(VLLM_MAX_QUEUE_SIZE)
_vllm_in_flight = {"count": 0}
_vllm_lock = asyncio.Lock()


@asynccontextmanager
async def openai_proxy_limiter() -> AsyncIterator[None]:
    if not _state["accepting"]:
        raise ShuttingDownError("Service is shutting down")

    queued = False
    try:
        _openai_queue.put_nowait(1)
        queued = True
    except asyncio.QueueFull as exc:
        raise QueueFullError("OpenAI proxy request queue is full") from exc

    acquired = False
    try:
        try:
            await asyncio.wait_for(_openai_semaphore.acquire(), timeout=OPENAI_QUEUE_TIMEOUT_SEC)
            acquired = True
        except TimeoutError as exc:
            raise QueueTimeoutError("Timed out waiting for OpenAI proxy slot") from exc

        async with _openai_lock:
            _openai_in_flight["count"] += 1
        try:
            yield
        finally:
            async with _openai_lock:
                _openai_in_flight["count"] = max(0, _openai_in_flight["count"] - 1)
    finally:
        if acquired:
            _openai_semaphore.release()
        if queued:
            with contextlib.suppress(Exception):
                _openai_queue.get_nowait()
                _openai_queue.task_done()


@asynccontextmanager
async def vllm_proxy_limiter() -> AsyncIterator[None]:
    if not _state["accepting"]:
        raise ShuttingDownError("Service is shutting down")

    queued = False
    try:
        _vllm_queue.put_nowait(1)
        queued = True
    except asyncio.QueueFull as exc:
        raise QueueFullError("vLLM proxy request queue is full") from exc

    acquired = False
    try:
        try:
            await asyncio.wait_for(_vllm_semaphore.acquire(), timeout=VLLM_QUEUE_TIMEOUT_SEC)
            acquired = True
        except TimeoutError as exc:
            raise QueueTimeoutError("Timed out waiting for vLLM proxy slot") from exc

        async with _vllm_lock:
            _vllm_in_flight["count"] += 1
        try:
            yield
        finally:
            async with _vllm_lock:
                _vllm_in_flight["count"] = max(0, _vllm_in_flight["count"] - 1)
    finally:
        if acquired:
            _vllm_semaphore.release()
        if queued:
            with contextlib.suppress(Exception):
                _vllm_queue.get_nowait()
                _vllm_queue.task_done()


def stop_accepting_upstream() -> None:
    _state["accepting"] = False


def start_accepting_upstream() -> None:
    _state["accepting"] = True


async def wait_for_drain_upstream(timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        async with _openai_lock:
            openai_active = _openai_in_flight["count"]
        async with _vllm_lock:
            vllm_active = _vllm_in_flight["count"]
        queue_backlog = _openai_queue.qsize() + _vllm_queue.qsize()
        total_active = openai_active + vllm_active
        if total_active == 0 and queue_backlog == 0:
            break
        if loop.time() >= deadline:
            break
        await asyncio.sleep(0.05)
