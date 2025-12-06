import logging

from fastapi import APIRouter

from app.concurrency.limiter import QUEUE_TIMEOUT_SEC, embedding_limiter
from app.routes import audio, chat, embeddings, health, rerank
from app.routes.audio import UPLOAD_CHUNK_BYTES, _save_upload
from app.routes.common import _await_executor_cleanup

logger = logging.getLogger(__name__)

router = APIRouter()
router.include_router(embeddings.router)
router.include_router(chat.router)
router.include_router(audio.router)
router.include_router(health.router)
router.include_router(rerank.router)

__all__ = [
    "router",
    "_await_executor_cleanup",
    "embedding_limiter",
    "QUEUE_TIMEOUT_SEC",
    "_save_upload",
    "UPLOAD_CHUNK_BYTES",
    "logger",
]
