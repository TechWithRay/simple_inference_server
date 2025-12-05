import logging

from fastapi import APIRouter

from app.concurrency.limiter import embedding_limiter, QUEUE_TIMEOUT_SEC
from app.routes import audio, chat, embeddings, health
from app.routes.audio import _save_upload, UPLOAD_CHUNK_BYTES
from app.routes.common import _await_executor_cleanup

logger = logging.getLogger(__name__)

router = APIRouter()
router.include_router(embeddings.router)
router.include_router(chat.router)
router.include_router(audio.router)
router.include_router(health.router)

__all__ = [
    "router",
    "_await_executor_cleanup",
    "embedding_limiter",
    "QUEUE_TIMEOUT_SEC",
    "_save_upload",
    "UPLOAD_CHUNK_BYTES",
    "logger",
]
