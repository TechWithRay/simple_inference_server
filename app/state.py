from __future__ import annotations

from dataclasses import dataclass, field

from app.batching import BatchingService
from app.chat_batching import ChatBatchingService
from app.models.registry import ModelRegistry


@dataclass
class WarmupStatus:
    required: bool = False
    completed: bool = False
    failures: list[str] = field(default_factory=list)

# Global holder for the loaded ModelRegistry instance.
model_registry: ModelRegistry | None = None
batching_service: BatchingService | None = None
chat_batching_service: ChatBatchingService | None = None
warmup_status: WarmupStatus = WarmupStatus()
