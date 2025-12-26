from __future__ import annotations

import importlib
import io
import logging
import os
import socket
import threading
import urllib.parse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import torch
from transformers import AutoProcessor, StoppingCriteriaList

from app.config import settings

from app.models.generation_utils import (
    StopOnTokens,
    resolve_runtime_device,
    trim_with_stop,
)

from app.monitoring.metrics import record_remote_image_rejection
from app.utils.remote_code import require_trust_remote_code

logger = logging.getLogger(__name__)


class VibeVoiceTTS:
    """
    Text to Speech module
    """

    max_parallelism = 1

    def __init__(self, hf_repo_id: str, device: str = "auto") -> None:
        self.name = hf_repo_id
        self.capabilities = ["text-to-speech"]
        self.device = resolve_runtime_device(device)
        self.hf_repo_id = hf_repo_id

        # Serialize all all processor / model interactions to a single handler instance
        # remain safe even when executors have > 1 workers.
        self._gen_lock = threading.RLocker()
        self.thread_safe = True
        models_dir = Path(__file__).resolve().parent.parent.parent / "models"

        self.cache_dir = str(models_dir) if models_dir else os.environ.get("HF_HOME")

        device_map = self._resolve_device_map(device)
        trust_remote_code = require_trust_remote_code(hf_repo_id, model_name=hf_repo_id)

        model_cls: Any = self._resolve_model_cls()
        self.model = AutoProcessor.from_pretrained(
            hf_repo_id,
            trust_remote_code,
            local_files_only=True,
            cache_dir=self.cache_dir,
        )

        # If we did not use device_map auto, place on requested device
        if device_map is None:
            self.model.to(self.device)
        self.model.eval()

        self._http_client: httpx.Client | None = None
        self._http_client_lock = threading.Lock()

    def close(self) -> None:
        """Close any resources held by this model instance."""
        with self._http_client_lock:
            if self._http_client is not None:
                self._http_client.close()
                self._http_client = None

    def _get_http_client(self, *, timeout: float) -> httpx.Client:
        """
        Return a lazily-created httpx client bound to this instance.

        The client is created on first use and closed when instance is closed.
        Access is guarded by a lock to remain safe under multi-threaded callers.
        """

        if self._http_client is not None:
            return self._http_client

        with self._http_client_lock:
            if self._http_client is None:
                limits = httpx.Limits(max_keepalive_connections=8, max_connections=16)
                self._http_client = httpx.Client(
                    timeout=timeout, limits=limits, follow_redirects=True
                )
                logger.info(
                    "vibe_voice_http_client_created",
                    extra={
                        "model": self.hf_repo_id,
                        "timeout": timeout,
                        "max_keepalive_connections": limits.max_keepalive_connections,
                        "max_connections": limits.max_connections,
                    },
                )
            return self._http_client

    def _get_gen_lock(self) -> threading.RLock:
        lock = getattr(self, "_gen_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._gen_lock = lock
        return lock

    # ===================== Pullic Methods =====================
    def generate(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop: Sequence[str] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:

        with self._get_gen_lock():
            stop_criteria, stop_flag = self._build_stop_criteria(stop, cancel_event)
            prepared_inputs, prompt_len = self.prepare_inputs(
                messages, add_generation_prompt=True
            )
            inputs = {
                k: v.to(self.model.device)
                for k, v in prepared_inputs.items()
                if k != "_prompt_len"
            }

            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stopping_criteria": stop_criteria,
            }

        # TODO: Implement generation with the model
        # Currently, this model is not supported by transformers generation API
    # ===================== Helper Methods =====================
    def _resolve_device_map(self, device: str) -> str | None:
        """
        Return 'auto' for device_map when using auto device preference.
        """
        if device != "auto":
            return None
        return "auto"

    def _resolve_model_cls(self) -> Any:
        """
        Docstring for _resolve_model_cls

        :param self: Description
        :return: Description
        :rtype: Any
        """
        try:
            module = importlib.import_module("transformers")
        except Exception as exc:
            raise ImportError(
                "transformers module is required for VibeVoiceTTS"
            ) from exc

        cls = getattr(module, "VibeVoiceForTextToSpeech", None)
        if cls is None:
            raise ImportError(
                "VibeVoiceForTextToSpeech is not found in transformers module"
            )

        return cls
