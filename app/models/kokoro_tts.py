from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
import requests
from kokoro_onnx import Kokoro

from app.models.base import TTSModel
from app.utils.device import resolve_device

logger = logging.getLogger(__name__)

KOKORO_ONNX_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
VOICES_BIN_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"


class KokoroTTS(TTSModel):
    """Kokoro TTS handler using ONNX Runtime."""

    # Serialize access to be safe, though ONNX Runtime is often thread-safe.
    max_parallelism = 1

    def __init__(self, hf_repo_id: str, device: str = "cpu") -> None:
        self.name = hf_repo_id
        self.device = resolve_device(device)
        self.capabilities = ["text-to-speech"]
        self.thread_safe = False
        self._lock = threading.Lock()

        # Model storage
        # Use a dedicated subfolder for Kokoro to avoid cluttering the main models dir
        models_dir = Path(__file__).resolve().parent.parent.parent / "models"
        self.model_dir = models_dir / "kokoro"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.onnx_path = self.model_dir / "kokoro-v1.0.onnx"
        self.voices_path = self.model_dir / "voices-v1.0.bin"

        self._ensure_models()

        # Initialize Kokoro
        # Note: Kokoro-onnx currently defaults to CPU.
        self.kokoro = Kokoro(str(self.onnx_path), str(self.voices_path))

    def _ensure_models(self) -> None:
        if not self.onnx_path.exists():
            logger.info("Downloading Kokoro ONNX model to %s...", self.onnx_path)
            self._download_file(KOKORO_ONNX_URL, self.onnx_path)

        if not self.voices_path.exists():
            logger.info("Downloading Kokoro voices to %s...", self.voices_path)
            self._download_file(VOICES_BIN_URL, self.voices_path)

    def _download_file(self, url: str, path: Path) -> None:
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with path.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        except Exception:
            # Clean up partial download
            if path.exists():
                path.unlink()
            raise

    def generate_speech(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        lang: str = "en-us",
        cancel_event: threading.Event | None = None,
    ) -> tuple[np.ndarray, int]:
        if cancel_event and cancel_event.is_set():
            raise InterruptedError("Cancelled")

        with self._lock:
            # Double check cancel
            if cancel_event and cancel_event.is_set():
                raise InterruptedError("Cancelled")

            samples, sample_rate = self.kokoro.create(
                text, voice=voice, speed=speed, lang=lang
            )
            return samples, sample_rate
