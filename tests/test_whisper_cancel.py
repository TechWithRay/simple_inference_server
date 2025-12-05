from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from app.models.whisper import WhisperASR


def test_whisper_transcribe_adds_cancel_stopper(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure transcribe passes a StopOnCancel stopping criterion into generate_kwargs."""

    # Build a lightweight WhisperASR instance without running heavy __init__
    obj = WhisperASR.__new__(WhisperASR)
    obj.hf_repo_id = "dummy"
    obj.name = "dummy"
    obj.capabilities = ["audio-transcription", "audio-translation"]
    obj.device = SimpleNamespace(type="cpu")  # minimal device mock
    obj._lock = threading.Lock()
    obj.processor = SimpleNamespace(get_prompt_ids=lambda *args, **kwargs: None)

    captured_kwargs = {}

    def fake_pipeline(_audio_path: str, *, return_timestamps: bool | str, generate_kwargs: dict) -> dict:
        captured_kwargs.update(generate_kwargs)
        return {"text": "ok", "language": "en", "chunks": []}

    obj.pipeline = fake_pipeline  # type: ignore[assignment]

    cancel_event = threading.Event()
    obj.transcribe(
        "audio.wav",
        language=None,
        prompt=None,
        temperature=None,
        task="transcribe",
        timestamp_granularity=None,
        cancel_event=cancel_event,
    )

    sc_list = captured_kwargs.get("stopping_criteria")
    assert sc_list is not None
    # StopOnCancel sets a .event attribute; ensure the instance holds our cancel_event
    assert any(getattr(sc, "event", None) is cancel_event for sc in sc_list)
