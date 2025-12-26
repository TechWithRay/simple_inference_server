from __future__ import annotations

import sys
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mock kokoro_onnx before importing app.models.kokoro_tts to avoid runtime dependencies
# and potential environment issues with onnxruntime/numpy
sys.modules["kokoro_onnx"] = MagicMock()

from app.models import kokoro_tts


@pytest.fixture
def mock_kokoro_lib(monkeypatch):
    mock_cls = MagicMock()
    monkeypatch.setattr(kokoro_tts, "Kokoro", mock_cls)
    return mock_cls


@pytest.fixture
def mock_resolve_device(monkeypatch):
    monkeypatch.setattr(kokoro_tts, "resolve_device", lambda d: "cpu")


def test_kokoro_generate_speech(mock_kokoro_lib, mock_resolve_device):
    # Mock _ensure_models to avoid filesystem checks/downloads
    with patch.object(kokoro_tts.KokoroTTS, "_ensure_models"):
        # This is the onnx model variants we want to test
        model = kokoro_tts.KokoroTTS("kokoro-onnx", device="cpu")

    # Setup mock instance
    mock_instance = mock_kokoro_lib.return_value
    # create returns (samples, sample_rate)
    # Note: numpy is mocked in conftest.py, so we can't use np.float32
    expected_samples = np.array([0.1, 0.2])
    expected_sr = 24000
    mock_instance.create.return_value = (expected_samples, expected_sr)

    samples, sr = model.generate_speech("Hello world", voice="af", speed=1.0)

    assert sr == expected_sr
    # Use np.array_equal or just equality depending on the mock implementation
    # The mock numpy.array returns _NDArray which implements __eq__
    assert samples == expected_samples
    mock_instance.create.assert_called_once_with(
        "Hello world", voice="af", speed=1.0, lang="en-us"
    )


def test_kokoro_cancellation(mock_kokoro_lib, mock_resolve_device):
    with patch.object(kokoro_tts.KokoroTTS, "_ensure_models"):
        model = kokoro_tts.KokoroTTS("hexgrad/Kokoro-82M", device="cpu")

    cancel_event = threading.Event()
    cancel_event.set()

    with pytest.raises(InterruptedError):
        model.generate_speech("Hello", voice="af", cancel_event=cancel_event)


def test_kokoro_ensure_models_downloads_if_missing(
    mock_kokoro_lib, mock_resolve_device
):
    # We want to test that _ensure_models calls _download_file when files don't exist.
    # We'll mock Path.exists to return False.

    with (
        patch("pathlib.Path.exists", return_value=False),
        patch("pathlib.Path.mkdir"),
        patch.object(kokoro_tts.KokoroTTS, "_download_file") as mock_download,
    ):

        kokoro_tts.KokoroTTS("hexgrad/Kokoro-82M", device="cpu")

        # Should be called twice: once for onnx, once for voices
        assert mock_download.call_count == 2


def test_kokoro_ensure_models_skips_download_if_exists(
    mock_kokoro_lib, mock_resolve_device
):
    # Mock Path.exists to return True
    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.mkdir"),
        patch.object(kokoro_tts.KokoroTTS, "_download_file") as mock_download,
    ):

        kokoro_tts.KokoroTTS("hexgrad/Kokoro-82M", device="cpu")

        mock_download.assert_not_called()

        mock_download.assert_not_called()
