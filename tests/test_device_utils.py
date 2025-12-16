"""Tests for device resolution utilities."""

import pytest
import torch

from app.utils.device import resolve_device, resolve_torch_device


class TestResolveDevice:
    """Tests for resolve_device function."""

    def test_auto_prefers_cuda_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert resolve_device("auto") == "cuda"

    def test_auto_prefers_mps_when_cuda_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        # Mock MPS availability
        class MockMPS:
            @staticmethod
            def is_available() -> bool:
                return True

        class MockBackends:
            mps = MockMPS

        monkeypatch.setattr(torch, "backends", MockBackends)
        assert resolve_device("auto") == "mps"

    def test_auto_falls_back_to_cpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr("app.utils.device._has_mps", lambda: False)
        assert resolve_device("auto") == "cpu"

    def test_cpu_always_works(self) -> None:
        assert resolve_device("cpu") == "cpu"

    def test_mps_raises_when_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("app.utils.device._has_mps", lambda: False)
        with pytest.raises(ValueError, match="MPS requested but not available"):
            resolve_device("mps", validate=True)

    def test_cuda_raises_when_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(ValueError, match="CUDA requested but not available"):
            resolve_device("cuda", validate=True)

    def test_cuda_index_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

        assert resolve_device("cuda:0") == "cuda:0"
        assert resolve_device("cuda:1") == "cuda:1"

        with pytest.raises(ValueError, match="only 2 device"):
            resolve_device("cuda:5")

    def test_invalid_cuda_format_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        with pytest.raises(ValueError, match="Invalid CUDA device format"):
            resolve_device("cuda:abc")

    def test_unknown_device_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown device preference"):
            resolve_device("tpu")

    def test_validate_false_skips_checks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        # Should not raise even though CUDA is unavailable
        assert resolve_device("cuda", validate=False) == "cuda"

    def test_none_treated_as_auto(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr("app.utils.device._has_mps", lambda: False)
        assert resolve_device(None) == "cpu"


class TestResolveTorchDevice:
    """Tests for resolve_torch_device function."""

    def test_returns_torch_device(self) -> None:
        # cpu is always available, no mocking needed
        device = resolve_torch_device("cpu")
        assert isinstance(device, torch.device)
        assert device.type == "cpu"

    def test_auto_resolution_cpu_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Mock to force CPU selection
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        # Create a mock backends that doesn't have MPS
        monkeypatch.setattr("app.utils.device._has_mps", lambda: False)

        device = resolve_torch_device("auto")
        assert device.type == "cpu"
