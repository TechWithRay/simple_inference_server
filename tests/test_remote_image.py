import pytest

from app.models.qwen_vl import QwenVLChat
from app.monitoring.metrics import REMOTE_IMAGE_REJECTIONS, record_remote_image_rejection


def _reset_metric(metric) -> None:
    # prometheus_client Counters cannot be reset directly; recreate collector values
    for labelset in list(metric._metrics.keys()):  # type: ignore[attr-defined]
        metric._metrics.pop(labelset, None)


def test_remote_image_disallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOW_REMOTE_IMAGES", "0")
    obj = QwenVLChat.__new__(QwenVLChat)

    with pytest.raises(ValueError):
        obj._load_image("http://example.com/image.png")


def test_remote_image_private_ip_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOW_REMOTE_IMAGES", "1")
    monkeypatch.setenv("REMOTE_IMAGE_HOST_ALLOWLIST", "example.com")
    obj = QwenVLChat.__new__(QwenVLChat)

    # Force DNS to resolve to a private address
    monkeypatch.setattr("socket.getaddrinfo", lambda *_args, **_kwargs: [("family", "type", "proto", "canon", ("127.0.0.1", 0))])

    with pytest.raises(ValueError):
        obj._load_image("http://example.com/image.png")


def test_data_uri_mime_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REMOTE_IMAGE_MIME_ALLOWLIST", "image/jpeg")
    obj = QwenVLChat.__new__(QwenVLChat)

    # Stub PIL Image to avoid dependency on real image loading
    class DummyImage:
        def __init__(self, fmt: str) -> None:
            self.format = fmt

        def convert(self, _mode: str) -> "DummyImage":
            return self

    class DummyImageModule:
        @staticmethod
        def open(_buf):
            return DummyImage("PNG")

    monkeypatch.setattr("app.models.qwen_vl.Image", DummyImageModule)

    # 1x1 PNG data URI
    data_uri = (
        "data:image/png;base64," "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9YxGDX4AAAAASUVORK5CYII="
    )

    with pytest.raises(ValueError):
        obj._load_image(data_uri)


def test_remote_image_rejection_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_metric(REMOTE_IMAGE_REJECTIONS)
    monkeypatch.setenv("ALLOW_REMOTE_IMAGES", "1")
    monkeypatch.setenv("REMOTE_IMAGE_HOST_ALLOWLIST", "example.com")
    obj = QwenVLChat.__new__(QwenVLChat)

    # Mock httpx client to raise size error via HEAD length
    class DummyResp:
        def __init__(self) -> None:
            self.headers = {"content-length": "9999999", "content-type": "image/png"}
            self.url = type("U", (), {"host": "example.com"})

        def close(self) -> None:  # pragma: no cover - trivial
            pass

    class DummyClient:
        def head(self, *_args, **_kwargs):
            return DummyResp()

        def stream(self, *_args, **_kwargs):  # pragma: no cover - unused because size reject on HEAD
            raise RuntimeError("should not stream")

    monkeypatch.setattr(obj, "_get_http_client", lambda timeout: DummyClient())

    with pytest.raises(ValueError):
        obj._load_image("http://example.com/img.png")

    # Expect size rejection recorded
    assert REMOTE_IMAGE_REJECTIONS._metrics  # type: ignore[attr-defined]
