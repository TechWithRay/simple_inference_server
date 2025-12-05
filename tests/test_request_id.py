"""Tests for Request ID middleware."""

from http import HTTPStatus

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware import RequestIDMiddleware, get_request_id

# UUID hex length (32 characters)
UUID_HEX_LENGTH = 32


def test_request_id_generated_when_not_provided() -> None:
    """Request ID is generated if not in headers."""
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)

    captured_id: str | None = None

    @app.get("/test")
    def test_route() -> dict[str, str | None]:
        nonlocal captured_id
        captured_id = get_request_id()
        return {"id": captured_id}

    client = TestClient(app)
    response = client.get("/test")

    assert response.status_code == HTTPStatus.OK
    assert "X-Request-ID" in response.headers
    assert response.headers["X-Request-ID"] == captured_id
    assert len(captured_id or "") == UUID_HEX_LENGTH


def test_request_id_propagated_from_header() -> None:
    """Request ID is propagated from incoming header."""
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)

    captured_id: str | None = None

    @app.get("/test")
    def test_route() -> dict[str, str | None]:
        nonlocal captured_id
        captured_id = get_request_id()
        return {"id": captured_id}

    client = TestClient(app)
    custom_id = "my-custom-request-id-123"
    response = client.get("/test", headers={"X-Request-ID": custom_id})

    assert response.status_code == HTTPStatus.OK
    assert response.headers["X-Request-ID"] == custom_id
    assert captured_id == custom_id


def test_request_id_not_leaked_between_requests() -> None:
    """Request ID is isolated between requests."""
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)

    captured_ids: list[str | None] = []

    @app.get("/test")
    def test_route() -> dict[str, str | None]:
        captured_ids.append(get_request_id())
        return {"id": get_request_id()}

    client = TestClient(app)

    # Make multiple requests
    client.get("/test", headers={"X-Request-ID": "id-1"})
    client.get("/test", headers={"X-Request-ID": "id-2"})
    client.get("/test")  # auto-generated

    assert captured_ids[0] == "id-1"
    assert captured_ids[1] == "id-2"
    assert captured_ids[2] is not None
    assert captured_ids[2] != "id-1"
    assert captured_ids[2] != "id-2"


def test_get_request_id_returns_none_outside_request() -> None:
    """get_request_id returns None when not in a request context."""
    # Outside of request context, should return None
    assert get_request_id() is None

