"""Unified device resolution utilities.

This module provides consistent device resolution logic across all model handlers,
supporting CPU, MPS (Apple Silicon), and CUDA (with multi-GPU indexing).
"""

from __future__ import annotations

import torch


def resolve_device(preference: str | None, *, validate: bool = True) -> str:  # noqa: PLR0911
    """Resolve device preference to a concrete device string.

    Args:
        preference: Device preference ("auto", "cpu", "mps", "cuda", "cuda:<idx>", or None).
                   None is treated as "auto".
        validate: If True, raise ValueError for unavailable devices.
                  If False, return the preference string directly for non-auto values.

    Returns:
        Resolved device string (e.g., "cpu", "cuda", "cuda:0", "mps").

    Raises:
        ValueError: If validate=True and requested device is not available.
    """
    pref = (preference or "auto").lower()
    has_cuda = torch.cuda.is_available()
    has_mps = _has_mps()

    if pref == "auto":
        if has_cuda:
            return "cuda"
        if has_mps:
            return "mps"
        return "cpu"

    if not validate:
        return preference or "cpu"

    if pref == "cpu":
        return "cpu"

    if pref == "mps":
        if not has_mps:
            raise ValueError("MPS requested but not available")
        return "mps"

    if pref.startswith("cuda"):
        if not has_cuda:
            raise ValueError("CUDA requested but not available")

        if ":" in pref:
            _, idx_str = pref.split(":", 1)
            if not idx_str.isdigit():
                raise ValueError(f"Invalid CUDA device format: {preference}")
            idx = int(idx_str)
            count = torch.cuda.device_count()
            if idx >= count:
                raise ValueError(f"Requested cuda:{idx} but only {count} device(s) visible")
            return f"cuda:{idx}"

        return "cuda"

    raise ValueError(f"Unknown device preference: {preference}")


def resolve_torch_device(preference: str | None, *, validate: bool = True) -> torch.device:
    """Resolve device preference to a torch.device object.

    Args:
        preference: Device preference ("auto", "cpu", "mps", "cuda", "cuda:<idx>", or None).
        validate: If True, raise ValueError for unavailable devices.

    Returns:
        torch.device instance.

    Raises:
        ValueError: If validate=True and requested device is not available.
    """
    return torch.device(resolve_device(preference, validate=validate))


def _has_mps() -> bool:
    """Check if MPS (Apple Silicon) is available."""
    backends = getattr(torch, "backends", None)
    if backends is None:
        return False
    mps = getattr(backends, "mps", None)
    if mps is None:
        return False
    return bool(mps.is_available())
