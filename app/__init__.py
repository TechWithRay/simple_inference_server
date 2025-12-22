from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import dotenv_values


def _running_under_pytest() -> bool:
    # Keep the test suite hermetic: do not implicitly load repo-local env files.
    return "pytest" in sys.modules or any("pytest" in arg for arg in sys.argv)


def _load_repo_env_defaults() -> None:
    """Load repo-local env defaults early (without overriding exported env vars).

    Precedence:
      - exported env vars (highest)
      - `.env` (local overrides; gitignored)
      - `env` (repo defaults; git-tracked)
    """
    if _running_under_pytest():
        return

    repo_root = Path(__file__).resolve().parent.parent
    merged: dict[str, str] = {}
    for filename in ("env", ".env"):
        path = repo_root / filename
        if not path.exists():
            continue
        for key, value in dotenv_values(path).items():
            if not key or value is None or value == "":
                continue
            # Later files override earlier ones inside this merged view.
            merged[key] = value

    # Apply merged dotenv values only when the process hasn't already set them.
    for key, value in merged.items():
        os.environ.setdefault(key, value)


_load_repo_env_defaults()
