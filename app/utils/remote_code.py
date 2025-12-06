from __future__ import annotations

from app.config import settings


def require_trust_remote_code(repo_id: str, *, model_name: str | None = None) -> bool:
    """Allow trust_remote_code only when explicitly configured.

    Returns True when repo_id or model_name appears in TRUST_REMOTE_CODE_ALLOWLIST,
    otherwise returns False so loaders run with trust_remote_code disabled.
    Models that require remote code will fail to load unless explicitly allowlisted.
    """

    allowlist = {item.lower() for item in settings.trust_remote_code_allowlist_set}
    repo_key = repo_id.lower()
    name_key = (model_name or "").lower()

    name_allowed = bool(name_key) and name_key in allowlist
    return repo_key in allowlist or name_allowed

