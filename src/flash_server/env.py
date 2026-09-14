"""Centralized reader for deployment options / environment variables.

Every component reads its options through this module instead of touching
``os.environ`` directly. Each key is resolved in this order (highest priority
first):

    1. process env      ``os.environ``
    2. ``.env.local``   per-machine overrides (do not commit)
    3. ``.env``         project defaults

So the surrounding process environment wins, then ``.env.local``, then
``.env`` -- matching the mainstream convention where real environment
variables take precedence over files and ``.env.local`` overrides ``.env``.
The base directory defaults to the current working directory and can be
overridden per call (used by tests).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import dotenv_values

__all__ = [
    "get_env",
    "get_env_str",
    "get_env_int",
    "get_env_float",
    "clear_cache",
]

_PRIORITY_FILES = (".env.local", ".env")


def _base_dir(base_dir: str | os.PathLike | None) -> Path:
    return Path(base_dir) if base_dir is not None else Path.cwd()


@lru_cache(maxsize=None)
def _load(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values = dict(dotenv_values(path))
    return {key: ("" if value is None else str(value)) for key, value in values.items()}


def get_env(
    key: str,
    default: str | None = None,
    base_dir: str | os.PathLike | None = None,
) -> str | None:
    """Return the value for ``key`` per the precedence above, else ``default``."""
    value = os.environ.get(key)
    if value is not None:
        return value
    root = _base_dir(base_dir)
    for name in _PRIORITY_FILES:
        value = _load(root / name).get(key)
        if value is not None:
            return value
    return default


def get_env_str(
    key: str,
    default: str | None = None,
    base_dir: str | os.PathLike | None = None,
) -> str | None:
    return get_env(key, default, base_dir)


def get_env_int(
    key: str,
    default: int | None = None,
    base_dir: str | os.PathLike | None = None,
) -> int | None:
    raw = get_env(key, base_dir=base_dir)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer (got {raw!r})") from exc


def get_env_float(
    key: str,
    default: float | None = None,
    base_dir: str | os.PathLike | None = None,
) -> float | None:
    raw = get_env(key, base_dir=base_dir)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be a number (got {raw!r})") from exc


def clear_cache() -> None:
    _load.cache_clear()
