"""Deployment configuration, resolved through the :mod:`env` module.

Precedence per key (highest first): process env, ``.env.local``, ``.env``.
Backend, device, dtype, and block size are deployment-only settings per the
plan and are not exposed as client controls.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import env
from .errors import ConfigError

DEFAULT_MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MiB


def _str(name, default, base_dir=None):
    return env.get_env_str(name, default, base_dir)


def _int(name, default, base_dir=None):
    try:
        return env.get_env_int(name, default, base_dir)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def _float(name, default, base_dir=None):
    try:
        return env.get_env_float(name, default, base_dir)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


@dataclass(frozen=True)
class Config:
    max_workers: int
    min_spare_workers: int
    keep_alive_ttl_seconds: float
    model_repo: str
    device: str
    backend: str
    dtype: str
    block_size: int
    output_dir: str
    max_upload_bytes: int

    @classmethod
    def from_env(cls, base_dir=None) -> "Config":
        cfg = cls(
            max_workers=_int("MAX_WORKERS", 1, base_dir),
            min_spare_workers=_int("MIN_SPARE_WORKERS", 1, base_dir),
            keep_alive_ttl_seconds=_float("KEEP_ALIVE_TTL_SECONDS", 300, base_dir),
            model_repo=_str("MODEL_REPO", "ResembleAI/chatterbox-flash", base_dir),
            device=_str("DEVICE", "cuda", base_dir),
            backend=_str("BACKEND", "auto", base_dir),
            dtype=_str("DTYPE", "bfloat16", base_dir),
            block_size=_int("BLOCK_SIZE", 16, base_dir),
            output_dir=_str("OUTPUT_DIR", "output", base_dir),
            max_upload_bytes=_int("MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES, base_dir),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.max_workers < 1:
            raise ConfigError(f"MAX_WORKERS must be >= 1 (got {self.max_workers})")
        if self.min_spare_workers < 0 or self.min_spare_workers > self.max_workers:
            raise ConfigError(
                "MIN_SPARE_WORKERS must be within [0, MAX_WORKERS] "
                f"(got {self.min_spare_workers}, MAX_WORKERS={self.max_workers})"
            )
        if self.keep_alive_ttl_seconds < 0:
            raise ConfigError(
                f"KEEP_ALIVE_TTL_SECONDS must be >= 0 (got {self.keep_alive_ttl_seconds})"
            )
        if self.block_size < 1:
            raise ConfigError(f"BLOCK_SIZE must be >= 1 (got {self.block_size})")
        if self.max_upload_bytes < 1:
            raise ConfigError(f"MAX_UPLOAD_BYTES must be >= 1 (got {self.max_upload_bytes})")
