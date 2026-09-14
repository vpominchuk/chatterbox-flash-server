"""Structured errors and their HTTP mapping for the Flash TTS server."""
from __future__ import annotations


class FlashError(Exception):
    """Base class for all structured, client-facing errors.

    Subclasses set a stable machine-readable ``code``, an HTTP ``status``,
    and carry an optional ``details`` mapping for diagnostics.
    """

    code: str = "internal_error"
    status: int = 500

    def __init__(self, message: str, *, code: str | None = None,
                 status: int | None = None, details: dict | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code
        if status is not None:
            self.status = status
        self.message = message
        self.details = details or {}

    def body(self) -> dict:
        return {"code": self.code, "message": self.message, "details": self.details}


class ConfigError(FlashError):
    code = "invalid_config"
    status = 500


class InvalidTextError(FlashError):
    code = "invalid_text"
    status = 422


class InvalidReferenceAudioError(FlashError):
    code = "invalid_audio"
    status = 422


class UploadTooLargeError(FlashError):
    code = "upload_too_large"
    status = 413


class SynthesisNotFoundError(FlashError):
    code = "not_found"
    status = 404


class ModelNotReadyError(FlashError):
    code = "unavailable"
    status = 503


class WorkerCapacityError(FlashError):
    code = "unavailable"
    status = 503


class WorkerFailureError(FlashError):
    code = "synthesis_failed"
    status = 500
