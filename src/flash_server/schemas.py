"""Pydantic response models (drives OpenAPI docs)."""
from __future__ import annotations

from pydantic import BaseModel

from .storage import SynthesisRecord


class SynthesisResponse(BaseModel):
    id: str
    status: str
    text: str
    audio_url: str
    sample_rate_hz: int
    duration_seconds: float
    created_at: str
    settings: dict

    @classmethod
    def from_record(cls, record: SynthesisRecord) -> "SynthesisResponse":
        return cls(
            id=record.id,
            status=record.status,
            text=record.text,
            audio_url=f"/v1/audio/{record.id}.wav",
            sample_rate_hz=record.sample_rate_hz,
            duration_seconds=record.duration_seconds,
            created_at=record.created_at,
            settings=record.settings,
        )
