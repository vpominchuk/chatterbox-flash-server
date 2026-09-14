"""Synthesis record storage and WAV persistence under a UUID-derived name."""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from .audio import write_wav
from .model import GenerationSettings


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class SynthesisRecord:
    id: str
    status: str
    text: str
    audio_path: str
    sample_rate_hz: int
    duration_seconds: float
    created_at: str
    settings: dict = field(default_factory=dict)


class Store:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._records: dict[str, SynthesisRecord] = {}

    def save(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        text: str,
        settings: GenerationSettings,
    ) -> SynthesisRecord:
        synthesis_id = str(uuid.uuid4())
        audio_path = os.path.join(self.output_dir, f"{synthesis_id}.wav")
        write_wav(audio_path, waveform, sample_rate)
        duration = round(len(waveform) / sample_rate, 3)
        record = SynthesisRecord(
            id=synthesis_id,
            status="completed",
            text=text,
            audio_path=audio_path,
            sample_rate_hz=int(sample_rate),
            duration_seconds=duration,
            created_at=_utc_now(),
            settings=settings.as_dict(),
        )
        self._records[synthesis_id] = record
        return record

    def get(self, synthesis_id: str) -> SynthesisRecord | None:
        return self._records.get(synthesis_id)
