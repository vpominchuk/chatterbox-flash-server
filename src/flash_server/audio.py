"""WAV writing helpers. Kept torch-free so the storage path is testable."""
from __future__ import annotations

import wave

import numpy as np


def write_wav(path: str, waveform: np.ndarray, sample_rate: int) -> None:
    """Write a mono 16-bit PCM WAV file from a float waveform in [-1, 1]."""
    data = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if data.size == 0:
        raise ValueError("waveform must not be empty")
    pcm = (np.clip(data, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())
