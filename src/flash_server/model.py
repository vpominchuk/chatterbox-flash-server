"""Model layer: generation settings plus a swappable TTS model interface.

The server depends only on the :class:`TTSModel` protocol and a zero-arg
loader callable. The real Chatterbox-Flash backend is imported lazily inside
:class:`ChatterboxModel` so the package imports and its tests run without
PyTorch / the multi-GB weights installed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .errors import InvalidReferenceAudioError


@dataclass(frozen=True)
class GenerationSettings:
    """Effective, validated generation parameters for one synthesis.

    Defaults follow the server plan (which intentionally differs from a few
    upstream defaults).
    """

    exaggeration: float = 0.5
    normalize_text: bool = True
    num_steps: int = 10
    temperature: float = 0.2
    time_shift_tau: float = 0.5
    omnivoice_schedule_t_shift: float = 0.5
    cfg_scale: float = 1.0
    position_temperature: float = 5.0
    max_speech_tokens: int | None = None
    n_cfm_timesteps: int = 2

    def as_dict(self) -> dict:
        result = {
            "exaggeration": self.exaggeration,
            "normalize_text": self.normalize_text,
            "num_steps": self.num_steps,
            "temperature": self.temperature,
            "time_shift_tau": self.time_shift_tau,
            "omnivoice_schedule_t_shift": self.omnivoice_schedule_t_shift,
            "cfg_scale": self.cfg_scale,
            "position_temperature": self.position_temperature,
            "n_cfm_timesteps": self.n_cfm_timesteps,
        }
        if self.max_speech_tokens is not None:
            result["max_speech_tokens"] = self.max_speech_tokens
        return result

    def to_upstream_kwargs(self) -> dict:
        return {
            "exaggeration": self.exaggeration,
            "normalize_text": self.normalize_text,
            "num_steps": self.num_steps,
            "temperature": self.temperature,
            "time_shift_tau": self.time_shift_tau,
            "omnivoice_schedule_t_shift": self.omnivoice_schedule_t_shift,
            "cfg_scale": self.cfg_scale,
            "position_temperature": self.position_temperature,
            "max_speech_tokens": self.max_speech_tokens,
            "n_cfm_timesteps": self.n_cfm_timesteps,
        }


class TTSModel(Protocol):
    """Minimal contract a model worker must satisfy."""

    @property
    def sample_rate(self) -> int: ...

    def warmup(self) -> None: ...

    def synthesize(
        self, text: str, reference_path: str, settings: GenerationSettings
    ) -> np.ndarray: ...

    def release(self) -> None: ...


class FakeTTSModel:
    """Deterministic stand-in used by tests and as a smoke backend.

    Emits a fixed-frequency tone whose length scales with the text so that
    ``duration_seconds`` is always positive. A reference is considered
    decodable only if it is a non-empty RIFF/WAVE file, which is what lets
    the undecodable-audio contract test drive a 422.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        generate_delay: float = 0.0,
        fail_generate: bool = False,
        require_wav_header: bool = True,
    ) -> None:
        self._sample_rate = sample_rate
        self._generate_delay = generate_delay
        self._fail_generate = fail_generate
        self._require_wav_header = require_wav_header
        self.released = False
        self.calls = 0

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def _reference_decodable(self, path: str) -> bool:
        try:
            with open(path, "rb") as handle:
                head = handle.read(12)
        except OSError:
            return False
        if len(head) < 12:
            return False
        if not self._require_wav_header:
            return True
        return head[:4] == b"RIFF" and head[8:12] == b"WAVE"

    def synthesize(
        self, text: str, reference_path: str, settings: GenerationSettings
    ) -> np.ndarray:
        self.calls += 1
        if not self._reference_decodable(reference_path):
            raise InvalidReferenceAudioError("reference_audio could not be decoded")
        if self._generate_delay:
            time.sleep(self._generate_delay)
        if self._fail_generate:
            raise RuntimeError("fake generation failure")
        duration = max(0.1, min(len(text) / 30.0, 3.0))
        n = int(duration * self._sample_rate)
        t = np.arange(n, dtype=np.float32) / self._sample_rate
        seed = sum(ord(ch) for ch in text) % 997
        f0 = 110.0 + seed
        wave = (
            0.5 * np.sin(2 * np.pi * f0 * t)
            + 0.3 * np.sin(2 * np.pi * 2 * f0 * t)
            + 0.2 * np.sin(2 * np.pi * 3 * f0 * t)
        )
        return wave.astype(np.float32)

    def warmup(self) -> None:
        pass

    def release(self) -> None:
        self.released = True


class ChatterboxModel:
    """Real Chatterbox-Flash backend. Imports heavy deps lazily."""

    def __init__(
        self,
        repo_id: str,
        device: str,
        dtype: str,
        block_size: int,
        backend: str = "auto",
    ) -> None:
        import torch

        from chatterbox_flash.tts import ChatterboxFlashTTS

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        resolved = dtype_map.get(dtype.lower(), torch.bfloat16)
        self._model = ChatterboxFlashTTS.from_pretrained(
            repo_id,
            device,
            dtype=resolved,
            drf_block_size=block_size,
        )
        self._sample_rate = int(self._model.sr)
        self._backend = backend
        self._torch = torch

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def synthesize(
        self, text: str, reference_path: str, settings: GenerationSettings
    ) -> np.ndarray:
        import librosa

        try:
            librosa.load(str(reference_path), sr=self._sample_rate)
        except Exception as exc:  # decode/codec failures -> structured 422
            raise InvalidReferenceAudioError(
                "reference_audio could not be decoded"
            ) from exc

        waveform = self._model.generate(
            text,
            audio_prompt_path=str(reference_path),
            backend=self._backend,
            **settings.to_upstream_kwargs(),
        )
        if hasattr(waveform, "detach"):
            waveform = waveform.detach().cpu().numpy()
        return np.asarray(waveform, dtype=np.float32).reshape(-1)

    def warmup(self) -> None:
        """Run one tiny synthesis so first-inference CUDA costs are paid now."""
        import os
        import tempfile

        from .audio import write_wav

        n = int(self._sample_rate * 0.25)
        t = np.arange(n, dtype=np.float32) / self._sample_rate
        tone = (0.2 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        fd, path = tempfile.mkstemp(prefix="flash_warm_", suffix=".wav")
        os.close(fd)
        try:
            write_wav(path, tone, self._sample_rate)
            self.synthesize(
                "warm", path, GenerationSettings(num_steps=2, n_cfm_timesteps=2)
            )
        finally:
            os.unlink(path)

    def release(self) -> None:
        import gc

        torch = self._torch
        self._model = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
