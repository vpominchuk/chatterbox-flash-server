"""Synthesis service: orchestrates the pool, model, and storage per request."""
from __future__ import annotations

import asyncio
import os
import tempfile

from .config import Config
from .errors import InvalidReferenceAudioError, WorkerFailureError
from .model import GenerationSettings
from .pool import WorkerPool
from .storage import Store, SynthesisRecord


class SynthesisService:
    def __init__(self, pool: WorkerPool, store: Store, cfg: Config) -> None:
        self._pool = pool
        self._store = store
        self._cfg = cfg

    def ready(self) -> bool:
        return self._pool.model_state == "ready"

    async def _run_worker(
        self,
        worker,
        text: str,
        audio_bytes: bytes,
        settings: GenerationSettings,
    ):
        fd, tmp_path = tempfile.mkstemp(prefix="flash_ref_", suffix=".wav")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(audio_bytes)
            waveform = await asyncio.to_thread(worker.run, text, tmp_path, settings)
            return waveform, worker.sample_rate
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    async def synthesize(
        self, text: str, audio_bytes: bytes, settings: GenerationSettings
    ) -> SynthesisRecord:
        worker = await self._pool.acquire()
        try:
            try:
                waveform, sample_rate = await self._run_worker(
                    worker, text, audio_bytes, settings
                )
            except InvalidReferenceAudioError:
                # The worker itself is fine; the reference was bad. Keep it.
                self._pool.mark_idle(worker)
                raise
            except Exception as exc:
                # Genuine worker failure: free it, drop it, then restore the
                # minimum baseline (which will load a replacement if possible).
                await asyncio.to_thread(worker.release)
                self._pool.drop(worker)
                await self._pool.ensure_baseline()
                raise WorkerFailureError(
                    "synthesis failed", details={"reason": str(exc)}
                ) from exc
            self._pool.mark_idle(worker)
            return self._store.save(waveform, sample_rate, text, settings)
        finally:
            self._pool.release_sem()
