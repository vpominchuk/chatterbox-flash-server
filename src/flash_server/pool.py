"""Elastic pool of independent model workers.

Concurrency is bounded by an ``asyncio.Semaphore`` sized to ``max_workers``:
each in-flight request holds exactly one permit and owns exactly one worker,
so the number of live workers can never exceed ``max_workers``. Idle workers
are always reused before a new one is loaded; when at capacity, additional
requests wait on the semaphore (FIFO) until a permit frees up.

All state mutation happens on the event loop. Each worker is a separate
OS process with its own CUDA context; blocking calls into workers run via
``asyncio.to_thread`` so the loop is never blocked.
"""
from __future__ import annotations

import asyncio
import time
from typing import Callable

from .errors import ModelNotReadyError
from .worker import WorkerProcess

LOADING = "loading"
READY = "ready"
BUSY = "busy"
IDLE = "idle"
RELEASING = "releasing"


class WorkerPool:
    def __init__(
        self,
        loader: Callable[[], object],
        *,
        max_workers: int,
        min_spare_workers: int,
        keep_alive_ttl_seconds: float,
    ) -> None:
        self._loader = loader
        self.max_workers = max_workers
        self.min_spare_workers = min_spare_workers
        self.keep_alive_ttl_seconds = keep_alive_ttl_seconds
        self._workers: list[Worker] = []
        self._next_index = 0
        self._sem = asyncio.Semaphore(max_workers)
        self.model_state = LOADING
        self._loader_calls = 0

    # ---- introspection ----------------------------------------------------
    def _alloc_index(self) -> int:
        index = self._next_index
        self._next_index += 1
        return index

    @property
    def loaded(self) -> int:
        return len(self._workers)

    @property
    def loader_calls(self) -> int:
        return self._loader_calls

    def counts(self) -> dict:
        busy = sum(1 for w in self._workers if w.state in (LOADING, BUSY, RELEASING))
        idle = sum(1 for w in self._workers if w.state == IDLE)
        return {
            "configured": self.max_workers,
            "loaded": len(self._workers),
            "busy": busy,
            "idle": idle,
            "min_spare": self.min_spare_workers,
            "available": idle,
        }

    def status(self) -> dict:
        return {"model_state": self.model_state, "workers": self.counts()}

    # ---- worker lifecycle -------------------------------------------------
    async def _load_worker(self) -> WorkerProcess:
        worker = WorkerProcess(self._alloc_index(), self._loader)
        worker.state = LOADING
        worker.idle_since = None
        self._workers.append(worker)
        try:
            await asyncio.to_thread(worker.load)
            self._loader_calls += 1
        except Exception:
            self._drop(worker)
            raise
        return worker

    def _pick_idle(self) -> WorkerProcess | None:
        for worker in self._workers:
            if worker.state == IDLE:
                return worker
        return None

    def _drop(self, worker: WorkerProcess) -> None:
        try:
            self._workers.remove(worker)
        except ValueError:
            pass
        worker.terminate()

    async def ensure_baseline(self) -> bool:
        """Load workers up to the minimum spare baseline.

        Returns True (and leaves state ``ready``) on success. On any load
        failure, marks the pool ``failed`` and returns False.
        """
        if self.model_state == "failed":
            return False
        while len(self._workers) < self.min_spare_workers:
            try:
                worker = await self._load_worker()
            except Exception:
                self.model_state = "failed"
                return False
            worker.state = IDLE
            worker.idle_since = time.monotonic()
        self.model_state = "ready"
        return True

    async def initialize(self) -> bool:
        return await self.ensure_baseline()

    # ---- request acquisition ---------------------------------------------
    async def acquire(self) -> WorkerProcess:
        if self.model_state != "ready":
            raise ModelNotReadyError("model pool is not ready")
        await self._sem.acquire()
        try:
            worker = self._pick_idle()
            if worker is None:
                # Safe to spawn: holding a permit implies < max_workers live
                # workers while no idle worker exists.
                worker = await self._load_worker()
            worker.state = BUSY
            worker.idle_since = None
            return worker
        except Exception:
            self._sem.release()
            raise

    def release(self, worker: WorkerProcess, *, ok: bool) -> None:
        if ok:
            worker.state = IDLE
            worker.idle_since = time.monotonic()
        else:
            self._drop(worker)
        self._sem.release()

    def mark_idle(self, worker: WorkerProcess) -> None:
        worker.state = IDLE
        worker.idle_since = time.monotonic()

    def drop(self, worker: WorkerProcess) -> None:
        self._drop(worker)

    def release_sem(self) -> None:
        self._sem.release()

    # ---- reaping ----------------------------------------------------------
    async def reap_once(self) -> int:
        """Unload idle workers past their TTL, never below the spare baseline."""
        if self.model_state != "ready":
            return 0
        now = time.monotonic()
        ttl = self.keep_alive_ttl_seconds
        idle = [
            w
            for w in self._workers
            if w.state == IDLE and (now - (w.idle_since or now)) >= ttl
        ]
        budget = max(0, len(self._workers) - self.min_spare_workers)
        to_unload = idle[:budget]
        for worker in to_unload:
            worker.state = RELEASING
            await asyncio.to_thread(worker.release)
            self._drop(worker)
        return len(to_unload)

    async def reap_loop(self, interval: float = 1.0) -> None:
        while True:
            await asyncio.sleep(interval)
            try:
                await self.reap_once()
            except Exception:
                # Reaping must never take the pool down.
                continue

    # ---- shutdown ---------------------------------------------------------
    async def shutdown(self) -> None:
        for worker in list(self._workers):
            worker.state = RELEASING
            await asyncio.to_thread(worker.release)
        self._workers.clear()
