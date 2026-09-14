"""Process-isolated model workers.

Each worker runs in its own OS process with its own CUDA context, so
concurrent workers can be scheduled on the GPU in parallel instead of
serializing on a single process's default stream. The model never crosses
the process boundary: the parent only sends requests over a pipe and
receives the waveform (or the raised exception) back.
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np

_ctx = mp.get_context("spawn")


class WorkerProcessError(RuntimeError):
    """Raised when a worker process dies or misbehaves."""


def worker_main(loader, conn) -> None:
    """Child-side message loop: load the model, serve, then exit on release."""
    model = None
    try:
        while True:
            msg = conn.recv()
            kind = msg[0]
            if kind == "load":
                try:
                    model = loader()
                    conn.send(("loaded", model.sample_rate))
                except Exception as exc:
                    conn.send(("error", exc))
                    return
            elif kind == "run":
                _, text, reference_path, settings = msg
                try:
                    waveform = model.synthesize(text, reference_path, settings)
                    conn.send(("ok", waveform))
                except Exception as exc:
                    conn.send(("error", exc))
            elif kind == "release":
                if model is not None:
                    model.release()
                conn.send(("released",))
                return
    except Exception as exc:
        try:
            conn.send(("error", exc))
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


class WorkerProcess:
    """Parent-side handle for one model worker process."""

    def __init__(self, index: int, loader) -> None:
        self.index = index
        self.sample_rate: int | None = None
        self._parent_conn, child_conn = _ctx.Pipe()
        self._proc = _ctx.Process(
            target=worker_main,
            args=(loader, child_conn),
            name=f"flash-worker-{index}",
            daemon=True,
        )
        self._proc.start()
        child_conn.close()

    @property
    def alive(self) -> bool:
        return self._proc.is_alive()

    def _recv(self):
        try:
            return self._parent_conn.recv()
        except EOFError as exc:
            raise WorkerProcessError(
                f"worker process {self.index} exited unexpectedly"
            ) from exc

    def load(self) -> None:
        self._parent_conn.send(("load",))
        resp = self._recv()
        if resp[0] == "loaded":
            self.sample_rate = int(resp[1])
            return
        raise resp[1]

    def run(self, text: str, reference_path: str, settings) -> np.ndarray:
        self._parent_conn.send(("run", text, reference_path, settings))
        resp = self._recv()
        if resp[0] == "ok":
            return resp[1]
        raise resp[1]

    def release(self) -> None:
        """Ask the child to unload its model, then reap the process."""
        try:
            self._parent_conn.send(("release",))
            self._recv()
        except Exception:
            pass
        self.terminate()

    def terminate(self) -> None:
        """Kill and reap the child process. Safe to call repeatedly."""
        proc = self._proc
        if proc.is_alive():
            proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5)
        try:
            self._parent_conn.close()
        except Exception:
            pass
