import io
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from flash_server import Config, FakeTTSModel, create_app


def make_wav_bytes(sample_rate: int = 24000, seconds: float = 0.5) -> bytes:
    n = max(1, int(sample_rate * seconds))
    t = np.arange(n) / sample_rate
    data = (0.5 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(data.tobytes())
    return buf.getvalue()


class LoaderFactory:
    """Zero-arg model loader for tests; records instances and load errors."""

    def __init__(self, **model_kwargs):
        self.kwargs = model_kwargs
        self.instances = []
        self.errors = []

    def __call__(self):
        inst = FakeTTSModel(**self.kwargs)
        self.instances.append(inst)
        return inst


_counter = 0


def make_config(tmp_path, **overrides) -> Config:
    global _counter
    _counter += 1
    base = {
        "max_workers": 2,
        "min_spare_workers": 1,
        "keep_alive_ttl_seconds": 300,
        "model_repo": "ResembleAI/chatterbox-flash",
        "device": "cpu",
        "backend": "auto",
        "dtype": "bfloat16",
        "block_size": 16,
        "output_dir": str(tmp_path / f"out_{_counter}"),
        "max_upload_bytes": 25 * 1024 * 1024,
    }
    base.update(overrides)
    return Config(**base)


@pytest.fixture
def wav_bytes():
    return make_wav_bytes()


@pytest.fixture
def make_client(tmp_path):
    def _make(**overrides):
        model_kwargs = overrides.pop("model_kwargs", {})
        cfg = make_config(tmp_path, **overrides)
        app = create_app(cfg=cfg, model_loader=LoaderFactory(**model_kwargs))
        return TestClient(app)

    return _make
