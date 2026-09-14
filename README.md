# Chatterbox Flash TTS Server

A synchronous zero-shot English text-to-speech API built on
[Chatterbox Flash](https://huggingface.co/ResembleAI/chatterbox-flash).

Send text plus a reference voice; get back a generated WAV.

- FastAPI service: `POST` a request, receive a `201` with a synthesis resource and a download URL
- Elastic pool of model workers — each worker is a **separate process with its own CUDA context**, warmed at startup, expanded under load, and reaped after an idle TTL
- Generated audio stored under a UUID (`output/`), retrievable any time via `GET`
- Structured errors: `{"code", "message", "details"}` on every failure path
- Server core and test suite run without PyTorch or model weights (fake backend)

## Requirements

- Python ≥ 3.10
- (Optional) An NVIDIA GPU with a CUDA driver for GPU inference

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"     # server + tests (fake backend)
.venv/bin/pip install -e ".[model]"   # real model: torch, librosa, chatterbox-flash
```

## GPU (CUDA) setup

The server defaults to `DEVICE=cuda`. For CPU-only machines, set `DEVICE=cpu`
and use the CPU PyTorch build. For GPU inference, install a CUDA-enabled
PyTorch that matches your driver, your architecture, and your GPU's compute
capability.

**1. Inspect your hardware**

```bash
nvidia-smi    # "CUDA Version: <max runtime>" (e.g. 13.0)
uname -m      # x86_64 or aarch64 (ARM)
```

**2. Install the model, then override the torch pair**

`chatterbox-tts` pins `torch==2.6.0` / `torchaudio==2.6.0`. If that version
doesn't support your GPU or architecture, install a matching `torch` +
`torchaudio` pair **after** the model extra — and keep both at the **same
version** (their C++ ABI must match). Pip's resolver warning about the pin is
expected and safe:

```bash
.venv/bin/pip install -e ".[model]"
# Example: CUDA 13.0 wheels — swap cu130 for your channel (see below):
.venv/bin/pip install "torch==2.11.0" "torchaudio==2.11.0" \
    --index-url https://download.pytorch.org/whl/cu130
```

**3. Confirm the device settings** (already the defaults, shown for completeness):

```bash
DEVICE=cuda
DTYPE=bfloat16
BACKEND=auto
```

**4. Verify**

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
curl -s localhost:8000/status | jq .runtime   # expect "device": "cuda"
```

### Choosing the PyTorch CUDA channel

PyTorch publishes one wheel set per CUDA channel:

| Channel | CUDA | Index |
| --- | --- | --- |
| `cu118` | 11.8 | `https://download.pytorch.org/whl/cu118` |
| `cu124` | 12.4 | `https://download.pytorch.org/whl/cu124` |
| `cu126` | 12.6 | `https://download.pytorch.org/whl/cu126` |
| `cu128` | 12.8 | `https://download.pytorch.org/whl/cu128` |
| `cu130` | 13.0 | `https://download.pytorch.org/whl/cu130` |

Pick the channel whose CUDA supports your GPU's compute capability and is ≤
your driver's max runtime. Both `x86_64` and `aarch64` wheels are published on
the official index (e.g. Blackwell `sm_121` needs `cu128`/`cu130`; `cu126`
fails with "no kernel image is available").

- PyTorch ↔ CUDA compatibility: <https://pytorch.org/docs/stable/notes/cuda.html>
- Interactive installer: <https://pytorch.org/get-started/locally/>

## Run

```bash
.venv/bin/python -m flash_server.main    # or: .venv/bin/chatterbox-flash-server
```

Interactive docs: `http://127.0.0.1:8000/docs`.

## Configuration

Precedence per option: real environment > `.env.local` > `.env` > built-in default.

| Variable | Default | Meaning |
| --- | --- | --- |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | Bind address. |
| `MAX_WORKERS` | `1` | Maximum concurrently loaded model workers (processes). |
| `MIN_SPARE_WORKERS` | `1` | Workers kept loaded at all times (baseline + reap floor). |
| `KEEP_ALIVE_TTL_SECONDS` | `300` | Idle TTL before excess workers are unloaded. |
| `MODEL_REPO` | `ResembleAI/chatterbox-flash` | Hugging Face checkpoint. |
| `DEVICE` | `cuda` | `cuda`, `cpu`, `mps`, … |
| `BACKEND` | `auto` | `auto`, `flashinfer`, `torch`, `mlx`. |
| `DTYPE` | `bfloat16` | `bfloat16`, `float16`, `float32`. |
| `BLOCK_SIZE` | `16` | Decoder block size (`drf_block_size`). |
| `OUTPUT_DIR` | `output` | Where generated WAVs are stored. |
| `MAX_UPLOAD_BYTES` | `26214400` | Max reference-audio upload (25 MiB). |

## API

### `POST /v1/syntheses` — generate speech

Multipart form. Required: `text` (≤ 5000 chars) and `reference_audio`
(≤ 25 MiB). Optional generation settings: `exaggeration`, `normalize_text`,
`num_steps`, `temperature`, `time_shift_tau`, `omnivoice_schedule_t_shift`,
`cfg_scale`, `position_temperature`, `max_speech_tokens`, `n_cfm_timesteps`.

```bash
curl -s -X POST http://127.0.0.1:8000/v1/syntheses \
  -F "text=Hello, world." \
  -F "reference_audio=@voice.mp3;type=audio/wav" | jq
```

```json
{
  "id": "3bae709f-40f9-46e2-82ea-f90fe4de6452",
  "status": "completed",
  "text": "Hello, world.",
  "audio_url": "/v1/audio/3bae709f-40f9-46e2-82ea-f90fe4de6452.wav",
  "sample_rate_hz": 24000,
  "duration_seconds": 1.43,
  "created_at": "2026-09-13T21:45:00Z",
  "settings": { "exaggeration": 0.5, "normalize_text": true, "num_steps": 10, "…": "…" }
}
```

### `GET /v1/syntheses/{id}` — fetch metadata

```bash
curl -s http://127.0.0.1:8000/v1/syntheses/3bae709f-40f9-46e2-82ea-f90fe4de6452 | jq
```

### `GET /v1/audio/{id}.wav` — download the audio

```bash
curl -s -o hello.wav http://127.0.0.1:8000/v1/audio/3bae709f-40f9-46e2-82ea-f90fe4de6452.wav
```

### `GET /status` — pool and runtime state

Returns `503` with `"model_state": "failed"` if the worker pool could not start.

```bash
curl -s http://127.0.0.1:8000/status | jq
```

```json
{
  "model_state": "ready",
  "workers": { "configured": 10, "loaded": 3, "busy": 0, "idle": 3, "min_spare": 3, "available": 3 },
  "runtime": { "backend": "auto", "device": "cuda", "dtype": "bfloat16" }
}
```

### Errors

| Code | When |
| --- | --- |
| `404` | Unknown synthesis id. |
| `413` | Reference audio exceeds `MAX_UPLOAD_BYTES`. |
| `422` | Invalid field, or reference audio that cannot be decoded. |
| `503` | Worker pool not ready / failed to initialize. |

## Tests

```bash
.venv/bin/python -m pytest
```

Covers the API contract, validation/error mapping, and worker-pool lifecycle
(startup, reuse, expansion, FIFO wait, TTL reaping, baseline survival, failure
recovery) using a deterministic fake backend — no PyTorch required.
