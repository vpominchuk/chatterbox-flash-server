# Chatterbox Flash — curl playbook

Copy-pasteable commands for every endpoint. Assumes the server is running on
the default `http://127.0.0.1:8000` and you have `curl` + `jq`.

```bash
BASE=http://127.0.0.1:8000
```

Start the server in another terminal:

```bash
cd /home/vasyl/apps/chatterbox/flash
.venv/bin/python -m flash_server.main      # or: .venv/bin/chatterbox-flash-server
```

Prepare a reference voice (any WAV/MP3 your reference should be cloned from).
The server decodes it with the model, so a normal audio file works.

```bash
# e.g. a 24 kHz mono clip
cp /path/to/your/reference.wav ref.wav
```

---

## 1. Health / pool state

`GET /status` — pool + runtime state. Returns `503` with
`"model_state":"failed"` if the baseline pool couldn't initialize.

```bash
curl -s $BASE/status | jq
```

Expected shape:

```json
{
  "model_state": "ready",
  "workers": {
    "configured": 5, "loaded": 3, "busy": 1, "idle": 2,
    "min_spare": 2, "available": 2
  },
  "runtime": { "backend": "auto", "device": "cuda", "dtype": "bfloat16" }
}
```

---

## 2. Synthesize (the main call)

`POST /v1/syntheses` — multipart form. Required: `text` (≤5000 chars) and
`reference_audio`. Everything else is an optional generation setting.
Returns `201`.

### Minimal

```bash
curl -s -X POST $BASE/v1/syntheses \
  -F "text=Hello, world." \
  -F "reference_audio=@ref.wav;type=audio/wav" \
  | jq
```

### With caller overrides

Any subset of the settings below can be sent; unset ones use the defaults.

```bash
curl -s -X POST $BASE/v1/syntheses \
  -F "text=The quick brown fox jumps over the lazy dog." \
  -F "reference_audio=@ref.wav;type=audio/wav" \
  -F "exaggeration=0.7" \
  -F "num_steps=20" \
  -F "temperature=1.5" \
  -F "cfg_scale=1.2" \
  -F "normalize_text=true" \
  | jq
```

### Capture the id for the follow-up calls

```bash
ID=$(curl -s -X POST $BASE/v1/syntheses \
  -F "text=Testing the API." \
  -F "reference_audio=@ref.wav;type=audio/wav" \
  | jq -r '.id')
echo "created: $ID"
```

### Generation settings reference

| Field | Range | Default |
| --- | --- | --- |
| `exaggeration` | 0–1 | `0.5` |
| `normalize_text` | `true`/`false` | `true` |
| `num_steps` | 1–50 | `10` |
| `temperature` | 0–2 | `0.2` |
| `time_shift_tau` | 0–1 | `0.5` |
| `omnivoice_schedule_t_shift` | 0–1 | `0.5` |
| `cfg_scale` | 0–3 | `1.0` |
| `position_temperature` | 0–10 | `5.0` |
| `max_speech_tokens` | 1–4096 | _(unset)_ |
| `n_cfm_timesteps` | 1–10 | `2` |

---

## 3. Fetch metadata

`GET /v1/syntheses/{id}` — the same resource. `status` is always `completed`
in v1 (field retained for future async values). `404` for unknown ids.

```bash
curl -s $BASE/v1/syntheses/$ID | jq
```

---

## 4. Download the audio

`GET /v1/audio/{id}.wav` — `Content-Type: audio/wav`. `404` for unknown ids.

```bash
# just download to a file
curl -s -OJ $BASE/v1/audio/$ID.wav

# or inspect headers + body size first
curl -s -D - -o out.wav $BASE/v1/audio/$ID.wav | grep -iE "HTTP/|content-type|content-length"

# save to a specific name
curl -s -o hello.wav $BASE/v1/audio/$ID.wav && ls -l hello.wav
```

---

## 5. OpenAPI

```bash
curl -s $BASE/openapi.json | jq '.paths | keys'     # list routes
curl -s $BASE/openapi.json | jq '.paths."/v1/syntheses".post'   # request schema
```

Interactive UI: `http://127.0.0.1:8000/docs` (ReDoc at `/redoc`).

---

## 6. Error cases to try

Each returns a consistent `{"code","message","details"}` body.

```bash
# 422 — blank text
curl -s -w "\nHTTP %{http_code}\n" -X POST $BASE/v1/syntheses \
  -F "text=   " -F "reference_audio=@ref.wav;type=audio/wav"

# 422 — oversized text (>5000 chars)
curl -s -w "\nHTTP %{http_code}\n" -X POST $BASE/v1/syntheses \
  -F "text=$(printf 'a%.0s' {1..5001})" -F "reference_audio=@ref.wav;type=audio/wav"

# 422 — out-of-range setting
curl -s -w "\nHTTP %{http_code}\n" -X POST $BASE/v1/syntheses \
  -F "text=hi" -F "num_steps=99" -F "reference_audio=@ref.wav;type=audio/wav"

# 422 — undecodable "audio" (not a real audio file)
curl -s -w "\nHTTP %{http_code}\n" -X POST $BASE/v1/syntheses \
  -F "text=hi" -F "reference_audio=@README.md;type=audio/wav"

# 413 — oversized upload (needs a small MAX_UPLOAD_BYTES, e.g. restart with
# MAX_UPLOAD_BYTES=1024) then:
curl -s -w "\nHTTP %{http_code}\n" -X POST $BASE/v1/syntheses \
  -F "text=hi" -F "reference_audio=@bigfile.bin;type=audio/wav"

# 404 — unknown id
curl -s -w "\nHTTP %{http_code}\n" $BASE/v1/syntheses/does-not-exist
curl -s -w "\nHTTP %{http_code}\n" $BASE/v1/audio/does-not-exist.wav
```

---

## 7. One-liner: synth + download

```bash
ID=$(curl -s -X POST $BASE/v1/syntheses \
  -F "text=Hello from curl." \
  -F "reference_audio=@ref.wav;type=audio/wav" | jq -r '.id')
curl -s -o "speech_$ID.wav" "$BASE/v1/audio/$ID.wav"
echo "wrote speech_$ID.wav"; ls -l "speech_$ID.wav"
```
