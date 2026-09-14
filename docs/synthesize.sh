#!/usr/bin/env bash
#
# synthesize.sh — one-shot synth + download against the Flash TTS server.
#
# Synthesizes a WAV from text + a reference voice, downloads it, and prints
# the server URL. Self-contained: needs only curl and jq.
set -euo pipefail

BASE="${BASE:-http://127.0.0.1:8000}"
OUT=""
TEXT=""
REF=""
SETTINGS=()

usage() {
  cat <<'USAGE'
Usage: synthesize.sh TEXT REFERENCE_AUDIO [options]

Synthesizes a WAV from text + a reference voice, downloads it, and prints the
server URL. Requires curl and jq.

Arguments:
  TEXT             Non-empty text to speak (max 5000 chars).
  REFERENCE_AUDIO  Path to a reference audio file to clone the voice from.

Options:
  -b, --base URL      Server base URL           (default: $BASE or http://127.0.0.1:8000)
  -o, --out FILE      Output WAV path           (default: speech_<id>.wav)
      --setting K=V   Extra generation setting  (repeatable), e.g. num_steps=20
  -h, --help          Show this help.

Generation settings (any subset):
  exaggeration(0-1)  normalize_text(true|false)  num_steps(1-50)
  temperature(0-2)   time_shift_tau(0-1)         omnivoice_schedule_t_shift(0-1)
  cfg_scale(0-3)     position_temperature(0-10)  max_speech_tokens(1-4096)
  n_cfm_timesteps(1-10)

Examples:
  ./synthesize.sh "Hello, world." ref.wav
  ./synthesize.sh "Hi" ref.wav --setting num_steps=20 --setting temperature=1.5 -o hi.wav
  BASE=http://localhost:8000 ./synthesize.sh "Hi" ref.wav
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -b|--base)   BASE="$2"; shift 2 ;;
    -o|--out)    OUT="$2"; shift 2 ;;
    --setting)   SETTINGS+=(-F "$2"); shift 2 ;;
    --setting=*) SETTINGS+=(-F "${1#--setting=}"); shift ;;
    -h|--help)   usage; exit 0 ;;
    -*)          echo "error: unknown option: $1" >&2; usage; exit 2 ;;
    *)
      if   [[ -z "$TEXT" ]]; then TEXT="$1"
      elif [[ -z "$REF"  ]]; then REF="$1"
      else echo "error: extra argument: $1" >&2; usage; exit 2; fi
      shift ;;
  esac
done

if [[ -z "$TEXT" || -z "$REF" ]]; then
  echo "error: TEXT and REFERENCE_AUDIO are required." >&2; usage; exit 2
fi

command -v curl >/dev/null 2>&1 || { echo "error: curl not found" >&2; exit 1; }
command -v jq   >/dev/null 2>&1 || { echo "error: jq not found"   >&2; exit 1; }
[[ -f "$REF" ]] || { echo "error: reference audio not found: $REF" >&2; exit 1; }

echo "→ POST $BASE/v1/syntheses"
RESP="$(curl -sS -X POST "$BASE/v1/syntheses" \
  -F "text=$TEXT" \
  -F "reference_audio=@${REF};type=audio/wav" \
  ${SETTINGS[@]+"${SETTINGS[@]}"})"

ID="$(jq -r '.id // empty' <<<"$RESP" 2>/dev/null || true)"
if [[ -z "$ID" ]]; then
  echo "✗ synthesis failed:" >&2
  jq . <<<"$RESP" 2>/dev/null || echo "$RESP" >&2
  exit 1
fi

URL="$(jq -r '.audio_url'          <<<"$RESP")"
STATUS="$(jq -r '.status'          <<<"$RESP")"
DUR="$(jq -r '.duration_seconds'   <<<"$RESP")"
SR="$(jq -r '.sample_rate_hz'      <<<"$RESP")"
echo "✓ id:       $ID"
echo "✓ status:   $STATUS"
echo "✓ duration: ${DUR}s @ ${SR} Hz"
echo "✓ url:      $BASE$URL"

if [[ -n "$OUT" ]]; then OUTPATH="$OUT"; else OUTPATH="speech_${ID}.wav"; fi

echo "↓ GET $BASE$URL → $OUTPATH"
curl -sS -f -o "$OUTPATH" "$BASE$URL"
echo "✓ saved:    $OUTPATH ($(wc -c < "$OUTPATH" | tr -d ' ') bytes)"
