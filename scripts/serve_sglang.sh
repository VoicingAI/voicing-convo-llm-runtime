#!/bin/bash
# Serve Voicing-Convo-V2-35B-MOE with SGLang.
#
# Usage:
#   MODEL_DIR=/models/Voicing-Convo-V2-35B-MOE scripts/serve_sglang.sh
#   MODEL_DIR=... PORT=8000 TP=2 CONTEXT_LEN=131072 scripts/serve_sglang.sh
#   MODEL_DIR=... scripts/serve_sglang.sh --attention-backend triton   # extra flags appended
#
# The stock SGLang entry point is used. This package's voicing_runtime is put on
# PYTHONPATH, which registers the model architecture and the reasoning /
# tool-call parsers at interpreter startup -- in the server and in every worker
# process SGLang spawns. Nothing is installed and no engine source is modified.
#
# Verified on SGLang 0.5.16. Run tests/test_model_registration.py and
# tests/test_sglang_parsers.py in your SGLang environment before first deploy.
set -euo pipefail

PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${MODEL_DIR:?set MODEL_DIR to the downloaded Voicing-Convo-V2-35B-MOE directory}"
[ -f "$MODEL_DIR/config.json" ] || { echo "error: no config.json in $MODEL_DIR" >&2; exit 1; }

SERVED_NAME="${SERVED_NAME:-voicing-ai/Voicing-Convo-V2-35B-MOE}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP="${TP:-1}"
CONTEXT_LEN="${CONTEXT_LEN:-65536}"
MEM_FRACTION="${MEM_FRACTION:-0.90}"
ATTN_BACKEND="${ATTN_BACKEND:-flashinfer}"

export PYTHONPATH="$PKG/voicing_runtime${PYTHONPATH:+:$PYTHONPATH}"

exec python3 -m sglang.launch_server \
  --model "$MODEL_DIR" \
  --served-model-name "$SERVED_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --tp "$TP" \
  --dtype bfloat16 \
  --mem-fraction-static "$MEM_FRACTION" \
  --attention-backend "$ATTN_BACKEND" \
  --sampling-backend pytorch \
  --reasoning-parser voicing \
  --tool-call-parser voicing \
  --context-length "$CONTEXT_LEN" \
  --chunked-prefill-size 4096 \
  "$@"
