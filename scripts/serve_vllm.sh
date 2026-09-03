#!/bin/bash
# Serve Voicing-Convo-V2-35B-MOE with vLLM.
#
# Usage:
#   MODEL_DIR=/models/Voicing-Convo-V2-35B-MOE scripts/serve_vllm.sh
#   MODEL_DIR=... PORT=8000 TP=2 MAX_MODEL_LEN=131072 scripts/serve_vllm.sh
#   MODEL_DIR=... scripts/serve_vllm.sh --enable-prefix-caching   # extra flags appended
#
# vLLM loads the two parsers from file paths given on the command line. The
# model architecture registers through this package's voicing_runtime on
# PYTHONPATH, which reaches vLLM's engine-core process as well. Nothing is
# installed and no engine source is modified.
#
# Verified on vLLM 0.28.0. Run tests/test_model_registration.py and
# tests/test_vllm_parsers.py in your vLLM environment before first deploy.
set -euo pipefail

PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${MODEL_DIR:?set MODEL_DIR to the downloaded Voicing-Convo-V2-35B-MOE directory}"
[ -f "$MODEL_DIR/config.json" ] || { echo "error: no config.json in $MODEL_DIR" >&2; exit 1; }

SERVED_NAME="${SERVED_NAME:-voicing-ai/Voicing-Convo-V2-35B-MOE}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP="${TP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
# This hybrid architecture allots one Mamba cache block per decode sequence.
# vLLM's default max_num_seqs (1024) does not fit alongside the weights and
# fails CUDA graph capture; 256 fits on a 96 GB card at TP=1.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"

PARSERS="$PKG/voicing_parsers/vllm"
export PYTHONPATH="$PKG/voicing_runtime${PYTHONPATH:+:$PYTHONPATH}"

exec vllm serve "$MODEL_DIR" \
  --served-model-name "$SERVED_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --dtype bfloat16 \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --reasoning-parser-plugin "$PARSERS/voicing_reasoning_parser.py" \
  --reasoning-parser voicing \
  --tool-parser-plugin "$PARSERS/voicing_tool_parser.py" \
  --enable-auto-tool-choice \
  --tool-call-parser voicing \
  "$@"
