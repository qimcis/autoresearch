#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/workspace/autoresearch
TARGET_REPO=/workspace/sglang
HERMES_ROOT=/workspace/hermes-agent
HERMES_PY="$HERMES_ROOT/.venv/bin/python"
PROMPT_FILE="$WORKSPACE/hermes_cron_prompt.txt"
LOG_DIR="$WORKSPACE/supervisor_logs"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
BATCH_DIR="$LOG_DIR/$TS"
mkdir -p "$BATCH_DIR"

export SGLANG_REPO=/workspace/sglang
export MARCONI_MODEL_PATH=nvidia/NVIDIA-Nemotron-Nano-9B-v2
export MARCONI_TP_SIZE=1
export MARCONI_MIN_GPUS=1
export MARCONI_EVAL_MODE=fast
export MARCONI_CONTEXT_LENGTH=32768
export MARCONI_RESULTS_DIR=/workspace/autoresearch/results
export MARCONI_EXTRA_SERVER_ARGS='--mamba-scheduler-strategy no_buffer'

cd "$WORKSPACE"

if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "Missing prompt file: $PROMPT_FILE" >&2
  exit 1
fi

PROMPT="$(cat "$PROMPT_FILE")"

echo "[$(date -u +%FT%TZ)] starting bounded Marconi batch" | tee "$BATCH_DIR/meta.log"
echo "workspace=$WORKSPACE" | tee -a "$BATCH_DIR/meta.log"
echo "target_repo=$TARGET_REPO" | tee -a "$BATCH_DIR/meta.log"

set +e
"$HERMES_PY" -m hermes_cli.main chat \
  --yolo \
  -q "$PROMPT" \
  > >(tee "$BATCH_DIR/hermes.out") \
  2> >(tee "$BATCH_DIR/hermes.err" >&2)
status=$?
set -e

echo "[$(date -u +%FT%TZ)] finished status=$status" | tee -a "$BATCH_DIR/meta.log"
exit $status
