#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SGLANG_REPO="${SGLANG_REPO:-/home/qi/sglang}"
export MARCONI_MODEL_PATH="${MARCONI_MODEL_PATH:-nvidia/NVIDIA-Nemotron-Nano-9B-v2}"
export MARCONI_TP_SIZE="${MARCONI_TP_SIZE:-1}"
export MARCONI_MIN_GPUS="${MARCONI_MIN_GPUS:-1}"
export MARCONI_EVAL_MODE="${MARCONI_EVAL_MODE:-fast}"
export MARCONI_CONTEXT_LENGTH="${MARCONI_CONTEXT_LENGTH:-32768}"
export MARCONI_RESULTS_DIR="${MARCONI_RESULTS_DIR:-$ROOT_DIR/results}"

cd "$ROOT_DIR"

printf '%s\n' "Workspace: $ROOT_DIR"
printf '%s\n' "Target repo: $SGLANG_REPO"
printf '%s\n' "Model: $MARCONI_MODEL_PATH"
printf '%s\n' "TP size: $MARCONI_TP_SIZE"
printf '%s\n' "Eval mode: $MARCONI_EVAL_MODE"
printf '\n%s\n\n' "Paste the contents of hermes_kickoff_prompt.txt into Hermes after it starts."

exec hermes
