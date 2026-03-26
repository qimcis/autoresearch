#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/workspace/autoresearch
RUN_SCRIPT="$WORKSPACE/run_marconi_batch.sh"
LOG_DIR="$WORKSPACE/supervisor_logs"
STATE_DIR="$WORKSPACE/.supervisor"
LOCK_DIR="$STATE_DIR/lock"
SUPERVISOR_LOG="$LOG_DIR/supervisor.log"
FAIL_SLEEP_SECONDS="${FAIL_SLEEP_SECONDS:-300}"
SUCCESS_SLEEP_SECONDS="${SUCCESS_SLEEP_SECONDS:-0}"

mkdir -p "$LOG_DIR" "$STATE_DIR"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Another Marconi supervisor appears to be running: $LOCK_DIR" >&2
  exit 1
fi
trap 'rmdir "$LOCK_DIR"' EXIT

echo "[$(date -u +%FT%TZ)] supervisor started" | tee -a "$SUPERVISOR_LOG"
while true; do
  echo "[$(date -u +%FT%TZ)] launching batch" | tee -a "$SUPERVISOR_LOG"
  if "$RUN_SCRIPT"; then
    echo "[$(date -u +%FT%TZ)] batch succeeded" | tee -a "$SUPERVISOR_LOG"
    if [[ "$SUCCESS_SLEEP_SECONDS" != "0" ]]; then
      sleep "$SUCCESS_SLEEP_SECONDS"
    fi
  else
    rc=$?
    echo "[$(date -u +%FT%TZ)] batch failed rc=$rc; sleeping ${FAIL_SLEEP_SECONDS}s" | tee -a "$SUPERVISOR_LOG"
    sleep "$FAIL_SLEEP_SECONDS"
  fi
done
