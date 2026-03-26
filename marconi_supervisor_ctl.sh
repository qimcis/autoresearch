#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/workspace/autoresearch
PID_FILE="$WORKSPACE/.supervisor/supervisor.pid"
SUPERVISOR_SCRIPT="$WORKSPACE/supervise_marconi_batches.sh"
LOG_OUT="$WORKSPACE/supervisor_logs/nohup.out"

cmd="${1:-status}"

case "$cmd" in
  start)
    if [[ -f "$PID_FILE" ]]; then
      pid="$(cat "$PID_FILE")"
      if kill -0 "$pid" 2>/dev/null; then
        echo "running pid=$pid"
        exit 0
      fi
    fi
    mkdir -p "$WORKSPACE/.supervisor" "$WORKSPACE/supervisor_logs"
    nohup "$SUPERVISOR_SCRIPT" >"$LOG_OUT" 2>&1 </dev/null &
    echo $! > "$PID_FILE"
    echo "started pid=$(cat "$PID_FILE")"
    ;;
  stop)
    if [[ ! -f "$PID_FILE" ]]; then
      echo "not running"
      exit 0
    fi
    pid="$(cat "$PID_FILE")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
      echo "stopped pid=$pid"
    else
      echo "stale pid file: $pid"
    fi
    rm -f "$PID_FILE"
    ;;
  restart)
    "$0" stop || true
    "$0" start
    ;;
  status)
    if [[ -f "$PID_FILE" ]]; then
      pid="$(cat "$PID_FILE")"
      if kill -0 "$pid" 2>/dev/null; then
        ps -p "$pid" -o pid=,ppid=,stat=,etime=,cmd=
        exit 0
      fi
      echo "stale pid file: $pid"
      exit 1
    fi
    echo "not running"
    ;;
  *)
    echo "usage: $0 {start|stop|restart|status}" >&2
    exit 2
    ;;
esac
