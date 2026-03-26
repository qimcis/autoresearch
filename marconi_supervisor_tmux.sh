#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/workspace/autoresearch
SESSION_NAME="${MARCONI_TMUX_SESSION:-marconi-supervisor}"
SUPERVISOR_SCRIPT="$WORKSPACE/supervise_marconi_batches.sh"
LEGACY_CTL="$WORKSPACE/marconi_supervisor_ctl.sh"
PID_FILE="$WORKSPACE/.supervisor/supervisor.pid"

cmd="${1:-status}"

case "$cmd" in
  start)
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      echo "tmux session already running: $SESSION_NAME"
      exit 0
    fi
    if [[ -x "$LEGACY_CTL" ]]; then
      "$LEGACY_CTL" stop >/dev/null 2>&1 || true
    fi
    rm -f "$PID_FILE"
    tmux new-session -d -s "$SESSION_NAME" "$SUPERVISOR_SCRIPT"
    echo "started tmux session: $SESSION_NAME"
    ;;
  stop)
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      tmux send-keys -t "$SESSION_NAME" C-c
      sleep 1
      tmux kill-session -t "$SESSION_NAME" || true
      echo "stopped tmux session: $SESSION_NAME"
    else
      echo "tmux session not running: $SESSION_NAME"
    fi
    ;;
  restart)
    "$0" stop || true
    "$0" start
    ;;
  attach)
    exec tmux attach-session -t "$SESSION_NAME"
    ;;
  status)
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      tmux list-sessions | awk -F: -v s="$SESSION_NAME" '$1==s {print $0}'
    else
      echo "tmux session not running: $SESSION_NAME"
      exit 1
    fi
    ;;
  tail)
    exec tmux capture-pane -pt "$SESSION_NAME" -S -200
    ;;
  *)
    echo "usage: $0 {start|stop|restart|status|attach|tail}" >&2
    exit 2
    ;;
esac
