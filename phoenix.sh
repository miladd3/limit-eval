#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDFILE="$SCRIPT_DIR/.phoenix.pid"
PORT=6006

case "${1:-status}" in
  start)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "Phoenix already running (PID $(cat "$PIDFILE"))."
      exit 0
    fi
    echo "Starting Phoenix on :$PORT..."
    "$SCRIPT_DIR/.venv/bin/python" -m phoenix.server.main serve > /tmp/phoenix.log 2>&1 &
    echo $! > "$PIDFILE"
    echo "Phoenix started (PID $!)."
    echo "UI: http://localhost:$PORT"
    ;;
  stop)
    if [ -f "$PIDFILE" ]; then
      PID=$(cat "$PIDFILE")
      if kill -0 "$PID" 2>/dev/null; then
        kill "$PID"
        echo "Phoenix stopped (PID $PID)."
      else
        echo "Phoenix not running (stale PID)."
      fi
      rm -f "$PIDFILE"
    else
      echo "Phoenix not running."
    fi
    ;;
  status)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "Phoenix running (PID $(cat "$PIDFILE")) on :$PORT"
    else
      echo "Phoenix not running."
      rm -f "$PIDFILE" 2>/dev/null
    fi
    ;;
  *)
    echo "Usage: ./phoenix.sh {start|stop|status}"
    ;;
esac
