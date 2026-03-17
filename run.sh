#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

API_PORT=2010
MCP_PORT=2009
PHOENIX_PORT=6006

API_DIR="$ROOT/limit-api"
MCP_DIR="$ROOT/limit-mcp"
EVAL_DIR="$SCRIPT_DIR"

PIDS=()

cleanup() {
  echo ""
  echo "Stopping services..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null
  echo "Done."
}
trap cleanup EXIT INT TERM

wait_for_http() {
  local url=$1 label=$2
  echo "Waiting for $label..."
  for i in $(seq 1 20); do
    if curl -sf "$url" > /dev/null 2>&1; then
      echo "$label is up."
      return 0
    fi
    sleep 1
  done
  echo "ERROR: $label did not become ready in time." >&2
  exit 1
}

wait_for_port() {
  local host=$1 port=$2 label=$3
  echo "Waiting for $label..."
  for i in $(seq 1 20); do
    if nc -z "$host" "$port" > /dev/null 2>&1; then
      echo "$label is up."
      return 0
    fi
    sleep 1
  done
  echo "ERROR: $label did not become ready in time." >&2
  exit 1
}

# 1. Start limit-api
if ! curl -sf "http://127.0.0.1:$API_PORT/accounts" > /dev/null 2>&1; then
  echo "Starting limit-api on :$API_PORT..."
  (cd "$API_DIR" && source .venv/bin/activate && uvicorn main:app --host 0.0.0.0 --port $API_PORT > /tmp/limit-api.log 2>&1) &
  PIDS+=($!)
  wait_for_http "http://127.0.0.1:$API_PORT/accounts" "limit-api"
else
  echo "limit-api already running."
fi

# 2. Start limit-mcp
if ! nc -z 127.0.0.1 $MCP_PORT > /dev/null 2>&1; then
  echo "Starting limit-mcp on :$MCP_PORT..."
  (cd "$MCP_DIR" && source .venv/bin/activate && python fastmcp_server.py > /tmp/limit-mcp.log 2>&1) &
  PIDS+=($!)
  wait_for_port "127.0.0.1" "$MCP_PORT" "limit-mcp"
else
  echo "limit-mcp already running."
fi

# 3. Start Phoenix
if ! curl -sf "http://127.0.0.1:$PHOENIX_PORT" > /dev/null 2>&1; then
  echo "Starting Phoenix on :$PHOENIX_PORT..."
  (cd "$EVAL_DIR" && source .venv/bin/activate && python -m phoenix.server.main serve > /tmp/phoenix.log 2>&1) &
  PIDS+=($!)
  wait_for_http "http://127.0.0.1:$PHOENIX_PORT" "Phoenix"
else
  echo "Phoenix already running."
fi

echo ""
echo "All services running. Starting eval..."
echo ""

# 4. Run eval
cd "$EVAL_DIR"
source .venv/bin/activate
python eval.py "$@"
