#!/usr/bin/env bash
# Portable local-development bootstrap for macOS/Linux (Bash or Zsh).
# Usage: bash run.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV="$ROOT/.venv"

command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  echo "Python 3.11 was not found. Install Python 3.11 or set PYTHON_BIN=/path/to/python3.11." >&2
  exit 1
}
command -v npm >/dev/null 2>&1 || {
  echo "Node.js/npm was not found. Install Node.js 20+ and npm." >&2
  exit 1
}

if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r "$ROOT/backend/requirements.txt"
(cd "$ROOT/frontend" && npm ci)

cleanup() {
  [[ -z "${BACKEND_PID:-}" ]] || kill "$BACKEND_PID" 2>/dev/null || true
  [[ -z "${FRONTEND_PID:-}" ]] || kill "$FRONTEND_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

(
  cd "$ROOT/backend"
  exec "$VENV/bin/python" -m uvicorn main:app --host 127.0.0.1 --port 8000
) &
BACKEND_PID=$!
(
  cd "$ROOT/frontend"
  exec npm run dev -- --host 127.0.0.1
) &
FRONTEND_PID=$!

echo "API docs: http://127.0.0.1:8000/docs"
echo "Frontend: http://127.0.0.1:5173"
echo "Press Ctrl+C to stop both processes."
wait "$BACKEND_PID" "$FRONTEND_PID"
