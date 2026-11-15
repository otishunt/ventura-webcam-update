#!/usr/bin/env bash
# Record on this computer. Double-click this file in Finder, or run ./run_local.command
#
# The server listens on 127.0.0.1 only: the recording page is reachable from this Mac and
# nowhere else (no tunnel, nothing exposed to the network). Close this window or press
# Ctrl+C to stop. Options:  PORT=8001 ./run_local.command   NO_BROWSER=1 ./run_local.command
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8000}"
say() { printf '\n%s\n' "$*"; }
fail() { say "ERROR: $*"; if [ -t 0 ]; then read -r -p "Press Return to close." _ || true; fi; exit 1; }

# Finder-launched shells can miss Homebrew/Anaconda on PATH.
for d in /opt/homebrew/bin /usr/local/bin /opt/anaconda3/bin; do
  case ":$PATH:" in *":$d:"*) ;; *) [ -d "$d" ] && PATH="$PATH:$d" ;; esac
done
export PATH

command -v ffmpeg >/dev/null 2>&1 || fail "ffmpeg not found. Install it with: brew install ffmpeg"

# ---- Python environment (reuses .venv; repairs it if it is missing or broken)
PY=.venv/bin/python
deps_ok() {
  "$PY" - <<'EOF' 2>/dev/null
import fastapi, uvicorn, yaml, multipart, numpy, cv2
assert int(cv2.__version__.split(".")[0]) < 5 and hasattr(cv2, "CascadeClassifier")
EOF
}
if ! deps_ok; then
  if ! "$PY" -c "import sys" 2>/dev/null; then
    command -v python3 >/dev/null || fail "python3 not found"
    say "Creating the Python environment (.venv)..."
    python3 -m venv --clear .venv || fail "could not create .venv"
  fi
  say "Installing requirements (first run only)..."
  "$PY" -m pip install -q --upgrade pip >/dev/null 2>&1 || true
  "$PY" -m pip install -q -r requirements.txt || fail "pip install failed"
  deps_ok || fail "requirements installed but imports still fail (OpenCV must be 4.x)"
fi

# ---- Admin token for /admin, generated once and kept in .admin_token (git-ignored)
if [ -z "${ADMIN_TOKEN:-}" ]; then
  if [ ! -s .admin_token ]; then
    (umask 077; "$PY" -c "import secrets; print(secrets.token_urlsafe(18))" > .admin_token)
  fi
  ADMIN_TOKEN="$(cat .admin_token)"
fi
export ADMIN_TOKEN

# ---- Port: reuse a recorder already running here, otherwise find a free port
is_recorder() { curl -fsS --max-time 2 "http://127.0.0.1:$1/api/protocol" 2>/dev/null | grep -q protocol_version; }
port_free() { "$PY" -c "import socket,sys; s=socket.socket(); s.bind(('127.0.0.1', int(sys.argv[1])))" "$1" 2>/dev/null; }
open_page() {
  [ -n "${NO_BROWSER:-}" ] && return 0
  if command -v open >/dev/null && [ "$(uname)" = Darwin ]; then open "$1"
  elif command -v xdg-open >/dev/null; then xdg-open "$1" >/dev/null 2>&1 || true
  fi
}
if is_recorder "$PORT"; then
  say "The recorder is already running at http://localhost:$PORT, opening it."
  open_page "http://localhost:$PORT/"
  exit 0
fi
for _ in $(seq 1 20); do port_free "$PORT" && break; PORT=$((PORT + 1)); done
port_free "$PORT" || fail "no free port found near ${PORT}"

# ---- Start the server (local only), wait until it answers, open the page
"$PY" -m uvicorn server.app:app --host 127.0.0.1 --port "$PORT" --workers 1 --log-level warning &
SERVER=$!
cleanup() { kill "$SERVER" 2>/dev/null || true; wait "$SERVER" 2>/dev/null || true; }
trap cleanup EXIT
trap 'exit 0' INT TERM HUP

for _ in $(seq 1 60); do
  is_recorder "$PORT" && break
  kill -0 "$SERVER" 2>/dev/null || fail "the server stopped while starting (see the messages above)"
  sleep 0.5
done
is_recorder "$PORT" || fail "the server did not start within 30 s"

# keep the Mac awake while the server runs
command -v caffeinate >/dev/null && caffeinate -dims -w "$SERVER" &

URL="http://localhost:$PORT"
say "Recorder running at $URL  (this computer only)"
echo "  Admin dashboard: $URL/admin   token: $ADMIN_TOKEN"
echo "  Data folder:     ${MEME_DATA_DIR:-$(pwd)/data}"
echo "  Keep this window open while recording. Ctrl+C or closing the window stops it."
open_page "$URL/"
wait "$SERVER"
