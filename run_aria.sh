#!/usr/bin/env bash
# A relocatable launcher: the directory containing this script IS Aria's root.
set -euo pipefail
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"
if [[ -x "$APP_DIR/.venv/bin/python" ]]; then
  PYTHON="$APP_DIR/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi
exec "$PYTHON" "$APP_DIR/app.py" "$@"
