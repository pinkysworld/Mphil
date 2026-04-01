#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT_DIR/.venv/bin/python"

if [ ! -x "$PY" ]; then
  PY=python3
fi

cd "$ROOT_DIR"

"$PY" scripts/12_open_set_family_holdout.py
"$PY" scripts/13_archive_token_case_studies.py
"$PY" scripts/14_retraining_trigger_analysis.py

echo "[note] external validation is not run here because no independent newer sample stream is present under data/raw/"
echo "[ok] future-work experiment bundle completed"
