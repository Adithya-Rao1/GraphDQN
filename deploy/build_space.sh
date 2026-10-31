#!/usr/bin/env bash
# Copy the static studio UI into deploy/space/web/ (the Space repo contents are deploy/space/ only).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/studio/web"
DST="$ROOT/deploy/space/web"
[ -f "$SRC/index.html" ] || { echo "missing $SRC/index.html" >&2; exit 1; }
rm -rf "$DST"
mkdir -p "$DST"
# screenshots are dev-only; vendor/ (3Dmol) and replay/ (recorded fallback) must ship.
rsync -a --exclude 'screenshots' --exclude '*.md' --exclude '__pycache__' --exclude '.DS_Store' "$SRC/" "$DST/"
[ -f "$DST/vendor/3Dmol-min.js" ] || { echo "vendored 3Dmol missing" >&2; exit 1; }
if [ ! -d "$DST/replay" ]; then
  echo "WARNING: studio/web/replay/ not present yet; the backend-down fallback will have nothing to play." >&2
fi
# the mock run is the last-resort offline demo and the UI's ?mock=1 path
if [ -f "$ROOT/studio/mock/mock_run.json" ]; then
  mkdir -p "$DST/mock" && cp "$ROOT/studio/mock/mock_run.json" "$DST/mock/mock_run.json"
fi
echo "built $DST ($(du -sh "$DST" | cut -f1))"
