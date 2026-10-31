#!/usr/bin/env bash
# Run bake_compare.py for each decoded mesh (EVALUATION venv). One process per mesh, so a failure stays local.
#   PYTHON=/path/to/evaluation-venv/bin/python bash tools/compare_all.sh MESH_DIR OUT_DIR prop[:decimation] ...
set -uo pipefail
ulimit -c 0
HERE=$(cd "$(dirname "$0")" && pwd)
MESHES=$1 OUT=$2; shift 2
mkdir -p "$OUT"
for spec in "$@"; do
  prop=${spec%%:*}; dec=500000; [ "$spec" != "$prop" ] && dec=${spec#*:}
  name=$prop; [ "$dec" != 500000 ] && name=$prop-d$dec
  echo "== $name $(date +%T)"
  timeout 1800 "${PYTHON:?set PYTHON to the evaluation venv python}" -I "$HERE/bake_compare.py" --mesh "$MESHES/$prop.pt" --out "$OUT/$name" --decimation "$dec" \
    > "$OUT/$name.log" 2>&1 || echo "FAILED $name (exit $?)"
  tail -1 "$OUT/$name.log" | cut -c1-200
done
