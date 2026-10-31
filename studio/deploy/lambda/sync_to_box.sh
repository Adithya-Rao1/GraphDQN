#!/bin/bash
# Push the app code to the Lambda box (run from your laptop; uses the lambda ssh key via ssh -i only).
#   sync_to_box.sh [SRC_REPO_ROOT]    default SRC = the checkout containing this script
# Extra overlay: STUDIO_DATA_OVERLAY=<dir containing studio/data> to add data not yet merged into SRC.
set -euo pipefail
HOST=ubuntu@150.136.42.38
KEY=${LAMBDA_KEY:-$HOME/.ssh/lambda_graphdqn}
SRC=${1:-$(cd "$(dirname "$0")/../../.." && pwd)}
SSH=(ssh -i "$KEY")
EXC=(--exclude .venv --exclude .git --exclude __pycache__ --exclude '*.pyc' --exclude .pytest_cache --exclude node_modules)
for d in studio binding_module molecular_modifications reward synthetic_accessibility dqn; do
  [ -d "$SRC/$d" ] && rsync -az -e "${SSH[*]}" "${EXC[@]}" "$SRC/$d" "$HOST:studio-app/"
done
if [ -n "${STUDIO_DATA_OVERLAY:-}" ]; then
  # data never overwrites precomputed poses produced on the box
  rsync -az -e "${SSH[*]}" "${EXC[@]}" --exclude precomputed_poses "$STUDIO_DATA_OVERLAY/studio/data" "$HOST:studio-app/studio/"
fi
echo "synced $SRC -> $HOST:studio-app"
