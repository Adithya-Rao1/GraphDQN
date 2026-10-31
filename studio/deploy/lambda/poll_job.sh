#!/bin/bash
# on the box: poll_job.sh <job_id> [port]  -> prints status lines until done/failed
J=$1; PORT=${2:-8100}; T0=$(date +%s)
for i in $(seq 1 200); do
  r=$(curl -s localhost:$PORT/jobs/$J | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['status'],d['progress'],d['note'][:70],d.get('pose_confidence'),d.get('affinity'),d.get('source'))")
  echo "+$(( $(date +%s)-T0 ))s $r"
  case "$r" in done*|failed*) break;; esac
  sleep 5
done
