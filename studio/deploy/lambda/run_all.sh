#!/bin/bash
# Run ON the Lambda box.  tmux sessions: dbsvc (DynamicBind service :8100), studio-api (:8000), tunnel (gated).
#   run_all.sh [start]       start dbsvc + studio-api (idempotent; never touches the tunnel)
#   run_all.sh stop          stop dbsvc + studio-api (tunnel untouched)
#   run_all.sh status
#   run_all.sh start_tunnel  EXPLICIT, public: opens a cloudflared quick tunnel to :8000, log -> ~/cloudflared.log
#   run_all.sh stop_tunnel
# Config comes from ~/.studio_env (chmod 600). STUDIO_API_KEY is intentionally unset (server caps protect the API).
set -uo pipefail
APP=$HOME/studio-app
VENV=$HOME/studio-venv
LAMBDA_DIR=$APP/studio/deploy/lambda
ENVF=$HOME/.studio_env
LOAD='set -a; [ -f ~/.studio_env ] && . ~/.studio_env; set +a; export PYTHONNOUSERSITE=1'

have() { tmux has-session -t "$1" 2>/dev/null; }

start_one() {  # name, cwd, command
  if have "$1"; then echo "$1: already running"; return; fi
  tmux new-session -d -s "$1" "$LOAD; cd $2 && $3 2>&1 | tee -a $HOME/$1.log"
  echo "$1: started"
}

case "${1:-start}" in
  start)
    start_one dbsvc "$LAMBDA_DIR" "$VENV/bin/uvicorn dynamicbind_service:app --host 127.0.0.1 --port 8100"
    start_one studio-api "$APP" "$VENV/bin/uvicorn studio.server.app:app --host 127.0.0.1 --port 8000"
    ;;
  stop)
    tmux kill-session -t studio-api 2>/dev/null; tmux kill-session -t dbsvc 2>/dev/null; echo stopped ;;
  status)
    tmux ls 2>/dev/null; ss -ltnp 2>/dev/null | grep -E ':(8000|8100) ' ;;
  start_tunnel)
    # quick tunnels on this box need an EMPTY config file, otherwise cloudflared picks up a stale default config
    : > "$HOME/cloudflared-metis.yml"
    if have tunnel; then echo "tunnel: already running"; exit 0; fi
    : > "$HOME/cloudflared.log"
    tmux new-session -d -s tunnel "cloudflared tunnel --config $HOME/cloudflared-metis.yml --url http://localhost:8000 > $HOME/cloudflared.log 2>&1"
    echo "tunnel: started; run print_tunnel_url.sh in ~10 s" ;;
  stop_tunnel)
    tmux kill-session -t tunnel 2>/dev/null; echo "tunnel stopped" ;;
  *) echo "usage: $0 start|stop|status|start_tunnel|stop_tunnel"; exit 2 ;;
esac
