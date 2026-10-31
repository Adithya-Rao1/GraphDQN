#!/bin/bash
# Run ON the box: print the quick-tunnel URL from ~/cloudflared.log (empty + exit 1 if not up yet).
u=$(grep -o 'https://[a-zA-Z0-9-]*\.trycloudflare\.com' "$HOME/cloudflared.log" 2>/dev/null | tail -n 1)
[ -n "$u" ] && echo "$u" || { echo "no tunnel URL yet (is 'run_all.sh start_tunnel' running?)" >&2; exit 1; }
