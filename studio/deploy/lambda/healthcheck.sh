#!/bin/bash
# Run ON the box: exit 0 only if both local services answer. Prints one line per check.
rc=0
chk() { # name url pattern
  out=$(curl -s -m 8 -w ' [http %{http_code}]' "$2" | tr '\n' ' ')
  if echo "$out" | grep -qi -- "$3"; then echo "OK   $1  ${out:0:100}"; else echo "FAIL $1  ${out:0:100}"; rc=1; fi
}
chk studio-api http://127.0.0.1:8000/api/health '"ok"'
chk studio-page http://127.0.0.1:8000/ '<html'
chk dynamicbind http://127.0.0.1:8100/health '"ok":true'
echo "gpu: $(nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader)"
echo "tunnel: $("$HOME/studio-app/studio/deploy/lambda/print_tunnel_url.sh" 2>/dev/null || echo 'not running')"
exit $rc
