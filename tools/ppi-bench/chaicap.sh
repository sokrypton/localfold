#!/bin/bash
# chaicap.sh: stop each Chai-1 design run at 120 attempts (5 a target) - both are clearly below every AF3 arm
D=/home/ubuntu/ppi-bench-2026-10-29
for r in NXH NX; do
  until [ "$(grep -c . $D/$r.jsonl)" -ge 120 ]; do sleep 30; done
  for p in $(ps aux | grep "[n]ative-bench.mjs $D/$r.jsonl" | awk '{print $2}'); do kill $p; done
  echo "capped $r at $(grep -c . $D/$r.jsonl)"
done
