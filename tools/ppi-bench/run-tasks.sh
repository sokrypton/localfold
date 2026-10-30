#!/bin/bash
# run-tasks.sh <config> <out> <workers> <target:attempt ...>: task-bench.mjs workers over one job list, dealt round-robin
cd "$(dirname "$0")"
c=$1; o=$2; n=$3; shift 3
jobs=("$@")
for w in $(seq 0 $((n - 1))); do
  mine=(); for i in "${!jobs[@]}"; do [ $((i % n)) -eq $w ] && mine+=("${jobs[$i]}"); done
  node task-bench.mjs "$o" "$c" "${mine[@]}" > "$o.log$w" 2>&1 &
done
wait; echo all-done
