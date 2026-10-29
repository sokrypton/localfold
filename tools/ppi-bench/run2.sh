#!/bin/bash
# run2.sh <jobs-prefix> <config> <out> <worker ...>: workers writing into one results file
cd "$(dirname "$0")"
export DISPLAY=:99 XDG_RUNTIME_DIR=/tmp/xdg
p=$1; c=$2; o=$3; shift 3
for w in "$@"; do python3 bench.py "$o" "$c" $(cat "$p.jobs$w") > "$p.log$w" 2>&1 & done
wait; echo all-done
