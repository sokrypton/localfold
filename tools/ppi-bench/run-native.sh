#!/bin/bash
# run-native.sh <jobs-prefix> <config> <out> <worker ...>: native-bench.mjs workers writing into one results file
cd "$(dirname "$0")"
p=$1; c=$2; o=$3; shift 3
for w in "$@"; do node native-bench.mjs "$o" "$c" $(cat "$p.jobs$w") > "$p.log$w" 2>&1 & done
wait; echo all-done
