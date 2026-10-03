#!/usr/bin/env bash
# Build the native ports for this machine's GPU:
#
#   bash native/build.sh [af3] [af2] [ef2]      (all three when none is named)
#
# Each port is compiled for the card nvidia-smi reports (sm_75 a T4, sm_89 an L4, sm_80 an A100), the three
# in parallel, each to a temporary file moved into place only when whole - so native/<port>/<port> either
# does not exist or is a finished binary, and a fold never starts on half of one. While it runs,
# /tmp/localfold-native-build holds this script's pid (tools/native_worker.py waits on it rather than
# refusing a fold that arrives mid-build); the log is /tmp/localfold-native-build.log. A port already
# built for this card from these sources is left alone.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
ports=("$@"); [ ${#ports[@]} -gt 0 ] || ports=(af3 af2 ef2)
cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')"
[ -n "$cc" ] || { echo "no NVIDIA GPU (nvidia-smi says nothing)" >&2; exit 1; }
arch="sm_$cc"
marker=/tmp/localfold-native-build log=/tmp/localfold-native-build.log
echo $$ > "$marker"; trap 'rm -f "$marker"' EXIT
: > "$log"
pids=()
for port in "${ports[@]}"; do
  case "$port" in af3|af2) fast=--use_fast_math ;; ef2) fast="" ;; *) echo "unknown port $port" >&2; exit 1 ;; esac
  out="$here/$port/$port"
  # (the stamp is the card AND the sources - every port includes native/af3/src's shared headers - so a
  # checkout that changed a kernel rebuilds rather than keep the last binary)
  stamp="$arch $(cat "$here/$port"/src/*.cu* "$here"/af3/src/*.cuh | sha256sum | cut -c1-16)"
  if [ -x "$out" ] && [ "$(cat "$out.arch" 2>/dev/null)" = "$stamp" ]; then continue; fi
  ( cd "$here/$port" && nvcc -O3 -std=c++17 -arch=$arch --default-stream per-thread $fast src/$port.cu \
      -lcublas -lcublasLt -lcupti -o "$out.building" >> "$log" 2>&1 \
    && mv "$out.building" "$out" && echo "$stamp" > "$out.arch" && echo "built $port for $arch" >> "$log" \
    || { echo "FAILED $port for $arch" >> "$log"; exit 1; } ) &
  pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[ $status = 0 ] || { echo "native build failed - $log:" >&2; tail -20 "$log" >&2; }
exit $status
