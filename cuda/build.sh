#!/usr/bin/env bash
# Build the native ports for this machine's GPU, and the native featurisers beside them:
#
#   bash cuda/build.sh [af3] [af2] [ef2]           (all three when none is named: cuda/<port>/localfold-<port>)
#   bash cuda/build.sh featurise                    (the featurisers alone: no GPU, no nvcc)
#
# The featurisers (cuda/featurise: one binary linked as af3-featurise, af2-featurise, ef2-featurise and
# resolve-templates - host C++, no GPU, a job JSON in and each port's input out, byte for byte the page's) are built
# first and always, so a fold needs no Node at all.
# Each port is compiled for the card nvidia-smi reports (sm_75 a T4, sm_89 an L4, sm_80 an A100), the three
# in parallel, each to a temporary file moved into place only when whole - so cuda/<port>/localfold-<port> either
# does not exist or is a finished binary, and a fold never starts on half of one. While it runs,
# /tmp/localfold-cuda-build holds this script's pid (cuda/worker.py waits on it rather than
# refusing a fold that arrives mid-build); the log is /tmp/localfold-cuda-build.log. A port already
# built for this card from these sources is left alone.
#
# For a prebuilt binary that runs on any card (the wheel, .github/workflows/wheel.yml - no GPU needed to build):
#   LOCALFOLD_CUDA_ARCHS="75 80 86 89 90 100 120"   compile for each of these, plus PTX of the last for newer cards
#   LOCALFOLD_RPATH='$ORIGIN/../../nvidia/cublas/lib'   where the binaries look for cuBLAS (pip's nvidia-cublas-cu12)
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
ports=("$@"); [ ${#ports[@]} -gt 0 ] || ports=(af3 af2 ef2)
for port in "${ports[@]}"; do      # (named before anything is forked: a typo must not leave half a build running)
  case "$port" in af3|af2|ef2|featurise) ;; *) echo "unknown port $port (af3, af2, ef2, featurise)" >&2; exit 1 ;; esac
done
cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')"
arch="sm_$cc"; archflags="-arch=$arch"
archs="${LOCALFOLD_CUDA_ARCHS:-}"
if [ -n "$archs" ]; then           # every listed card's SASS, and the newest's PTX for a card newer still
  archflags="--threads 0"; for a in $archs; do archflags+=" -gencode arch=compute_$a,code=sm_$a"; done
  archflags+=" -gencode arch=compute_${archs##* },code=compute_${archs##* }"
  arch="sm_${archs// /,sm_}"; cc="multi"
fi
rpath=(); [ -z "${LOCALFOLD_RPATH:-}" ] || rpath=(-Xlinker -rpath -Xlinker "$LOCALFOLD_RPATH")
marker=/tmp/localfold-cuda-build log=/tmp/localfold-cuda-build.log
echo $$ > "$marker"; trap 'rm -f "$marker"' EXIT
: > "$log"
pids=()
# the featurisers: ONE binary (cuda/featurise/featurise.cpp) linked as each tool's name, rebuilt when any of its
# sources changed - g++, the host's only compiler need
fstamp="$(cat "$here"/featurise/*.h "$here"/featurise/*.inc "$here"/featurise/*.cpp | sha256sum | cut -c1-16)"
fbin="$here/featurise/featurise"
if [ ! -x "$fbin" ] || [ "$(cat "$fbin.stamp" 2>/dev/null)" != "$fstamp" ]; then
  ( g++ -std=c++17 -O2 -ffp-contract=off -pthread "$here/featurise/featurise.cpp" -o "$fbin.building" >> "$log" 2>&1 \
    && mv "$fbin.building" "$fbin" && echo "$fstamp" > "$fbin.stamp" && echo "built the featurisers" >> "$log" \
    || { echo "FAILED the featurisers" >> "$log"; exit 1; } ) &
  pids+=($!)
fi
# ...and the same code as an object each port links (cuda/featurise/standalone.cpp: `af3 --job=...` featurises in
# its own process), by the same g++ with the same flags - so a standalone fold reads the featuriser's own bytes
sobj="$here/featurise/standalone.o"
if [ ! -f "$sobj" ] || [ "$(cat "$sobj.stamp" 2>/dev/null)" != "$fstamp" ]; then
  rm -f "$sobj.stamp"
  ( g++ -std=c++17 -O2 -ffp-contract=off -pthread -c "$here/featurise/standalone.cpp" -o "$sobj.building" >> "$log" 2>&1 \
    && mv "$sobj.building" "$sobj" && echo "$fstamp" > "$sobj.stamp" \
    || { echo "FAILED the standalone object" >> "$log"; echo failed > "$sobj.stamp"; exit 1; } ) &
  pids+=($!)
fi
for name in af3-featurise af2-featurise ef2-featurise resolve-templates chem-probe fetch-weights; do
  ln -sfn featurise "$here/featurise/$name"
done
# (no GPU: the featurisers still build, the ports cannot)
if [ "${ports[*]}" = featurise ]; then
  status=0; for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  [ $status = 0 ] || { echo "featuriser build failed - $log:" >&2; tail -20 "$log" >&2; }
  exit $status
fi
if [ -z "$cc" ]; then for pid in "${pids[@]}"; do wait "$pid"; done; echo "no NVIDIA GPU (nvidia-smi says nothing)" >&2; exit 1; fi
for port in "${ports[@]}"; do
  case "$port" in af3|af2) fast=--use_fast_math ;; ef2) fast="" ;; *) echo "unknown port $port" >&2; exit 1 ;; esac
  # AF3 - the page's default model, so usually the first fold - at full priority and the others niced: a
  # Colab VM has two cores and three compiles, and the first fold waits only on its own port's binary
  prio=""; [ "$port" = af3 ] || prio="nice -n 10"
  out="$here/$port/localfold-$port"
  # (the stamp is the card AND the sources - every port includes cuda/af3/src's shared headers - so a
  # checkout that changed a kernel rebuilds rather than keep the last binary)
  # (and the weight walks in cuda/featurise, which af3 and af2 include)
  stamp="$arch ${LOCALFOLD_RPATH:-} $(cat "$here/$port"/src/*.cu* "$here"/af3/src/*.cuh "$here"/plm/*.cuh "$here"/featurise/*.h "$here"/featurise/*.inc "$here"/featurise/*.cpp | sha256sum | cut -c1-16)"
  if [ -x "$out" ] && [ "$(cat "$out.arch" 2>/dev/null)" = "$stamp" ]; then continue; fi
  # (-O1: nvcc's -O is the HOST code's level - the device code is optimised either way, its SASS byte-identical
  # - and host -O3 was 40% of AF3's compile for no measurable run time; -O0 costs a fold 7%)
  # (compiled to an object while standalone.o builds, then linked once it has)
  ( cd "$here/$port" && $prio nvcc -O1 -std=c++17 $archflags --default-stream per-thread $fast -c src/$port.cu \
      -o "$out.o" >> "$log" 2>&1 \
    && until [ "$(cat "$sobj.stamp" 2>/dev/null)" = "$fstamp" ]; do
         [ "$(cat "$sobj.stamp" 2>/dev/null)" != failed ] || { echo "FAILED $port: no standalone object" >> "$log"; exit 1; }; sleep 1; done \
    && nvcc $archflags "$out.o" "$sobj" -lcublas -lcublasLt -ldl "${rpath[@]}" -o "$out.building" >> "$log" 2>&1 \
    && rm -f "$out.o" && mv "$out.building" "$out" && echo "$stamp" > "$out.arch" && echo "built $port for $arch" >> "$log" \
    || { echo "FAILED $port for $arch" >> "$log"; exit 1; } ) &
  pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[ $status = 0 ] || { echo "native build failed - $log:" >&2; tail -20 "$log" >&2; }
exit $status
