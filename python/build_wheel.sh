#!/usr/bin/env bash
# Build the localfold wheel: the three native ports compiled for every current NVIDIA card, in one platform wheel.
#
#   bash python/build_wheel.sh [<out dir>]          (default python/dist)
#
# Needs nvcc (CUDA >= 12.8 for Blackwell's sm_100/sm_120), cuBLAS's headers, g++ and a Python with pip - and NO GPU.
# .github/workflows/wheel.yml runs it in pypa's manylinux_2_28 image, so the binaries need glibc 2.28 at most;
# that is checked here, as is the RPATH that finds pip's nvidia-cublas-cu12 next to the package.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; repo="$(cd "$here/.." && pwd)"
out="$(mkdir -p "${1:-$here/dist}" && cd "${1:-$here/dist}" && pwd)"
py="${PYTHON:-python3}"
# every card from a T4 (sm_75) to Blackwell (sm_100 B200, sm_120 RTX 50xx / RTX PRO 6000), and the newest's PTX for
# a card newer still (its driver compiles that on first launch)
export LOCALFOLD_CUDA_ARCHS="${LOCALFOLD_CUDA_ARCHS:-75 80 86 89 90 100 120}"
# the binaries sit in site-packages/localfold/bin and pip puts cuBLAS in site-packages/nvidia/cublas/lib
export LOCALFOLD_RPATH='$ORIGIN/../../nvidia/cublas/lib'
# one port at a time: each compiles its architectures in parallel (nvcc --threads 0), and three at once would
# hold ~3x the memory of a CI runner's worth
for port in af3 af2 ef2; do
  bash "$repo/cuda/build.sh" "$port" || { tail -30 /tmp/localfold-cuda-build.log >&2; exit 1; }
done
bin="$here/localfold/bin"
rm -f "$bin"/localfold-*
for port in af3 af2 ef2; do
  install -m 755 "$repo/cuda/$port/localfold-$port" "$bin/localfold-$port"
  strip "$bin/localfold-$port"             # (host symbols only: the device code is the fatbinary's data)
done
# localfold-fetch: the featuriser binary under the name that runs its weight and CCD fetcher (no GPU, no cuBLAS)
install -m 755 "$repo/cuda/featurise/featurise" "$bin/localfold-fetch"; strip "$bin/localfold-fetch"
# what the wheel's tag promises: nothing newer than glibc 2.28, and cuBLAS found through the RPATH
for f in "$bin"/localfold-*; do
  glibc="$(objdump -T "$f" | grep -o 'GLIBC_[0-9.]*' | sed 's/GLIBC_//' | sort -V | tail -1)"
  [ "$(printf '%s\n2.28\n' "$glibc" | sort -V | tail -1)" = 2.28 ] || { echo "$f needs glibc $glibc, past manylinux_2_28" >&2; exit 1; }
  [ "$(basename "$f")" = localfold-fetch ] || readelf -d "$f" | grep -q 'nvidia/cublas/lib' || { echo "$f has no RPATH to nvidia-cublas-cu12" >&2; exit 1; }
  if readelf -d "$f" | grep NEEDED | grep -q cupti; then echo "$f links CUPTI" >&2; exit 1; fi
  echo "$(basename "$f"): glibc $glibc, $(du -m "$f" | cut -f1) MB"
done
work="$(mktemp -d)"; trap 'rm -rf "$work"' EXIT
"$py" -m pip wheel --no-deps -w "$work" "$here"
"$py" -m pip install -q "wheel>=0.40"
"$py" -m wheel tags --remove --python-tag py3 --abi-tag none --platform-tag manylinux_2_28_x86_64 "$work"/localfold-*.whl
mv "$work"/localfold-*-py3-none-manylinux_2_28_x86_64.whl "$out/"
ls -la "$out"/localfold-*.whl
