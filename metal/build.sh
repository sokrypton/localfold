#!/usr/bin/env bash
# Build the native Metal ports for this Mac:
#
#   bash metal/build.sh [ef2] [af2] [af3] [selftest]      ->  metal/<port>/localfold-<port>
#
# A port is metal/core (the device, memory, kernels, GEMM, weight loader) and its own directory's .mm/.cpp files,
# linked with cuda/featurise's standalone object (the native featuriser, built by clang here). Kernels are Metal
# source compiled at run time (the command-line tools have no offline Metal compiler): metal/core/args.h, the core's
# kernels and the port's are embedded as one string, the GEMM's as another.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(dirname "$here")"
ports=("$@"); [ ${#ports[@]} -gt 0 ] || ports=(ef2)
opt="${LOCALFOLD_METAL_OPT:--O2}"
flags=(-std=c++17 -fobjc-arc -Wall -Wno-unused-function -Wno-deprecated-declarations -Wno-unused-result)
link=(-framework Metal -framework Foundation)
build="$here/build"
mkdir -p "$build"

embed() {   # embed <symbol> <file>... : the files joined, as a C++ string (raw literals in pieces)
  python3 - "$@" <<'PY'
import sys, re
name, files = sys.argv[1], sys.argv[2:]
src = ""
for f in files:
    text = open(f).read()
    text = re.sub(r'^#pragma once\s*$', '', text, flags=re.M)
    text = re.sub(r'^#include "[^"]*"\s*$', '', text, flags=re.M)
    src += f"// ---- {f.split('/metal/')[-1]}\n" + text + "\n"
parts = [src[i:i + 60000] for i in range(0, len(src), 60000)]
print(f"extern const char* {name};\nconst char* {name} = " + "\n".join(f'R"LFSRC({p})LFSRC"' for p in parts) + ";")
PY
}

# zstd's decompressor (the featuriser's fetch reads af3-any-model's .bin.zst blobs; macOS has no libzstd): pinned,
# built once
zstd_lib="$build/zstd/libzstd-dec.a"
if [ ! -f "$zstd_lib" ]; then
  zdir="$build/zstd"; mkdir -p "$zdir"
  [ -f "$zdir/zstd.tar.gz" ] || curl -sL -o "$zdir/zstd.tar.gz" https://github.com/facebook/zstd/releases/download/v1.5.6/zstd-1.5.6.tar.gz
  echo "8c29e06cf42aacc1eafc4077ae2ec6c6fcb96a626157e0593d5e82a34fd403c1  $zdir/zstd.tar.gz" | shasum -a 256 -c - >/dev/null \
    || { echo "zstd-1.5.6.tar.gz: checksum mismatch" >&2; exit 1; }
  tar xzf "$zdir/zstd.tar.gz" -C "$zdir"
  zsrc="$zdir/zstd-1.5.6/lib"; objs=()
  for c in common/debug.c common/entropy_common.c common/error_private.c common/fse_decompress.c common/xxhash.c \
           common/zstd_common.c decompress/huf_decompress.c decompress/zstd_ddict.c decompress/zstd_decompress.c \
           decompress/zstd_decompress_block.c; do
    o="$zdir/$(basename "$c" .c).o"
    clang -O2 -DZSTD_DISABLE_ASM -DZSTD_MULTITHREAD=0 -I "$zsrc" -I "$zsrc/common" -c "$zsrc/$c" -o "$o"
    objs+=("$o")
  done
  ar rcs "$zstd_lib" "${objs[@]}"
fi

# the core, once
core_objs=()
for f in core.mm model.cpp host.cpp; do
  o="$build/core-${f%.*}.o"
  clang++ "${flags[@]}" $opt -I "$here/core" -x objective-c++ -c "$here/core/$f" -o "$o"
  core_objs+=("$o")
done
embed "mt::GEMM_SOURCE" "$here/core/args.h" "$here/core/gemm.metal" | sed 's/extern const char\* mt::GEMM_SOURCE;/namespace mt { extern const char* GEMM_SOURCE; }/' > "$build/gemm-source.cpp"
clang++ -std=c++17 -c "$build/gemm-source.cpp" -o "$build/gemm-source.o"
core_objs+=("$build/gemm-source.o")

for port in "${ports[@]}"; do
  pdir="$here/$port"
  [ -d "$pdir" ] || { echo "no metal/$port" >&2; exit 1; }
  metals=("$here/core/args.h"); [ -f "$pdir/kernels.h" ] && metals+=("$pdir/kernels.h")
  metals+=("$here/core/common.metal")
  for m in "$pdir"/*.metal; do [ -f "$m" ] && metals+=("$m"); done
  embed "PORT_SOURCE" "${metals[@]}" > "$build/$port-source.cpp"
  srcs=()
  for s in "$pdir"/*.mm "$pdir"/*.cpp; do [ -f "$s" ] && srcs+=("$s"); done
  extra=(-Wl,-dead_strip)
  if [ "$port" != selftest ]; then
    sobj="$repo/cuda/featurise/standalone.o"
    if [ ! -f "$sobj" ] || [ -n "$(find "$repo/cuda/featurise" -newer "$sobj" \( -name '*.h' -o -name '*.inc' -o -name '*.cpp' \) | head -1)" ]; then
      clang++ -std=c++17 -O2 -ffp-contract=off -pthread -c "$repo/cuda/featurise/standalone.cpp" -o "$sobj"
    fi
    extra=("$sobj" -Wl,-force_load,"$zstd_lib")
  fi
  clang++ "${flags[@]}" $opt -I "$here/core" -I "$pdir" -I "$repo/cuda/featurise" -x objective-c++ "${srcs[@]}" "$build/$port-source.cpp" \
    -x none "${core_objs[@]}" "${extra[@]}" "${link[@]}" -o "$pdir/localfold-$port.building"
  mv "$pdir/localfold-$port.building" "$pdir/localfold-$port"
  echo "built metal/$port/localfold-$port"
done
