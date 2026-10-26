#!/usr/bin/env bash
# Build the native ports for this machine's GPU, and the native featurisers beside them:
#
#   bash cuda/build.sh [af3] [af2] [ef2] [bioemu2] (af3, af2 and ef2 when none is named: cuda/<port>/localfold-<port>)
#   bash cuda/build.sh msap                         (MSA Pairformer: contacts from an alignment, built only when named)
#   bash cuda/build.sh featurise                    (the featurisers alone: no GPU, no nvcc)
#
# The featurisers (cuda/featurise: one binary linked as af3-featurise, af2-featurise, ef2-featurise and
# resolve-templates - host C++, no GPU, a job JSON in and each port's input out, byte for byte the page's) are built
# first and always, so a fold needs no Node at all.
# Each port is compiled for the card nvidia-smi reports (sm_75 a T4, sm_89 an L4, sm_80 an A100), the three
# in parallel, each to a temporary file moved into place only when whole - so cuda/<port>/localfold-<port> either
# does not exist or is a finished binary, and a fold never starts on half of one. While it runs,
# /tmp/localfold-cuda-build holds this script's pid (python/localfold/worker.py waits on it rather than
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
  case "$port" in af3|af2|ef2|msap|bioemu2|rfd4p|featurise) ;; *) echo "unknown port $port (af3, af2, ef2, msap, bioemu2, rfd4p, featurise)" >&2; exit 1 ;; esac
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
# CUPTI's header (af3/src/profile.cuh; the library itself is dlopen'd only for --profile): a toolkit from NVIDIA's
# runfile or .deb puts it in <cuda>/extras/CUPTI/include, which nvcc does not search - python/manylinux_build.sh adds it
# through CPATH, and this is the same directory for a local build. Absent (a conda toolkit has it in include/), nothing.
cupti=(); nvcc_bin="$(command -v nvcc 2>/dev/null || true)"
for root in "${CUDA_HOME:-}" "${CUDA_PATH:-}" "${nvcc_bin:+$(dirname "$(dirname "$nvcc_bin")")}"; do
  [ -n "$root" ] && [ -f "$root/extras/CUPTI/include/cupti.h" ] && { cupti=(-I"$root/extras/CUPTI/include"); break; }
done
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
for name in af3-featurise af2-featurise ef2-featurise bioemu2-featurise resolve-templates chem-probe fetch-weights msa-search; do
  ln -sfn featurise "$here/featurise/$name"
done
# RFD4-Proteina's featuriser is its OWN binary, not a link to the one above: its pipeline shares
# nothing with AF3's - a contig and a selection language rather than a job of chains - and the only
# code the two have in common is jsmath and the JSON reader, which it includes.
rfbin="$here/rfd4p/featurise/rfd4p-featurise"
# 🔴 EVERY HEADER IT INCLUDES, NOT JUST THE .cpp. The stamp named five files and the structure
# path lives in three more (`atoms.h`, `cif.h`, `chirality.h`), so a change to a transform stage
# left the binary untouched and the gate diffed the PREVIOUS build against the new reference -
# green or red for reasons nothing in the diff explains.
# 🔴 GLOBBED, NOT ENUMERATED, BECAUSE THIS LIST WENT STALE TWICE AND BOTH TIMES THE GATE DIFFED
# A BINARY NOBODY HAD BUILT. First it named five files while `atoms.h`, `cif.h` and
# `chirality.h` were outside it; then it named those and the structure path landed in
# `selection.h`, `contig.h` and `conditions.h`, which were outside it too - so a mutation left
# in `selection.h` survived a clean `build.sh rfd4p` and the `cif` arm reported the SELECTION
# LANGUAGE inverted on nine structures with the source plainly correct. A hand-written list of
# inputs is a list that falls behind the code; the FOLD binary's stamp below already globs this
# directory, and these two must not disagree about what the featuriser is made of.
#
# 🔴 AND A STAMP SAYS THE SOURCES HAVE NOT MOVED, NOT THAT THIS BINARY CAME FROM THEM. A
# falsification run that compiles `featurise.cpp` by hand writes the binary and leaves the stamp
# alone, so the next `build.sh` matches the stamp, skips the build, and the gate measures the
# MUTATION with every source restored and `git status` clean - which is how the `cif` arm was
# read twice as a defect in one sitting. `rm <binary>.stamp` after any hand build; mutate
# through a scratch copy where you can.
rfstamp="$(cat "$here"/rfd4p/featurise/*.cpp "$here"/rfd4p/featurise/*.h \
  "$here"/rfd4p/featurise/*.inc \
  "$here"/rfd4p/src/assign.h "$here"/rfd4p/src/superpose.h "$here"/rfd4p/src/schedule.h \
  "$here"/featurise/*.h \
  | sha256sum | cut -c1-16)"
if [ ! -x "$rfbin" ] || [ "$(cat "$rfbin.stamp" 2>/dev/null)" != "$rfstamp" ]; then
  # ...and the same code as an object the fold links, by the same g++ with the same flags, so a
  # `--job` fold reads the featuriser's own bytes.
  # (and the assignment probe beside it, which the gate diffs against output.js's own solver)
  ( g++ -std=c++17 -O2 "$here/rfd4p/featurise/assign-probe.cpp" \
      -o "$here/rfd4p/featurise/assign-probe" >> "$log" 2>&1; \
    g++ -std=c++17 -O2 "$here/rfd4p/featurise/superpose-probe.cpp" \
      -o "$here/rfd4p/featurise/superpose-probe" >> "$log" 2>&1; \
    g++ -std=c++17 -O2 -ffp-contract=off "$here/rfd4p/featurise/featurise.cpp" -o "$rfbin.building" \
      >> "$log" 2>&1 \
    && g++ -std=c++17 -O2 -ffp-contract=off -DRFD4P_FEATURISE_NO_MAIN \
         -c "$here/rfd4p/featurise/featurise.cpp" -o "$here/rfd4p/featurise/featurise.o" \
         >> "$log" 2>&1 \
    && mv "$rfbin.building" "$rfbin" && echo "$rfstamp" > "$rfbin.stamp" \
    && echo "built rfd4p-featurise" >> "$log" \
    || { echo "FAILED rfd4p-featurise" >> "$log"; exit 1; } ) &
  pids+=($!)
fi
# (no GPU: the featurisers still build, the ports cannot)
if [ "${ports[*]}" = featurise ]; then
  status=0; for pid in ${pids[@]+"${pids[@]}"}; do wait "$pid" || status=1; done
  [ $status = 0 ] || { echo "featuriser build failed - $log:" >&2; tail -20 "$log" >&2; }
  exit $status
fi
if [ -z "$cc" ]; then for pid in ${pids[@]+"${pids[@]}"}; do wait "$pid"; done; echo "no NVIDIA GPU (nvidia-smi says nothing)" >&2; exit 1; fi
for port in "${ports[@]}"; do
  case "$port" in featurise) continue ;; af3|af2) fast=--use_fast_math ;; ef2|msap|bioemu2|rfd4p) fast="" ;; *) echo "unknown port $port" >&2; exit 1 ;; esac
  # AF3 - the page's default model, so usually the first fold - at full priority and the others niced: a
  # Colab VM has two cores and three compiles, and the first fold waits only on its own port's binary
  prio=""; [ "$port" = af3 ] || prio="nice -n 10"
  out="$here/$port/localfold-$port"
  # (the stamp is the card AND the sources - every port includes cuda/af3/src's shared headers - so a
  # checkout that changed a kernel rebuilds rather than keep the last binary)
  # (and the weight walks in cuda/featurise, which af3 and af2 include)
  # (msap is built on cuda/ef2's kernels, so their headers are its sources too; it links no featuriser)
  # (bioemu2 is built on them too, and links the featuriser: `localfold-bioemu2 --job=...`)
  # (the object a port links, and the stamp file that says it is the current one: the link waits
  # for that stamp, so a featuriser compiling beside it is never linked half-built or stale)
  extra=(); link=("$sobj"); linkstamp="$sobj.stamp"; linkwant="$fstamp"
  [ "$port" != msap ] || { extra=("$here"/ef2/src/*.cuh); link=(); }
  [ "$port" != bioemu2 ] || extra=("$here"/ef2/src/*.cuh)
  # (rfd4p is built on cuda/ef2's ops and cuda/af3's flash, and links its OWN featuriser object -
  # cuda/rfd4p/featurise, whose pipeline shares nothing with AF3's - so `--job=<spec>` featurises
  # and folds in one process)
  # 🔴 AND ITS FEATURISER'S SOURCES ARE IN THE STAMP, because it LINKS that object. The stamp
  # below hashes `cuda/featurise` - the AF3 lineage's - and rfd4p's pipeline lives in
  # `cuda/rfd4p/featurise`, so a change there rebuilt `rfd4p-featurise` and left
  # `localfold-rfd4p` linked against the PREVIOUS object. A structure job featurised correctly
  # through the standalone binary and refused through the fold one, with nothing in the diff to
  # explain it. Same shape as the stamp that hashed five files while the structure path lived
  # in three more.
  [ "$port" != rfd4p ] || { extra=("$here"/ef2/src/*.cuh "$here"/rfd4p/featurise/*.cpp \
    "$here"/rfd4p/featurise/*.h "$here"/rfd4p/featurise/*.inc); \
    link=("$here/rfd4p/featurise/featurise.o"); linkstamp="$rfbin.stamp"; linkwant="$rfstamp"; }
  stamp="$arch ${LOCALFOLD_RPATH:-} $(cat "$here/$port"/src/*.cu* "${extra[@]}" "$here"/af3/src/*.cuh "$here"/plm/*.cuh "$here"/featurise/*.h "$here"/featurise/*.inc "$here"/featurise/*.cpp | sha256sum | cut -c1-16)"
  if [ -x "$out" ] && [ "$(cat "$out.arch" 2>/dev/null)" = "$stamp" ]; then continue; fi
  # (-O1: nvcc's -O is the HOST code's level - the device code is optimised either way, its SASS byte-identical
  # - and host -O3 was 40% of AF3's compile for no measurable run time; -O0 costs a fold 7%)
  # (compiled to an object while standalone.o builds, then linked once it has)
  ( cd "$here/$port" && $prio nvcc -O1 -std=c++17 $archflags --split-compile 0 --default-stream per-thread $fast "${cupti[@]}" -c src/$port.cu \
      -o "$out.o" >> "$log" 2>&1 \
    && until [ ${#link[@]} = 0 ] || [ "$(cat "$linkstamp" 2>/dev/null)" = "$linkwant" ]; do
         [ "$(cat "$linkstamp" 2>/dev/null)" != failed ] || { echo "FAILED $port: no featuriser object" >> "$log"; exit 1; }
         grep -q "FAILED rfd4p-featurise" "$log" 2>/dev/null && [ "$port" = rfd4p ] && { echo "FAILED $port: no featuriser object" >> "$log"; exit 1; }
         sleep 1; done \
    && nvcc $archflags "$out.o" "${link[@]}" -lcublas -lcublasLt -ldl "${rpath[@]}" -o "$out.building" >> "$log" 2>&1 \
    && rm -f "$out.o" && mv "$out.building" "$out" && echo "$stamp" > "$out.arch" && echo "built $port for $arch" >> "$log" \
    || { echo "FAILED $port for $arch" >> "$log"; exit 1; } ) &
  pids+=($!)
done
status=0
for pid in ${pids[@]+"${pids[@]}"}; do wait "$pid" || status=1; done
[ $status = 0 ] || { echo "native build failed - $log:" >&2; tail -20 "$log" >&2; }
exit $status
