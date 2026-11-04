#!/usr/bin/env bash
# The headers this port derives from cuda/rfd4p at build time:  bash derive.sh <build-dir> <repo>
#
# 🔴 DERIVED, NOT COPIED, AND NOT MOVED EITHER. The design metrics and the structure writer are host-only C++
# that both ports must agree on to the byte - `cuda/rfd4p/check-writer.mjs` holds that writer to the page's, and
# `gate.py` holds the metrics to `design-metrics.mjs` value for value - so this port wants THAT code and not a
# second implementation of it. What stops a plain include is one line: `metrics.h` pulls in `generate.cuh` for
# `AtomRecord`, the insertion mapping and the record builder, and that file carries three `__global__` kernels
# which clang cannot compile without nvcc.
#
# Copying the host half into metal/rfd4p duplicates declarations that must not drift; moving it into its own
# header edits cuda/, which another agent is working in. Deriving it HERE, on every build, drifts never: a
# symbol that moves or changes shape fails this build rather than going quietly stale.
set -euo pipefail
build="$1"; repo="$2"
src="$repo/cuda/rfd4p/src"

# generate.cuh's host surface - everything in the file but its three kernels and its `jsNumber`.
python3 "$repo/tools/metal-extract-host.py" "$src/generate.cuh" "$build/rfd4p-gen-host.h" rfd4p \
  --include=rfd4p/featurise/tables.inc --include=rfd4p/src/superpose.h --include=rfd4p/src/assign.h \
  --include=featurise/jsmath.h --include=featurise/jsnum.h --alias=lf::jsNumber \
  GenOut postprocess OutTables unpackString unpackDouble inferElement tokenAtomName OutNaming chainFromCode \
  Insertion matchInsertedAtoms predictedInsertionMapping applyInsertion resolveInsertion alignToInputFrame \
  chainName AtomRecord buildRecords jsToFixed cifAtomName cifCoord toCif pdbAtomField toPdb

# 🔴 AND `metrics.h` IS TAKEN WHOLE, WITH ITS INCLUDES REPOINTED AND NOTHING ELSE. It is already CUDA-free, so
# there is no host surface to pick out of it - only the five quoted includes to aim at the derived header above
# and at this build's search path. A rewrite that does more than that is an edit, and an edit here would drift.
sed -e 's|#include "generate.cuh"|#include "rfd4p-gen-host.h"|' \
    -e 's|#include "metrics_tags.h"|#include "rfd4p/src/metrics_tags.h"|' \
    -e 's|#include "superpose.h"|#include "rfd4p/src/superpose.h"|' \
    -e 's|#include "symmetry.h"|#include "rfd4p/src/symmetry.h"|' \
    -e 's|#include "../../featurise/jsmath.h"|#include "featurise/jsmath.h"|' \
    "$src/metrics.h" > "$build/rfd4p-metrics.h"
# 🔴 AND THE REWRITE IS CHECKED BY WHAT IT PRODUCED, NOT BY WHAT IT REMOVED. A guard that only asks whether the
# old include names are gone passes when one of them is RENAMED - the sed then matches nothing, the header keeps
# an include this build cannot resolve, and the failure arrives later as a missing file naming cuda/rfd4p rather
# than as this script saying its reading of metrics.h is out of date. So every target is required to be present.
want=(rfd4p-gen-host.h rfd4p/src/metrics_tags.h rfd4p/src/superpose.h rfd4p/src/symmetry.h featurise/jsmath.h)
for w in "${want[@]}"; do
  grep -q "^#include \"$w\"" "$build/rfd4p-metrics.h" || {
    echo "derive.sh: $src/metrics.h no longer includes what this rewrite aims at $w - re-read its includes" >&2
    exit 1
  }
done
# ...and nothing quoted is left pointing into cuda/rfd4p's own directory, which is not on this build's path.
if grep -qE '^#include "(\.\./)|^#include "(generate\.cuh|metrics_tags\.h|superpose\.h|symmetry\.h)"' \
     "$build/rfd4p-metrics.h"; then
  echo "derive.sh: $src/metrics.h has an include this rewrite does not cover" >&2
  exit 1
fi
echo "$build/rfd4p-metrics.h: metrics.h with its includes repointed"
