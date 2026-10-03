#!/usr/bin/env bash
# Set up the native CUDA ports on a fresh machine (a Colab VM: T4, L4 or A100) from the published bundles:
#
#   bash native/colab_setup.sh [ef2] [af3]         (both when none is named)
#
# Node 22 if the machine's is older (the exporters run the page's own featurisers), the bundles from the
# registry's `remote:` URLs (native/fetch_bundles.py), each port's weights exported from them once, and
# each port built for this GPU (nvidia-smi's compute capability). Idempotent: a step whose output exists
# is skipped. The weights are the published int5/int3 bundles decoded to float32 - what the page folds
# with - so a fold matches the page's numbers, not native/<port>/gate-baseline.json's (exported from the
# float32 bundles, which are not published).
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
ports=("$@"); [ ${#ports[@]} -gt 0 ] || ports=(ef2 af3)
cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')"; arch="sm_$cc"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ($arch)"

major="$(node -v 2>/dev/null | sed 's/^v\([0-9]*\).*/\1/' || echo 0)"
if [ "${major:-0}" -lt 22 ]; then
  echo "node $(node -v 2>/dev/null || echo none) -> 22"
  curl -fsSL https://nodejs.org/dist/v22.12.0/node-v22.12.0-linux-x64.tar.xz | tar -xJ -C /usr/local --strip-components=1
fi

for port in "${ports[@]}"; do
  case "$port" in
    ef2)
      python3 "$repo/native/fetch_bundles.py" ef2-fast-600m esmc
      [ -f "$repo/native/ef2/weights/model.idx" ] || node --js-float16array "$repo/native/ef2/export_weights.mjs" \
        "$repo/native/ef2/weights" --fold="$repo/model-esmfold2-int5" --esmc="$repo/model-esmc-600m-int3"
      (cd "$repo/native/ef2" && nvcc -O3 -std=c++17 -arch=$arch --default-stream per-thread src/ef2.cu \
         -lcublas -lcublasLt -lcupti -o ef2) ;;
    af3)
      python3 "$repo/native/fetch_bundles.py" af3
      [ -f "$repo/native/af3/weights/model.idx" ] || (cd "$repo/native/af3" && node --js-float16array \
        --max-old-space-size=24000 export-model.mjs weights --weights-only --bundle="$repo/model-af3-int5/manifest.json")
      (cd "$repo/native/af3" && nvcc -O3 -std=c++17 -arch=$arch --default-stream per-thread --use_fast_math src/af3.cu \
         -lcublas -lcublasLt -lcupti -o af3) ;;
    *) echo "unknown port $port (ef2, af3)" >&2; exit 1 ;;
  esac
  echo "$port ready"
done
