#!/usr/bin/env bash
# Set up the native CUDA ports on a fresh machine (a Colab VM: T4, L4 or A100) from the published bundles:
#
#   bash native/colab_setup.sh [ef2] [af3] [af2]    (all three when none is named)
#
# Node 22 if the machine's is older (the input exporters run the page's own featurisers), the bundles
# from the registry's `remote:` URLs (native/fetch_bundles.py) - the weights the page folds with, read
# as they are, decoded on the device - and each port built for this GPU (nvidia-smi's compute
# capability). Idempotent: a bundle already on disk is not fetched again.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
ports=("$@"); [ ${#ports[@]} -gt 0 ] || ports=(ef2 af3 af2)
echo "GPU: $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1)"

major="$(node -v 2>/dev/null | sed 's/^v\([0-9]*\).*/\1/' || echo 0)"
if [ "${major:-0}" -lt 22 ]; then
  echo "node $(node -v 2>/dev/null || echo none) -> 22"
  curl -fsSL https://nodejs.org/dist/v22.12.0/node-v22.12.0-linux-x64.tar.xz | tar -xJ -C /usr/local --strip-components=1
fi

for port in "${ports[@]}"; do
  case "$port" in
    ef2) python3 "$repo/native/fetch_bundles.py" ef2-fast-600m esmc ;;    # (read as they are: no export)
    af3) python3 "$repo/native/fetch_bundles.py" af3 ;;                  # (read through native/af3/maps/af3.map)
    af2) python3 "$repo/native/fetch_bundles.py" monomer multimer ;;     # (read through native/af2/maps/*.map)
    *) echo "unknown port $port (ef2, af3, af2)" >&2; exit 1 ;;
  esac
done
bash "$repo/native/build.sh" "${ports[@]}"
echo "ready: ${ports[*]}"
