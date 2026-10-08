#!/usr/bin/env bash
# Set up the native CUDA ports on a fresh machine (a Colab VM: T4, L4 or A100) from the published bundles:
#
#   bash cuda/colab_setup.sh [ef2] [af3] [af2]    (all three when none is named)
#
# The weights fetched natively (cuda/featurise/fetch-weights, by model: the registry's `remote:` bundles for ESMFold2
# and AF2, af3-any-model's int8 blob for AF3 - which asks for DeepMind's terms, or LOCALFOLD_ACCEPT_MODEL_TERMS=alphafold3),
# read as they are, decoded on the device - each port built for this GPU (nvidia-smi's compute capability), and the
# native featurisers beside them (cuda/featurise: no Node - a job is featurised in C++, byte for byte the page's).
# Idempotent: a bundle already on disk is not fetched again.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
ports=("$@"); [ ${#ports[@]} -gt 0 ] || ports=(ef2 af3 af2)
echo "GPU: $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1)"

bash "$repo/cuda/build.sh" featurise          # (the fetcher is one of the featuriser's names)
fetch="$repo/cuda/featurise/fetch-weights"
for port in "${ports[@]}"; do
  case "$port" in
    ef2) "$fetch" ef2-fast-600m ;;              # (read as they are: no export)
    af3) "$fetch" af3 ;;                                  # (read through its weight walk)
    af2) "$fetch" model_1_ptm model_1_multimer_v3 ;;      # (read through its weight walk)
    *) echo "unknown port $port (ef2, af3, af2)" >&2; exit 1 ;;
  esac
done
bash "$repo/cuda/build.sh" "${ports[@]}"
echo "ready: ${ports[*]}"
