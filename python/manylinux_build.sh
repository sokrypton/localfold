#!/usr/bin/env bash
# The wheel's build inside pypa's manylinux_2_28 image (AlmaLinux 8): CUDA from NVIDIA's RHEL 8 repository, then
# python/build_wheel.sh. One script for .github/workflows/wheel.yml and for a local run:
#
#   docker run --rm -v "$PWD:/src" -w /src quay.io/pypa/manylinux_2_28_x86_64 bash python/manylinux_build.sh dist
set -euo pipefail
cuda="${CUDA_VERSION:-12.8}"; v="${cuda/./-}"
dnf -q config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
# nvcc, the static runtime, cuBLAS's headers and link libraries, and CUPTI's header (the profiler loads the library
# itself, so only the header is needed)
dnf -q install -y "cuda-nvcc-$v" "cuda-cudart-devel-$v" "libcublas-devel-$v" "cuda-cupti-$v"
export PATH="/usr/local/cuda-$cuda/bin:$PATH" CPATH="/usr/local/cuda-$cuda/extras/CUPTI/include${CPATH:+:$CPATH}"
nvcc --version | tail -1; g++ --version | head -1
PYTHON=/opt/python/cp311-cp311/bin/python bash "$(dirname "$0")/build_wheel.sh" "${1:-dist}"
