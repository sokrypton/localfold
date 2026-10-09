#!/usr/bin/env bash
# metal/check/build-bench-gemm.sh: metal/build/bench-gemm (bench-gemm.mm), the runtime's GEMM on its own
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$here/build"
python3 - "$here/shim/shim.metal" "$here/shim/prelude.metal" > "$here/build/shimsrc.mm" <<'PY'
import sys
src = open(sys.argv[1]).read().replace('#include "prelude.metal"', open(sys.argv[2]).read())
parts = [src[i:i + 60000] for i in range(0, len(src), 60000)]
print("namespace lf { extern const char* SHIM_SOURCE = " + "\n".join(f'R"LFSRC({p})LFSRC"' for p in parts) + "; }")
PY
clang++ -std=c++17 -fobjc-arc -O2 -Wno-deprecated-declarations -Wno-unused-result -framework Metal -framework Foundation \
  -I "$here/shim/include" -I "$here/shim" -x objective-c++ "$here/shim/lfcuda.mm" "$here/build/shimsrc.mm" \
  "$here/check/bench-gemm-port.mm" "$here/check/bench-gemm.mm" -o "$here/build/bench-gemm"
echo "built $here/build/bench-gemm"
