#!/usr/bin/env bash
# metal/tools/build-bench-gemm.sh: metal/build/bench-gemm (bench-gemm.mm), the runtime's GEMM on its own
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$here/build"
python3 - "$here/runtime/runtime.metal" "$here/runtime/prelude.metal" > "$here/build/runtimesrc.mm" <<'PY'
import sys
src = open(sys.argv[1]).read().replace('#include "prelude.metal"', open(sys.argv[2]).read())
parts = [src[i:i + 60000] for i in range(0, len(src), 60000)]
print("namespace lf { extern const char* RUNTIME_SOURCE = " + "\n".join(f'R"LFSRC({p})LFSRC"' for p in parts) + "; }")
PY
clang++ -std=c++17 -fobjc-arc -O2 -Wno-deprecated-declarations -Wno-unused-result -framework Metal -framework Foundation \
  -I "$here/runtime/include" -I "$here/runtime" -x objective-c++ "$here/runtime/lfcuda.mm" "$here/build/runtimesrc.mm" \
  "$here/tools/bench-gemm-port.mm" "$here/tools/bench-gemm.mm" -o "$here/build/bench-gemm"
echo "built $here/build/bench-gemm"
