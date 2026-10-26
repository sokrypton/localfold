#!/usr/bin/env python3
"""The native ports compile for a T4 (sm_75) on whatever card this machine has - the instructions a T4 lacks caught here.

    python3 tools/check-t4-build.py              # af3, af2, ef2 and bioemu2, compiled to objects for sm_75
    python3 tools/check-t4-build.py af2 ef2      # just these

🔴 WHY THIS EXISTS: A KERNEL BUILT ONLY FOR THIS A100 SAYS NOTHING ABOUT COLAB'S FREE GPU. cuda/build.sh compiles for
the card nvidia-smi reports, so every build and every gate here is sm_80, and a construct sm_75 lacks compiles cleanly
and folds correctly on this box. Branch argon-ptx routed every f32 -> f16x2/bf16 conversion and every f16 max through
hand-written PTX (cvt.rn.f16x2.f32, cvt.rn.bf16x2.f32, max.f16x2: sm_80 instructions), and AF2 then failed ptxas for
sm_75 with 13,688 errors, AF3 with 15,068 - the T4, and the wheel's multi-arch build (75 80 86 89 90 100 120) with it -
while every gate on this machine stayed green. Compiling is the cheap half: whether a T4 FOLDS right is rehearsed on
the A100 with compute_75 PTX and LOCALFOLD_EMULATE_CC=75 (cuda/af2/README.md), or run on Colab.

Each port's one translation unit is compiled as cuda/build.sh compiles it (the same flags, -arch=sm_75) to an object
in a temporary directory, the ports in parallel; nothing is linked or installed. A failure prints the compiler's
first errors and exits 1.
"""
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAST = {"af3", "af2"}            # (cuda/build.sh: --use_fast_math for these two, not for ef2 or bioemu2)


def compile_port(port, out):
    cmd = ["nvcc", "-O1", "-std=c++17", "-arch=sm_75", "--split-compile", "0", "--default-stream", "per-thread",
           *(["--use_fast_math"] if port in FAST else []), "-c", f"src/{port}.cu", "-o", os.path.join(out, f"{port}.o")]
    proc = subprocess.run(cmd, cwd=os.path.join(REPO, "cuda", port), capture_output=True, text=True)
    errors = [line for line in (proc.stdout + proc.stderr).splitlines() if " error" in line or "error:" in line]
    return port, proc.returncode, errors


def main():
    ports = [a for a in sys.argv[1:] if not a.startswith("-")] or ["af3", "af2", "ef2", "bioemu2"]
    with tempfile.TemporaryDirectory(prefix="t4-build-") as out, ThreadPoolExecutor(len(ports)) as pool:
        results = list(pool.map(lambda p: compile_port(p, out), ports))
    failed = False
    for port, code, errors in results:
        if code == 0:
            print(f"{port:8} compiles for sm_75")
            continue
        failed = True
        print(f"{port:8} FAILED for sm_75 ({len(errors)} error lines), the first:")
        for line in errors[:5]:
            print(f"    {line.strip()[:200]}")
    print("t4 build: " + ("FAILED" if failed else "ok"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
