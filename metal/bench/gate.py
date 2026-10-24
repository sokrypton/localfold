"""The Metal GEMM speed gate: localfold-bench's plain and extended-epilogue arms, held to this chip's own times.

    python3 metal/bench/gate.py [--write] [--bench=<another build's localfold-bench>]

The answer gates (metal/<port>/gate.py) cannot see a kernel getting slower - the answers never move. 5646656b made
every lf_gemm 15x slower on the M2 and 0de9becb's extended instance 1.7x, and both passed every gate on the M5, whose
dynamic registers never felt them (docs/M2-TO-M5.md). This gate is the bench those were bisected with, at the shapes
that showed them, recorded per chip: it fails at 1.3x its baseline, and only on a chip that regressed - so an lf_gemm
change wants a run on an M1/M2 (static registers) as well as on the chip that wrote it.

The bench's EP arms are lf_gemm on every chip (the M1-M4's path, the M5's control; the matrix units are its own arm).
Each shape three times, the best of the three medians kept (a drifting clock moves the slow runs, not the fast one).
Baselines: metal/bench/bench-baseline.<chip>-<GPU cores>c.json - speed differs between an M5 and an M5 Pro, so unlike
the answer gates a chip without its own file is not held to anyone's: it prints NEW, and --write records it.
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from gate_baseline import chip  # noqa: E402

BENCH = os.path.join(HERE, "localfold-bench")
LIMIT = 1.3
# (out, rows, in, arms): the M2's regression shape (whole 64-row tiles), the same with partial tiles, and a small one
SHAPES = [(1024, 66816, 256, "0,64"), (1024, 66850, 256, "0"), (1024, 1024, 256, "0,64")]


def gpu_cores():
    out = subprocess.run(["ioreg", "-rc", "AGXAccelerator"], capture_output=True, text=True).stdout
    m = re.search(r'"gpu-core-count" = (\d+)', out)
    return m.group(1) if m else "unknown"


def times(bench, shape):
    out, rows, k, arms = shape
    best = {}
    for _ in range(3):
        p = subprocess.run([bench, str(out), str(rows), str(k), arms, "hhh", "7"], capture_output=True, text=True)
        if p.returncode:
            sys.exit(f"localfold-bench {out} {rows} {k} {arms} failed:\n{p.stderr}")
        for arm, ms, wrong in re.findall(r"EP\+(\d+)\s+([\d.]+) ms.*?(WRONG)?$", p.stdout, re.M):
            if wrong:
                sys.exit(f"localfold-bench {out} {rows} {k}: arm {arm} WRONG against arm 0\n{p.stdout}")
            best[arm] = min(best.get(arm, float("inf")), float(ms))
    return {f"{out}x{rows}x{k} EP+{arm}": ms for arm, ms in best.items()}


def main():
    write = "--write" in sys.argv
    # (--bench= times another build against this checkout's baseline: a commit's bench, to see the gate catch it)
    bench = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--bench=")), BENCH)
    if not os.path.exists(bench):
        sys.exit(f"no {bench}: bash metal/build.sh bench")
    path = os.path.join(HERE, f"bench-baseline.{chip()}-{gpu_cores()}c.json")
    base = json.load(open(path)) if os.path.exists(path) else {}
    measured, failures = {}, 0
    for shape in SHAPES:
        for key, ms in times(bench, shape).items():
            measured[key] = ms
            if key not in base:
                verdict = "NEW (no baseline)"
            elif ms > LIMIT * base[key]:
                verdict = f"FAIL: {ms / base[key]:.2f}x its baseline {base[key]:.3f} ms"
                failures += 1
            else:
                verdict = f"ok ({ms / base[key]:.2f}x)"
            print(f"{key:28s} {ms:9.3f} ms  {verdict}", flush=True)
    if write:
        json.dump(dict(sorted(measured.items())), open(path, "w"), indent=1)
        print(f"wrote {path}")
    elif failures:
        sys.exit(f"{failures} GEMM arm(s) slower than {LIMIT}x this chip's baseline ({os.path.basename(path)})")


if __name__ == "__main__":
    main()
