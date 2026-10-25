#!/usr/bin/env python3
"""The rfd4p port's gate: every trunk stage against the CPU reference, both arithmetics.

    python3 cuda/rfd4p/gate.py [--write] [--only=<case>] [--weights=<dir>] [--no-build]

🔴 THIS GATE IS THE ONLY THING THAT RE-RUNS THE PORT'S NUMBERS. The README's residuals and its speed
table are prose, and this repository's own recurring lesson is that a figure in prose is re-run when
somebody decides to and a figure in a file is re-run by the gate.

Two arms:

  oracles  every stage of `forwardTrunkCpu` against the port's, at one recycle and at two, in the
           exact arithmetic (bound 1e-5) and under --fast (5e-3). The two-recycle case is not a
           duplicate: it is the only arm where c0 and c1 differ, so it is what catches a cycle
           projection or a conditioning that has been mixed up.
  speed    a steady-state trunk pass at 100 tokens and 6 recycles, both arithmetics, recorded in
           gate-baseline.json.

🔴 A BASELINE IS THIS MACHINE'S. A fold does not reproduce to the digit on another GPU and a timing
reproduces even less, so a moved number on a different card is a question and not a failure - the
report says which card recorded it.

It needs the float32 weights (`export_weights.mjs`, from NVIDIA's own checkpoint under NVIDIA's own
terms) and the exported inputs. Both are rebuilt on demand if absent; the exporter runs the JS
reference, which takes a couple of minutes a case.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BINARY = HERE / "localfold-rfd4p"
WEIGHTS = HERE / "weights-f32"
BASELINE = HERE / "gate-baseline.json"

# (name, input dir, oracle dir, exporter flags)
CASES = [
    ("trunk-1", HERE / "data", HERE / "oracle", []),
    ("trunk-2", HERE / "data-r2", HERE / "oracle-r2", ["--recycles=2"]),
]
SPEED = ("speed-n100-r6", HERE / "data-n100", ["--length=100"], ["--recycles=6", "--repeat=30"])
BOUND = {"exact": 1e-5, "fast": 5e-3}
# A residual is this machine's arithmetic, not its luck: hold it to the bound AND to the baseline.
TOL_RES = 2.0   # a stage may drift this many times its recorded residual before it is a finding
TOL_MS = 0.25   # a timing is noisy and machine-specific; 25% is a regression worth printing


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)


def ensure_inputs(case_dir: Path, oracle_dir: Path | None, flags: list[str]) -> None:
    if (case_dir / "model.idx").exists() and (oracle_dir is None or (oracle_dir / "model.idx").exists()):
        return
    cmd = ["node", "--max-old-space-size=12288", HERE / "export_input.mjs", case_dir, *flags]
    if oracle_dir is not None:
        cmd.append(f"--oracle={oracle_dir}")
    print(f"  building {case_dir.name} (the JS reference, a couple of minutes)...", flush=True)
    r = run(cmd, cwd=REPO)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:], file=sys.stderr)
        sys.exit(f"could not export {case_dir}")


def worst_of(out: str) -> tuple[float, int]:
    m = re.search(r"worst stage (\S+) at relRMS ([\d.e+-]+); (\d+) stages? past", out)
    if not m:
        return (float("nan"), -1)
    return (float(m.group(2)), int(m.group(3)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="re-record the baseline")
    ap.add_argument("--only", default="")
    ap.add_argument("--weights", default=str(WEIGHTS))
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()

    if not args.no_build:
        r = run(["bash", REPO / "cuda" / "build.sh", "rfd4p"])
        if r.returncode != 0:
            print(r.stdout[-3000:], r.stderr[-3000:], file=sys.stderr)
            return 1
    if not BINARY.exists():
        return print(f"no {BINARY}") or 1
    if not (Path(args.weights) / "model.idx").exists():
        return print(f"no weights at {args.weights}; run cuda/rfd4p/export_weights.mjs") or 1

    card = run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).stdout.strip()
    base = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
    got: dict[str, dict] = {}
    failed = 0

    for name, case_dir, oracle_dir, flags in CASES:
        if args.only and args.only not in name:
            continue
        ensure_inputs(case_dir, oracle_dir, flags)
        row = {}
        for arm in ("exact", "fast"):
            cmd = [BINARY, case_dir, f"--weights={args.weights}", f"--oracle={oracle_dir}"]
            if arm == "fast":
                cmd.append("--fast")
            r = run(cmd)
            worst, past = worst_of(r.stdout)
            if r.returncode != 0 and past <= 0:
                print(f"{name}/{arm}: FAILED to run\n{r.stdout[-1500:]}{r.stderr[-1500:]}")
                failed += 1
                continue
            row[arm] = worst
            mark = "ok "
            if past > 0:
                mark, failed = "OVER BOUND", failed + 1
            elif name in base and arm in base[name] and worst > base[name][arm] * TOL_RES:
                mark, failed = "MOVED", failed + 1
            print(f"  {name:10s} {arm:6s} relRMS {worst:.2e}  (bound {BOUND[arm]:.0e})  {mark}")
        got[name] = row

    name, case_dir, export_flags, fold_flags = SPEED
    if not args.only or args.only in name:
        ensure_inputs(case_dir, None, export_flags)
        row = {}
        for arm in ("exact", "fast"):
            cmd = [BINARY, case_dir, f"--weights={args.weights}", *fold_flags]
            if arm == "fast":
                cmd.append("--fast")
            times = []
            for _ in range(3):
                r = run(cmd)
                m = re.search(r"trunk: ([\d.]+) ms a pass", r.stdout)
                if m:
                    times.append(float(m.group(1)))
            if not times:
                print(f"{name}/{arm}: no timing")
                failed += 1
                continue
            ms = sorted(times)[len(times) // 2]
            row[f"{arm}_ms"] = round(ms, 2)
            mark = "ok "
            was = base.get(name, {}).get(f"{arm}_ms")
            if was and ms > was * (1 + TOL_MS):
                mark, failed = f"SLOWER (was {was})", failed + 1
            elif was:
                mark = f"ok (was {was})"
            print(f"  {name:10s} {arm:6s} {ms:7.2f} ms a pass  {mark}")
        got[name] = row

    if args.write:
        got["_card"] = card
        BASELINE.write_text(json.dumps(got, indent=1, sort_keys=True) + "\n")
        print(f"recorded for {card}")
        return 0
    if base.get("_card") and base["_card"] != card:
        print(f"🔴 the baseline is {base['_card']}'s, this is {card} - a moved number is a question")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
