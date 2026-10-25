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
# The two stages a trunk oracle cannot reach, each with its own input and its own reference.
#
# 🔴 THE ENCODER NEEDS A REAL STRUCTURE, which is why it is the 1PRW partial-diffusion preset and
# not the monomer every other case uses: it turns clean all-atom coordinates into latents, and a
# contig has none. Its oracle is the DETERMINISTIC arm (z is the mean), because the sampled one
# would compare two Box-Muller implementations rather than the encoder.
#
# 🔴 AND THE TRAJECTORY IS THE ONLY ARM THAT CHECKS THE LOOP. Every stage above is held to the
# reference one call at a time; what is left over is whether they are WIRED the way
# `generateRfd4pCpu` wires them - the order of the noise draws, which features move between steps,
# when the emit masks apply, which recycle depth each step gets. Two steps exercise all of it, and
# it caught the port running four recycles a step against the reference's six.
ENCODER = ("encoder", HERE / "data-pd", ["--preset=partial_diffusion_1prw", "--encoder"])
# (name, input dir, exporter flags, needs the encoder beside --generate)
#
# 🔴 FOUR LOOPS, NOT ONE, BECAUSE THEY ARE FOUR DIFFERENT LOOPS. The monomer starts from noise and
# runs the ordinary schedule. `partial` starts from a STRUCTURE through the encoder, corrupts it to
# the midpoint and runs an EXPANDED schedule's tail with a different draw order. `inverse` keeps the
# backbone and redraws it from its forward marginal at every step. `smallmol` is the only shipped
# preset with chiral frames, which are the one sequence feature besides the noised state that moves
# between steps. Each of the last three has a branch the first cannot reach.
TRAJECTORIES = [
    ("trajectory", HERE / "data-g2", ["--generate", "--gen-oracle", "--steps=2"], False),
    ("partial", HERE / "data-pdg",
     ["--preset=partial_diffusion_1prw", "--generate", "--encoder", "--gen-oracle", "--steps=4"],
     True),
    ("inverse", HERE / "data-ifg",
     ["--preset=inverse_folding_1prw", "--generate", "--encoder", "--gen-oracle", "--steps=2"],
     True),
    ("smallmol", HERE / "data-smg",
     ["--preset=sm_sam", "--generate", "--gen-oracle", "--steps=2"], False),
]
SPEED = ("speed-n100-r6", HERE / "data-n100", ["--length=100"], ["--recycles=6", "--repeat=30"])
# The trunk's bounds; the DECODER's stages take 1e-4 on both arms and the binary applies that
# itself - it runs in f32 whatever the arm, because its near-uniform softmax costs far more in
# reduced precision than the trunk's does and it is one pass a generation. See src/rfd4p.cu.
BOUND = {"exact": 1e-5, "fast": 5e-3, "int8": 2.5e-2}   # and the same for each one's `-big` arm
# 🔴 A TRAJECTORY'S BOUND IS NOT A STAGE'S, because it compounds: the trunk's ~1e-3 a pass is
# amplified by the first step's division by t = 0.001 and the next step starts from the result. The
# number that says what that is worth is the structure, which the arm measures beside it.
TRAJECTORY_BOUND = {"exact": 1e-5, "fast": 5e-2}
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


def ca_rms(a: Path, b: Path) -> float:
    """CA root-mean-square deviation between two PDBs, as written (no superposition).

    They are two arithmetics over ONE trajectory from ONE seed, so they are already in the same
    frame - fitting them would hide exactly the drift being measured.
    """
    def cas(p: Path) -> list[tuple[float, float, float]]:
        return [(float(l[30:38]), float(l[38:46]), float(l[46:54]))
                for l in p.read_text().splitlines()
                if l.startswith("ATOM") and l[12:16].strip() == "CA"]
    x, y = cas(a), cas(b)
    if not x or len(x) != len(y):
        return float("nan")
    total = sum((px - qx) ** 2 + (py - qy) ** 2 + (pz - qz) ** 2
                for (px, py, pz), (qx, qy, qz) in zip(x, y))
    return (total / len(x)) ** 0.5


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
        # 🔴 AND THE LONG FORM IS RUN AT 54 TOKENS, which is the only way it is checked at all: two
        # of the triangle's choices invert with length and the long branch only fires past ~40,000
        # pairs, where no oracle case reaches. LOCALFOLD_BIG forces it, as cuda/af3's big-input
        # paths are forced - an untested branch is a second, wrong implementation waiting to happen.
        # 🔴 THE int8 ARM IS HELD TO THE BUNDLE'S QUANTISATION, NOT THE PORT'S ARITHMETIC. Reading
        # the published int8 weights against an oracle computed from the FLOAT checkpoint measures
        # how good int8 is, which is upstream's decision and not this port's; what it catches here
        # is a reader that puts a tensor in the wrong place. It is skipped where the shards are
        # not on disk, which is everywhere but a machine that fetched them.
        arms = ["exact", "fast", "exact-big", "fast-big"]
        if (REPO / "model-rfd4p-int8" / "shards" / "shard_000.bin").exists():
            arms.append("int8")
        for arm in arms:
            weights = (f"--int8={REPO / 'model-rfd4p-int8'}" if arm == "int8"
                       else f"--weights={args.weights}")
            cmd = [BINARY, case_dir, weights, f"--oracle={oracle_dir}"]
            if arm.startswith("fast") or arm == "int8":
                cmd.append("--fast")
            env = dict(os.environ)
            if arm.endswith("-big"):
                env["LOCALFOLD_BIG"] = "1"
            r = run(cmd, env=env)
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
            print(f"  {name:10s} {arm:10s} worst {worst:.2e}  (trunk {BOUND[arm.split('-')[0]]:.0e},"
                  f" decoder 1e-04)  {mark}")
        got[name] = row

    # ---- the encoder, against its own reference -----------------------------------------------
    name, case_dir, export_flags = ENCODER
    if not args.only or args.only in name:
        if not (case_dir.with_name(case_dir.name + "-enc") / "model.idx").exists():
            print(f"  building {case_dir.name} (the encoder's structure case)...", flush=True)
            r = run(["node", "--max-old-space-size=12288", HERE / "export_input.mjs", case_dir,
                     *export_flags], cwd=REPO)
            if r.returncode != 0:
                print(r.stdout[-2000:], r.stderr[-2000:], file=sys.stderr)
                sys.exit(f"could not export {case_dir}")
        r = run([BINARY, case_dir, f"--weights={args.weights}",
                 f"--encoder={case_dir}-enc", f"--enc-oracle={case_dir}-enc-oracle"])
        m = re.search(r"encoder: (\w+), worst ([\d.e+-]+)", r.stdout)
        worst = float(m.group(2)) if m else float("nan")
        mark = "ok "
        if not m or m.group(1) != "ok":
            mark, failed = "OVER BOUND", failed + 1
            if not m:
                print(r.stdout[-1200:], r.stderr[-800:])
        elif name in base and worst > base[name].get("exact", 0) * TOL_RES:
            mark, failed = "MOVED", failed + 1
        print(f"  {name:10s} {'exact':10s} worst {worst:.2e}  (1e-04)  {mark}")
        got[name] = {"exact": worst}

    # ---- the whole sampling loop, against the reference's own --------------------------------
    for name, case_dir, export_flags, needs_encoder in TRAJECTORIES:
      if not args.only or args.only in name:
          if not (case_dir.with_name(case_dir.name + "-gen-oracle") / "model.idx").exists():
              print(f"  building {case_dir.name} (two steps of the JS sampler)...", flush=True)
              r = run(["node", "--max-old-space-size=12288", HERE / "export_input.mjs", case_dir,
                       *export_flags], cwd=REPO)
              if r.returncode != 0:
                  print(r.stdout[-2000:], r.stderr[-2000:], file=sys.stderr)
                  sys.exit(f"could not export {case_dir}")
          row = {}
          for arm in ("exact", "fast"):
              cmd = [BINARY, case_dir, f"--weights={args.weights}", f"--generate={case_dir}-gen",
                       f"--gen-oracle={case_dir}-gen-oracle", f"--out={case_dir}-gate.pdb"]
              if needs_encoder:
                  cmd.append(f"--encoder={case_dir}-enc")
              if arm == "fast":
                  cmd.append("--fast")
              r = run(cmd)
              vals = [float(v) for v in re.findall(r"gen/x(?:Bb|Ll)\s+relRMS ([\d.e+-]+)", r.stdout)]
              worst = max(vals) if vals else float("nan")
              mark = "ok "
              if not vals or "DIVERGES" in r.stdout:
                  mark, failed = "DIVERGES", failed + 1
                  if not vals:
                      print(r.stdout[-1200:], r.stderr[-800:])
              row[arm] = worst
              print(f"  {name:10s} {arm:10s} worst {worst:.2e}  "
                    f"({TRAJECTORY_BOUND[arm]:.0e})  {mark}")
          # 🔴 AND WHAT --fast COSTS A FINISHED STRUCTURE, which is the only form of that question
          # anybody can act on. A trajectory compounds, so its relRMS says the arithmetic differs and
          # not by how much it matters; the preset's own 60 steps both ways, scored on CA, says that.
          full, full_gen = HERE / "data", HERE / "data-gen"
          if name == "trajectory" and (full_gen / "model.idx").exists():
              pdbs = []
              for arm in ("exact", "fast"):
                  out = HERE / f"gate-full-{arm}.pdb"
                  cmd = [BINARY, full, f"--weights={args.weights}", f"--generate={full_gen}",
                         f"--out={out}"]
                  if arm == "fast":
                      cmd.append("--fast")
                  run(cmd)
                  pdbs.append(out)
              rms = ca_rms(*pdbs)
              mark = "ok "
              was = base.get(name, {}).get("fast_ca_rms")
              if rms != rms or rms > 0.2:
                  mark, failed = "MOVED", failed + 1
              elif was:
                  mark = f"ok (was {was})"
              row["fast_ca_rms"] = round(rms, 4)
              print(f"  {name:10s} {'fast vs exact':10s} {rms:8.3f} A of CA over 60 steps  {mark}")
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
