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
# 🔴 THE NATIVE FEATURISER IS HELD TO BYTES, NOT TO A TOLERANCE. `export_input.mjs` drives the
# page's own shared/rfd4p/ and stays as the reference; `rfd4p-featurise` writes the same
# model.idx/model.bin and a single differing float fails this. A sweep of LENGTHS, not one: the
# first version was exact at 53 and 100 residues and one ulp out at 1, 2, 7 and 257, the centering
# scale written `0.2f` where the reference multiplies in double.
FEATURISE_LENGTHS = [1, 2, 7, 53, 100, 257, 400]
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


def write_nucleic_ss_case(cif: Path, job: Path) -> None:
    """The DNA example with a nucleic state on every residue and base pairs across its strands.

    `condition_na-ss-state_1_residue` cycles PAIR, PAIR, LOOP, UNSPECIFIED down each strand, and
    `_condition_basepair_2_residue` pairs strand A's residue i with strand B's residue n-1-i on
    each residue's first atom - one row a pair, as upstream writes it, which its reader expands
    both ways.
    """
    import zstandard

    text = zstandard.ZstdDecompressor().decompressobj().decompress(
        (REPO / "web" / "examples" / "dna_7m5w.cif.zst").read_bytes()).decode()
    lines = text.splitlines()
    cols, first = [], None
    for i, ln in enumerate(lines):
        if ln.startswith("_atom_site."):
            first = i if first is None else first
            cols.append(ln.split(".", 1)[1].strip())
    body = first + len(cols)
    end = body
    while end < len(lines) and not lines[end].startswith(("#", "loop_", "_")):
        end += 1
    rows = [ln.split() for ln in lines[body:end]]
    ci, si = cols.index("label_asym_id"), cols.index("label_seq_id")
    residues: dict[str, list[tuple[str, int]]] = {}
    first_atom: dict[tuple[str, str], int] = {}
    for k, f in enumerate(rows):
        key = (f[ci], f[si])
        if key not in first_atom:
            first_atom[key] = k
            residues.setdefault(f[ci], []).append(key)
    cycle = [1, 1, 2, 0]
    state = {}
    for chain in ("A", "B"):
        for r, key in enumerate(residues.get(chain, [])):
            state[key] = cycle[r % 4]
    out = lines[:body - 0]
    out.insert(first + len(cols), "_atom_site.condition_na-ss-state_1_residue")
    for f in rows:
        out.append(" ".join(f) + f" {state.get((f[ci], f[si]), 0)}")
    out.extend(lines[end:end + 1] if end < len(lines) and lines[end].startswith("#") else ["#"])
    out += ["loop_", "_condition_basepair_2_residue.idx0", "_condition_basepair_2_residue.idx1",
            "_condition_basepair_2_residue.val0"]
    a, b = residues.get("A", []), residues.get("B", [])
    for i in range(min(len(a), len(b))):
        out.append(f"{first_atom[a[i]]} {first_atom[b[len(b) - 1 - i]]} 1")
    out.append("#")
    rest = lines[end + 1:] if end < len(lines) and lines[end].startswith("#") else lines[end:]
    cif.write_text("\n".join(out + rest) + "\n")
    job.write_text(json.dumps({"input": str(cif), "nsteps": 4}))


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


def chain_steps(pdb: Path) -> list[float]:
    """Consecutive CA-CA distances, which say whether a fold is a chain at all."""
    if not pdb.exists():
        return []
    ca = [(float(l[30:38]), float(l[38:46]), float(l[46:54]))
          for l in pdb.read_text().splitlines()
          if l.startswith("ATOM") and l[12:16].strip() == "CA"]
    return [sum((a - b) ** 2 for a, b in zip(ca[i], ca[i + 1])) ** 0.5
            for i in range(len(ca) - 1)]


def compare_dirs(native: Path, js: Path) -> list[str]:
    """Every tensor the NATIVE directory writes, against the JavaScript's, as bytes.

    🔴 THE JAVASCRIPT SIDE MAY CARRY MORE, AND THAT IS NOT A DIFFERENCE. `--oracle=` makes the
    exporter write `in/zLatent` and `in/decSeq`, which are the reference TRUNK's outputs and not a
    featurisation; the featuriser rightly does not produce them. What is compared is what the
    native side claims, and a key it claims that the JavaScript lacks IS a difference.
    """
    import struct

    def load(d: Path) -> dict:
        out: dict = {}
        raw = (d / "model.bin").read_bytes()
        for ln in (d / "model.idx").read_text().splitlines():
            p = ln.split()
            if p[0] == "m":
                out[p[1]] = ("m", float(p[2]))
                continue
            off, cnt = int(p[2]), int(p[3])
            fmt = ("<%df" if p[0] == "t" else "<%di") % cnt
            out[p[1]] = (p[0], struct.unpack(fmt, raw[off * 4:(off + cnt) * 4]))
        return out

    if not (native / "model.idx").exists() or not (js / "model.idx").exists():
        return ["one side was not written"]
    a, b = load(native), load(js)
    differ = []
    for k, (kind, va) in sorted(a.items()):
        if k not in b:
            differ.append(f"{k} is only in the native directory")
            continue
        vb = b[k][1]
        if kind == "m":
            if va != vb:
                differ.append(f"{k} is {va} native and {vb} in JS")
        elif len(va) != len(vb):
            differ.append(f"{k} is {len(va)} long native and {len(vb)} in JS")
        else:
            # 🔴 NaN EQUALS NaN HERE. A conditioned coordinate that nobody set is NaN on both
            # sides, and `x != y` is TRUE for two NaNs - so a column that is absent by design
            # reads as every element differing. The first tensor to carry NaN through this
            # comparison reported 111 of 111 wrong while the two files were byte-identical.
            n = sum(1 for x, y in zip(va, vb)
                    if x != y and not (x != x and y != y))
            if n:
                differ.append(f"{k}: {n} of {len(va)} differ")
    return differ


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
          # 🔴 AND THE TWO FLAGS TOGETHER, ONCE. `--designs=N` makes every design after the first
          # draw its OWN noise, so an oracle comparison must be pinned to design 0 - and when it
          # was not, the gate would have read two different noise streams as a port defect. No arm
          # ran both until this one.
          if name == "trajectory":
              r = run([BINARY, case_dir, f"--weights={args.weights}",
                       f"--generate={case_dir}-gen", f"--gen-oracle={case_dir}-gen-oracle",
                       f"--out={case_dir}-gate2.pdb", "--designs=2"], cwd=REPO)
              matched = r.stdout.count("the trajectory matches")
              diverged = "DIVERGES" in r.stdout
              both_ok = matched == 1 and not diverged
              if not both_ok:
                  failed += 1
              print(f"  {'designs+oracle':10s} compared {matched} time(s), "
                    f"{'pinned to design 0' if both_ok else 'WRONG DESIGN'}  "
                    f"{'ok ' if both_ok else 'WRONG'}")
              row["designs_oracle"] = matched
          got[name] = row

    # ---- the native featuriser, against the JavaScript exporter's own bytes -------------------
    if not args.only or args.only in "featurise":
        binary = HERE / "featurise" / "rfd4p-featurise"
        if not binary.exists():
            print("  featurise  no rfd4p-featurise; run bash cuda/build.sh rfd4p")
            failed += 1
        else:
            # 🔴 THE RECYCLE SCHEDULE IS PASSED TO BOTH SIDES RATHER THAN DEFAULTED ON EITHER.
            # `export_input.mjs --length=N` replaces only the CONTIG, so it keeps the default
            # preset's `low` schedule, where the C++ has no preset and defaults to `none`. That is
            # a difference in the harness, not in the featuriser, and saying it out loud is
            # cheaper than teaching one side about the other's menu.
            cases = [(f"n{L}", [f"--length={L}", "--recycle=none"],
                      [f"--length={L}", "--recycle=none"]) for L in FEATURISE_LENGTHS]
            # And a CONDITIONED case, because a plain segment reaches neither the selection
            # language nor the CATH table. The preset is turned into the plain condition spec the
            # C++ reads, so neither side learns about the other's menu.
            spec = HERE / "featurise" / "gate-cath.json"
            r = run(["node", HERE / "featurise" / "preset-spec.mjs", "custom_cath_monomer"],
                    cwd=REPO)
            if r.returncode == 0:
                spec.write_text(r.stdout)
                cases.append(("cath", [f"--spec={spec}"], ["--preset=custom_cath_monomer"]))
            # 🔴 AND A RANGED CONTIG AT FOUR SEEDS, which is the only arm that reaches the
            # PIPELINE's random source. It is a different generator from the sampler's - mulberry32
            # where the sampler is xorshift128+ - and it decides how long each segment comes out:
            # these four seeds give 24, 30, 27 and 33 tokens, so a wrong draw order is a wrong
            # protein rather than a wrong digit.
            ranged = HERE / "featurise" / "gate-ranged.json"
            for sd in (0, 1, 7, 42):
                cases.append((f"ranged{sd}", [f"--spec={ranged}", f"--seed={sd}"],
                              [f"--spec={ranged}", f"--seed={sd}"]))
            # 🔴 AND THREE CHAINS, because a one-chain job cannot see the permutations at all -
            # `permutation(1)`'s Fisher-Yates loop runs zero times - and a TWO-chain one agreed by
            # coincidence while the featuriser was skipping a draw. Three is where "the segment
            # length is drawn even when min == max" showed: skipping it moved every later draw up
            # one and relabelled the chains.
            chains = HERE / "featurise" / "gate-chains.json"
            for sd in (0, 3):
                cases.append((f"chains{sd}", [f"--spec={chains}", f"--seed={sd}"],
                              [f"--spec={chains}", f"--seed={sd}"]))
            # And a protein, a DNA and an RNA chain in one job: for a contig the chain type reaches
            # the batch as `token_type` and nothing else, which is exactly why it is worth an arm -
            # one wrong integer a token, in a field no geometry check could see.
            mixed = HERE / "featurise" / "gate-mixed.json"
            cases.append(("mixed", [f"--spec={mixed}"], [f"--spec={mixed}"]))
            # 🔴 AND TWO JOBS WITH A REAL INPUT STRUCTURE, which is a different pipeline from
            # everything above it: `buildSpecBatch` is the contig-only specialisation where most
            # of `convertToProteinaBatch` is a no-op, and a structure runs the whole of it - the
            # reader, the contig, the conditions and the fourteen transform stages. One without a
            # contig (take the structure whole) and one with, because the contig is what makes
            # the conditions' residue numbers differ from the assembled ones.
            #
            # 🔴 AND `gate-structure.json` CARRIES NO CONDITIONS, BECAUSE THE ONES IT USED TO
            # CARRY DID NOTHING. Without `specMode` a job's `input` is a STRUCTURE and not a
            # spec, so `applyContig`/`applyConditions` are never called and the file's own
            # annotations are kept instead - measured, the fold is BYTE-IDENTICAL with the two
            # conditions deleted. That is correct and load-bearing rather than a bug: eight of
            # the ten shipped presets have exactly this shape, their `.cif.zst` already carrying
            # the conditions as annotations, which is why `input_as_annotated_cif` is true on
            # that path and why refusing here would break them. What was wrong was the FIXTURE,
            # which copied a preset's shape and so looked like it tested conditions on a
            # structure. `gate-structure-contig.json` and `gate-crdgroups.json` set `specMode`
            # and are where a structure's conditions are actually gated.
            for tag in ("structure", "structure-contig"):
                spec = HERE / "featurise" / f"gate-{tag}.json"
                cases.append((tag, [f"--spec={spec}"], [f"--spec={spec}"]))
            # 🔴 AND THREE COORDINATE GROUPS, WHICH NO SHIPPED EXAMPLE HAS. `coord_group_seq`
            # permutes its group LABELS at the top of every step, off the SAMPLER's stream and
            # in RAW uniforms rather than normals - so the draw both relabels the feature and
            # shifts every later draw of the run. At one group the permutation is the identity
            # and costs no stream, which is why the whole mechanism is invisible on the corpus:
            # `export_input.mjs` refused it outright and this side wrote frozen labels.
            crdgrp = HERE / "featurise" / "gate-crdgroups.json"
            cases.append(("crdgroups", [f"--spec={crdgrp}"], [f"--spec={crdgrp}"]))
            # 🔴 AND THE RUNS THAT DO NOT START FROM NOISE, WHICH THE NATIVE FEATURISER DROPPED.
            # A job naming partial diffusion or inverse folding was featurised as an ordinary
            # design and generated from noise with exit status 0: the spec reader had no field
            # for either and wrote `partial = 0` as a constant. Every form `resolve_partial_t`
            # takes - a fraction, both start times, one COUPLED through the schedules, inverse
            # folding - and the encoder's inputs (`-enc`), which only the exporter wrote, so
            # those two runs needed Node however the rest of a job was featurised.
            for tag in ("partial", "inverse", "partial-t", "partial-couple"):
                spec = HERE / "featurise" / f"gate-{tag}.json"
                cases.append((tag, [f"--spec={spec}"], [f"--spec={spec}", "--encoder"]))
            # 🔴 AND A CHIRAL CENTRE, WHOSE FRAMES THE NATIVE FEATURISER NEVER WROTE. Its comment
            # said "a contig has no chiral frames; a structure input will", and nothing did - so
            # a small molecule through `--job` sampled with its chirality feature FROZEN at the
            # clean frame, and the step-0 features copied the clean vectors where the page
            # rotates them onto the noisy CA (252 values on `sm_sam`).
            chiral = HERE / "featurise" / "gate-chirality.json"
            cases.append(("chirality", [f"--spec={chiral}"], [f"--spec={chiral}"]))
            # 🔴 AND NUCLEIC SECONDARY STRUCTURE, WHICH BOTH PORTS WROTE AS ZEROS. The model reads
            # `na_ss_state` and `na_ss_pair`; the reader did not keep the state column and nothing
            # read the base-pair loop. No shipped example carries either, so one is SYNTHESISED
            # from the DNA example - a state a residue and pairs across the two strands - and the
            # two ports are held to each other on it. (Against upstream itself the page's batch
            # matches its own tokeniser on `data/benchmarks/rna/pseudoknot/W05`: 90 tokens, 74
            # directed pairs, 16 loop tokens, every array exact; that file is upstream's and is
            # not copied here.)
            nss = HERE / "featurise" / "gate-nucleic-ss.json"
            write_nucleic_ss_case(HERE / "featurise" / "gate-nucleic-ss.cif", nss)
            cases.append(("nucleic-ss", [f"--spec={nss}"], [f"--spec={nss}"]))
            # 🔴 AND THE GENERATION DIRECTORY, NOT ONLY THE TRUNK'S INPUT. `--generate` writes the
            # per-step conditioning, the emit masks, the recycle depths and the NOISE - which the
            # C++ now draws rather than replays, so Box-Muller is compared here too. Without this
            # arm the featuriser could be exact on one step and wrong about the whole run.
            cases = [(tag, nf + ["--generate"], jf + ["--generate"]) for tag, nf, jf in cases]
            first, bad_fields = "", 0
            for tag, native_flags, js_flags in cases:
                nat, js = HERE / f"gate-nat-{tag}", HERE / f"gate-js-{tag}"
                run([binary, nat, f"--weights={args.weights}",
                     f"--config={REPO}/model-rfd4p-int8/config.json", *native_flags], cwd=REPO)
                if not (js / "model.idx").exists():
                    run(["node", "--max-old-space-size=12288", HERE / "export_input.mjs", js,
                         *js_flags], cwd=REPO)
                differ = compare_dirs(nat, js)
                differ += [f"-gen {d}" for d in compare_dirs(
                    Path(str(nat) + "-gen"), Path(str(js) + "-gen"))]
                if (Path(str(nat) + "-enc") / "model.idx").exists():
                    differ += [f"-enc {d}" for d in compare_dirs(
                        Path(str(nat) + "-enc"), Path(str(js) + "-enc"))]
                if differ:
                    bad_fields += len(differ)
                    first = first or f"{tag}: {differ[0]}"
            mark = "ok " if not bad_fields else "DIFFERS"
            if bad_fields:
                failed += 1
            print(f"  {'featurise':10s} {len(cases)} cases, {bad_fields} field(s) differ"
                  f"{'  ' + first if first else ''}  {mark}")
            got["featurise"] = {"cases": len(cases), "differing": bad_fields}

            # 🔴 AND WHAT IT REFUSES, BY NAME. A featuriser that covers a subset has to say which
            # subset at the moment it is handed something outside it - a condition quietly matched
            # to nothing is a condition the model is told is absent, which is a different job
            # folded under the right name. Each of these must name the thing it cannot take.
            # 🔴 A PURE CONTIG RUNS THE STRUCTURE PIPELINE NOW, from an empty structure as the
            # reference does, so the refusals that were this port's SUBSET are gone - a selection
            # that constrains a chain is evaluated, an unnamed segment is taken - and the two
            # contig refusals are the reference's own, in its own words.
            refusals = [
                ('{"contig":"a1:5-5P,A16-35"}', "matched no atoms"),
                ('{"contig":"a1:5-5X"}', "Malformed expandable-segment token"),
                ('{"contig":"a1:5-5P","conditions":{"C_SEQ":{"ALA":[{"select":"sel(\'*/a1/*/*\')"}]}}}',
                 "drops silently"),
                # 🔴 A CENTROID WITH FEWER THAN THREE NUMBERS. A deliberate divergence: the
                # reference stores `parts.slice(0, 3)`, so `"1.5,-2.25"` becomes a TWO-element
                # entry in a column every stage after it reads as a point, silently. This port
                # cannot represent that and would not want to - padding the third with NaN
                # invents a coordinate. MORE than three is not refused, because both sides
                # truncate and `check-cif.mjs` holds them to each other on a four-component
                # value. Found because gcov said the whole `centroid` branch had never run.
                #
                # 🔴 AND IT NEEDS AN INPUT STRUCTURE **AND** `specMode`, both of which the
                # first two attempts missed, each failing differently. On a pure contig a
                # centroid is ALREADY refused for needing a structure, so that earlier refusal
                # fired and this arm read NOT REFUSED with the rule it was testing never
                # reached - two refusals on one input test the first of them. Then with an
                # `input` and no `specMode` the job is a STRUCTURE rather than a spec, so
                # `applyConditions` is never called and the condition is dropped before it can
                # be refused - the exact trap the comment above the featurise arm describes,
                # walked into while writing that comment.
                ('{"input":"web/examples/ppi_pdl1.cif.zst","specMode":true,"conditions":'
                 '{"C_CENTROID":{"1.5,-2.25":[{"select":"sel(\'B/*/*/*\')"}]}}}', "not three"),
            ]
            bad_refusals = []
            probe = HERE / "featurise" / "gate-refusal.json"
            for text, wanted in refusals:
                probe.write_text(text)
                r = run([binary, HERE / "gate-refused", f"--weights={args.weights}",
                         f"--config={REPO}/model-rfd4p-int8/config.json", f"--spec={probe}"],
                        cwd=REPO)
                if r.returncode == 0 or wanted not in r.stderr:
                    bad_refusals.append(wanted)
            if bad_refusals:
                failed += 1
            print(f"  {'refusals':10s} {len(refusals)} inputs outside the subset  "
                  f"{'ok ' if not bad_refusals else 'NOT REFUSED: ' + ', '.join(bad_refusals)}")
            got["refusals"] = {"checked": len(refusals), "wrong": len(bad_refusals)}

            # 🔴 AND THE ONE-PROCESS FORM, because linking the featuriser in is a second way to
            # reach it and a second way to get it wrong. `--job=<spec>` must produce the SAME PDB
            # as featurising to a directory and folding that directory - byte for byte, since both
            # routes run the same code over the same bytes.
            jobdir = HERE / "gate-job"
            jobpdb, viapdb = HERE / "gate-job.pdb", HERE / "gate-via.pdb"
            r = run([BINARY, f"--job={ranged}", f"--weights={args.weights}",
                     f"--work-dir={jobdir}", f"--out={jobpdb}", "--seed=0"], cwd=REPO)
            run([BINARY, HERE / "gate-nat-ranged0", f"--weights={args.weights}",
                 f"--generate={HERE / 'gate-nat-ranged0'}-gen", f"--out={viapdb}"], cwd=REPO)
            same = (jobpdb.exists() and viapdb.exists()
                    and jobpdb.read_bytes() == viapdb.read_bytes())
            if not same:
                failed += 1
                if r.returncode != 0:
                    print(r.stdout[-800:], r.stderr[-600:])
            print(f"  {'job':10s} one process against featurise-then-fold  "
                  f"{'ok ' if same else 'DIFFERS'}")
            got["job"] = {"same_pdb": bool(same)}

            # 🔴 AND THE SAME JOB FROM THE PUBLISHED BUNDLE ALONE, which is the run anybody can
            # make: no float checkpoint and so no weight terms to accept. It is scored on the
            # CHAIN and not against the float arm's coordinates - int8 perturbs the velocity field
            # and the sampler is chaotic, so the trajectory lands on a DIFFERENT member of the
            # distribution (2.1 A of CA at 25 steps), which is not a defect and not a tolerance
            # anybody should pin.
            bundle = REPO / "model-rfd4p-int8"
            if (bundle / "shards" / "shard_000.bin").exists():
                i8pdb = HERE / "gate-job-int8.pdb"
                # 🔴 A CONVERGED JOB, NOT THE FEATURISER'S. The ranged case above runs 10 steps
                # at one recycle because it exists to compare BYTES; folded, it is not a chain on
                # either arithmetic (CA-CA 4.32 A, worst gap 14.6) and reading that as int8's
                # fault would be reading the step count as a defect.
                fold = HERE / "featurise" / "gate-fold.json"
                # `--config=` is the featuriser's flag and must not reach the fold, which would
                # refuse it AFTER the featuriser had written its directory - a failure in silence,
                # since the only lines anybody reads are the featuriser's.
                run([BINARY, f"--job={fold}", f"--int8={bundle}",
                     f"--config={bundle}/config.json",
                     f"--work-dir={HERE / 'gate-job-int8'}", f"--out={i8pdb}", "--seed=0",
                     "--fast"], cwd=REPO)
                caca = chain_steps(i8pdb)
                ok = bool(caca) and 3.4 <= (sum(caca) / len(caca)) <= 4.1 and min(caca) > 2.8
                if not ok:
                    failed += 1
                print(f"  {'int8 job':10s} the published bundle alone, "
                      f"{len(caca)} CA-CA mean "
                      f"{sum(caca) / len(caca):.3f} A  {'ok ' if ok else 'NOT A CHAIN'}"
                      if caca else f"  {'int8 job':10s} no structure  FAILED")
                got["int8_job"] = {"caca_mean": round(sum(caca) / len(caca), 3) if caca else None}

                # 🔴 AND THE WRITER, AGAINST THE PAGE'S, ON THE SAME SAMPLE. Every arm above
                # scores a structure as GEOMETRY, and the native writer was wrong everywhere
                # geometry cannot look: a nucleotide's atoms named by the protein layout (P as
                # `N`, C1' as `CA`), a ligand atom written `UNK CA`, element C, `ATOM`, no
                # B-factor, occupancy, charge or entity from the input, an unindexed motif grafted
                # pair by pair where the page grafts all or nothing, and its never-written tokens
                # used as alignment anchors. `--dump-sample=` writes what `processSampleToAtoms`
                # takes; check-writer.mjs runs the page's writer on it and requires the SAME TEXT,
                # every preset in both formats, and the unindexed one converged as well, because
                # at six steps no mapping can be grafted and the predicted one goes unexercised.
                writer_bad = []
                writer_cases = [(pid, 6) for pid in (
                    "uncond_short", "ppi_pdl1", "motif_idx_1prw", "motif_unidx_1prw", "sm_sam",
                    "dna_7m5w", "yaml_5o45_binder", "partial_diffusion_1prw",
                    "inverse_folding_1prw", "custom_cath_monomer")] + [("motif_unidx_1prw", 60)]
                for pid, steps in writer_cases:
                    r = run(["node", HERE / "featurise" / "preset-spec.mjs", pid], cwd=REPO)
                    spec = json.loads(r.stdout)
                    spec["nsteps"] = steps
                    job = HERE / "featurise" / f"gate-writer-{pid}-{steps}.json"
                    job.write_text(json.dumps(spec))
                    dump = HERE / f"gate-writer-{pid}-{steps}"
                    for ext in ("cif", "pdb"):
                        outf = HERE / f"gate-writer-{pid}-{steps}.{ext}"
                        run([BINARY, f"--job={job}", f"--int8={bundle}",
                             f"--work-dir={dump}-work", f"--out={outf}", "--seed=0", "--fast",
                             f"--dump-sample={dump}"], cwd=REPO)
                        r = run(["node", HERE / "check-writer.mjs", f"--spec={job}", dump, outf],
                                cwd=REPO)
                        if r.returncode != 0:
                            writer_bad.append(f"{pid}@{steps}.{ext}")
                if writer_bad:
                    failed += 1
                print(f"  {'writer':10s} {len(writer_cases) * 2} files against the page's writer"
                      f"  {'ok ' if not writer_bad else 'DIFFERS: ' + ', '.join(writer_bad)}")
                got["writer"] = {"files": len(writer_cases) * 2, "differing": len(writer_bad)}

            # 🔴 THE ASSIGNMENT SOLVER, ON ITS OWN. It decides which designed token an
            # unindexed motif is grafted onto, so a wrong answer is a plausible structure with
            # the motif in the WRONG PLACE - not a crash and not a number anybody reads. It is
            # Jonker-Volgenant with potentials, which agrees with the textbook Hungarian on the
            # optimum and may differ where the optimum is not unique, so it is compared against
            # the reference's own rather than against a property like "the cost is minimal".
            # 🔴 AND THE MATRIX IS GENERATED ON BOTH SIDES, not passed between them: a double
            # written and re-read as text differs in the last place, and an assignment is a
            # DISCRETE answer that can flip on exactly that.
            probe = HERE / "featurise" / "assign-probe"
            if probe.exists():
                bad_assign, n_assign = 0, 0
                shapes = [(4, 4), (3, 7), (7, 3), (1, 5), (5, 1), (12, 12), (9, 14), (20, 11)]
                for sd in (1, 7, 42):
                    for rows, cols in shapes:
                        for mode in ("n", "x"):
                            for every in ("", "3", "5"):
                                # (not `args`: that is argparse's namespace in this scope, and
                                # shadowing it made the NEXT arm fail with
                                # `'list' object has no attribute 'weights'`)
                                probe_args = [str(sd), str(rows), str(cols), mode]
                                if every:
                                    probe_args.append(every)
                                a = run([probe, *probe_args], cwd=REPO).stdout.strip()
                                b = run(["node", HERE / "featurise" / "assign-probe.mjs",
                                         *probe_args], cwd=REPO).stdout.strip()
                                n_assign += 1
                                if a != b:
                                    bad_assign += 1
                if bad_assign:
                    failed += 1
                print(f"  {'assign':10s} {n_assign} matrices against the reference solver  "
                      f"{'ok ' if not bad_assign else f'DIFFER {bad_assign}'}")
                got["assign"] = {"cases": n_assign, "differing": bad_assign}

            # 🔴 AND THE KABSCH SUPERPOSITION, which puts a generated structure back into the
            # INPUT's frame. Without it a motif design is a correct structure in the wrong
            # place, which superposes on nothing and which no bond length, clash score or chain
            # check can see. Four of its conventions are the reference's own and not the
            # textbook's, and this sweep is what prices them: mutated one at a time it is 114 of
            # 240 for forcing the eigenvectors right-handed, 99 for U's third column being a
            # cross product, 90 for det(H)'s sign, and 29 - all of them n = 2, where the rotation
            # is not determined by the data - for the partial Jacobi update. A fifth, clamping a
            # negative eigenvalue, is 0 of 240; see the comment in src/superpose.h for why it
            # cannot be anything else.
            sprobe = HERE / "featurise" / "superpose-probe"
            if sprobe.exists():
                bad_sp, n_sp = 0, 0
                for sd in (1, 2, 7, 13, 42):
                    for pts in (1, 2, 3, 4, 8, 12, 40, 200):
                        for noise in ("0", "0.5", "4"):
                            # 🔴 AND A REFLECTED SET, because det(H) is otherwise always
                            # positive and the sign that stops Kabsch returning an IMPROPER
                            # rotation is never exercised.
                            for refl in ("0", "1"):
                                pa = [str(sd), str(pts), noise, refl]
                                a = run([sprobe, *pa], cwd=REPO).stdout.strip()
                                b = run(["node", HERE / "featurise" / "superpose-probe.mjs",
                                         *pa], cwd=REPO).stdout.strip()
                                n_sp += 1
                                if a != b:
                                    bad_sp += 1
                if bad_sp:
                    failed += 1
                print(f"  {'superpose':10s} {n_sp} point sets against the reference Kabsch  "
                      f"{'ok ' if not bad_sp else f'DIFFER {bad_sp}'}")
                got["superpose"] = {"cases": n_sp, "differing": bad_sp}

            # 🔴 AND A JOB THAT NAMES AN INPUT STRUCTURE MUST FOLD, not merely featurise. Every
            # arm above is a contig - a pure design job where most of the structure pipeline is a
            # no-op - and this one runs the whole of it in the fold binary: the mmCIF reader, the
            # fourteen transform stages and `convertToProteinaBatch` on real atoms. It is scored
            # on the CHAIN, because the reference's own trajectory for a structure job is what
            # the featuriser arm already compares byte for byte; what this adds is that the batch
            # those bytes describe folds into a protein.
            if (REPO / "model-rfd4p-int8" / "shards" / "shard_000.bin").exists():
                sfold = HERE / "featurise" / "gate-structure-fold.json"
                spdb = HERE / "gate-structure-fold.pdb"
                bundle = REPO / "model-rfd4p-int8"
                run([BINARY, f"--job={sfold}", f"--int8={bundle}",
                     f"--work-dir={HERE / 'gate-structure-fold'}", f"--out={spdb}", "--seed=0",
                     "--fast"], cwd=REPO)
                caca = chain_steps(spdb)
                ok = bool(caca) and 3.4 <= (sum(caca) / len(caca)) <= 4.1 and min(caca) > 2.8
                if not ok:
                    failed += 1
                print(f"  {'structure':10s} a job with an input structure folds, "
                      f"{len(caca)} CA-CA mean {sum(caca) / len(caca):.3f} A  "
                      f"{'ok ' if ok else 'NOT A CHAIN'}"
                      if caca else f"  {'structure':10s} no structure  FAILED")
                got["structure_job"] = {
                    "caca_mean": round(sum(caca) / len(caca), 3) if caca else None}

                # 🔴 AND THE OUTPUT KEEPS THE INPUT'S NUMBERING AND CHAIN LETTERS. A motif's
                # residues are deposited under their own numbers, and writing `i + 1` and chain
                # A renumbers them - the design then no longer superposes on the target it was
                # built against, which no RMSD or bond check would ever report. Compared per
                # TOKEN against the reference pipeline's own `res_id` and `chain_id`.
                # 🔴 AND AN UNINDEXED MOTIF, WHICH IS A DIFFERENT TOKEN SET ENTIRELY. Its 40
                # unindexed tokens are in-context conditioning the sampler never writes, so the
                # output must carry only the 72 INDEXED ones - the motif reaches it by being
                # grafted onto them. The writer emitted all 112, the extra 40 as `UNK`, and the
                # fold's own count did not report it because it counts tokens with a known
                # residue type and those have none.
                # 🔴 AND AT 60 STEPS, NOT 25. At 25 this fold's shortest CA-CA step is 2.76 A
                # and the band below rejects it - which is the step count, not the port: at 60
                # the same job is 3.66 to 3.88. The int8 arm above records the same trap in the
                # same words, and this arm was written at 25 anyway.
                ufold = HERE / "featurise" / "gate-unindexed.json"
                updb = HERE / "gate-unindexed.pdb"
                run([BINARY, f"--job={ufold}", f"--int8={bundle}",
                     f"--work-dir={HERE / 'gate-unindexed'}", f"--out={updb}", "--seed=0",
                     "--fast"], cwd=REPO)
                uca = chain_steps(updb)
                ures = len({ln[22:26] for ln in updb.read_text().splitlines()
                            if ln.startswith("ATOM")})
                unk = sum(1 for ln in updb.read_text().splitlines()
                          if ln.startswith("ATOM") and ln[17:20].strip() == "UNK")
                # 🔴 AND THE FOLD'S OWN COUNT MUST MATCH ITS OWN FILE. It counted every token
                # where it writes only the emitted ones, so it said "72 residues" over a file
                # holding 112 and "602 atoms" over 562 - and that disagreement is exactly what
                # made the residue bug look like it was not there.
                ustdout = run([BINARY, f"--job={ufold}", f"--int8={bundle}",
                               f"--work-dir={HERE / 'gate-unindexed'}", f"--out={updb}",
                               "--seed=0", "--fast"], cwd=REPO).stdout
                m = re.search(r"generated (\d+) residues, (\d+) atoms", ustdout)
                said = (int(m.group(1)), int(m.group(2))) if m else (-1, -1)
                wrote = (ures, sum(1 for ln in updb.read_text().splitlines()
                                   if ln.startswith("ATOM")))
                uok = (ures == 72 and unk == 0 and bool(uca) and min(uca) > 2.8
                       and said == wrote)
                if not uok:
                    failed += 1
                print(f"  {'unindexed':10s} the indexed tokens only, {ures} residues and "
                      f"{unk} UNK, the count matches the file  "
                      f"{'ok ' if uok else f'WRONG: said {said} wrote {wrote}'}")
                got["unindexed"] = {"residues": ures, "unk": unk, "count_matches": said == wrote}

                # 🔴 AND A SECOND STRUCTURE WITH TWO CHAINS, because `motif_idx_1prw`'s only
                # chain is A and `chainName(0)` is A too - the chain half of this arm cannot
                # fail on it. `ppi_pdl1` carries A and B, and the batch's `chain_iid` is
                # PERMUTED, so naming from it rather than from the input disagrees.
                naming_cases = [("motif_idx_1prw", spdb)]
                sfold2 = HERE / "featurise" / "gate-structure-fold2.json"
                spdb2 = HERE / "gate-structure-fold2.pdb"
                run([BINARY, f"--job={sfold2}", f"--int8={bundle}",
                     f"--work-dir={HERE / 'gate-structure-fold2'}", f"--out={spdb2}",
                     "--seed=0", "--fast"], cwd=REPO)
                naming_cases.append(("ppi_pdl1", spdb2))
                ref = run(["node", "--input-type=module", "-e", r'''
                    import { buildBatch, decodeTextOrZstd } from "./shared/rfd4p/pipeline/index.js";
                    import { getTokenStarts } from "./shared/rfd4p/pipeline/bonds.js";
                    import { readFileSync } from "node:fs";
                    const out = {};
                    for (const nm of ["motif_idx_1prw", "ppi_pdl1"]) {
                      const text = decodeTextOrZstd(new Uint8Array(
                        readFileSync(`web/examples/${nm}.cif.zst`)));
                      const { atoms } = buildBatch(text, { seed: 0 });
                      const starts = getTokenStarts(atoms, false);
                      out[nm] = starts.map((a) => [atoms.res_id[a], atoms.chain_id[a]]);
                    }
                    console.log(JSON.stringify(out));
                '''], cwd=REPO)
                wants = json.loads(ref.stdout.strip().splitlines()[-1])
                bad, total = 0, 0
                for nm, pdb_path in naming_cases:
                    want = wants[nm]
                    seen_tok, got_tok = [], []
                    for line in pdb_path.read_text().splitlines():
                        if not line.startswith("ATOM"):
                            continue
                        key = (int(line[22:26]), line[21])
                        if not seen_tok or seen_tok[-1] != key:
                            seen_tok.append(key)
                            got_tok.append([key[0], key[1]])
                    total += len(got_tok)
                    bad += (abs(len(got_tok) - len(want))
                            + sum(1 for g, w in zip(got_tok, want) if g != w))
                if bad:
                    failed += 1
                print(f"  {'out names':10s} the input's own numbering and chains, "
                      f"{total} tokens over {len(naming_cases)}  "
                      f"{'ok ' if not bad else f'DIFFER {bad}'}")
                got["out_naming"] = {"tokens": total, "differing": int(bad)}

            # 🔴 AND A TWO-CHAIN DESIGN MUST COME OUT AS TWO CHAINS, in both formats. The writer
            # used to put every atom in chain A - the chains were in the batch and simply not
            # carried to it - which for a binder or a complex is a structure no viewer and no
            # scorer can read correctly, and which no RMSD or bond check would ever notice.
            pair = HERE / "featurise" / "gate-pair.json"
            seen = {}
            for ext in ("pdb", "cif"):
                out_pdb = HERE / f"gate-pair.{ext}"
                run([BINARY, f"--job={pair}", f"--weights={args.weights}",
                     f"--work-dir={HERE / 'gate-pair'}", f"--out={out_pdb}", "--fast"], cwd=REPO)
                text = out_pdb.read_text() if out_pdb.exists() else ""
                if ext == "pdb":
                    seen[ext] = sorted({l[21] for l in text.splitlines()
                                        if l.startswith("ATOM")})
                else:
                    seen[ext] = sorted({l.split()[6] for l in text.splitlines()
                                        if l.startswith("ATOM ")})
            ok = seen.get("pdb") == ["A", "B"] and seen.get("cif") == ["A", "B"]
            if not ok:
                failed += 1
            print(f"  {'chains':10s} a two-chain job writes {seen.get('pdb')} as PDB and "
                  f"{seen.get('cif')} as mmCIF  {'ok ' if ok else 'WRONG'}")
            got["chains"] = {"pdb": seen.get("pdb"), "cif": seen.get("cif")}

            # 🔴 AND `--designs=N` MUST GIVE N DIFFERENT PROTEINS, not one protein N times. It used
            # to replay a single noise stream so a warm fold could be timed, and the flag now draws
            # a fresh one per design - which is only meaningful if the designs DIFFER and each is
            # still a chain. Both halves are asserted: a bug that returned the same structure N
            # times would otherwise pass every geometry check there is.
            campaign = HERE / "gate-campaign.pdb"
            for old in HERE.glob("gate-campaign*.pdb"):
                old.unlink()
            run([BINARY, f"--job={fold}", f"--weights={args.weights}",
                 f"--work-dir={HERE / 'gate-campaign'}", f"--out={campaign}", "--designs=3",
                 "--fast"], cwd=REPO)
            made = sorted(HERE.glob("gate-campaign*.pdb"))
            steps_ok = all(3.4 <= sum(c) / len(c) <= 4.1
                           for c in (chain_steps(f) for f in made) if c)
            spread = []
            if len(made) > 1:
                first = chain_steps(made[0])
                for f in made[1:]:
                    a = (made[0]).read_text()
                    b = f.read_text()
                    spread.append(a != b)
            ok = len(made) == 3 and steps_ok and all(spread)
            if not ok:
                failed += 1
            print(f"  {'designs':10s} --designs=3 wrote {len(made)} structures, "
                  f"{'all different and all chains' if ok else 'NOT 3 DIFFERENT CHAINS'}  "
                  f"{'ok ' if ok else 'WRONG'}")
            got["designs"] = {"written": len(made)}

            # 🔴 AND THE mmCIF READER, which is the structure path's first half and is gated
            # BEFORE its second half exists. What it claims is what comes straight from the file;
            # the chemistry the reference derives from that - bonds, leaving atoms, chain types,
            # the id and entity annotations - is still the exporter's, and the checker compares
            # only what the reader claims rather than letting omission look like agreement.
            r = run(["node", HERE / "featurise" / "check-cif.mjs"], cwd=REPO)
            m = re.search(r"(\d+) structures, (\d+) differing", r.stdout)
            if not m or m.group(2) != "0":
                failed += 1
                if not m:
                    print(r.stdout[-1000:], r.stderr[-600:])
            print(f"  {'cif':10s} {m.group(1) if m else '?'} shipped structures, "
                  f"{m.group(2) if m else '?'} differing  "
                  f"{'ok ' if m and m.group(2) == '0' else 'DIFFERS'}")
            got["cif"] = {"structures": int(m.group(1)) if m else 0,
                          "differing": int(m.group(2)) if m else -1}

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
