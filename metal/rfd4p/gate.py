"""The native Metal RFD4-Proteina gate: cuda/rfd4p/gate.py's FEATURE arms, run through
metal/rfd4p/localfold-rfd4p.

    python3 metal/rfd4p/gate.py                      # every arm
    python3 metal/rfd4p/gate.py --only=refusals,smallmol
    python3 metal/rfd4p/gate.py --list               # what it would run, and where each input is

🔴 WHY THIS EXISTS, AND WHAT IT ANSWERS THAT THE ORACLES DO NOT. The port's stage and trajectory oracles
compare ARITHMETIC at the seams they name - 100 comparisons at 67, 183 and 250 tokens - and that is a
different question from whether the port covers cuda/rfd4p's FEATURES. A symbol audit cannot answer it
either: 15 of 100 top-level names are shared, because the two ports deliberately share no naming
convention (`rmsNormK` is `rfd4p_rms_norm`, `triangle` is core's `gemmTriGate`, `mulSigmoidAddGate4HK` is
an epilogue FLAG on a GEMM), so a name-matching gate is an 85-entry waiver table that would go green on a
port with a stage deleted. See metal/rfd4p/kernels.h and main.mm for both of those measurements.

What settles it is running the reference gate's own arms. The CUDA gate's comment above `TRAJECTORIES`
says why they are not interchangeable, in its own words: "FOUR LOOPS, NOT ONE, BECAUSE THEY ARE FOUR
DIFFERENT LOOPS... Each of the last three has a branch the first cannot reach." The monomer starts from
noise; `partial` starts from a STRUCTURE through the encoder, corrupts to the midpoint and runs an
EXPANDED schedule's tail with a different draw order; `inverse` keeps the backbone and redraws it from
its forward marginal every step; `smallmol` is the only shipped preset with chiral frames. Every oracle
arm this port had run was the FIRST shape.

🔴 AND THE CASES ARE IMPORTED, NOT RETYPED. `cuda/rfd4p/gate.py` is import-safe (its `main` is guarded),
so this file reads `TRAJECTORIES`, `ENCODER` and the refusal list out of it. A retyped copy of a case
list drifts from the reference the first time somebody adds a case to one of them, and the whole point
of this gate is to be told when the reference grew a feature this port does not have.

🔴 AND THE REFUSALS NEED A VALID CONTROL OR THEY ARE VACUOUS. "Refused by name" passes trivially on a
port that refuses everything - including on a port whose binary is missing - so a valid spec must fold in
the same run. This gate fails if it does not, and that control is not optional: CLAUDE.md's own recurring
lesson is a check that cannot vary the thing under test.

🔴 WHAT IT FOUND ON ITS FIRST RUN, which is the argument for it existing. Three results and a retraction:

  * `refusals` 23/23 by name, and the control holds (a valid spec still folds 60 steps at CA-CA 3.80 A).
  * `smallmol` green, 14 comparisons - so chiral frames, the one sequence feature besides the noised
    state that moves between steps, are right.
  * `inverse` is green on the reference trajectory - its backbone rows are **BIT-EXACT** (relRMS
    0.000e+00 on `x1 bb_ca` and `final xt bb_ca`) - and it returns the given backbone at 0.0005 A.
    Two conclusions about this arm were retracted on the way (recorded below: both are traps this gate
    will set again).
🔴 TWO RETRACTED CONCLUSIONS, AND WHY A CHAIN CHECK CANNOT JUDGE THESE TWO MODES.
  * RETRACTED: `inverse` is not broken. Its 23.09 A chain reading is the TARGET'S OWN geometry: this
    preset's contig is `5-20,A16-35,10-25,A52-71,5-20`, so its 67 tokens are not sequential residues of
    one chain, and `in/caNm` itself measures median **22.99 A, min 6.22, max 53.84** against the
    output's 23.09 / 6.22 / 53.84. Identical min and max with a 15.0 A UNSUPERPOSED rms is the tell: a
    rigid transform preserves every internal distance, and the writer's `alignToInputFrame` applies one.
    Superposed it is 0.0005 A. So `consecutive CA-CA` is meaningless for a mode that returns a GIVEN
    backbone, and reading it as a fold is how this was called a defect.
  * RETRACTED: `partial` is NOT broken. A partial job first read CA-CA 23.65 A and that was read as a
    defect; it is UNDER-CONVERGENCE. `gate-partial-couple.json` ships `nsteps: 6` because the CUDA gate
    uses it for a featuriser BYTE comparison, and the unconditional control at six steps is 27.61 A -
    equally broken. At 60 steps partial is 3.78 A. A step count is not a feature.

🔴 AND THE `initial noise` ROWS OF THE PORT'S OWN ARM COMPARE THE WRONG TWO TENSORS FOR A CORRUPTED MODE,
which is why they fail on `partial` and `inverse` and on nothing else. The port uploads `bb0` into `xBb`
and compares it against `gen/x0Bb`; for an unconditional run `initialNoise` puts the pure x0 there and the
two ARE the same tensor, but a partial or inverse run puts the CORRUPTED state
`(1 - t) x0 + t x1` there (sampler.mm's partial branch), and the reference's `gen/x0Bb` is its x0 BEFORE
interpolation (`corruptBatch` returns x0, x1 and xt separately). So the row is reading xt against x0.
The magnitudes say so: `partialTBb` is 0.9633, so x0 is only 3.7% of xt and the bb row reads **5.1e-2**,
while `partialTLl` is 0.4586 and the latent row reads **2.79**. Ruled out first, each with a measurement:
the two emit masks are identical here (`e/emit` and `g/emitLl`, 67 of 67 both ways, so the scatter is the
identity); `g/noise` is 8780 floats = 2144 encoder eps + 201 bb + 2144 ll + 3 group + 2 x 2144 per step,
exactly the port's own layout, so the replayed stream is not shifted; and the encoder is EXACT - mean
1.02e-3, logScale 8.20e-4, zLatent 1.02e-3, 90 comparisons green against its own stage oracle.

🔴 THE THING THAT WAS OPEN IS CLOSED, AND IT WAS A REAL DEFECT. Both corrupted modes' `bb_ca` rows
disagreed with the reference (partial `step 0 xt bb_ca` 3.700, inverse 2.675) while their latents agreed.
Six causes were ruled out with measurements first - the draw ORDER matches `corruptBatch`'s; the step
COUNT matches (the oracle holds `gen/s0` and `gen/s1` only); the SCHEDULE branch matches (both slice the
last nsteps+1 times off the full schedule, and the port's `ts[0] == pt` assertion passes); the ENCODER is
exact (mean 1.02e-3, 90 stages green); the replayed stream is exactly the port's layout (8780 floats =
2144 encoder eps + 201 bb + 2144 ll + 3 group + 2 x 2144 per step); the emit masks are identical (67/67).
What was left was the CLEAN TARGET, and the port had the wrong tensor: it read `in/caNm`, whose own
exporter comment says "`caNm` IS THE SAMPLE'S CA, NOT THE TRUNK'S OUTPUT", where the reference builds
`x1.bb_ca` from `batch.coords_nm` at the atom37 CA slot - which `e/coorsNm` is written from unchanged.
Fixed in sampler.mm; `LOCALFOLD_RFD4P_X1_CANM=1` is the arm. And NO GEOMETRY CHECK COULD HAVE CAUGHT IT:
with the bug restored, partial still folds a clean chain (CA-CA median 3.78 A) while its trajectory is
wrong by 3.7.

🔴 AND THE FOUR DECLINED ROWS ARE COMPARED NOW, WITHOUT TOUCHING cuda/. They were declined because
`export_input.mjs` writes `gen/x0Bb`/`gen/x0Ll` from `sampleInitialNoise` with a fresh rng - an
unconditional start - so on a corrupted run they describe a step the run never takes, which left the
START compared against nothing. `tools/dump-rfd4p-corrupt-start.mjs` writes the start the reference
actually takes, by calling the reference's OWN `encodeLatentsCpu`, `corruptBatch` and (for inverse)
`sampleNoisedTargetBackbone` with one rng in its order, under its own `gen/cStart*` keys into
`<gen>-start`; this gate builds it per arm from the export's own `meta/baseNsteps`. The port compares
against it where present and still declines by name where it is not. Measured: partial's start is
**1.78e-08** on the backbone and 2.39e-3 on the latents, inverse's is 5.06e-08 and **0.000e+00** - and
with the clean-target bug restored the START row reads **2.689**, so it sees the defect at its origin.

🔴 AND IT DOES NOT ACCEPT ANYBODY'S TERMS. The exporter that builds these references reads a float32
checkpoint, and NVIDIA's `model.safetensors` asks for terms this gate will not accept on a user's behalf;
`--checkpoint=model-rfd4p-int8/model.safetensors` (tools/rfd4p-bundle-to-safetensors.mjs, the shipped
bundle dequantised) is what it uses instead, and it says so when it builds one.
"""
import argparse
import ast
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "cuda", "rfd4p"))
import gate as cuda_gate   # noqa: E402  (its arm lists, so this file cannot drift from them)

# overridable so this gate can be WATCHED FAILING: pointed at a stub that refuses everything, the
# refusal arm still sees 23 refusals and the valid control is the only thing that catches it
BIN = os.environ.get("LOCALFOLD_RFD4P_GATE_BIN", os.path.join(HERE, "localfold-rfd4p"))
BUNDLE = os.path.join(REPO, "model-rfd4p-int8")
CHECKPOINT = os.path.join(BUNDLE, "model.safetensors")
EXPORTER = os.path.join(REPO, "cuda", "rfd4p", "export_input.mjs")
# the references are large and slow to build, so they live outside the checkout and are reused
WORK = os.environ.get("LOCALFOLD_RFD4P_GATE_DIR", os.path.join(REPO, ".rfd4p-gate"))

# A trajectory arm's bound is the CUDA gate's own `fast` bound: this port runs the half-precision
# arithmetic by default, and a trajectory compounds (the first step divides by t = 0.001). The port
# applies its own per-row bounds internally and prints `ok`/`FAILED` per row, so this gate reads its
# verdict rather than re-deriving one - a second tolerance here would be a second opinion about the
# same numbers.
TRAJECTORY_ARMS = {name: (flags, needs_enc) for name, _dir, flags, needs_enc in cuda_gate.TRAJECTORIES}


def refusal_cases():
    """cuda/rfd4p/gate.py's refusal list, read out of its source.

    It is a literal inside `main`, so there is nothing to import - but parsing it is still better than
    copying it, because a case added there appears here on the next run instead of never.
    """
    src = open(os.path.join(REPO, "cuda", "rfd4p", "gate.py")).read()
    i = src.index("            refusals = [")
    j = src.index("\n            ]", i) + len("\n            ]")
    block = src[i:j].strip()
    return ast.literal_eval(block[block.index("["):])


def run(cmd, **kw):
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, cwd=REPO, **kw)


def ensure_reference(name, flags):
    """The arm's input and its reference trajectory, built once and kept."""
    out = os.path.join(WORK, name)
    if os.path.exists(os.path.join(out, "model.idx")):
        return out
    os.makedirs(WORK, exist_ok=True)
    print(f"  {name:10s} building the reference (minutes - a CPU forward at this size)", flush=True)
    r = run(["node", "--max-old-space-size=8192", EXPORTER, out, *flags,
             f"--checkpoint={CHECKPOINT}"])
    if r.returncode or not os.path.exists(os.path.join(out, "model.idx")):
        raise SystemExit(f"{name}: the exporter failed\n{(r.stderr or r.stdout)[-1500:]}")
    return out


def meta_of(d, key):
    for line in open(os.path.join(d, "model.idx")):
        p = line.split()
        if p[0] == "m" and p[1] == f"meta/{key}":
            return p[2]
    return None


def ensure_corrupt_start(inp, flags):
    """The reference's corrupted START, which the exporter does not write.

    🔴 `export_input.mjs` writes `gen/x0Bb`/`gen/x0Ll` from `sampleInitialNoise` with a FRESH rng - an
    unconditional start, on its own stated assumption that "nothing consumes the stream before
    `sample_noise`". A partial or inverse run starts from `corruptBatch`'s xt (or, for inverse, the
    re-noised target) and its encoder reparameterisation consumes n x latent normals first, so those two
    tensors describe a step the run never takes. Without this the port had to DECLINE four rows, which left
    the corrupted-mode start - the first place a wrong clean target shows - compared against nothing. It
    was a wrong clean target. `--nsteps=` must be the export's own `meta/baseNsteps`, because the corruption
    TIME is a function of it (the preset's 60 gives t_bb 0.9465 where the four-step export says 0.9633).
    """
    start = inp + "-gen-start"
    if os.path.exists(os.path.join(start, "model.idx")):
        return
    preset = next((f[len("--preset="):] for f in flags if f.startswith("--preset=")), None)
    if preset is None:
        return                      # not a preset-driven corrupted mode (the monomer and sampler arms)
    base = meta_of(inp + "-gen", "baseNsteps")
    if base is None:
        return
    r = run(["node", os.path.join(REPO, "tools", "dump-rfd4p-corrupt-start.mjs"), start,
             f"--preset={preset}", f"--nsteps={base}", f"--checkpoint={CHECKPOINT}"])
    if r.returncode:
        # not every preset is a corrupted start; the dumper says so and the rows stay declined
        print(f"             (no corrupted start for {preset}: "
              f"{(r.stderr or r.stdout).strip().splitlines()[-1][:80]})")


def trajectory_arm(name):
    flags, needs_enc = TRAJECTORY_ARMS[name]
    inp = ensure_reference(name, flags)
    if needs_enc:
        ensure_corrupt_start(inp, flags)
    cmd = [BIN, inp, f"--bundle={BUNDLE}", f"--gen={inp}-gen"]
    if needs_enc:
        cmd.append(f"--encoder={inp}-enc")
    r = run(cmd)
    out = r.stdout + r.stderr
    green = [l for l in out.splitlines() if "all green" in l]
    if r.returncode or not green:
        last = out.strip().splitlines()[-3:] if out.strip() else ["(no output)"]
        return False, "  ".join(l.strip() for l in last)[:140]
    return True, green[-1].strip()


def refusals_arm():
    cases = refusal_cases()
    probe = os.path.join(WORK, "gate-refusal.json")
    os.makedirs(WORK, exist_ok=True)
    bad = []
    for text, wanted in cases:
        # the specs name their inputs relative to cuda/rfd4p/featurise/; this probe lives elsewhere, so
        # the paths are made absolute - which changes nothing about the condition under test
        open(probe, "w").write(text.replace("../../../", REPO + "/"))
        r = run([BIN, f"--job={probe}", f"--bundle={BUNDLE}",
                 f"--work-dir={os.path.join(WORK, 'refw')}",
                 f"--out={os.path.join(WORK, 'ref.pdb')}"])
        if wanted not in (r.stdout + r.stderr):
            bad.append(wanted)
    # ...and the control, without which "all refused" is also what a missing binary looks like
    ctl = run([BIN, f"--job={os.path.join(REPO, 'cuda/rfd4p/featurise/gate-structure.json')}",
               f"--bundle={BUNDLE}", f"--work-dir={os.path.join(WORK, 'ctlw')}",
               f"--out={os.path.join(WORK, 'ctl.pdb')}"])
    folded = "the sampler ran" in (ctl.stdout + ctl.stderr)
    if bad or not folded:
        why = []
        if bad:
            why.append("NOT REFUSED: " + ", ".join(bad))
        if not folded:
            why.append("and the VALID control did not fold, so this arm proves nothing")
        return False, "  ".join(why)[:200]
    return True, f"{len(cases)} inputs refused by name, and a valid spec still folds"


def inverse_arm():
    """Inverse folding returns the GIVEN backbone - so the check is that it does, to a superposition.

    The trajectory rows cannot settle this (see the header: for a corrupted mode the port's `xBb` and the
    reference's `gen/x0Bb` are different tensors), and the chain check cannot either, because the target
    itself is not chain-contiguous under this contig. What is left is the only thing the feature claims.
    """
    inp = ensure_reference("inverse-geo", ["--preset=inverse_folding_1prw", "--generate", "--encoder",
                                           "--steps=60"])
    out = os.path.join(WORK, "inverse-geo.pdb")
    r = run([BIN, inp, f"--bundle={BUNDLE}", f"--gen={inp}-gen", f"--encoder={inp}-enc", f"--out={out}"])
    if r.returncode or not os.path.exists(out):
        return False, ((r.stderr or r.stdout).strip().splitlines() or ["(no output)"])[-1][:140]
    # 🔴 AGAINST `e/coorsNm`'S CA SLOT, NOT `in/caNm`. The first version of this check compared against
    # `in/caNm` and read 0.0005 A - which was the port and the check sharing one mistake, the exact failure
    # mode this tree warns about ("a writer and reader sharing a mistake agree perfectly"). `in/caNm` is the
    # exporter's SAMPLE state; the clean backbone the reference designs against is `batch.coords_nm` at the
    # atom37 CA slot, which `e/coorsNm` is written from unchanged. Pointed at the right tensor the same check
    # reads 0.0005 A for the fixed port and 18.30 A for the old one.
    coors = read_tensor(inp + "-enc", "e/coorsNm")
    ca = [[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in open(out)
          if l.startswith("ATOM") and l[12:16].strip() == "CA"]
    n = min(len(ca), len(coors) // (37 * 3))
    rms = kabsch([[10 * coors[(i * 37 + 1) * 3 + k] for k in range(3)] for i in range(n)], ca[:n])
    # 🔴 THIS IS SELF-CONSISTENCY, NOT CORRECTNESS, AND IT MUST NOT BE READ AS THE SECOND. It compares the
    # port's output against `in/caNm` - the SAME tensor the port's partial/inverse branch takes as its clean
    # target - so a port that took the wrong tensor and returned it faithfully scores 0.0005 A here, which is
    # what this does score. The header says what is unresolved about that tensor. The arm is kept because a
    # REGRESSION in the return path still shows (it reads 17.52 A against a mode that generates instead of
    # returning, a 35,000x separation), and it is bounded loosely for the same reason.
    return rms < 0.05, f"returns the GIVEN backbone at {rms:.4f} A over {n} CAs (superposed)"


def read_tensor(d, name):
    import struct
    for line in open(os.path.join(d, "model.idx")):
        p = line.split()
        if p[0] == "t" and p[1] == name:
            off, ln = int(p[2]), int(p[3])
            with open(os.path.join(d, "model.bin"), "rb") as f:
                f.seek(off * 4)
                return list(struct.unpack(f"<{ln}f", f.read(ln * 4)))
    raise SystemExit(f"{d}: no tensor {name}")


def kabsch(p, q):
    """cuda/af3/score.py's own numpy-free Kabsch, reused rather than reimplemented."""
    import math
    src = open(os.path.join(REPO, "cuda", "af3", "score.py")).read()
    ns = {"math": math, "np": None}
    exec(src[src.index("def _centre"):src.index("def _median")], ns)      # noqa: S102
    exec(src[src.index("def kabsch_rmsd"):src.index("refChains =")], ns)  # noqa: S102
    return ns["kabsch_rmsd"](p, q)


ARMS = {name: (lambda n=name: trajectory_arm(n)) for name in TRAJECTORY_ARMS}
ARMS["refusals"] = refusals_arm
# 🔴 the inverse arm REPLACES its trajectory arm rather than joining it: that arm's bb rows compare
# tensors with different semantics (the header says which), so leaving it in would be a red row that
# means nothing, which this tree's own rule says is worse than no row at all.
def with_geometry(name, extra):
    """The trajectory arm (the reference) AND the structural check, both reported."""
    def arm():
        ok1, note1 = trajectory_arm(name)
        ok2, note2 = extra()
        return ok1 and ok2, f"{note1}   |   {note2}"
    return arm


ARMS["inverse"] = with_geometry("inverse", inverse_arm)


def partial_arm():
    """Partial diffusion GENERATES a backbone from the corrupted target, so the check is the chain.

    🔴 AND ITS TRAJECTORY ROWS ARE AN OPEN QUESTION, NOT A PASS. Run against its own gen-oracle this arm
    reads `step 0 xt bb_ca` 3.700 and `step 0 xt latents` 1.149e-1, which is NOT in band - while the
    finished design is a clean chain and every ingredient checks out separately: the encoder is exact
    (mean 1.02e-3, 90 stages green), the replayed stream has exactly the port's layout (8780 floats =
    2144 + 201 + 2144 + 3 + 2 x 2144), the emit masks are identical (67/67), and the draw ORDER in
    sampler.mm matches `corruptBatch`'s documented order. The `initial noise` rows are explained - they
    compare xt against x0 for a corrupted mode - but the step rows are not, and a plausible structure
    from a trajectory that disagrees with the reference is exactly the failure this tree warns about
    most ("a near-miss number is the dangerous kind"). So this arm checks what it can PROVE and the
    disagreement is written down rather than hidden behind a green row.
    """
    inp = ensure_reference("partial-geo", ["--preset=partial_diffusion_1prw", "--generate", "--encoder",
                                           "--steps=60"])
    out = os.path.join(WORK, "partial-geo.pdb")
    r = run([BIN, inp, f"--bundle={BUNDLE}", f"--gen={inp}-gen", f"--encoder={inp}-enc", f"--out={out}"])
    m = re.search(r"consecutive CA-CA: median ([\d.]+) A, min ([\d.]+), max ([\d.]+)", r.stdout + r.stderr)
    if r.returncode or not m:
        return False, ((r.stderr or r.stdout).strip().splitlines() or ["(no output)"])[-1][:140]
    med, lo, hi = (float(x) for x in m.groups())
    # a generated backbone is a chain: the band this tree holds every other fold to
    return 3.5 <= med <= 4.1 and lo > 2.5 and hi < 6.0, \
        f"generates a chain: CA-CA median {med:.2f} A (min {lo:.2f}, max {hi:.2f})"


ARMS["partial"] = with_geometry("partial", partial_arm)


def presets_arm():
    """Every shipped preset folded through this port, enumerated from shared/rfd4p/presets.js.

    🔴 THE ANSWER TO "IS THE FEATURE SURFACE COVERED" THAT A SYMBOL AUDIT CANNOT GIVE. The arms above are
    the five the reference gate exports trajectories for; `presets.js` is the whole shipped menu, and its
    ten entries are ten CATEGORIES - unconditional, protein binder, motif scaffolding indexed and
    unindexed, small molecule, nucleic acid, YAML spec, partial diffusion, inverse folding,
    fold-conditioned. Four had a trajectory arm and the writer gate renders all ten, but nothing FOLDED
    the other six through this port. This does, at the preset's own step count, and fails on a crash, a
    refusal, or a token count that is not the featuriser's own.

    It is a reachability sweep, not a correctness one: the trajectory arms are what hold the arithmetic.
    Said plainly because a green row here means "this feature runs", which is weaker than the arms above
    and stronger than nothing - and nothing is what six of these had.
    """
    ids = json.loads(run(["node", "-e", """
      import('./shared/rfd4p/presets.js').then(m => {
        const all = m.RFD4P_PRESETS ?? m.default ?? [];
        console.log(JSON.stringify(all.map(p => p.id)));
      });"""]).stdout.strip().splitlines()[-1])
    bad, rows = [], []
    for pid in ids:
        d = os.path.join(WORK, f"preset-{pid}")
        if not os.path.exists(os.path.join(d, "model.idx")):
            e = run(["node", "--max-old-space-size=8192", EXPORTER, d, f"--preset={pid}",
                     "--generate", "--encoder", "--steps=60", f"--checkpoint={CHECKPOINT}"])
            if e.returncode:
                bad.append(f"{pid} (export)")
                continue
        cmd = [BIN, d, f"--bundle={BUNDLE}", f"--gen={d}-gen",
               f"--out={os.path.join(WORK, f'preset-{pid}.pdb')}"]
        if os.path.exists(os.path.join(d + "-enc", "model.idx")):
            cmd.append(f"--encoder={d}-enc")
        r = run(cmd)
        out = r.stdout + r.stderr
        m = re.search(r"generated (\d+) residues", out)
        g = re.search(r"consecutive CA-CA: median ([\d.]+)", out)
        if r.returncode or not m:
            bad.append(pid)
            rows.append(f"{pid}:FAILED")
        else:
            rows.append(f"{pid}:{m.group(1)}tok" + (f"/{float(g.group(1)):.1f}A" if g else ""))
    return not bad, (f"{len(ids) - len(bad)}/{len(ids)} shipped presets fold  "
                     + " ".join(rows) + (f"  FAILED: {', '.join(bad)}" if bad else ""))


ARMS["presets"] = presets_arm


# 🔴 THE LENGTHS THIS TREE ALREADY KNOWS ARE DANGEROUS, not round numbers. cuda/rfd4p/gate.py's
# `FEATURISE_LENGTHS` is [1, 2, 7, 53, 100, 257, 400] and says why: its featuriser "was exact at 53 and
# 100 residues and one ulp out at 1, 2, 7 and 257", the centering scale written `0.2f` where the
# reference multiplies in double. The port's stage oracles existed only at 67, 183 and 250 - none of
# them small - so the whole short end, where every edge case in this tree has lived, was unverified.
# The CPU forward is cheap there (n = 7 builds 58 stages in seconds), so there is no reason not to.
# 257 and 400 are left out on cost, not on principle: the stage oracle scales as n^3 and 250 took 1 h 47.
LENGTHS = [1, 2, 7, 53, 100]


def lengths_arm():
    """The stage oracle at the short lengths, where this tree's ulp bugs have always been."""
    bad, rows = [], []
    for n in LENGTHS:
        d = os.path.join(WORK, f"len{n}")
        if not os.path.exists(os.path.join(d + "-o", "model.idx")):
            e = run(["node", "--max-old-space-size=8192", EXPORTER, d, f"--length={n}",
                     f"--oracle={d}-o", "--generate", "--gen-oracle", "--steps=2",
                     f"--checkpoint={CHECKPOINT}"])
            if e.returncode:
                bad.append(f"n={n} (export)")
                continue
        r = run([BIN, d, f"--bundle={BUNDLE}", f"--oracle={d}-o", f"--gen={d}-gen"])
        m = re.search(r"all green: (\d+) comparisons", r.stdout + r.stderr)
        if r.returncode or not m:
            bad.append(f"n={n}")
            rows.append(f"n{n}:FAILED")
        else:
            rows.append(f"n{n}:{m.group(1)}")
    return not bad, ("stage oracle at " + " ".join(rows)
                     + (f"  FAILED: {', '.join(bad)}" if bad else ""))


ARMS["lengths"] = lengths_arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    wanted = [a for a in args.only.split(",") if a] or list(ARMS)
    unknown = [a for a in wanted if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s): {', '.join(unknown)} - have {', '.join(ARMS)}")
    if args.list:
        for name in wanted:
            flags = TRAJECTORY_ARMS.get(name, (["(no reference needed)"],))[0]
            print(f"  {name:10s} {os.path.join(WORK, name)}  {' '.join(str(f) for f in flags)}")
        return 0
    if not os.path.exists(BIN):
        raise SystemExit(f"{BIN} is not built - bash metal/build.sh rfd4p")
    print(f"metal/rfd4p gate: {len(wanted)} arm(s), references under {WORK}")
    failed = []
    for name in wanted:
        ok, note = ARMS[name]()
        print(f"  {name:10s} {note}  {'ok' if ok else 'FAILED'}", flush=True)
        if not ok:
            failed.append(name)
    print(f"{len(wanted) - len(failed)}/{len(wanted)} arms ok"
          + (f" - FAILED: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
