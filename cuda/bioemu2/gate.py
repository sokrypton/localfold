"""The native BioEmu2's regression gate, two arms:

  oracles  - every tools/fixtures/bioemu2 job that has a dump in oracle-dumps/bioemu2-<name>.json
             (tools/oracle/dump_bioemu2.py: af3-any-model's own port, stage by stage) folded with --oracle on the
             float32 blob, in float32 and in --fast: every stage's relRMS (s_inputs, z_init, each trunk pass, the
             distogram, the denoiser's static context, every denoise level and the propagation arm) held to
             1e-5 in float32 (they are 1e-7..5e-6) and 5e-3 in --fast (2e-3..3e-3). A dump that is missing is
             skipped and says so: dumps are gitignored, made by hand.
  folds    - real targets through cuda/bioemu2/fold (--fast, the int8 blob, 200 steps, 5 samples, seed 0), held
             to gate-baseline.json: 6MRR's CA RMSD to its crystal (each sample's, the mean and the best), the
             bond geometry of a ligand, an atomised phosphoserine and two nucleic chains (cuda/af3/bonds.mjs: rms
             against the CCD's ideals), the equilibrium mode and a propagation chain from a sample.

    python3 cuda/bioemu2/gate.py                 # both arms against the baseline
    python3 cuda/bioemu2/gate.py --write         # re-record the fold baseline
    python3 cuda/bioemu2/gate.py --only=folds    # one arm (or a case name)
    python3 cuda/bioemu2/gate.py --weights-f32=<dir> --weights=<dir>   # (the oracle's float32 blob; the folds' int8)

A baseline is this machine's: a fold does not reproduce to the digit on another GPU.
"""
import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FIX = os.path.join(REPO, "tools", "fixtures")
JOBS = os.path.join(FIX, "bioemu2")
DUMPS = os.path.join(REPO, "oracle-dumps")
BASELINE = os.path.join(HERE, "gate-baseline.json")
TMP = os.environ.get("GATE_TMP", "/tmp/bioemu2-gate")
ORACLE_BOUND = {"f32": 1e-5, "fast": 5e-3}
# the denoiser's VJP (src/vjp.cuh, edm_fkc's reverse pass) against jax.grad: tools/check-bioemu2-denoiser.js's bars - the
# network part of d/dx 1e-3, d/dsigma 5e-3 - because the reference's own float32 backward is ~1.4e-4 off its float64 and
# both sums cancel (docs/BIOEMU2.md)
VJP_BOUND = {"d/dx": 1e-3, "d/dsigma": 5e-3}
# a fold's RMSD may move 0.15 A (a BioEmu2 sample is a sample: five of them at one seed span ~0.5 A), its mean 0.1,
# a bond rms 0.01
TOL_RMSD, TOL_MEAN, TOL_BOND = 0.15, 0.10, 0.01


def option(name, fallback):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), fallback)


def oracle_cases():
    # (name, job, extra export flags, extra binary flags)
    out = []
    for name, job, export, flags in [
        ("6mrr", "6mrr", [], []),
        ("6mrr-gol-sep", "6mrr-gol-sep", [], []),
        ("rna-dna", "rna-dna", [], []),
        ("smiles", "smiles", ["--smiles-from=" + os.path.join(DUMPS, "bioemu2-smiles.json")], []),
        ("6mrr-eq-contacts", "6mrr", ["--mode=equilibrium", "--contacts=" + os.path.join(JOBS, "contacts.json")], []),
        ("heteromer", "heteromer", [], ["--passes=1"]),        # (its dump runs one trunk pass)
    ]:
        out.append((name, os.path.join(JOBS, job + ".json"), os.path.join(DUMPS, f"bioemu2-{name}.json"), export, flags))
    return out


def run_oracle(name, job, dump, export, flags, weights, fast):
    data, odir = os.path.join(TMP, "in-" + name), os.path.join(TMP, "oracle-" + name)
    if not os.path.exists(os.path.join(odir, "model.idx")):
        subprocess.run(["node", os.path.join(HERE, "oracle_input.mjs"), dump, odir], check=True, capture_output=True)
    if not os.path.exists(os.path.join(data, "model.idx")):
        subprocess.run(["node", os.path.join(HERE, "export_input.mjs"), data, "--job=" + job, *export], check=True,
                       capture_output=True)
    cmd = [os.path.join(HERE, "localfold-bioemu2"), data, f"--weights={weights}", f"--oracle={odir}", "--samples=0",
           *flags] + (["--fast"] if fast else [])
    p = subprocess.run(cmd, capture_output=True, text=True)
    text = p.stdout + p.stderr
    got = {m.group(1).strip(): float(m.group(2)) for m in re.finditer(r"^\s+(\S.*?)\s+relRMS ([\d.e+-]+|nan)", text, re.M)}
    if p.returncode or not got:
        return None, text.strip().splitlines()[-1] if text.strip() else f"exit {p.returncode}"
    return got, None


def ca(pdb, chain=None):
    out = []
    for line in open(pdb):
        if line.startswith(("ATOM", "HETATM")) and line[12:16] == " CA " and (chain is None or line[21] == chain):
            out.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return out


def kabsch_rmsd(p, q):
    import numpy as np
    p, q = np.asarray(p, float), np.asarray(q, float)
    p, q = p - p.mean(0), q - q.mean(0)
    u, _, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt(((p @ r - q) ** 2).sum(1).mean()))


def chain_ok(pdb):
    """Every consecutive CA within 3.4-4.2 A, the chain check every port's gate makes (cis-prolines aside)."""
    import numpy as np
    x = np.asarray(ca(pdb))
    d = np.linalg.norm(x[1:] - x[:-1], axis=1)
    return float(np.median(d)), bool(np.median(d) > 3.6 and np.median(d) < 4.0)


def fold(name, inputs, flags=()):
    os.makedirs(TMP, exist_ok=True)
    pdb = os.path.join(TMP, name + ".pdb")
    for f in glob.glob(os.path.join(TMP, name + "*.pdb")):
        os.remove(f)
    p = subprocess.run([os.path.join(HERE, "fold"), pdb, *inputs, "--", "--samples=5", "--steps=200", "--seed=0", *flags],
                       capture_output=True, text=True)
    text = p.stdout + p.stderr
    if p.returncode or not os.path.exists(pdb):
        return None, text, None
    samples = [pdb] + [os.path.join(TMP, f"{name}_sample-{k}.pdb") for k in range(1, 100)
                       if os.path.exists(os.path.join(TMP, f"{name}_sample-{k}.pdb"))]
    return samples, text, re.search(r"sampler ([\d.]+) ms", text)


def bonds(pdb, codes):
    b = subprocess.run(["node", os.path.join(REPO, "cuda", "af3", "bonds.mjs"), pdb, codes], capture_output=True, text=True).stdout
    return {cls: float(m.group(1)) for cls in ("ligand", "nucleic") for m in [re.search(cls + r" ([\d.]+)", b)] if m}


def fold_cases(weights):
    crystal = os.path.join(FIX, "6mrr-crystal.pdb")
    w = [f"--weights={weights}"] if weights else []
    cases = {}

    def score6mrr(samples):
        ref = ca(crystal)
        r = [kabsch_rmsd(ca(s)[:len(ref)], ref) for s in samples]
        return {"rmsd": round(r[0], 3), "mean": round(sum(r) / len(r), 3), "best": round(min(r), 3),
                "caca": round(chain_ok(samples[0])[0], 3)}

    cases["6mrr"] = (["--job=" + os.path.join(JOBS, "6mrr.json")], w, score6mrr)
    cases["6mrr-equilibrium"] = (["--job=" + os.path.join(JOBS, "6mrr.json"), "--mode=equilibrium"], w, score6mrr)
    cases["6mrr-gol-sep"] = (["--job=" + os.path.join(JOBS, "6mrr-gol-sep.json")], w,
                             lambda s: {**score6mrr(s), **bonds(s[0], "GOL,SEP")})
    cases["rna-dna"] = (["--job=" + os.path.join(JOBS, "rna-dna.json")], w, lambda s: bonds(s[0], "A,C,G,U,DA,DC,DG,DT"))
    return cases


def propagation(weights):
    """Two 1 ns steps from the 6MRR fold's first sample: each frame stays a chain and near where it started."""
    start = os.path.join(TMP, "6mrr.pdb")
    if not os.path.exists(start):
        return {"error": "no 6mrr fold to propagate from (run the folds arm)"}
    w = [f"--weights={weights}"] if weights else []
    samples, text, timing = fold("6mrr-propagate", ["--job=" + os.path.join(JOBS, "6mrr.json"), "--propagate-from=" + start,
                                                    "--lag-ns=1", "--propagation-steps=2"], w)
    if not samples:
        return {"error": text.strip().splitlines()[-1] if text.strip() else "failed"}
    ref = ca(start)
    # sample i * 2 + k is trajectory i, step k + 1
    first = [kabsch_rmsd(ca(samples[i * 2]), ref) for i in range(len(samples) // 2)]
    return {"frames": len(samples), "step1": round(sum(first) / len(first), 3), "caca": round(chain_ok(samples[-1])[0], 3),
            "ms": float(timing.group(1)) if timing else None}


def main():
    write = "--write" in sys.argv
    only = option("only", None)
    only = only.split(",") if only else None
    weights_f32 = option("weights-f32", os.path.expanduser("~/ported/bioemu2"))
    weights = option("weights", "")
    base = json.load(open(BASELINE)) if os.path.exists(BASELINE) else {}
    failed = 0
    if not only or "oracles" in only:
        for name, job, dump, export, flags in oracle_cases():
            if not os.path.exists(dump):
                print(f"{name:24s} skipped: no {os.path.relpath(dump, REPO)} (tools/oracle/dump_bioemu2.py)")
                continue
            for arm, fast in (("f32", False), ("fast", True)):
                got, error = run_oracle(name, job, dump, export, flags, weights_f32, fast)
                if error:
                    print(f"{name + ' ' + arm:24s} FAILED: {error}"); failed += 1; continue
                vjp = {k: v for k, v in got.items() if k.startswith("vjp")}
                got = {k: v for k, v in got.items() if not k.startswith("vjp")}
                worst = max(got.values())
                ok = worst == worst and worst <= ORACLE_BOUND[arm]
                failed += not ok
                print(f"{name + ' ' + arm:24s} {len(got)} stages, worst {worst:.2e} ({max(got, key=got.get)})"
                      f"  (bound {ORACLE_BOUND[arm]:.0e})  {'ok' if ok else 'FAILED'}", flush=True)
                for k, v in vjp.items():
                    bound = VJP_BOUND["d/dx" if "d/dx" in k else "d/dsigma"]
                    good = v == v and v <= bound
                    failed += not good
                    print(f"{'':24s} {k}: {v:.2e} (bound {bound:.0e})  {'ok' if good else 'FAILED'}", flush=True)
    if not only or "folds" in only or any(c in only for c in ("6mrr", "6mrr-equilibrium", "6mrr-gol-sep", "rna-dna", "propagation")):
        for name, (inputs, flags, score) in fold_cases(weights).items():
            if only and "folds" not in only and name not in only:
                continue
            samples, text, timing = fold(name, inputs, flags)
            got = {"error": text.strip().splitlines()[-1] if text.strip() else "failed"} if not samples else score(samples)
            if samples and timing: got["ms"] = float(timing.group(1))
            failed += report(name, got, base, write)
        if not only or "folds" in only or "propagation" in only:
            failed += report("6mrr-propagation", propagation(weights), base, write)
    if write:
        json.dump(base, open(BASELINE, "w"), indent=1, sort_keys=True)
        print(f"wrote {BASELINE}")
    if failed:
        print(f"{failed} case(s) failed")
        sys.exit(1)


def report(name, got, base, write):
    if "error" in got:
        print(f"{name:24s} FAILED: {got['error']}")
        return 1
    want = base.get(name)
    verdict, bad = ("recorded", 0) if write else ("NEW (no baseline)", 0)
    if want and not write:
        ok = True
        for k, v in want.items():
            if k == "ms" or k not in got or got[k] is None:
                continue
            tol = TOL_MEAN if k == "mean" else TOL_RMSD if k in ("rmsd", "best", "step1") else TOL_BOND if k in ("ligand", "nucleic") \
                else 0.05 if k == "caca" else 0
            ok = ok and abs(got[k] - v) <= tol
        verdict, bad = ("ok" if ok else "MOVED") + f" (baseline {json.dumps({k: v for k, v in want.items() if k != 'ms'})})", int(not ok)
    if write:
        base[name] = got
    print(f"{name:24s} " + "  ".join(f"{k} {v}" for k, v in got.items()) + f"  {verdict}", flush=True)
    return bad


if __name__ == "__main__":
    main()
