"""The Metal port's AF3 gate: cuda/af3/gate.py's cases - every AF3-lineage model on 6MRR from its sequence, a
ligand and an atomised modified residue, DNA and modified bases, AF3's kitchen sink, a glycan, 5CAJ and
barnase-barstar with their crystals as templates - folded by metal/af3/localfold-af3 and scored the same way.

    python3 metal/af3/gate.py                   # against metal/af3/gate-baseline.json (this Mac's)
    python3 metal/af3/gate.py --write           # re-record it
    python3 metal/af3/gate.py --only=boltz2,af3-6mrr

Each case is held to this Mac's own baseline (0.05 A, 0.5 pLDDT, as the CUDA gate holds its) and REPORTED beside the
A100's (cuda/af3/gate-baseline.json): a fold does not reproduce to the digit on another GPU, so the A100's numbers
are the reference a change is read against, not the bar.

The af3 cases fold AlphaFold 3's published int5 bundle (model-af3-int5/, as the website does) through the
featuriser and the port's --bundle path: its af3-any-model blob asks for DeepMind's terms to be accepted, which a
gate does not do on anyone's behalf. Every other model folds the way a user's command does, its weights fetched
into the checkout (af3am-<model>/).
"""
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "cuda", "af3"))
import gate as cuda_gate   # noqa: E402  (its cases and its scoring)

BIN = os.path.join(REPO, "metal", "af3", "localfold-af3")
FEATURISE = os.path.join(REPO, "cuda", "featurise", "af3-featurise")
BASELINE = os.path.join(HERE, "gate-baseline.json")
LEGACY = os.path.join(REPO, "metal", "legacy", "af3", "gate-baseline.json")
A100 = cuda_gate.BASELINE
TMP = os.environ.get("GATE_TMP", os.path.join(tempfile.gettempdir(), "metal-af3-gate"))
AF3_BUNDLE = os.path.join(REPO, "model-af3-int5")


def fold_command(name, model, inputs, fold, pdb):
    if model == "af3":
        data = os.path.join(TMP, name + ".input")
        subprocess.run(["rm", "-rf", data])
        f = subprocess.run([FEATURISE, data, "--no-weights", "--family=af3", *inputs], capture_output=True, text=True)
        if f.returncode:
            return None, f.stdout + f.stderr
        return [BIN, data, "--fold", "--fast", f"--bundle={AF3_BUNDLE}", "--family=af3", f"--out={pdb}", *fold], ""
    return [BIN, f"--out={pdb}", f"--model={model}", f"--weights-dir={REPO}", *inputs, *fold], ""


def run(name, model, inputs, fold, ref, codes=None):
    os.makedirs(TMP, exist_ok=True)
    pdb = os.path.join(TMP, name + ".pdb")
    if os.path.exists(pdb):
        os.remove(pdb)
    cmd, err = fold_command(name, model, inputs, fold, pdb)
    if cmd is None:
        return {"error": err.strip().splitlines()[-1] if err.strip() else "featurisation failed"}
    p = subprocess.run(cmd, capture_output=True, text=True)
    text = p.stdout + p.stderr
    m = re.search(r"mean pLDDT ([\d.]+)\s+pTM ([\d.nan]+)", text)
    if p.returncode or not m or not os.path.exists(pdb):
        return {"error": text.strip().splitlines()[-1] if text.strip() else f"exit {p.returncode}"}
    r = None
    if ref:
        s = subprocess.run([sys.executable, os.path.join(REPO, "cuda", "af3", "score.py"), pdb, *ref],
                           capture_output=True, text=True)
        r = re.search(r"CA RMSD ([\d.]+) A", s.stdout)
    timing = re.search(r"total ([\d.]+) ms", text)
    got = {"rmsd": float(r.group(1)) if r else None, "plddt": float(m.group(1)), "ptm": m.group(2),
           "ms": float(timing.group(1)) if timing else None}
    if codes is not None:
        b = subprocess.run(["node", os.path.join(REPO, "cuda", "af3", "bonds.mjs"), pdb, codes],
                           capture_output=True, text=True).stdout
        for cls in ("ligand", "nucleic"):
            v = re.search(cls + r" ([\d.]+)", b)
            if v:
                got[cls] = float(v.group(1))
    return got


def main():
    write = "--write" in sys.argv
    only = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--only=")), None)
    base = json.load(open(BASELINE)) if os.path.exists(BASELINE) else {}
    a100 = json.load(open(A100)) if os.path.exists(A100) else {}
    failed = 0
    for case in cuda_gate.cases():
        name, model, inputs, fold, ref = case[:5]
        codes = case[5] if len(case) > 5 else None
        if only and model not in only and name not in only:
            continue
        got = run(name, model, inputs, fold, ref, codes)
        want = base.get(name)
        verdict = "recorded" if write else "NEW (no baseline)"
        if "error" in got:
            verdict, failed = "FAILED: " + got["error"], failed + 1
        elif want and not write:
            ok = abs(got["plddt"] - want["plddt"]) <= cuda_gate.TOL_PLDDT
            if want.get("rmsd") is not None:
                ok = ok and got["rmsd"] is not None and abs(got["rmsd"] - want["rmsd"]) <= cuda_gate.TOL_RMSD
            for cls in ("ligand", "nucleic"):
                if cls in want:
                    ok = ok and cls in got and abs(got[cls] - want[cls]) <= 0.01
            verdict = "ok" if ok else f"MOVED (baseline {json.dumps(want)})"
            failed += not ok
        if write and "error" not in got:
            base[name] = {k: got[k] for k in ("rmsd", "plddt", "ptm", "ligand", "nucleic") if k in got}
        ref100 = a100.get(name, {})
        theirs = (f"  [A100 RMSD {ref100['rmsd']:.3f} pLDDT {ref100['plddt']:.2f}]" if ref100.get("rmsd") is not None
                  else (f"  [A100 pLDDT {ref100['plddt']:.2f}]" if ref100 else ""))
        extra = "".join(f"{cls} {got[cls]:.3f}  " for cls in ("ligand", "nucleic") if cls in got)
        rmsd = f"RMSD {got['rmsd']:.3f} A  " if got.get("rmsd") is not None else "RMSD   -      "
        line = (f"{name:22s} " + (f"{rmsd}pLDDT {got['plddt']:6.2f}  pTM {got['ptm']:>6s}  {extra}"
                                   f"{got['ms']:7.1f} ms  " if "error" not in got else "") + verdict + theirs)
        print(line, flush=True)
    if write:
        json.dump(base, open(BASELINE, "w"), indent=1, sort_keys=True)
        print(f"wrote {BASELINE}")
    if failed:
        print(f"{failed} case(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
