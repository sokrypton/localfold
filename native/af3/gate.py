"""The native port's regression gate: every AF3-lineage model folds 6MRR from its sequence, plus
AlphaFold 3 on 5CAJ with its crystal as a template and on the barnase-barstar complex, each
scored against the deposited structure. The signature of every case (CA RMSD, mean pLDDT, pTM)
is held to gate-baseline.json; a fold that moves past the tolerance fails the gate.

    python3 native/af3/gate.py                    # check against the baseline
    python3 native/af3/gate.py --write            # re-record it (after a deliberate change)
    python3 native/af3/gate.py --only=boltz2,af3  # a subset

A baseline is this machine's: a fold does not reproduce to the digit on another GPU.
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FIX = os.path.join(REPO, "tools", "fixtures")
BASELINE = os.path.join(HERE, "gate-baseline.json")
TMP = os.environ.get("GATE_TMP", "/tmp/af3-gate")
SEQ_6MRR = "GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE"
# RMSD may move this much (A), pLDDT this much (points) before the gate fails: an f16 path
# reorders sums, so a kernel change that is right still moves the last digits
TOL_RMSD, TOL_PLDDT = 0.05, 0.5


def chain_sequence(pdb, chain):
    three = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
             "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
             "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M"}
    seen, out = set(), ""
    for line in open(pdb):
        if line.startswith(("ATOM", "HETATM")) and line[12:16] == " CA " and line[21] == chain:
            key = line[22:27]       # residue number + insertion code; alternate locations count once
            if key in seen:
                continue
            seen.add(key)
            out += three.get(line[17:20], "X")
    return out


def cases():
    crystal_5caj = os.path.join(FIX, "5caj-crystal.pdb")
    crystal_1brs = os.path.join(FIX, "1brs-crystal.pdb")
    out = []
    for model in ["af3", "protenix2", "boltz2", "intellifold2", "rosettafold3", "openbind0", "opendde"]:
        fold = ["--steps=16"] if model == "opendde" else []
        out.append((f"{model}-6mrr", model, [f"--sequence={SEQ_6MRR}"], fold,
                    [os.path.join(FIX, "6mrr-crystal.pdb")]))
    seq = chain_sequence(crystal_5caj, "A")
    out.append(("af3-5caj-template", "af3", [f"--sequence={seq}", f"--template={crystal_5caj}:A"], [],
                [crystal_5caj, "A"]))
    if os.path.exists(crystal_1brs):
        a, d = chain_sequence(crystal_1brs, "A"), chain_sequence(crystal_1brs, "D")
        out.append(("af3-1brs-template", "af3",
                    [f"--sequence={a}:{d}", f"--template={crystal_1brs}:A@0+{crystal_1brs}:D@1"], [],
                    [crystal_1brs, "A,D"]))
    return out


def run(name, model, inputs, fold, ref):
    os.makedirs(TMP, exist_ok=True)
    pdb = os.path.join(TMP, name + ".pdb")
    if os.path.exists(pdb):
        os.remove(pdb)
    cmd = [os.path.join(HERE, "fold"), pdb, f"--model={model}", *inputs, "--", *fold]
    p = subprocess.run(cmd, capture_output=True, text=True)
    text = p.stdout + p.stderr
    m = re.search(r"mean pLDDT ([\d.]+)\s+pTM ([\d.nan]+)", text)
    if p.returncode or not m or not os.path.exists(pdb):
        return {"error": text.strip().splitlines()[-1] if text.strip() else f"exit {p.returncode}"}
    s = subprocess.run([sys.executable, os.path.join(HERE, "score.py"), pdb, *ref], capture_output=True, text=True)
    r = re.search(r"CA RMSD ([\d.]+) A", s.stdout)
    timing = re.search(r"total ([\d.]+) ms", text)
    return {"rmsd": float(r.group(1)) if r else None, "plddt": float(m.group(1)), "ptm": m.group(2),
            "ms": float(timing.group(1)) if timing else None}


def main():
    write = "--write" in sys.argv
    only = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--only=")), None)
    base = json.load(open(BASELINE)) if os.path.exists(BASELINE) else {}
    failed = 0
    for name, model, inputs, fold, ref in cases():
        if only and model not in only and name not in only:
            continue
        got = run(name, model, inputs, fold, ref)
        want = base.get(name)
        verdict = "recorded" if write else "NEW (no baseline)"
        if "error" in got:
            verdict, failed = "FAILED: " + got["error"], failed + 1
        elif want and not write:
            dr, dp = abs(got["rmsd"] - want["rmsd"]), abs(got["plddt"] - want["plddt"])
            ok = dr <= TOL_RMSD and dp <= TOL_PLDDT
            verdict = ("ok" if ok else "MOVED") + f" (baseline {want['rmsd']:.3f} A, pLDDT {want['plddt']:.2f})"
            failed += not ok
        if write and "error" not in got:
            base[name] = {k: got[k] for k in ("rmsd", "plddt", "ptm")}
        line = (f"{name:22s} " + (f"RMSD {got['rmsd']:.3f} A  pLDDT {got['plddt']:6.2f}  pTM {got['ptm']:>6s}  "
                                   f"{got['ms']:7.1f} ms  " if "error" not in got else "") + verdict)
        print(line, flush=True)
    if write:
        json.dump(base, open(BASELINE, "w"), indent=1, sort_keys=True)
        print(f"wrote {BASELINE}")
    if failed:
        print(f"{failed} case(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
