"""The native ESMFold2's regression gate, two arms:

  folds    - real targets through cuda/ef2/fold (--fast), scored against the deposited structure
             (cuda/af3/score.py: CA RMSD after superposition, the chain check). Each case's
             signature (CA RMSD, mean pLDDT, pTM) is held to gate-baseline.json.
  oracles  - every cuda/ef2/data-*/ that carries an oracle-f32att/ (oracle.py --float32-attention:
             biohub's forward with its bf16 atom attention neutralised, Synthyra's confidence head),
             run in float32 (--atom-f32) and in --fast: the last trunk pass, the first denoiser call
             and the per-atom pLDDT held to it - 1e-5 in float32 (they are 1e-7..1e-6), 5e-3 in --fast.
             The data directories are gitignored and built by hand (export_input.mjs + oracle.py).

    python3 cuda/ef2/gate.py                 # both arms against the baseline
    python3 cuda/ef2/gate.py --write         # re-record the fold baseline
    python3 cuda/ef2/gate.py --only=folds    # one arm (or a case name)

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
TMP = os.environ.get("GATE_TMP", "/tmp/ef2-gate")
SEQ_6MRR = "GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE"
# RMSD may move 0.05 A or 1% of itself, whichever is more: the no-template arms are unfolded answers
# (17-22 A), whose RMSD swings a twentieth of an angstrom with an f16 path's rounding; pLDDT 0.5 points
TOL_RMSD, TOL_RMSD_REL, TOL_PLDDT = 0.05, 0.01, 0.5
# atoms, and the Evoformer's pair (the seam a kernel change reaches first): on an unconverged fold
# (1BRS from its sequences, pLDDT 36) the structure module turns a 3e-4 change in its input into
# 4e-3 in its atoms, so atoms alone cannot tell a reordered sum from a defect there
ORACLE_BOUND = {"f32": 1e-5, "fast": 5e-3}
ORACLE_SEAMS = ["trunk pass 3 out", "denoiser, step 0", "confidence pLDDT per atom"]


def chain_sequence(pdb, chain):
    three = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
             "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
             "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M"}
    seen, out = set(), ""
    for line in open(pdb):
        if line.startswith(("ATOM", "HETATM")) and line[12:16] == " CA " and line[21] == chain:
            key = line[22:27]
            if key in seen:
                continue
            seen.add(key)
            out += three.get(line[17:20], "X")
    return out


def fold_cases():
    c6mrr, c1qys, c5caj, c1brs = (os.path.join(FIX, f"{t}-crystal.pdb") for t in ("6mrr", "1qys", "5caj", "1brs"))
    return [
        ("6mrr", [f"--sequence={SEQ_6MRR}"], [c6mrr]),
        ("1qys", [f"--sequence={chain_sequence(c1qys, 'A')}"], [c1qys, "A"]),
        ("5caj", [f"--sequence={chain_sequence(c5caj, 'A')}"], [c5caj, "A"]),
        ("1brs-complex", [f"--sequence={chain_sequence(c1brs, 'A')}:{chain_sequence(c1brs, 'D')}"], [c1brs, "A,D"]),
        # what a plain protein never reaches - a ligand and an atomised modified residue, a DNA duplex, an
        # RNA hairpin - scored on bond geometry too (cuda/af3/bonds.mjs: rms against CCD ideals, A)
        ("6mrr-gol-sep3", [f"--sequence={SEQ_6MRR}", "--ligands=GOL", "--modify=SEP@3"], [c6mrr], "GOL,SEP"),
        ("dna-duplex", ["--sequence=GCGATCGATCGC:GCGATCGATCGC", "--kinds=dna,dna"], None, "DA,DC,DG,DT"),
        ("rna-hairpin", ["--sequence=GGCGCUUCGGCGCC", "--kinds=rna"], None, "A,C,G,U"),
        # AlphaFold 3's own covalent-inhibitor job: sotorasib bonded to Cys12 (a declared bond)
        ("kras-sotorasib", [f"--job={os.path.join(FIX, 'af3-jobs', 'kras_g12c_sotorasib.json')}"], None, "MOV"),
    ]


def run_fold(name, inputs, ref, codes=None):
    os.makedirs(TMP, exist_ok=True)
    pdb = os.path.join(TMP, name + ".pdb")
    if os.path.exists(pdb):
        os.remove(pdb)
    p = subprocess.run([os.path.join(HERE, "fold"), pdb, *inputs], capture_output=True, text=True)
    text = p.stdout + p.stderr
    m = re.search(r"mean pLDDT ([\d.]+)\s+pTM ([\d.nan]+)", text)
    if p.returncode or not m or not os.path.exists(pdb):
        return {"error": text.strip().splitlines()[-1] if text.strip() else f"exit {p.returncode}"}
    rmsd = None
    if ref:
        s = subprocess.run([sys.executable, os.path.join(REPO, "cuda", "af3", "score.py"), pdb, *ref],
                           capture_output=True, text=True)
        r = re.search(r"CA RMSD ([\d.]+) A", s.stdout)
        if not r:
            return {"error": "unscored: " + (s.stdout + s.stderr).strip().splitlines()[-1]}
        rmsd = float(r.group(1))
    timing = re.search(r"sampler ([\d.]+) ms", text)
    got = {"rmsd": rmsd, "plddt": float(m.group(1)), "ptm": m.group(2), "ms": float(timing.group(1)) if timing else None}
    if codes:
        b = subprocess.run(["node", os.path.join(REPO, "cuda", "af3", "bonds.mjs"), pdb, codes], capture_output=True, text=True).stdout
        for cls in ("ligand", "nucleic"):
            v = re.search(cls + r" ([\d.]+)", b)
            if v: got[cls] = float(v.group(1))
    if name == "kras-sotorasib":           # the declared covalent bond, Cys12 SG to the ligand's C25
        atoms = {}
        for line in open(pdb):
            if line.startswith(("ATOM", "HETATM")):
                key = ("SG" if line[17:20] == "CYS" and int(line[22:26]) == 12 and line[12:16].strip() == "SG" else
                       "C25" if line.startswith("HETATM") and line[12:16].strip() == "C25" else None)
                if key: atoms[key] = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        got["covalent"] = round(sum((a - b) ** 2 for a, b in zip(atoms["SG"], atoms["C25"])) ** 0.5, 2)
    return got


def oracle_cases():
    return [(e, os.path.join(HERE, e)) for e in sorted(os.listdir(HERE))
            if e.startswith("data-") and os.path.isfile(os.path.join(HERE, e, "oracle-f32att", "model.idx"))]


def run_oracle(data, fast):
    cmd = [os.path.join(HERE, "esmfold2"), data, f"--weights={os.path.join(HERE, 'weights')}",
           f"--oracle={data}/oracle-f32att", "--atom-f32", f"--out={TMP}/oracle.pdb"] + (["--fast"] if fast else [])
    p = subprocess.run(cmd, capture_output=True, text=True)
    text = p.stdout + p.stderr
    got = {}
    for seam in ORACLE_SEAMS:
        m = re.search(re.escape(seam) + r"\s+relRMS ([\d.e+-]+)", text)
        if m:
            got[seam] = float(m.group(1))
    if p.returncode or len(got) != len(ORACLE_SEAMS):
        return None, text.strip().splitlines()[-1] if text.strip() else f"exit {p.returncode}"
    return got, None


def main():
    write = "--write" in sys.argv
    only = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--only=")), None)
    base = json.load(open(BASELINE)) if os.path.exists(BASELINE) else {}
    failed = 0
    if not only or "folds" in only or any(c[0] in only for c in fold_cases()):
        for case in fold_cases():
            name, inputs, ref = case[:3]
            codes = case[3] if len(case) > 3 else None
            if only and "folds" not in only and name not in only:
                continue
            got, want = run_fold(name, inputs, ref, codes), base.get(name)
            verdict = "recorded" if write else "NEW (no baseline)"
            if "error" in got:
                verdict, failed = "FAILED: " + got["error"], failed + 1
            elif want and not write:
                ok = abs(got["plddt"] - want["plddt"]) <= TOL_PLDDT
                if want.get("rmsd") is not None:
                    ok = ok and got["rmsd"] is not None and abs(got["rmsd"] - want["rmsd"]) <= max(TOL_RMSD, TOL_RMSD_REL * want["rmsd"])
                for cls in ("ligand", "nucleic"):        # bond rms may move 0.01 A
                    if cls in want: ok = ok and cls in got and abs(got[cls] - want[cls]) <= 0.01
                if "covalent" in want: ok = ok and got.get("covalent", 99) < 2.2     # bonded, not merely near
                verdict = ("ok" if ok else "MOVED") + f" (baseline {json.dumps(want)})"
                failed += not ok
            if write and "error" not in got:
                base[name] = {k: got[k] for k in ("rmsd", "plddt", "ptm", "ligand", "nucleic", "covalent") if k in got}
            extra = "".join(f"{k} {got[k]:.3f}  " for k in ("ligand", "nucleic", "covalent") if k in got)
            rmsd = f"RMSD {got['rmsd']:.3f} A  " if got.get("rmsd") is not None else "RMSD   -      "
            body = "" if "error" in got else \
                f"{rmsd}pLDDT {got['plddt']:6.2f}  pTM {got['ptm']:>6s}  {extra}{got['ms']:7.1f} ms  "
            print(f"{name:24s} {body}{verdict}", flush=True)
    if not only or "oracles" in only:
        for name, data in oracle_cases():
            for arm, fast in (("f32", False), ("fast", True)):
                got, error = run_oracle(data, fast)
                if error:
                    print(f"{name + ' ' + arm:24s} FAILED: {error}"); failed += 1; continue
                ok = all(v <= ORACLE_BOUND[arm] for v in got.values())
                failed += not ok
                print(f"{name + ' ' + arm:24s} " + "  ".join(f"{k.split(',')[0]} {v:.1e}" for k, v in got.items())
                      + f"  (bound {ORACLE_BOUND[arm]:.0e})  {'ok' if ok else 'FAILED'}", flush=True)
    if write:
        json.dump(base, open(BASELINE, "w"), indent=1, sort_keys=True)
        print(f"wrote {BASELINE}")
    if failed:
        print(f"{failed} case(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
