"""The native AlphaFold 2's regression gate, two arms:

  folds    - real targets through cuda/af2/fold (--fast), scored against the deposited structure
             (cuda/af3/score.py: CA RMSD after superposition, the chain check). Each case's
             signature (CA RMSD, mean pLDDT, pTM) is held to gate-baseline.json.
  oracles  - every cuda/af2/data-*/ that carries an oracle/ (oracle.py: af3-any-model's JAX AF2 on
             the same features), run in float32 and in --fast, its atoms held to the reference:
             relRMS 1e-4 in float32 (it is 1e-7..2e-5), 1e-2 in --fast (1e-4..1.5e-3). The data
             directories are gitignored and built by hand (export_input.mjs + oracle.py), so this arm
             checks what this machine has.

    python3 cuda/af2/gate.py                 # both arms against the baseline
    python3 cuda/af2/gate.py --write         # re-record the fold baseline
    python3 cuda/af2/gate.py --only=folds    # one arm (or a case name)

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
TMP = os.environ.get("GATE_TMP", "/tmp/af2-gate")
SEQ_6MRR = "GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE"
# RMSD may move 0.05 A or 1% of itself, whichever is more: the no-template arms are unfolded answers
# (17-22 A), whose RMSD swings a twentieth of an angstrom with an f16 path's rounding; pLDDT 0.5 points
TOL_RMSD, TOL_RMSD_REL, TOL_PLDDT = 0.05, 0.01, 0.5
# atoms, and the Evoformer's pair (the seam a kernel change reaches first): on an unconverged fold
# (1BRS from its sequences, pLDDT 36) the structure module turns a 3e-4 change in its input into
# 4e-3 in its atoms, so atoms alone cannot tell a reordered sum from a defect there
ORACLE_BOUND = {"f32": (1e-4, 1e-4), "fast": (2e-2, 5e-3)}   # (f32 pair: the template torsions are 1.2e-5)
MULTIMER = "model_1_multimer_v3"


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
    c6mrr, c5caj, c1brs = (os.path.join(FIX, f"{t}-crystal.pdb") for t in ("6mrr", "5caj", "1brs"))
    s5caj = chain_sequence(c5caj, "A")
    a, d = chain_sequence(c1brs, "A"), chain_sequence(c1brs, "D")
    out = [
        ("6mrr", [f"--sequence={SEQ_6MRR}"], [c6mrr]),
        # a template must MOVE the fold: 5CAJ is ~20 A from its sequence alone (both arms run)
        ("5caj-seq", [f"--sequence={s5caj}", "--recycles=0"], [c5caj, "A"]),
        ("5caj-template", [f"--sequence={s5caj}", f"--template={c5caj}:A", "--recycles=0"], [c5caj, "A"]),
        ("1brs-multimer", [f"--sequence={a}:{d}", f"--model={MULTIMER}"], [c1brs, "A,D"]),
        ("1brs-multimer-template", [f"--sequence={a}:{d}", f"--model={MULTIMER}", f"--template={c1brs}:A+D"],
         [c1brs, "A,D"]),
    ]
    a3m = os.path.join(REPO, "oracle-dumps", "5caj-a.a3m")
    if os.path.exists(a3m):     # the MSA and extra stacks at depth (512 + 1024 of 7907 rows)
        out.append(("5caj-msa", [f"--sequence={s5caj}", f"--a3m={a3m}"], [c5caj, "A"]))
    return out


def run_fold(name, inputs, ref):
    os.makedirs(TMP, exist_ok=True)
    pdb = os.path.join(TMP, name + ".pdb")
    if os.path.exists(pdb):
        os.remove(pdb)
    p = subprocess.run([os.path.join(HERE, "fold"), pdb, *inputs], capture_output=True, text=True)
    text = p.stdout + p.stderr
    m = re.search(r"mean pLDDT ([\d.]+)\s+pTM ([\d.nan]+)", text)
    if p.returncode or not m or not os.path.exists(pdb):
        return {"error": text.strip().splitlines()[-1] if text.strip() else f"exit {p.returncode}"}
    s = subprocess.run([sys.executable, os.path.join(REPO, "cuda", "af3", "score.py"), pdb, *ref],
                       capture_output=True, text=True)
    r = re.search(r"CA RMSD ([\d.]+) A", s.stdout)
    if not r:
        return {"error": "unscored: " + (s.stdout + s.stderr).strip().splitlines()[-1]}
    timing = re.search(r"passes, ([\d.]+) ms", text)
    return {"rmsd": float(r.group(1)), "plddt": float(m.group(1)), "ptm": m.group(2),
            "ms": float(timing.group(1)) if timing else None}


def oracle_cases():
    out = []
    for entry in sorted(os.listdir(HERE)):
        data = os.path.join(HERE, entry)
        if not entry.startswith("data-") or not os.path.isfile(os.path.join(data, "oracle", "model.idx")):
            continue
        meta = open(os.path.join(data, "oracle", "model.idx")).read()
        passes = len(set(re.findall(r"o/pass(\d+)/", meta))) or 1
        out.append((entry, data, passes))
    return out


def run_oracle(data, passes, weights, fast):
    cmd = [os.path.join(HERE, "af2"), data, f"--weights={weights}", f"--oracle={data}/oracle",
           f"--recycles={passes - 1}", f"--out={TMP}/oracle.pdb"] + (["--fast"] if fast else [])
    p = subprocess.run(cmd, capture_output=True, text=True)
    text = p.stdout + p.stderr
    worst = [float(v) for v in re.findall(r"atom37 positions\s+relRMS ([\d.e+-]+)", text)]
    pair = [float(v) for v in re.findall(r"evoformer pair\s+relRMS ([\d.e+-]+)", text)]
    if p.returncode or not worst or not pair:
        return None, text.strip().splitlines()[-1] if text.strip() else f"exit {p.returncode}"
    return (max(worst), max(pair)), None


def main():
    write = "--write" in sys.argv
    only = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--only=")), None)
    base = json.load(open(BASELINE)) if os.path.exists(BASELINE) else {}
    failed = 0
    if not only or "folds" in only or any(c[0] in only for c in fold_cases()):
        for name, inputs, ref in fold_cases():
            if only and "folds" not in only and name not in only:
                continue
            got, want = run_fold(name, inputs, ref), base.get(name)
            verdict = "recorded" if write else "NEW (no baseline)"
            if "error" in got:
                verdict, failed = "FAILED: " + got["error"], failed + 1
            elif want and not write:
                ok = abs(got["plddt"] - want["plddt"]) <= TOL_PLDDT and got["rmsd"] is not None \
                    and abs(got["rmsd"] - want["rmsd"]) <= max(TOL_RMSD, TOL_RMSD_REL * want["rmsd"])
                verdict = ("ok" if ok else "MOVED") + f" (baseline {json.dumps(want)})"
                failed += not ok
            if write and "error" not in got:
                base[name] = {k: got[k] for k in ("rmsd", "plddt", "ptm")}
            body = "" if "error" in got else \
                f"RMSD {got['rmsd']:.3f} A  pLDDT {got['plddt']:6.2f}  pTM {got['ptm']:>6s}  {got['ms']:7.1f} ms  "
            print(f"{name:24s} {body}{verdict}", flush=True)
    if not only or "oracles" in only:
        for name, data, passes in oracle_cases():
            weights = os.path.join(HERE, "weights-" + read_model(data))
            for arm, fast in (("f32", False), ("fast", True)):
                worst, error = run_oracle(data, passes, weights, fast)
                if error:
                    print(f"{name + ' ' + arm:24s} FAILED: {error}"); failed += 1; continue
                (atoms, pair), (atomBound, pairBound) = worst, ORACLE_BOUND[arm]
                ok = atoms <= atomBound and pair <= pairBound
                failed += not ok
                print(f"{name + ' ' + arm:24s} atoms {atoms:.2e} (bound {atomBound:.0e})  pair {pair:.2e} (bound "
                      f"{pairBound:.0e})  {passes} pass{'es' if passes > 1 else ''}  {'ok' if ok else 'FAILED'}", flush=True)
    if write:
        json.dump(base, open(BASELINE, "w"), indent=1, sort_keys=True)
        print(f"wrote {BASELINE}")
    if failed:
        print(f"{failed} case(s) failed")
        sys.exit(1)


def read_model(data):
    """The parameter set an input was exported for (export_input.mjs records it as meta/model)."""
    for line in open(os.path.join(data, "model.idx")):
        if line.startswith("m meta/model "):
            return line.split()[2]
    sys.exit(f"{data}: no meta/model - re-export it with export_input.mjs")


if __name__ == "__main__":
    main()
