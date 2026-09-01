"""Score a fold from tools/gpu/fold.js against a reference structure.

    node tools/gpu-chrome.mjs tools/gpu/fold.js ... > out.json
    python3 tools/score_fold.py out.json --reference 6mrr-crystal.pdb

🔴 CA RMSD AND TM-SCORE, NOT pLDDT. pLDDT comes off the trunk and is a
prediction of confidence, so a quantisation that damages only the diffusion
head leaves it looking healthy. The distance to a crystal structure does not
care how confident anything was.
"""
import argparse, json, sys
import numpy as np


def ca_from_text(text):
    return np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
                     for l in text.split("\n")
                     if l.startswith("ATOM") and l[12:16].strip() == "CA"])


def ca_from_file(path, chain=None):
    rows = []
    for line in open(path):
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        if line[16] not in (" ", "A"):          # first altloc only
            continue
        if chain and line[21] != chain:
            continue
        rows.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(rows)


def superimpose(P, Q):
    """Kabsch. Returns per-atom distances after the best rigid fit."""
    Pc = P - P.mean(0)
    Qc = Q - Q.mean(0)
    V, _, W = np.linalg.svd(Pc.T @ Qc)
    R = V @ np.diag([1, 1, np.sign(np.linalg.det(V @ W))]) @ W
    return np.sqrt(((Pc @ R - Qc) ** 2).sum(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fold")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--chain", default="A")
    parser.add_argument("--field", default="pdb",
                        help="which structure in the result to score:"
                             " pdb (the sample) or denoisedPdb (the prediction)")
    arguments = parser.parse_args()

    text = open(arguments.fold).read()
    start = text.find('{\n  "sequence"')
    if start < 0:
        sys.exit(f"{arguments.fold} has no fold result - did the run fail?")
    result = json.loads(text[start:])
    P = ca_from_text(result[arguments.field])
    Q = ca_from_file(arguments.reference, arguments.chain)
    if len(P) != len(Q):
        sys.exit(f"{len(P)} predicted CA against {len(Q)} in the reference")

    distances = superimpose(P, Q)
    length = len(P)
    d0 = 1.24 * (length - 15) ** (1 / 3) - 1.8
    print(f"  RMSD      {np.sqrt((distances ** 2).mean()):.2f} A")
    print(f"  TM-score  {np.sum(1 / (1 + (distances / d0) ** 2)) / length:.3f}")
    print(f"  CA within 1 A {int((distances < 1).sum())}/{length}"
          f"   2 A {int((distances < 2).sum())}/{length}")
    print(f"  mean pLDDT {result['meanPlddt']:.1f}")
    geometry = result.get("geometry", {})
    if geometry:
        print(f"  backbone CA-CA {geometry['caca']['median']:.2f} A (ideal 3.80)"
              f"   radius of gyration {result.get('gyration', float('nan')):.1f} A")


if __name__ == "__main__":
    main()
