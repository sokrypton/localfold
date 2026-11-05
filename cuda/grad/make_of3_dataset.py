#!/usr/bin/env python3
"""OpenFold3's training set for cuda/grad/train-of3, from a list of PDB entries: each fetched from RCSB, filtered as
AF3 SI 2.5 filters the weighted PDB set, built into an example (make_of3_example.py, with AF3's structure filters and
optionally the featuriser's MSA and template search), and every chain and interface weighted by Algorithm 1 with its
real N_clust - the number of datapoints of the set in its cluster.

    ~/venv/bin/python cuda/grad/make_of3_dataset.py --ids=1BRS,5K9P,... --out=<dir> [--val=6MRR,...] [--search]
        [--max-resolution=9] [--released-before=2021-09-30] [--family=openbind0]

Writes <out>/train.txt (the --examples file: a line a chain and a line an interface) and <out>/val.txt (one whole line
an entry, for --val), and <out>/clusters.txt (each chain's cluster).

Clusters (AF3 SI 2.5.3): a protein chain of 10 or more residues at 40% sequence identity, a shorter one (a peptide) and
a nucleic chain at 100%, a ligand by its components. AF3 clusters with MMseqs2, which this box lacks: identity here is
an affine-gap semi-global alignment's identical pairs over its aligned pairs at 80% coverage of the shorter chain,
greedily against cluster representatives, longest
first - the same threshold, a simpler aligner. An interface's cluster is the pair of its chains' clusters, unordered.
"""
import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import urllib.request

import gemmi
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def fetch(pdb, raw):
    path = os.path.join(raw, f"{pdb.lower()}.cif")
    if not os.path.exists(path):
        with urllib.request.urlopen(f"https://files.rcsb.org/download/{pdb.upper()}.cif.gz", timeout=120) as r:
            data = gzip.decompress(r.read())
        with open(path, "wb") as f: f.write(data)
    return path


def release_date(path):
    block = gemmi.cif.read(path).sole_block()
    dates = block.find_values("_pdbx_audit_revision_history.revision_date")
    return min(dates) if dates else None


def identity(a, b):
    """MMseqs2's sense of it, approximately: a semi-global alignment with affine gaps (match +2, mismatch -1, gap open
    -5, extend -1, end gaps free), its identical pairs over its aligned pairs - and 0 unless the aligned pairs cover 80%
    of the shorter sequence (MMseqs2's coverage). Unrelated proteins come out near 0.1; haemoglobin alpha/beta 0.43."""
    n, m = len(a), len(b)
    NEG = -1e9
    # three states (match, gap in b, gap in a), each with (score, identical, aligned) carried along its best path
    M = [[(NEG, 0, 0)] * (m + 1) for _ in range(n + 1)]
    X = [[(NEG, 0, 0)] * (m + 1) for _ in range(n + 1)]
    Y = [[(NEG, 0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): M[i][0] = (0, 0, 0)        # (free leading gaps)
    for j in range(m + 1): M[0][j] = (0, 0, 0)
    best = max
    for i in range(1, n + 1):
        ai = a[i - 1]
        Mi, Mp, Xi, Xp, Yi = M[i], M[i - 1], X[i], X[i - 1], Y[i]
        for j in range(1, m + 1):
            eq = ai == b[j - 1]
            prev = best(Mp[j - 1], Xp[j - 1], Yi[0] if False else Y[i - 1][j - 1])
            Mi[j] = (prev[0] + (2 if eq else -1), prev[1] + eq, prev[2] + 1)
            o, e = Mp[j], Xp[j]
            Xi[j] = best((o[0] - 5, o[1], o[2]), (e[0] - 1, e[1], e[2]))
            o, e = Mi[j - 1], Yi[j - 1]
            Yi[j] = best((o[0] - 5, o[1], o[2]), (e[0] - 1, e[1], e[2]))
    # (free trailing gaps: the best cell on the last row or column)
    cells = [best(M[n][j], X[n][j], Y[n][j]) for j in range(m + 1)] + [best(M[i][m], X[i][m], Y[i][m]) for i in range(n + 1)]
    sc, ident, aligned = best(cells)
    if aligned < 0.8 * min(n, m): return 0.0
    return ident / max(1, aligned)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="comma-separated PDB ids, or a file of them")
    ap.add_argument("--val", default="", help="comma-separated PDB ids for validation (whole entries, no weights)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--family", default="openbind0")
    ap.add_argument("--search", action="store_true", help="the featuriser's MSA and template search (network)")
    ap.add_argument("--max-resolution", type=float, default=9.0)
    ap.add_argument("--released-before", default="", help="YYYY-MM-DD: AF3's training cutoff is 2021-09-30")
    ap.add_argument("--reuse", action="store_true", help="keep an entry already built under --out (re-cluster only)")
    a = ap.parse_args()
    ids = open(a.ids).read().split() if os.path.exists(a.ids) else [x for x in a.ids.split(",") if x]
    val = [x for x in a.val.split(",") if x]
    raw = os.path.join(a.out, "raw"); os.makedirs(raw, exist_ok=True)
    built = {}
    for pdb in ids + val:
        path = fetch(pdb, raw)
        st = gemmi.read_structure(path)
        res = st.resolution
        if pdb in ids:
            if not res or res > a.max_resolution:
                print(f"{pdb}: resolution {res} - left out (above {a.max_resolution} A or none)"); continue
            date = release_date(path)
            if a.released_before and (date is None or date >= a.released_before):
                print(f"{pdb}: released {date} - left out"); continue
        out = os.path.join(a.out, pdb.lower())
        if a.reuse and os.path.exists(os.path.join(out, "example.txt")):
            built[pdb] = out; continue
        cmd = [sys.executable, os.path.join(HERE, "make_of3_example.py"), path, out, f"--family={a.family}", "--af3-filters"]
        if pdb in val: cmd.append("--whole")
        if a.search: cmd += ["--search", "--rcsb-templates"]   # (alignments, and templates released 60 days before the entry)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            print(f"{pdb}: the example builder failed:\n{r.stderr[-2000:]}"); continue
        print(f"{pdb}: {r.stdout.strip().splitlines()[-1]}")
        built[pdb] = out
    # ---- clusters
    chains = {}       # (pdb, chain index) -> (kind, key material)
    for pdb, out in built.items():
        if pdb not in ids: continue
        job = json.load(open(os.path.join(out, "job.json")))
        for i, s in enumerate(job["sequences"], start=1):
            kind, body = next(iter(s.items()))
            chains[(pdb, i)] = (kind, body.get("sequence") or "+".join(body.get("ccdCodes", [])))
    reps, cluster = [], {}
    order = sorted(chains, key=lambda k: -len(chains[k][1]))
    for k in order:
        kind, seq = chains[k]
        if kind == "protein" and len(seq) >= 10:
            for ci, (rk, rs) in enumerate(reps):
                if rk == "protein" and identity(seq, rs) >= 0.4: cluster[k] = f"p{ci}"; break
            else:
                cluster[k] = f"p{len(reps)}"; reps.append((kind, seq))
        else:
            cluster[k] = f"{kind}:{seq}"            # (a peptide, a nucleic chain at 100%; a ligand by its components)
    with open(os.path.join(a.out, "clusters.txt"), "w") as f:
        for (pdb, i), c in sorted(cluster.items()): f.write(f"{pdb} {i} {c}\n")
    # ---- the datapoints, N_clust from the set
    lines = []
    for pdb, out in built.items():
        if pdb not in ids: continue
        for l in open(os.path.join(out, "example.txt")).read().split("\n"):
            if l.strip(): lines.append((pdb, l))
    def key(pdb, line):
        pref = [f for f in line.split() if f.startswith("preferred=")]
        if not pref: return None
        idx = [int(x) for x in pref[0].split("=")[1].split(",")]
        return tuple(sorted(cluster[(pdb, i)] for i in idx))
    counts = {}
    for pdb, l in lines: counts[key(pdb, l)] = counts.get(key(pdb, l), 0) + 1
    with open(os.path.join(a.out, "train.txt"), "w") as f:
        for pdb, l in lines:
            fields = l.split()
            w = [x for x in fields if x.startswith("weight=")]
            if w:   # (the builder's weight is Algorithm 1's at N_clust 1: divided by the set's N_clust)
                weight = float(w[0].split("=")[1]) / counts[key(pdb, l)]
                fields = [x for x in fields if not x.startswith("weight=")] + [f"weight={weight:.5g}"]
            f.write(" ".join(fields) + "\n")
    with open(os.path.join(a.out, "val.txt"), "w") as f:
        for pdb in val:
            if pdb in built: f.write(open(os.path.join(built[pdb], "example.txt")).read())
    print(f"{len(lines)} datapoints from {sum(1 for p in built if p in ids)} entries, {len(set(cluster.values()))} clusters; "
          f"{sum(1 for p in val if p in built)} validation entries -> {a.out}")


if __name__ == "__main__":
    main()
