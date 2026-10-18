#!/usr/bin/env python3
"""Re-dump every reference batch tools/check-batch-fields.js reads, from the af3-any-model installed in ~/.venv-lfjax.

    python3 tools/oracle/regen_batch_dumps.py --reference=<dir with dev/ and converters/> [--only=SUBSTRING] [--jobs=N]

🔴 THE COMMANDS WERE NOT RECORDED, so this file is where they live now. Each target's flags are the ones its entry in
check-batch-fields.js describes (TARGETS, and the job files under tools/fixtures/af3-jobs/); the sequence of a target
comes from the dump being replaced - each dump carries its own `sequence` - so a target is reproduced, not reinvented.
The set of (model, target) pairs is the set already in oracle-dumps/: a model the gate reads without a dump stays without
one. Every component in tools/fixtures/ccd/ is handed over as --user-ccd (the pip CCD is minimal).

--reference: a directory holding just `dev/` and `converters/` of an af3-any-model checkout (a checkout's `src/` would
shadow the installed package and its compiled extension). af3-any-model's weights: AF3_WEIGHTS_DIR=~/lfjax/weights.
"""
import argparse, glob, json, os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
DUMPS = os.path.join(REPO, "oracle-dumps")
FIX = os.path.join(REPO, "tools", "fixtures")
GLYCAN_BONDS = "A:34:ND2-B:1:C1;B:1:O4-B:2:C1;B:2:O4-B:3:C1;B:3:O3-B:4:C1;B:3:O6-B:5:C1"


def target_args(target, sequence):
    """The dump flags of a check-batch-fields.js target"""
    if target.startswith("job-"):
        return ["--job", os.path.join(FIX, "af3-jobs", target[4:] + ".json")]
    simple = {
        "": [],
        "gol-sep3": ["--ligand", "GOL", "--ptm", "SEP@3"],
        "dna-5cm": ["--kind", "dna", "--mods", "5CM@5,5CM@9"],
        "dna-unk": ["--kind", "dna"],
        "rna-unk": ["--kind", "rna"],
        "glycan": ["--ligand", "NAG,NAG,BMA,MAN,MAN", "--bonds", GLYCAN_BONDS],
        "hy3": ["--ptm", "HY3@4"],
        "dna-5prime": ["--kind", "dna", "--mods", "6OG@1,6MA@2"],
        "rna-5prime": ["--kind", "rna", "--mods", "2MG@1,5MC@4"],
        "rna-mods": ["--kind", "rna", "--mods", "OMG@4,PSU@13,5MC@18"],
    }
    if target in simple:
        return ["--sequence", sequence, *simple[target]]
    if target == "rna-msa":
        return ["--chains", f"rna:{sequence}:{FIX}/msa/rna-17.a3m"]
    if target == "prot-rna-msa":
        protein, rna = sequence.split(":")
        return ["--chains", f"protein:{protein}:{FIX}/msa/protein-21.a3m,rna:{rna}:{FIX}/msa/rna-17.a3m"]
    if target == "kitchen-sink":
        return ["--job", os.path.join(FIX, "kitchen-sink-relabelled.json")]
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True)
    p.add_argument("--only", default="")
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--python", default=os.path.expanduser("~/.venv-lfjax/bin/python"))
    a = p.parse_args()
    ccd = [x for f in sorted(glob.glob(os.path.join(FIX, "ccd", "*.cif"))) for x in ("--user-ccd", f)]
    work = []
    for path in sorted(glob.glob(os.path.join(DUMPS, "af3-batch-*-6mrr*.json"))):
        m = re.match(r"af3-batch-(\w+?)-6mrr-?(.*)\.json$", os.path.basename(path))
        if not m or a.only not in os.path.basename(path):
            continue
        model, target = m.group(1), m.group(2)
        with open(path) as f:
            sequence = json.load(f)["sequence"]
        args = target_args(target, sequence)
        if args is None:
            print(f"skip {os.path.basename(path)}: no recorded target {target!r}")
            continue
        # (chai1 with an alignment: featurised as run_alphafold.py does with one, which fold_check's has_msa=False is not)
        msa = ["--has-msa"] if model == "chai1" and target == "kitchen-sink" else []
        work.append((path, [a.python, os.path.join(REPO, "tools", "oracle", "dump_af3_batch.py"), model,
                            "--reference", a.reference, *args, *msa, *ccd, "--out", path + ".new"]))
    env = dict(os.environ, AF3_WEIGHTS_DIR=os.environ.get("AF3_WEIGHTS_DIR", os.path.expanduser("~/lfjax/weights")))

    def run(item):
        path, cmd = item
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=REPO)
        if r.returncode == 0 and os.path.exists(path + ".new"):
            os.replace(path + ".new", path)
            return f"ok   {os.path.basename(path)}"
        return f"FAIL {os.path.basename(path)}: {(r.stderr or r.stdout).strip().splitlines()[-1:]}"

    with ThreadPoolExecutor(a.jobs) as pool:
        for line in pool.map(run, work):
            print(line, flush=True)


if __name__ == "__main__":
    sys.exit(main())
