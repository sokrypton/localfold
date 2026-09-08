#!/usr/bin/env python3
"""Fold twice under two configurations and compare the COORDINATES.

    python3 tools/diff-fold-coords.py --a="--attn-splits=1" --b="--attn-splits=4"
    python3 tools/diff-fold-coords.py --b="--tune=diffusionBatchedGates=false" --steps=200

🔴 pLDDT IS NOT A SUFFICIENT GATE, AND THIS IS THE MEASUREMENT THAT SHOWS IT.
Every arm of a night's tuning reported `meanPlddt` identical to SIXTEEN DIGITS -
84.20887255253277 - including one that moves thirty-three atoms. Measured here
at 68 tokens and 200 steps:

    tile change, reorders nothing     max |dx| 0.000000 A   574/574 identical
    the conditioning hoist            max |dx| 0.000000 A   574/574 identical
    attnSplits 1 -> 4, regroups a sum max |dx| 0.001000 A   541/574 identical

The third arm changes the structure and pLDDT does not notice. pLDDT is a
PREDICTED confidence, averaged over every atom, and it is flat to well past the
digits a small coordinate change reaches - so "pLDDT unchanged" means "not
obviously broken", never "bit-exact". Claims of bit-exactness need this, or
`tools/gpu/bench-difftx-splits.js`'s relRMS on the raw tensor.

🔴 AND HOLD THE OTHER KNOBS FIXED, WHICH COST A WRONG CONCLUSION ONCE.
Comparing the hoist on against off ALSO flips normKSplits, because the shader
factory forces it to 1 when the projection is batched - so the naive comparison
reads 538/574 and looks like the hoist is inexact. Pin `--norm-splits=1` on both
sides and it is 574/574. A differential gate is only differential in the thing
you actually varied.

🔴 PDB IS THREE DECIMALS, so this resolves 0.001 A and no finer. A difference it
calls zero is below a millipascal of an angstrom, not proven absent; for a real
bit-exactness claim the raw-tensor relRMS is the instrument.
"""
import argparse, json, re, subprocess, sys, os

SEQ = "GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE"


def fold(extra, args):
    cmd = ["node", "tools/gpu-chrome.mjs", "tools/gpu/fold.js",
           f"--model={args.model}", f"--sequence={args.sequence}",
           f"--steps={args.steps}", "--recycles=0", "--folds=1", "--trajectory=off"]
    cmd += [a for a in extra.split(" ") if a]
    env = dict(os.environ, DISPLAY=os.environ.get("DISPLAY", ":99"))
    out = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600).stdout
    # 🔴 --tune PRINTS A LINE BEFORE THE JSON, so the first "{" is not the body.
    at = out.index("{\n")
    body = json.loads(out[at:])
    subprocess.run(["pkill", "-9", "-f", "gpu-chrome[-]"], capture_output=True)
    return body.get("pdb") or "", body.get("meanPlddt")


def atoms(pdb):
    return [(float(l[30:38]), float(l[38:46]), float(l[46:54]))
            for l in pdb.splitlines() if l.startswith(("ATOM", "HETATM"))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", default="", help="flags for the reference arm")
    p.add_argument("--b", required=True, help="flags for the arm under test")
    p.add_argument("--steps", default="200")
    p.add_argument("--sequence", default=SEQ)
    p.add_argument("--model", default="/model-af3-int5/manifest.json")
    args = p.parse_args()

    left, plddtA = fold(args.a, args)
    right, plddtB = fold(args.b, args)
    a, b = atoms(left), atoms(right)
    if len(a) != len(b) or not a:
        print(f"different atom counts: {len(a)} vs {len(b)}")
        return 1
    per = [max(abs(x - y) for x, y in zip(p, q)) for p, q in zip(a, b)]
    same = sum(1 for x in per if x == 0.0)
    print(f"  A: {args.a or '(default)'}")
    print(f"  B: {args.b}")
    print(f"  meanPlddt   A {plddtA!r}")
    print(f"              B {plddtB!r}")
    print(f"              {'IDENTICAL - which proves nothing on its own' if plddtA == plddtB else 'differ'}")
    print(f"  coordinates max |dx| {max(per):.6f} A   identical {same}/{len(per)}")
    print("  => bit-exact at PDB resolution" if same == len(per)
          else f"  => {len(per) - same} atoms moved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
