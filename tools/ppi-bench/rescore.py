# Re-scores benchmark designs with the native CUDA AlphaFold 3 (cuda/af3, af3-any-model's int8 weights) instead
# of the page's int5 WebGPU fold - an independent evaluator for the same criteria: single sequence, target
# templated, 5 recycles, 200 steps, one sample.
# usage: rescore.py <in.jsonl> <out.jsonl> [seed]
import json, os, subprocess, sys
from native_eval import ROOT, fold_line, terms
HERE = os.path.dirname(os.path.abspath(__file__))
src, dst = sys.argv[1], sys.argv[2]
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 20261029
targets = json.load(open(os.path.join(HERE, "targets.json")))
work = os.path.join(HERE, "rescore-" + os.path.basename(dst).replace(".jsonl", ""))
os.makedirs(os.path.join(work, "t"), exist_ok=True)
done = set()
if os.path.exists(dst):
    done = {(d["target"], d["attempt"]) for d in map(json.loads, open(dst))}
every = [d for d in map(json.loads, open(src)) if (d["target"], d["attempt"]) not in done]
rows = [d for d in every if d.get("binder")]
lines = []
for d in rows:
    t = targets[d["target"]]
    tp = os.path.join(work, "t", d["target"] + ".pdb")
    if not os.path.exists(tp): open(tp, "w").write(t["pdb"])
    out = os.path.join(work, f"{d['target']}_{d['attempt']}.pdb")
    lines.append(" ".join(fold_line(t, d["binder"], tp, out)))
batch = os.path.join(work, "batch.txt")
open(batch, "w").write("\n".join(lines) + "\n")
if lines:
    subprocess.run([os.path.join(ROOT, "cuda/af3/fold"), f"--batch={batch}", "--",
                    "--recycles=5", "--steps=200", f"--seed={seed}"], check=True)

with open(dst, "a") as f:
    for d in rows:
        out = os.path.join(work, f"{d['target']}_{d['attempt']}.pdb")
        r = terms(out, d["designPdb"])
        f.write(json.dumps({"target": d["target"], "attempt": d["attempt"], "binder": d["binder"],
                            "binderPtm": r["binderPtm"], "ipae": r["ipae"], "rmsd": r["rmsd"],
                            "designPdb": d["designPdb"], "pageBinderPtm": d.get("binderPtm"),
                            "pageIpae": d.get("ipae"), "pageRmsd": d.get("rmsd")}) + "\n")
    # an attempt that produced no design is a failure under either evaluator
    for d in every:
        if not d.get("binder") and not d.get("error"):
            f.write(json.dumps({"target": d["target"], "attempt": d["attempt"], "binderPtm": None}) + "\n")
print("rescored", len(rows))
