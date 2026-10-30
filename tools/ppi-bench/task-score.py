# task-score.py <run.jsonl> ...: DNA-, small-molecule-binder and AME pass rates (task-bench.mjs rows) per task, mean
# over targets with a bootstrap 95% CI over attempts within targets, beside the paper's models on the same targets -
# one-shot, and RFD4-Proteina's Search pipeline. A target's variant is its prefix: "U-" is AME's unfixed-ligand set
# (mcsa_unfixed), "F-" a fixed target (the *_fixed datasets); none is the figure-2 dataset.
import json, os, sys, collections, random
HERE = os.path.dirname(os.path.abspath(__file__))
ref = json.load(open(os.path.join(HERE, "rfd4-paper-data.json")))
CRIT = {"dna": "dna", "sm": "sm_binder", "ame": "ame"}
DATASET = {("dna", ""): "dna_unfixed", ("dna", "F-"): "dna_fixed", ("sm", ""): "studio_figure_2c_unfixed",
           ("sm", "F-"): "studio_figure_2c_fixed", ("ame", ""): "mcsa_fixed", ("ame", "U-"): "mcsa_unfixed"}
paper = collections.defaultdict(lambda: collections.defaultdict(dict))
for r in ref["rows"]:
    if r["criterion"] in CRIT.values():
        paper[(r["criterion"], r["dataset"])][r["model"]][r["target"]] = r["pass_percent"]
def variant(target):
    return next((p for p in ("U-", "F-") if target.startswith(p)), "")
rows = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
errors = [d for d in rows if d.get("error")]
by = collections.defaultdict(lambda: collections.defaultdict(list))
for d in rows:
    if not d.get("error"): by[(d["task"], variant(d["target"]))][d["target"][len(variant(d["target"])):]].append(bool(d.get("pass")))
for (task, prefix), targets in by.items():
    crit, dataset = CRIT[task], DATASET[(task, prefix)]
    m = sum(sum(v) / len(v) for v in targets.values()) / len(targets)
    rng = random.Random(0)
    s = sorted(sum(sum(rng.choice(v) for _ in v) / len(v) for v in targets.values()) / len(targets) for _ in range(2000))
    n = sum(map(len, targets.values()))
    print(f"{task} ({dataset}): {len(targets)} targets, {n} attempts, mean {100 * m:.1f}% (CI {100 * s[50]:.1f}-{100 * s[1949]:.1f}), "
          f"targets with a pass {sum(any(v) for v in targets.values())}")
    for model in ("rfd4_proteina_high", "rfd4_proteina_high_search", "protein_hunter", "disco", "boltzgen", "proteina_complexa"):
        p = paper[(crit, dataset)].get(model)
        if not p: continue
        same = [p[t] for t in targets if t in p]
        print(f"   {model:26s} on these targets: {sum(same) / max(len(same), 1):.1f}%  "
              f"(targets with a pass {sum(1 for x in same if x > 0)})")
print("errors", len(errors), *(e["error"][:120] for e in errors[:3]))
