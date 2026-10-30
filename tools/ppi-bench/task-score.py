# task-score.py <run.jsonl> ...: DNA- and small-molecule-binder pass rates (task-bench.mjs rows) per task, mean over
# targets with a bootstrap 95% CI over attempts within targets, beside the paper's models on the same targets
import json, os, sys, collections, random
HERE = os.path.dirname(os.path.abspath(__file__))
ref = json.load(open(os.path.join(HERE, "rfd4-paper-data.json")))
CRIT = {"dna": "dna", "sm": "sm_binder"}
paper = collections.defaultdict(lambda: collections.defaultdict(dict))
for r in ref["rows"]:
    if r.get("figure2") and r["criterion"] in CRIT.values():
        paper[r["criterion"]][r["model"]][r["target"]] = r["pass_percent"]
rows = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
errors = [d for d in rows if d.get("error")]
by = collections.defaultdict(lambda: collections.defaultdict(list))
for d in rows:
    if not d.get("error"): by[d["task"]][d["target"]].append(bool(d.get("pass")))
for task, targets in by.items():
    crit = CRIT[task]
    m = sum(sum(v) / len(v) for v in targets.values()) / len(targets)
    rng = random.Random(0)
    s = sorted(sum(sum(rng.choice(v) for _ in v) / len(v) for v in targets.values()) / len(targets) for _ in range(2000))
    n = sum(map(len, targets.values()))
    print(f"{task}: {len(targets)} targets, {n} attempts, mean {100 * m:.1f}% (CI {100 * s[50]:.1f}-{100 * s[1949]:.1f}), "
          f"targets with a pass {sum(any(v) for v in targets.values())}")
    for model in ("rfd4_proteina_high", "protein_hunter", "disco", "boltzgen", "proteina_complexa"):
        p = paper[crit].get(model)
        if not p: continue
        same = [p[t] for t in targets if t in p]
        print(f"   {model:20s} on these targets: {sum(same) / max(len(same), 1):.1f}%  "
              f"(targets with a pass {sum(1 for x in same if x > 0)})")
print("errors", len(errors), *(e["error"][:120] for e in errors[:3]))
