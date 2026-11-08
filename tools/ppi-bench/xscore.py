# xscore.py: cross-evaluation summary - each design set's pass rate under each re-predictor, and how many of the
# designs that pass the AF3 evaluation also pass the other model (mean over targets, bootstrap 95% CI).
import json, os, sys, collections, random
D = os.path.expanduser("~/ppi-bench-2026-10-29")
def passes(m, ptm=True):
    if m is None or m.get("ipae") is None: return False
    if ptm and (m.get("binderPtm") is None or m["binderPtm"] < 0.8): return False
    return m["ipae"] <= 1.5 and m["rmsd"] <= 2.5
def load(path):
    if not os.path.exists(path): return None
    return {(d["target"], d["attempt"]): d for d in map(json.loads, open(path)) if not d.get("error")}
def rate(flags):
    by = collections.defaultdict(list)
    for (t, _), f in flags.items(): by[t].append(f)
    m = sum(sum(v) / len(v) for v in by.values()) / len(by)
    rng = random.Random(0); s = sorted(sum(sum(rng.choice(v) for _ in v) / len(v) for v in by.values()) / len(by) for _ in range(1000))
    return f"{100*m:5.1f}% ({100*s[25]:4.1f}-{100*s[974]:4.1f}) n={sum(map(len, by.values()))}"
sets = {"C": f"{D}/C-native.jsonl", "NC": f"{D}/NC.jsonl", "NH": f"{D}/NH.jsonl"}
for name, path in sets.items():
    base = load(path)
    if base is None: continue
    base = {k: (v.get("native", v) if "native" in v else v) for k, v in base.items()}
    print(f"== {name} designs, AF3 evaluation: {rate({k: passes(v) for k, v in base.items()})}")
    for model in ("boltz2", "chai1", "protenix2", "ef2"):
        x = load(f"{D}/x-{model}-{name}.jsonl")
        if not x: continue
        ptm = model != "ef2"
        keys = [k for k in base if k in x]
        alone = {k: passes(x[k], ptm) for k in keys}
        both = {k: passes(base[k]) and passes(x[k], ptm) for k in keys}
        print(f"   {model:9}{'' if ptm else ' (iPAE+RMSD only)'}: {rate(alone)}   AF3 and {model}: {rate(both)}")
