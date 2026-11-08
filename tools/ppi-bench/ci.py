# ci.py <run.jsonl> ...: mean per-target pass rate, a bootstrap 95% CI over attempts within targets, and coverage
import json, sys, collections, random
P = lambda d: d.get("binderPtm") is not None and d["binderPtm"] >= 0.8 and d["ipae"] <= 1.5 and d["rmsd"] <= 2.5
for f in sys.argv[1:]:
    by = collections.defaultdict(list)
    for l in open(f):
        d = json.loads(l)
        if d.get("error"): continue
        by[d["target"]].append(P(d.get("native", d)) if "native" in d else P(d))
    if not by: continue
    rng = random.Random(0); means = []
    for _ in range(2000):
        means.append(sum(sum(rng.choice(v) for _ in v) / len(v) for v in by.values()) / len(by))
    means.sort(); m = sum(sum(v) / len(v) for v in by.values()) / len(by)
    n = sum(map(len, by.values()))
    print("%-28s n=%4d targets=%2d  mean %5.1f%%  CI %5.1f-%5.1f%%  targets with a pass %2d" % (f.split("/")[-1], n, len(by), 100 * m, 100 * means[50], 100 * means[1949], sum(any(v) for v in by.values())))
