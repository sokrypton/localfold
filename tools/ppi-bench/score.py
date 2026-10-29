# score.py <run.jsonl> ...: pass rates per target against the paper's (data.json, figure-2 PPI rows)
import json, os, sys, collections
import cluster
ref = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rfd4-paper-data.json")))
paper = collections.defaultdict(dict)
for r in ref["rows"]:
    if r["criterion"] == "ppi" and r.get("figure2"):
        paper[r["target"]][r["model"]] = (r["pass_percent"], r["clustered_percent"])
def key(name):
    return name if name in paper else "rfd3_" + name
rows = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
# an attempt that died on the machine (out of memory, a server gone) is no design and no failure: it is re-run
rows = [d for d in rows if not d.get("error")]
by = collections.defaultdict(list)
for d in rows: by[d["target"]].append(d)
def passes(d):
    return d.get("binderPtm") is not None and d["binderPtm"] >= 0.8 and d["ipae"] <= 1.5 and d["rmsd"] <= 2.5
models = ["rfd4_proteina_high", "bindcraft", "protein_hunter"]
print("%-12s %5s %6s %s" % ("target", "n", "ours%", "  ".join("%s%%" % m[:10] for m in models)))
ours, oursc, theirs, theirsc = [], [], collections.defaultdict(list), collections.defaultdict(list)
for name in sorted(by):
    ds = by[name]; p = sum(map(passes, ds)); n = len(ds)
    ours.append(100 * p / n)
    passing = [cluster.ca_of(d["designPdb"]) for d in ds if passes(d)]
    c = cluster.clusters(passing) if passing else 0
    oursc.append(100 * c / n)
    for m in models: theirs[m].append(paper[key(name)][m][0]); theirsc[m].append(paper[key(name)][m][1])
    near = [f"{d.get('binderPtm', 0) or 0:.2f}/{d.get('ipae', 99) if d.get('ipae') is not None else 99:.1f}/{d.get('rmsd', 99) if d.get('rmsd') is not None else 99:.1f}" for d in ds[:4]]
    print("%-12s %5d %6.1f %s   %s" % (name, n, 100 * p / n, "  ".join("%10.1f" % paper[key(name)][m][0] for m in models), " ".join(near)))
print("mean over these %d targets: ours %.1f%% | %s" % (len(ours), sum(ours) / len(ours),
      " | ".join("%s %.1f%%" % (m, sum(v) / len(v)) for m, v in theirs.items())))
print("clustered:                   ours %.1f%% | %s" % (sum(oursc) / len(oursc),
      " | ".join("%s %.1f%%" % (m, sum(v) / len(v)) for m, v in theirsc.items())))
errs = [d for d in rows if d.get("error")]
print("errors", len(errs), "no-design", sum(1 for d in rows if not d.get("binder") and not d.get("error")),
      "mean wall %.0f s" % (sum(d.get("wall", 0) for d in rows) / len(rows)))
