# engine-probe.py <out.jsonl> <calls> <recycles> <arm> [<limit>] [--source=<rows.jsonl>] [--seeds=11,12] [--attempts=0-2]:
# the same protein-binder complexes - a run's submitted binders (NC's by default, attempts 0-2), at each seed, the
# target templated - folded by one engine, to compare the page's AF3 with the native one on identical inputs
# (docs/DESIGN-TODO.md, "the page against native").
#   arm: "native" (cuda/af3/fold, its int8 weights, as native-bench.mjs folds) | "int5" (the page's AF3 on its shipped
#   bundle) | a bundle directory served from the checkout, e.g. "/model-af3-full-f32/" (the page's AF3 on it)
import json, os, socket, subprocess, sys, tempfile, time
ROOT = os.path.expanduser("~/localfold"); sys.path.insert(0, os.path.join(ROOT, "tools")); import cdp
HERE = os.path.dirname(os.path.abspath(__file__))
opts = {a.split("=", 1)[0][2:]: a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--")}
pos = [a for a in sys.argv[1:] if not a.startswith("--")]
out, calls, recycles, arm = pos[0], int(pos[1]), int(pos[2]), pos[3]
limit = int(pos[4]) if len(pos) > 4 else None
source = opts.get("source", os.path.expanduser("~/ppi-bench-2026-10-29/NC.jsonl"))
seeds = [int(x) for x in opts.get("seeds", "11,12").split(",")]
lo, hi = (int(x) for x in opts.get("attempts", "0-2").split("-"))
targets = json.load(open(os.path.join(HERE, "targets.json")))
items = []
for line in open(source):
    d = json.loads(line)
    if d.get("error") or not d.get("binder") or not lo <= d["attempt"] <= hi: continue
    t = targets[d["target"]]
    for seed in seeds:
        items.append({"key": f"{d['target']}:{d['attempt']}", "seed": seed, "binder": d["binder"], "sequences": t["sequences"], "pdb": t["pdb"]})
items = items[:limit]
rows = []
if arm == "native":
    work = tempfile.mkdtemp()
    for it in items:
        pdb = os.path.join(work, "t.pdb"); open(pdb, "w").write(it["pdb"])
        slots = ",".join(f"{pdb}:{'BCDEFG'[i]}@{i + 1}" for i in range(len(it["sequences"])))
        f = os.path.join(work, "f.pdb")
        subprocess.run([f"{ROOT}/cuda/af3/fold", f, f"--sequence={':'.join([it['binder'], *it['sequences']])}", f"--template={slots}",
                        "--", f"--steps={calls}", f"--recycles={recycles}", f"--seed={it['seed']}"], check=True, capture_output=True)
        s = json.load(open(f.replace(".pdb", "_summary_confidences.json")))
        k = len(s["chain_pair_pae_min"])
        ipae = min((s["chain_pair_pae_min"][0][j] + s["chain_pair_pae_min"][j][0]) / 2 for j in range(1, k))
        rows.append({"key": it["key"], "seed": it["seed"], "ipae": ipae, "binderPtm": s["chain_ptm"][0]})
else:
    def free():
        s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p
    port = free()
    srv = subprocess.Popen([sys.executable, os.path.join(ROOT, "tools/serve.py"), str(port)], cwd=ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    prof = f"/tmp/localfold-probe-{os.getpid()}"
    proc, ws = cdp.launch(free(), prof)
    try:
        ws.call("Page.enable"); ws.call("Runtime.enable")
        ws.call("Page.navigate", url=f"http://127.0.0.1:{port}/design.html")
        cdp.wait_for(ws, "document.querySelectorAll('#preset option').length > 2", 90, "page")
        cdp.evaluate(ws, open(os.path.join(HERE, "engine-probe.js")).read())
        cdp.evaluate(ws, "window.__r = null; window.__e = null; window.__probe(%s, %s, %d, %d).then(r => window.__r = r, e => window.__e = String(e && e.stack || e)); 1"
                     % (json.dumps(items), json.dumps(arm), calls, recycles), await_promise=False)
        while True:
            time.sleep(5)
            state = cdp.evaluate(ws, "window.__e !== null ? 'err' : window.__r !== null ? 'done' : 'run'")
            if state != "run": break
        if state == "err": raise RuntimeError(cdp.evaluate(ws, "window.__e"))
        rows = cdp.evaluate(ws, "window.__r")
    finally:
        proc.kill(); srv.kill(); subprocess.run(["rm", "-rf", prof])
with open(out, "w") as f:
    for r in rows: f.write(json.dumps({**r, "arm": arm, "calls": calls, "recycles": recycles}) + "\n")
print(arm, len(rows), "folds;", "check-worthy (pTM >= 0.8, iPAE <= 2.5):", sum(r["binderPtm"] >= 0.8 and r["ipae"] <= 2.5 for r in rows))
