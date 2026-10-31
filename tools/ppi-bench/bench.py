# Drives design.html in headless Chrome through Protein Hunter attempts on the RFD4-Proteina PPI benchmark.
# usage: bench.py <out.jsonl> <config.json> <target>:<attempt> ...   (one browser; run several in parallel)
import json, os, random, socket, subprocess, sys, tempfile, time
ROOT = os.path.expanduser("~/localfold"); sys.path.insert(0, os.path.join(ROOT, "tools")); import cdp
from native_eval import evaluate
HERE = os.path.dirname(os.path.abspath(__file__))
def free():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p
# --root=<dir>: serve the page from another checkout (a worktree carrying a change to test), not this one
args = [a for a in sys.argv[1:] if not a.startswith("--root=")]
SITE = next((a[7:] for a in sys.argv[1:] if a.startswith("--root=")), ROOT)
out, config = args[0], json.load(open(args[1]))
jobs = [a.split(":") for a in args[2:]]
targets = json.load(open(os.path.join(HERE, "targets.json")))
done = set()
if os.path.exists(out):
    for line in open(out):
        d = json.loads(line); done.add((d["target"], d["attempt"]))
bench_js = open(os.path.join(HERE, "bench.js")).read()
port = free()
srv = subprocess.Popen([sys.executable, os.path.join(SITE, "tools/serve.py"), str(port)], cwd=SITE,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
prof = f"/tmp/localfold-bench-{os.getpid()}"
proc, ws = None, None
def start():
    global proc, ws
    proc, ws = cdp.launch(free(), prof)
    ws.call("Page.enable"); ws.call("Runtime.enable")
    ws.call("Page.navigate", url=f"http://127.0.0.1:{port}/design.html")
    cdp.wait_for(ws, "document.querySelectorAll('#preset option').length > 2", 90, "page")
    cdp.evaluate(ws, bench_js)
def stop():
    if proc is not None: proc.kill()
    subprocess.run(["rm", "-rf", prof], stderr=subprocess.DEVNULL)
try:
    start()
    for name, attempt in jobs:
        attempt = int(attempt)
        if (name, attempt) in done: continue
        t = targets[name]
        # `pair`: arms of one A/B draw the same lengths and evaluation seeds
        rng = random.Random(f"{config.get('pair', config['tag'])}:{name}:{attempt}")
        lo, hi = t["binder"]
        hi = min(hi, lo + config.get("lengthSpan", 10**9))
        cfg = dict(config, target=t, length=rng.randint(lo, hi), seed=1000 * attempt + 7,
                   evalSeed=rng.randint(1, 2**30))
        cfg.pop("tag", None); cfg.pop("lengthSpan", None); cfg.pop("pair", None)
        if cfg.pop("msaDir", None) is not None:
            import hashlib
            cfg["targetMsas"] = [open(os.path.join(config["msaDir"], hashlib.sha1(s.encode()).hexdigest()[:12] + ".a3m")).read()
                                 for s in t["sequences"]]
        t0 = time.time()
        error = None
        try:
            cdp.evaluate(ws, "window.__job = null; window.__err = null; window.__bench(%s).then(r => window.__job = r, e => window.__err = String(e && e.stack || e)); 1" % json.dumps(cfg), await_promise=False)
            while True:
                time.sleep(2)
                state = cdp.evaluate(ws, "window.__err !== null ? 'err' : window.__job !== null ? 'done' : 'run'")
                if state != "run": break
                if time.time() - t0 > 3600: raise RuntimeError("attempt over an hour")
            if state == "err":
                error = cdp.evaluate(ws, "window.__err")
                result = None
            else:
                result = json.loads(cdp.evaluate(ws, "JSON.stringify(window.__job)"))
        except Exception as e:
            error = repr(e); result = None
            stop(); start()
        row = {"target": name, "attempt": attempt, "length": cfg["length"], "config": config["tag"],
               "wall": round(time.time() - t0, 1), "error": error}
        if result is not None:
            row.update(result)
            if result.get("best") is not None or result.get("binder"):
                ptm = result["chainPtm"]
                row["binderPtm"] = ptm.get(str(result["coi"])) if isinstance(ptm, dict) else None
        # 🔴 THE EVALUATION IS NATIVE AF3, at the attempt's own evaluation seed - the page's own score of its design is
        # kept beside it (page*) but a single page-AF3 seed is a noisy label: on 48 paired attempts the page scored its
        # designs 12 where native scored the same designs 16 (docs/DESIGN-TODO.md, "the page against native")
        if row.get("binder") and row.get("designPdb"):
            for k in ("binderPtm", "ipae", "rmsd"): row["page" + k[0].upper() + k[1:]] = row.get(k)
            with tempfile.TemporaryDirectory() as work:
                row.update(evaluate(t, row["binder"], row["designPdb"], cfg["evalSeed"], work))
            row["evaluator"] = "native"
        with open(out, "a") as f: f.write(json.dumps(row) + "\n")
        print(name, attempt, row.get("binderPtm"), row.get("ipae"), row.get("rmsd"), row["wall"], error and error[:200], flush=True)
finally:
    stop(); srv.kill()
