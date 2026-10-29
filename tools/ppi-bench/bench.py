# Drives design.html in headless Chrome through Protein Hunter attempts on the RFD4-Proteina PPI benchmark.
# usage: bench.py <out.jsonl> <config.json> <target>:<attempt> ...   (one browser; run several in parallel)
import json, os, random, socket, subprocess, sys, time
ROOT = os.path.expanduser("~/localfold"); sys.path.insert(0, os.path.join(ROOT, "tools")); import cdp
HERE = os.path.dirname(os.path.abspath(__file__))
def free():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p
out, config = sys.argv[1], json.load(open(sys.argv[2]))
jobs = [a.split(":") for a in sys.argv[3:]]
targets = json.load(open(os.path.join(HERE, "targets.json")))
done = set()
if os.path.exists(out):
    for line in open(out):
        d = json.loads(line); done.add((d["target"], d["attempt"]))
bench_js = open(os.path.join(HERE, "bench.js")).read()
port = free()
srv = subprocess.Popen([sys.executable, os.path.join(ROOT, "tools/serve.py"), str(port)], cwd=ROOT,
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
        rng = random.Random(f"{config['tag']}:{name}:{attempt}")
        lo, hi = t["binder"]
        hi = min(hi, lo + config.get("lengthSpan", 10**9))
        cfg = dict(config, target=t, length=rng.randint(lo, hi), seed=1000 * attempt + 7,
                   evalSeed=rng.randint(1, 2**30))
        cfg.pop("tag", None); cfg.pop("lengthSpan", None)
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
        with open(out, "a") as f: f.write(json.dumps(row) + "\n")
        print(name, attempt, row.get("binderPtm"), row.get("ipae"), row.get("rmsd"), row["wall"], error and error[:200], flush=True)
finally:
    stop(); srv.kill()
