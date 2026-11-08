"""Run a tools/gpu module's `main(null, args)` in the REAL Safari, through safaridriver, and print what it returns.

    python3 tools/run-in-safari.py tools/gpu/probe-webgl-uniform-budget.js
    python3 tools/run-in-safari.py tools/gpu/fold-af3-webgl.js --a3m=/tools/fixtures/test.a3m --no-reference

🔴 gpu-chrome.mjs IS CHROME, AND CHROME IS NOT WHAT THE WebGL2 PATH IS FOR. Its readers are on Safari before 26 -
no WebGPU - and Safari's WebGL is WebKit's own ANGLE over Metal, which differs in ways no Chrome run can show: it
refuses `readPixels` as FLOAT from an RGBA16F attachment, and it draws with under 16 KiB of uniform blocks while
reporting 192 (probe-webgl-uniform-budget.js) - a limit that made a probe report a 2.22x speedup from draws that never
ran. So a module's `device` is null here (Safari 16.3 has no WebGPU) and its result comes back as JSON from
tools/safari-run.html. Needs Safari's "Allow Remote Automation" (Develop menu) and `safaridriver --enable` once.
"""
import importlib.util, json, os, subprocess, sys, time, urllib.parse, urllib.request
REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
module, args = sys.argv[1], sys.argv[2:]
repo = REPO
spec = importlib.util.spec_from_file_location("fip", os.path.join(repo, "tools", "fold-in-page.py"))
fip = importlib.util.module_from_spec(spec); sys.path.insert(0, os.path.join(repo, "tools")); spec.loader.exec_module(fip)
httpd = fip.serve(local_weights=False)
query = urllib.parse.urlencode([("module", "/" + module.lstrip("/"))] + [("arg", a) for a in args])
url = f"http://127.0.0.1:{fip.PORT}/tools/safari-run.html?{query}"
drv = subprocess.Popen(["safaridriver", "--port", "4448"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1.5)
def call(method, path, body=None):
    req = urllib.request.Request(f"http://127.0.0.1:4448{path}", method=method,
        data=None if body is None else json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r: return json.loads(r.read() or b"{}").get("value")
sid = None
try:
    sid = call("POST", "/session", {"capabilities": {"alwaysMatch": {"browserName": "safari"}}})["sessionId"]
    call("POST", f"/session/{sid}/url", {"url": url})
    t0 = time.time()
    while time.time() - t0 < 1200:
        r = call("POST", f"/session/{sid}/execute/sync", {"script": "return window.__result;", "args": []})
        if r: print(r); break
        time.sleep(1)
finally:
    if sid: call("DELETE", f"/session/{sid}")
    drv.terminate(); httpd.shutdown()
