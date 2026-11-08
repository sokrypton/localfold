"""Fold in the real page in the REAL Safari, through safaridriver's WebDriver HTTP API.

    python3 tools/fold-in-safari.py --msa search --repeat 2
    python3 tools/fold-in-safari.py --url https://localfold.org/ --msa search

The Safari twin of fold-in-page.py's press-Fold path: it sets the controls the page's own way, presses Fold, and
reports the browser (WebGPU? WebGL2? CompressionStream?) and, per fold, when each status line first appeared. Safari
16.3 has no WebGPU and no DecompressionStream, so this is the one check of the page's two fallbacks in the browser
they are for - and the one that found a deep MMseqs2 search refused there ("the WebGL2 fold does not draw from an
alignment (8077 rows)"). Needs Safari's "Allow Remote Automation" and `safaridriver --enable` once.

Serves <repo> with fold-in-page.py's own server (local weights, no-store source), opens
the page in Safari, sets the controls the way fold-in-page.py does, presses Fold and
reports the browser's capabilities and the final status line.
"""
import argparse, importlib.util, json, os, re, subprocess, sys, time, urllib.request

p = argparse.ArgumentParser()
p.add_argument("--msa", default="none")
p.add_argument("--model", default="af3")
p.add_argument("--steps", default="4")
p.add_argument("--timeout", type=float, default=600)
p.add_argument("--url", default=None)
p.add_argument("--repeat", type=int, default=1)
p.add_argument("--delay", type=float, default=0, help="seconds to wait after the page is ready, as a reader typing would")
p.add_argument("--sequence", default="GWSTELEKHREELKEFLKKEGITLGFTNAEKQEQAQKLGLGKKVSPELLIKAFAILKK")
a = p.parse_args()
a.repo = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

spec = importlib.util.spec_from_file_location("fip", os.path.join(a.repo, "tools", "fold-in-page.py"))
fip = importlib.util.module_from_spec(spec)
sys.path.insert(0, os.path.join(a.repo, "tools"))
spec.loader.exec_module(fip)
httpd = fip.serve(local_weights=True)
URL = a.url or f"http://127.0.0.1:{fip.PORT}/index.html"

DPORT = 4447
drv = subprocess.Popen(["safaridriver", "--port", str(DPORT)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
time.sleep(1.5)


def call(method, path, body=None):
    req = urllib.request.Request(f"http://127.0.0.1:{DPORT}{path}", method=method,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read() or b"{}").get("value")
    except urllib.error.HTTPError as e:
        raise SystemExit(f"webdriver {method} {path}: {e.code} {e.read()[:400]!r}")


sid = None
try:
    sid = call("POST", "/session", {"capabilities": {"alwaysMatch": {"browserName": "safari"}}})["sessionId"]
    js = lambda src, *args: call("POST", f"/session/{sid}/execute/sync", {"script": src, "args": list(args)})
    call("POST", f"/session/{sid}/url", {"url": URL})
    t0 = time.time()
    while time.time() - t0 < 60:
        if js("return !!document.getElementById('predict') && /Ready|Paste/i.test(document.getElementById('status-message')?.textContent||'')"):
            break
        time.sleep(0.5)
    if a.delay > 0:
        time.sleep(a.delay)
    print("browser:", json.dumps(js("""return {
        ua: navigator.userAgent,
        webgpu: 'gpu' in navigator,
        webgl2: !!document.createElement('canvas').getContext('webgl2'),
        decompressionStream: typeof DecompressionStream,
        compressionStream: typeof CompressionStream,
        status: document.getElementById('status-message')?.textContent };""")))
    js("""
      const seq = arguments[0];
      const field = document.querySelector('.entity-field [contenteditable], .entity-field textarea, .entity-field input');
      if ('value' in field && field.tagName !== 'DIV') field.value = seq; else field.textContent = seq;
      field.dispatchEvent(new Event('input', { bubbles: true }));
      const set = (id, v) => { const el = document.getElementById(id); if (!el) return;
        el.value = v; el.dispatchEvent(new Event('change', { bubbles: true })); };
      set('model-family', arguments[1]); set('recycles', '1'); set('msa-mode', arguments[2]);
      set('af3-count', arguments[3]);
      try { localStorage.setItem('localfold.modelTerms.alphafold3', 'accepted'); } catch (e) {}
    """, a.sequence, a.model, a.msa, a.steps)
    time.sleep(0.5)
    print("controls:", json.dumps(js("""const v = id => (document.getElementById(id)||{}).value;
        return {model: v('model-family'), msa: v('msa-mode'), steps: v('af3-count')};""")))
    js("window.__statusLog = []; const el = document.getElementById('status-message');"
       "new MutationObserver(() => { const t = el.textContent; const l = window.__statusLog;"
       " if (!l.length || l[l.length-1][1] !== t) l.push([Math.round(performance.now() - (window.__t0 ?? 0)), t]); })"
       ".observe(el, {childList: true, subtree: true, characterData: true});")
    for fold in range(a.repeat):
        js("window.__statusLog.length = 0; window.__t0 = performance.now(); document.getElementById('predict').click();")
        t0, last = time.time(), ""
        while time.time() - t0 < a.timeout:
            st = js("""const s = document.getElementById('status-message');
                return {text: s ? s.textContent : '', error: !!(s && s.classList.contains('error')),
                        downloads: (document.getElementById('downloads')||{style:{}}).style.display};""")
            last = st["text"]
            if st["error"] or (st["downloads"] != "none" and any(w in last for w in ("pLDDT", "Done"))
                               and time.time() - t0 > 2):
                break
            time.sleep(0.5)
        log = js("return window.__statusLog.slice();")
        print(f"fold {fold + 1}: {last}")
        # one line a stage: when each status shape first appeared, in ms from the click
        seen = {}
        for at, text in log:
            key = re.sub(r"[0-9]+", "#", text)
            seen.setdefault(key, at)
        for key, at in seen.items():
            print(f"   {at:7d} ms  {key}")
finally:
    if sid:
        try: call("DELETE", f"/session/{sid}")
        except SystemExit: pass
    drv.terminate()
    httpd.shutdown()
