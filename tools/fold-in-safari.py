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
p.add_argument("--recycles", default="1", help="the recycle row (the page's own default is 3)")
p.add_argument("--remote-weights", action="store_true",
               help="the registry's published bundles rather than this checkout's exports (as fold-in-page.py's)")
p.add_argument("--bioemu2-samples", default=None, help="BioEmu2's sample count (the page's default is 1)")
p.add_argument("--stop-after", type=float, default=0,
               help="press Stop (the Fold button again) this many seconds into each fold, to check what a stop keeps")
p.add_argument("--delay", type=float, default=0, help="seconds to wait after the page is ready, as a reader typing would")
p.add_argument("--sequence", default="GWSTELEKHREELKEFLKKEGITLGFTNAEKQEQAQKLGLGKKVSPELLIKAFAILKK")
a = p.parse_args()
a.repo = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

spec = importlib.util.spec_from_file_location("fip", os.path.join(a.repo, "tools", "fold-in-page.py"))
fip = importlib.util.module_from_spec(spec)
sys.path.insert(0, os.path.join(a.repo, "tools"))
spec.loader.exec_module(fip)
httpd = fip.serve(local_weights=not a.remote_weights)
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
      set('model-family', arguments[1]); set('recycles', arguments[4]); set('msa-mode', arguments[2]);
      if (arguments[5] !== null) set('bioemu2-samples', arguments[5]);
      set('af3-count', arguments[3]);
      try { localStorage.setItem('localfold.modelTerms.alphafold3', 'accepted'); } catch (e) {}
    """, a.sequence, a.model, a.msa, a.steps, a.recycles, a.bioemu2_samples)
    time.sleep(0.5)
    print("controls:", json.dumps(js("""const v = id => (document.getElementById(id)||{}).value;
        return {model: v('model-family'), msa: v('msa-mode'), steps: v('af3-count'),
                recycles: v('recycles'), samples: v('bioemu2-samples')};""")))
    js("window.__statusLog = []; const el = document.getElementById('status-message');"
       "new MutationObserver(() => { const t = el.textContent; const l = window.__statusLog;"
       " if (!l.length || l[l.length-1][1] !== t) l.push([Math.round(performance.now() - (window.__t0 ?? 0)), t]); })"
       ".observe(el, {childList: true, subtree: true, characterData: true});")
    for fold in range(a.repeat):
        js("window.__statusLog.length = 0; window.__t0 = performance.now(); document.getElementById('predict').click();")
        t0, last, pressed = time.time(), "", False
        while time.time() - t0 < a.timeout:
            if a.stop_after > 0 and not pressed and time.time() - t0 > a.stop_after:
                js("document.getElementById('predict').click();")
                pressed = True
                print(f"   stop pressed at {round((time.time() - t0) * 1000)} ms")
            st = js("""const s = document.getElementById('status-message');
                return {text: s ? s.textContent : '', error: !!(s && s.classList.contains('error')),
                        downloads: (document.getElementById('downloads')||{style:{}}).style.display};""")
            last = st["text"]
            # (a BioEmu2 fold has no pLDDT: its finished line ends "... · <n> steps · ... · <seconds>s")
            finished = any(w in last for w in ("pLDDT", "Done")) or re.search(r" steps · .*[0-9]s\b", last)
            if st["error"] or (st["downloads"] != "none" and finished and time.time() - t0 > 2):
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
        # ...and every progress reading the reader saw, in order: how smoothly the bar moved
        steps = [(at, m.group(1)) for at, text in log for m in [re.search(r"Folding[^0-9]*([0-9]+)%", text)] if m]
        if steps:
            print("   progress:", " ".join(f"{p}%@{at}" for at, p in steps))
        # ...and a BioEmu2 fold's, which says its trunk and its sampler step rather than a percentage
        steps = [(at, m.group(1)) for at, text in log for m in [re.search(r"sampling(?: batch [^·]*·[^·]*·)? (?:step )?([0-9]+/[0-9]+)", text)] if m]
        if steps:
            print("   sampler:", " ".join(f"{p}@{at}" for at, p in steps))
finally:
    if sid:
        try: call("DELETE", f"/session/{sid}")
        except SystemExit: pass
    drv.terminate()
    httpd.shutdown()
