#!/usr/bin/env python3
"""Does the page tell a reader what a fold will cost, and act on it, before anything is downloaded?

    python3 tools/check-compute-plan.py [--shots DIR]

Runs no fold on a GPU. Every case below is read from the line beside the button, or from what a press of Fold does when
the plan refuses, and the sizes that would fold are never pressed, so this is safe on a display's own card. It reads what
this machine has, so the GPU cases say what they saw and skip nothing: on a machine with no adapter they are the no-GPU
case, and that is checked instead.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import cdp  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--shots", default="")
parser.add_argument("--port", type=int, default=4198)
parser.add_argument("--debug-port", type=int, default=9398)
parser.add_argument("--flags", default="--headless=new")
parser.add_argument("--profile", default="")
args = parser.parse_args()

UBI = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
REPO = Path(__file__).parent.parent


def sequence(size):
    return (UBI * (size // len(UBI) + 1))[:size]


def read_plan(ws):
    return json.loads(cdp.evaluate(ws, """JSON.stringify((() => {
      const line = document.getElementById('fold-plan');
      return { hidden: line.hidden, text: line.textContent, tone: line.dataset.tone, title: line.title };
    })())"""))


def set_rows(ws, size):
    cdp.evaluate(ws, "(() => { const list = window.__entityList; list.set([{ ...list.read()[0], value: %s, copies: 1 }]); })()"
                 % json.dumps(sequence(size)))
    time.sleep(0.4)


def set_control(ws, element, value):
    cdp.evaluate(ws, """(() => { const el = document.getElementById(%s); el.value = %s;
                                 el.dispatchEvent(new Event('change', { bubbles: true })); })()"""
                 % (json.dumps(element), json.dumps(value)))
    time.sleep(0.4)


def shot(ws, name):
    if args.shots:
        data = ws.call("Page.captureScreenshot", format="png")["data"]
        import base64
        Path(args.shots, name).write_bytes(base64.b64decode(data))


server = subprocess.Popen([sys.executable, str(REPO / "tools" / "serve.py"), str(args.port)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=REPO)
profile = args.profile or str(Path(os.environ.get("TEMP", "/tmp")) / f"plan-profile-{os.getpid()}")
browser = subprocess.Popen([cdp.chrome_binary(), *args.flags.split(), f"--user-data-dir={profile}", "--no-first-run",
                            f"--remote-debugging-port={args.debug_port}", "--window-size=1400,900", "about:blank"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
checks = []


def check(name, ok, seen):
    checks.append((name, ok, seen))
    print(("ok  " if ok else "FAIL"), name, "--", seen, flush=True)


try:
    ws = None
    for _ in range(120):
        try:
            pages = json.load(urllib.request.urlopen(f"http://127.0.0.1:{args.debug_port}/json/list"))
            ws = cdp.WS(next(p for p in pages if p["type"] == "page")["webSocketDebuggerUrl"])
            break
        except Exception:
            time.sleep(0.5)
    assert ws is not None, "no debugging port"
    ws.call("Page.enable")
    ws.call("Network.enable")
    ws.call("Page.navigate", url=f"http://127.0.0.1:{args.port}/index.html")
    cdp.wait_for(ws, "typeof window.processFiles === 'function' && !document.getElementById('predict').disabled",
                 what="the page", timeout=120)
    adapter = json.loads(cdp.evaluate(ws, """(async () => {
      const a = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
      return JSON.stringify(a ? { vendor: a.info.vendor, fallback: a.info.isFallbackAdapter === true } : null);
    })()"""))
    usable = adapter is not None and not adapter["fallback"]
    print("adapter", adapter, flush=True)

    set_control(ws, "model-family", "ef2-fast-600m")
    set_rows(ws, 300)
    plan = read_plan(ws)
    if usable:
        check("a fold that fits says what it holds and how long", not plan["hidden"] and "GiB" in plan["text"]
              and "about" in plan["text"] and plan["tone"] == "plain", plan["text"])
    else:
        check("no GPU: the line offers the CPU", "CPU thread" in plan["text"], plan["text"])
    shot(ws, "plan-fits.png")

    set_rows(ws, 2100)
    plan = read_plan(ws)
    if usable:
        check("past the binding ceiling the line warns", plan["tone"] == "warn" and "Too long" in plan["text"], plan["text"])
    shot(ws, "plan-too-long.png")

    if usable:
        # A limit in Settings is a card any size, and refusing it starts nothing.
        cdp.evaluate(ws, "document.getElementById('settings-toggle').click()")
        time.sleep(0.6)
        options = json.loads(cdp.evaluate(ws, """JSON.stringify([...document.getElementById('settings-gpu-limit').options]
          .map((o) => [o.value, o.textContent]))"""))
        check("Settings offers an automatic limit and sizes up to the card", options[0][0] == "" and len(options) > 1,
              str(options[:3]))
        shot(ws, "plan-settings.png")
        set_control(ws, "settings-gpu-limit", str(2 * 1024 ** 3))
        cdp.evaluate(ws, "document.getElementById('settings-toggle').click()")
        set_rows(ws, 1000)
        plan = read_plan(ws)
        check("a limit in Settings moves the plan", plan["tone"] == "warn" and "of 2.0 GiB" in plan["text"], plan["text"])
        shot(ws, "plan-over.png")

        before = float(cdp.evaluate(ws, "performance.getEntriesByType('resource').length"))
        cdp.evaluate(ws, "document.getElementById('predict').click()")
        time.sleep(1.0)
        state = json.loads(cdp.evaluate(ws, """JSON.stringify({
          running: window.__foldState?.running, status: document.getElementById('status-message')?.textContent ?? '',
          buttons: [...document.querySelectorAll('#status-message button')].map((b) => b.textContent),
          resident: window.__foldState?.since !== undefined })"""))
        check("Fold on a fold that does not fit stops and offers the ways on", state["running"] is False
              and "needs" in state["status"] and any("CPU" in b for b in state["buttons"])
              and any("anyway" in b for b in state["buttons"]), json.dumps(state)[:200])
        after = float(cdp.evaluate(ws, """performance.getEntriesByType('resource')
          .filter((r) => r.name.includes('huggingface')).length"""))
        check("and nothing was downloaded", after == 0, f"{after} model requests ({before} resources before)")
        shot(ws, "plan-refused.png")
        cdp.evaluate(ws, "document.getElementById('settings-toggle').click()")
        time.sleep(0.4)
        set_control(ws, "settings-gpu-limit", "")
        cdp.evaluate(ws, "document.getElementById('settings-toggle').click()")
        set_rows(ws, 300)
        plan = read_plan(ws)
        check("back to automatic, it fits again", plan["tone"] == "plain", plan["text"])

    # A model that folds on a GPU only: an estimate, or that it needs one.
    set_control(ws, "model-family", "boltz2")
    plan = read_plan(ws)
    if usable:
        check("another model says what it holds and how long", "GiB" in plan["text"] and "about" in plan["text"], plan["text"])
        set_rows(ws, 6000)
        plan = read_plan(ws)
        check("another model past what the card can hold says so", plan["tone"] == "warn" and "Needs" in plan["text"], plan["text"])
        set_rows(ws, 300)
    else:
        check("no GPU: another model says it folds on one only", plan["text"] == "Folds on a GPU only", plan["text"])
    set_control(ws, "model-family", "ef2-fast-600m")

    # A second tab waits its turn, and Stop ends the wait. Held from outside the page, so no fold runs.
    cdp.evaluate(ws, """(() => { window.__heldBy = new Promise(() => {});
      navigator.locks.request('localfold-fold', () => window.__heldBy); })()""")
    set_rows(ws, 76)
    cdp.evaluate(ws, "document.getElementById('predict').click()")
    time.sleep(0.5)
    waiting = cdp.evaluate(ws, "document.getElementById('status-message').textContent")
    check("a fold waits for the one in another tab", "another LocalFold tab" in waiting, waiting)
    cdp.evaluate(ws, "document.getElementById('predict').click()")
    time.sleep(0.5)
    state = json.loads(cdp.evaluate(ws, """JSON.stringify({ running: window.__foldState?.running,
      status: document.getElementById('status-message').textContent })"""))
    check("and Stop ends the wait", state["running"] is False and "stopped" in state["status"], str(state))
finally:
    browser.kill()
    server.kill()

failed = [name for name, ok, _ in checks if not ok]
print(f"{len(checks) - len(failed)}/{len(checks)} passed")
sys.exit(1 if failed else 0)
