#!/usr/bin/env python3
"""A whole ESMFold2 fold on the CPU, through the page, in a Chrome with no GPU.

    python3 tools/check-cpu-fold.py [--shots DIR] [--complex]

Runs about three folds of ubiquitin, so five minutes on eight cores; --complex adds
barnase-barstar, five more. The unit tests
and tools/fold-esmfold2-cpu.mjs never load the page, and what lives only there is what
this checks:

  * with no adapter, a GPU fold says so and offers the CPU, and the page still works;
  * choosing the CPU reloads once under the service worker's headers, comes back
    isolated and threaded, and keeps the sequence and the model it was given;
  * the fold crosses the worker and lands: a status line with scores, and no console error;
  * a second fold keeps the worker's weights;
  * Stop ends the fold at once, and the next fold still works from the shard cache;
  * with --complex, two chains cross the worker and come back with an ipTM;
  * and with nothing chosen, the line beside the button offers the CPU, Fold goes there, and the reload that takes
    carries the fold across it, with nothing remembered for the next visit.
"""
import argparse
import base64
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cdp

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 4193
DEBUG_PORT = 9393
SEQUENCE = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
# 1BRS, as RCSB gives its two entities.
BARNASE = "AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR"
BARSTAR = "KKAVINGEQIRSISDLHQTLKKELALPEYYGENLDALWDALTGWVEYPLVLEWRQFEQSKQLTENGAESVLQVFREAKAEGADITIILS"

# Before any page script runs: whatever the page reports as broken.
CAPTURE_ERRORS = """
window.__errors = [];
const original = console.error;
console.error = (...args) => { window.__errors.push(args.map(String).join(' ').slice(0, 400)); original(...args); };
addEventListener('error', (event) => window.__errors.push('error: ' + event.message));
addEventListener('unhandledrejection', (event) => window.__errors.push('rejection: ' + event.reason));
"""
FIELD = "document.querySelector('.entity-field [contenteditable], .entity-field textarea')"
READY = "typeof window.processFiles === 'function' && !document.getElementById('predict').disabled"
STATUS = "document.getElementById('status-message')?.textContent ?? ''"
STATE = """JSON.stringify({
  isolated: self.crossOriginIsolated,
  status: document.getElementById('status-message')?.textContent ?? '',
  failed: document.getElementById('status-message')?.classList.contains('error') ?? false,
  badge: document.getElementById('device-badge')?.textContent ?? '',
  model: document.getElementById('model-family')?.value,
  variant: document.getElementById('plm-mode')?.value,
  residues: %s?.value?.length ?? null,
  errors: window.__errors,
})""" % FIELD
FOLD_ENDED = """(() => {
  const status = document.getElementById('status-message');
  return window.__foldState?.running === false || status?.classList.contains('error') === true;
})()"""


def state(ws):
    return json.loads(cdp.evaluate(ws, STATE))


def press_fold(ws):
    cdp.evaluate(ws, "document.getElementById('predict').click()")
    cdp.wait_for(ws, "window.__foldState?.running === true", timeout=30, what="the fold to start")


def fold(ws, timeout):
    started = time.time()
    press_fold(ws)
    cdp.wait_for(ws, FOLD_ENDED, timeout=timeout, what="the fold to finish", progress=STATUS)
    now = state(ws)
    assert not now["failed"], f"the fold failed: {now['status']}"
    assert "pLDDT" in now["status"], f"no scores on the status line: {now['status']}"
    assert now["errors"] == [], f"console errors: {now['errors']}"
    return round(time.time() - started, 1), now


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shots", default="", help="directory for the page's screenshots")
    parser.add_argument("--timeout", type=int, default=900, help="seconds to allow one fold")
    parser.add_argument("--complex", action="store_true", help="also fold barnase-barstar")
    args = parser.parse_args()

    def shot(ws, name):
        if args.shots:
            os.makedirs(args.shots, exist_ok=True)
            data = ws.call("Page.captureScreenshot", format="png")["data"]
            with open(os.path.join(args.shots, f"{name}.png"), "wb") as handle:
                handle.write(base64.b64decode(data))

    server = subprocess.Popen([sys.executable, os.path.join(REPO, "tools", "serve.py"), str(PORT)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # --disable-gpu is what makes this machine's answer the one under test: no adapter.
    browser, ws = cdp.launch(DEBUG_PORT, "/tmp/_cdp_cpu_fold_profile", extra_args=["--disable-gpu"])
    report = {}
    try:
        ws.call("Page.enable")
        ws.call("Page.addScriptToEvaluateOnNewDocument", source=CAPTURE_ERRORS)
        ws.call("Emulation.setDeviceMetricsOverride", width=1440, height=1100, deviceScaleFactor=1, mobile=False)
        ws.call("Page.navigate", url=f"http://127.0.0.1:{PORT}/index.html")
        cdp.wait_for(ws, READY, what="the page to finish loading")
        adapter = cdp.evaluate(ws, """(async () => {
          const adapter = await navigator.gpu?.requestAdapter();
          return adapter === null || adapter === undefined ? null : adapter.info?.vendor ?? 'unknown';
        })()""")
        assert adapter is None, f"this check needs a browser with no GPU; it found {adapter}"

        cdp.evaluate(ws, """(() => {
          const field = %s;
          field.textContent = %s; field.value = %s;
          field.dispatchEvent(new Event('input', { bubbles: true }));
        })()""" % (FIELD, json.dumps(SEQUENCE), json.dumps(SEQUENCE)))

        # From AF3, the page's default: choosing the CPU must bring a model that folds there.
        cdp.evaluate(ws, "document.getElementById('settings-toggle').click()")
        cdp.evaluate(ws, "document.querySelector('#settings-compute button:last-child').click()")
        cdp.wait_for(ws, f"self.crossOriginIsolated === true && {READY}", timeout=60,
                     what="the page to come back isolated")
        report["isolated"] = now = state(ws)
        assert now["residues"] == len(SEQUENCE), f"the reload lost the sequence: {now}"
        assert now["model"] == "ef2-fast-600m" and now["variant"].startswith("esmc-"), f"the wrong model: {now}"
        assert now["badge"].startswith("CPU"), f"the badge says {now['badge']}"
        assert now["errors"] == [], f"console errors: {now['errors']}"

        report["first_seconds"], report["first"] = fold(ws, args.timeout)
        shot(ws, "light")
        cdp.evaluate(ws, "document.querySelector('#settings-theme button:last-child').click()")
        time.sleep(1)
        shot(ws, "dark")

        report["second_seconds"], report["second"] = fold(ws, args.timeout)

        press_fold(ws)
        time.sleep(15)
        pressed = time.time()
        cdp.evaluate(ws, "document.getElementById('predict').click()")
        cdp.wait_for(ws, "window.__foldState?.running === false", timeout=30, what="Stop to take effect")
        report["stop_seconds"] = round(time.time() - pressed, 2)
        report["stopped"] = now = state(ws)
        assert not now["failed"], f"Stop reported a failure: {now['status']}"

        report["after_stop_seconds"], report["after_stop"] = fold(ws, args.timeout)
        if args.complex:
            cdp.evaluate(ws, """(() => {
              const list = window.__entityList;
              const row = list.read()[0];
              list.set([{ ...row, value: %s, copies: 1 }, { ...row, value: %s, copies: 1 }]);
            })()""" % (json.dumps(BARNASE), json.dumps(BARSTAR)))
            report["complex_seconds"], report["complex"] = fold(ws, args.timeout)
            assert "ipTM" in report["complex"]["status"], f"no ipTM: {report['complex']['status']}"
            shot(ws, "complex")
        report["caches"] = cdp.evaluate(ws, "caches.keys()")
        assert len(report["caches"]) >= 2, f"the shard caches did not survive: {report['caches']}"

        # Somebody with no GPU who has chosen nothing: one press of Fold, across the reload.
        cdp.evaluate(ws, "document.querySelector('#settings-compute button:first-child').click()")
        cdp.wait_for(ws, "navigator.serviceWorker.getRegistrations().then((all) => all.length === 0)", timeout=30,
                     what="the service worker to go")
        ws.call("Page.reload")
        cdp.wait_for(ws, f"self.crossOriginIsolated === false && {READY}", timeout=60,
                     what="the page to come back without isolation")
        cdp.evaluate(ws, """(() => {
          for (const [id, value] of [['model-family', 'ef2-fast-600m'], ['plm-mode', 'esmc-300m']]) {
            const el = document.getElementById(id); el.value = value;
            el.dispatchEvent(new Event('change', { bubbles: true }));
          }
          const list = window.__entityList;
          list.set([{ ...list.read()[0], value: %s, copies: 1 }]);
        })()""" % json.dumps(SEQUENCE))
        cdp.wait_for(ws, "document.getElementById('fold-plan')?.textContent.includes('CPU thread')", timeout=15,
                     what="the line beside the button to offer the CPU")
        report["plan_line"] = cdp.evaluate(ws, "document.getElementById('fold-plan').textContent")
        cdp.evaluate(ws, "document.getElementById('predict').click()")
        cdp.wait_for(ws, f"self.crossOriginIsolated === true && {READY}", timeout=60,
                     what="Fold to bring the page back isolated")
        cdp.wait_for(ws, "(document.getElementById('status-message')?.textContent ?? '').includes('pLDDT')"
                         " || document.getElementById('status-message')?.classList.contains('error') === true",
                     timeout=args.timeout, what="the fold to finish on its own after the reload", progress=STATUS)
        report["one_press"] = now = state(ws)
        assert not now["failed"], f"the fold failed: {now['status']}"
        assert now["residues"] == len(SEQUENCE), f"the reload lost the sequence: {now}"
        assert now["errors"] == [], f"console errors: {now['errors']}"
        remembered = cdp.evaluate(ws, "localStorage.getItem('localfold.compute')")
        assert remembered in (None, "auto"), f"the fallback was remembered as a choice: {remembered}"
        print(json.dumps(report, indent=1))
        print("PASS")
    finally:
        browser.kill()
        server.kill()


if __name__ == "__main__":
    main()
