#!/usr/bin/env python3
"""Does the weight-download dial only ever go FORWARDS?

    python3 tools/download-dial.py                 # AF2-mono, model 2
    python3 tools/download-dial.py --number 4 --model multimer

🔴 A DELTA FAMILY DOWNLOADS TWO BUNDLES AND THE DIAL IS ONE. AlphaFold 2's
models 2 to 5 are 43 MiB of difference added to a 73 MiB base, and handing each
store the caller's `onProgress` lets them take turns owning the arc: it reads
"4 of 43 MiB", then "20 of 73", then back. Reported by a visitor as a flicker,
and only ever visible when the base is NOT already cached - which is exactly the
visitor who picks model 2 before ever running model 1.

WHAT IT MEASURES. Every change to `#model-load` while the weights load, with the
byte counts parsed out of the label. Two things fail it: a loaded count that
DECREASES, and more than one TOTAL - the second is the subtler half, because the
arc snaps back when the denominator grows even though the numerator did not.

🔴 AND IT IS WATCHED FAILING. With `mine = onProgress` put back in
web/model.js - both stores on one callback - it reports **273 backwards steps
across totals of 43 and 73 MiB**, and ends at "43 / 43" because the smaller
download finishes last and overwrites. Fixed, it is one total from the first
update to the last.

🔴 AND IT WAITS A FIXED WINDOW RATHER THAN FOR A WORD IN THE STATUS LINE. What
is being watched finishes long before the fold does, and polling for "Done" hung
this probe past its own timeout twice.
"""
import argparse, http.server, json, os, re, socketserver, sys, threading, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cdp import launch, evaluate  # noqa: E402

PORT, DBG = 9678, 9243
SAMPLE = """(() => {
  const set = (id, v) => { const e = document.getElementById(id); if (!e) return;
    e.value = v; e.dispatchEvent(new Event('change', { bubbles: true })); };
  set('model-family', %s);
  set('af2Model', %s);
  set('msa-mode', 'none');
  window.__dial = [];
  const node = document.getElementById('model-load');
  new MutationObserver(() => {
    const label = document.getElementById('model-load-text');
    window.__dial.push({ text: (label || {}).textContent || '' });
  }).observe(node, { attributes: true, subtree: true, childList: true, characterData: true });
  document.getElementById('predict').click();
})()"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="monomer", help="the #model-family value")
    parser.add_argument("--number", default="2", help="the AF2 model number")
    parser.add_argument("--seconds", type=int, default=45)
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    httpd = socketserver.ThreadingTCPServer(
        ("127.0.0.1", PORT), http.server.SimpleHTTPRequestHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    process, ws = launch(DBG, "/tmp/download-dial-profile")
    try:
        ws.call("Page.enable")
        ws.call("Page.navigate", url=f"http://127.0.0.1:{PORT}/index.html")
        time.sleep(3)
        evaluate(ws, SAMPLE % (json.dumps(args.model), json.dumps(args.number)), False)
        time.sleep(args.seconds)
        samples = json.loads(evaluate(ws, "JSON.stringify(window.__dial || [])", False))
    finally:
        httpd.shutdown()
        process.kill()

    seen = []
    for sample in samples:
        found = re.search(r"([\d.]+)\s*/\s*([\d.]+)\s*MiB", sample.get("text") or "")
        if found:
            seen.append((float(found.group(1)), float(found.group(2))))
    if not seen:
        print("the dial never showed a byte count - nothing to judge", file=sys.stderr)
        return 1
    backwards = sum(1 for (a, _), (b, _) in zip(seen, seen[1:]) if b < a - 0.01)
    totals = sorted({total for _, total in seen})
    print(f"{args.model} model {args.number}: {len(seen)} updates with a byte count,"
          f" first {seen[0][0]:.0f}/{seen[0][1]:.0f} MiB, last {seen[-1][0]:.0f}/{seen[-1][1]:.0f}")
    print(f"  totals offered: {totals}")
    print(f"  backwards steps: {backwards}")
    if backwards or len(totals) > 1:
        print("🔴 the dial flickers: two stores are reporting into one arc", file=sys.stderr)
        return 1
    print("one total, never backwards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
