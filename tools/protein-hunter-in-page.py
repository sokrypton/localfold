"""Drive a REAL design run in proteinhunter.html, and check what it produced.

    python3 tools/protein-hunter-in-page.py                  # monomer, 2 cycles
    python3 tools/protein-hunter-in-page.py --target MKV...  # a binder
    python3 tools/protein-hunter-in-page.py --remote-weights # from Hugging Face

🔴 IT EXISTS BECAUSE THE CPU TESTS CANNOT SEE THE LOOP CLOSE. test/hunter-loop
drives the schedule with stub functions and test/mpnn-bridge designs off a
fixed PDB, so between them they prove that the loop's arithmetic is right and
that MPNN reads a structure. Neither can say that AF3's PDB is one MPNN
accepts, that the designed sequence is one AF3 will fold, or that the two
agree about which chain is which - which is the entire join, and the only
place it exists is the page.

So this presses Hunt and reads the table back: one row per cycle, a sequence
that actually changed between them, and a finite score. If cycle 1's sequence
still contains X the design step did nothing; if it equals cycle 0's the
structure never reached MPNN.

🔴 THE WEIGHTS COME OFF THE DISK BY DEFAULT, WHICH IS NOT THE PAGE'S OWN
BEHAVIOUR. src/reference/manifests/index.js gives the af3 bundle a `remote`,
and a deployed page is right to use it - but a check that runs on demand
should not spend 150 MB of somebody's connection every time. The local server
below rewrites that ONE FILE as it serves it, dropping the `remote:` line so
`bundleBaseUrl` falls back to ./model-af3-int5/. Nothing on disk changes: this
is the harness choosing what to serve, which is what serving locally is for.
Pass --remote-weights to exercise the real path.
"""
import argparse
import http.server
import json
import os
import re
import socketserver
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cdp                                                   # noqa: E402

PORT, DBG = 9668, 9231
REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MANIFESTS = "/src/reference/manifests/index.js"
REMOTE_LINE = re.compile(rb'^\s*remote:\s*"[^"]*",\s*$', re.MULTILINE)


def serve(local_weights):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=REPO, **kw)

        def log_message(self, *a):
            pass

        def do_GET(self):
            if local_weights and self.path.split("?")[0] == MANIFESTS:
                source = open(os.path.join(REPO, MANIFESTS.lstrip("/")), "rb").read()
                body = REMOTE_LINE.sub(b"", source)
                self.send_response(200)
                self.send_header("Content-Type", "text/javascript")
                self.send_header("Content-Length", str(len(body)))
                # 🔴 AND NO CACHING, or the second run of this tool serves the
                # UNPATCHED module from Chrome's heuristic cache and quietly
                # goes to the network anyway. A fresh profile per launch makes
                # that unlikely; saying so makes it impossible.
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", PORT), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


SET_CONTROLS = """(() => {
  const set = (id, value) => {
    const el = document.getElementById(id);
    if (!el) return `missing #${id}`;
    if (el.type === 'checkbox') el.checked = Boolean(value);
    else el.value = String(value);
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
    return null;
  };
  const problems = [%s].map(([id, value]) => set(id, value)).filter(Boolean);
  return JSON.stringify(problems);
})()"""

READ_TABLE = """(() => {
  const rows = [...document.querySelectorAll('#results-body tr')].map((row) => {
    const cells = [...row.children].map((cell) => cell.textContent);
    return { run: cells[0], cycle: cells[1], score: cells[2], objective: cells[3],
             plddt: cells[4], iptm: cells[5], alanine: cells[6], sequence: cells[7],
             best: row.className === 'best' };
  });
  return JSON.stringify(rows);
})()"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="",
                        help="the target chain(s); empty hallucinates a monomer")
    parser.add_argument("--length", type=int, default=24,
                        help="the designed chain's length, both ends of the range")
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8, help="AF3 sampler steps")
    parser.add_argument("--percent-x", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alanine-bias", action="store_true",
                        help="ramp the alanine bias, as the reference does")
    parser.add_argument("--remote-weights", action="store_true",
                        help="fetch the AF3 bundle from its pinned remote"
                             " (~150 MB) instead of ./model-af3-int5/")
    parser.add_argument("--url", default=None,
                        help="drive a DEPLOYED page instead of this checkout")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    local = not args.remote_weights and args.url is None
    if local and not os.path.isdir(os.path.join(REPO, "model-af3-int5")):
        print("model-af3-int5/ is not in this checkout;"
              " re-run with --remote-weights", file=sys.stderr)
        return 1

    httpd = serve(local)
    proc, ws = cdp.launch(DBG, "/tmp/_cdp_hunter_profile")
    failures = []
    try:
        ws.call("Page.enable")
        ws.call("Page.navigate",
                url=args.url or ("http://127.0.0.1:%d/proteinhunter.html" % PORT))
        # 🔴 THE BUTTON BEING ENABLED IS NOT THE SIGNAL HERE - it never is
        # disabled on this page. web/hunter.js is a module, so it runs after
        # py2Dmol's blocking script; until its listener is bound a click does
        # nothing at all, silently. The status line it writes is the proof it
        # ran, and `__hunterReady` is set by nothing, so wait on the listener
        # having replaced the initial text is wrong too - it has not yet.
        # What IS true once the module has run: #results exists and the
        # length-range block has been synced.
        cdp.wait_for(ws, "document.getElementById('results') !== null"
                         " && document.getElementById('length-range') !== null"
                         " && typeof window.py2Dmol !== 'undefined'",
                     what="the page to finish loading")
        time.sleep(0.5)

        controls = [
            ("target", args.target),
            ("start-sequence", ""),
            ("min-length", args.length),
            ("max-length", args.length),
            ("runs", args.runs),
            ("cycles", args.cycles),
            ("percent-x", args.percent_x),
            ("steps", args.steps),
            ("seed", args.seed),
            ("alanine-bias", args.alanine_bias),
        ]
        problems = cdp.evaluate(ws, SET_CONTROLS % ", ".join(
            "[%s, %s]" % (json.dumps(k), json.dumps(v)) for k, v in controls))
        if json.loads(problems):
            print("controls:", problems, file=sys.stderr)
            return 1
        print("controls: ok ·", ", ".join("%s=%s" % (k, v) for k, v in controls))
        print("weights :", "./model-af3-int5/" if local else "the pinned remote")

        cdp.evaluate(ws, "document.getElementById('hunt').click()")
        cdp.wait_for(ws, """(() => {
          const text = (document.getElementById('status-message') || {}).textContent || '';
          return /^Done in|^Stopped after/.test(text) ||
            document.getElementById('status-message').classList.contains('error');
        })()""", timeout=args.timeout, what="the hunt to finish")
        time.sleep(0.5)

        status = cdp.evaluate(
            ws, "(document.getElementById('status-message')||{}).textContent")
        errored = cdp.evaluate(
            ws, "document.getElementById('status-message').classList.contains('error')")
        print("status  :", status)
        if errored:
            return 1

        rows = json.loads(cdp.evaluate(ws, READ_TABLE))
        for row in rows:
            print("row     : run %s cycle %s  %s %s  plddt %s  ala %s  %s%s"
                  % (row["run"], row["cycle"], row["objective"], row["score"],
                     row["plddt"], row["alanine"], row["sequence"][:44],
                     "  <- best" if row["best"] else ""))

        expected = args.runs * (args.cycles + 1)
        if len(rows) != expected:
            failures.append("%d rows, expected %d" % (len(rows), expected))
        # 🔴 THE ASSERTIONS ARE ABOUT THE JOIN, NOT ABOUT THE DESIGN. Whether
        # the binder is any good is a question for a benchmark; whether the two
        # models reached each other is answerable here, and these are the three
        # ways it has failed while everything else passed.
        if rows:
            if "X" in rows[0]["sequence"] and args.percent_x == 0:
                failures.append("cycle 0 has X at percent-x 0")
            for row in rows[1:]:
                if "X" in row["sequence"]:
                    failures.append("cycle %s still contains X: MPNN designed nothing"
                                    % row["cycle"])
                    break
            changed = {row["sequence"] for row in rows}
            if len(changed) == 1 and len(rows) > 1:
                failures.append("every cycle has the same sequence:"
                                " the structure never reached the designer")
            for row in rows:
                try:
                    float(row["score"])
                except ValueError:
                    failures.append("cycle %s has no score (%r)"
                                    % (row["cycle"], row["score"]))
                    break
            if sum(1 for row in rows if row["best"]) != 1:
                failures.append("%d rows marked best, expected 1"
                                % sum(1 for row in rows if row["best"]))

        print("frames  :", cdp.evaluate(ws, """(() => {
          const reg = window.py2dmol_viewers || {};
          const key = Object.keys(reg)[0];
          const v = key && reg[key].renderer;
          if (!v) return 'no viewer';
          const frames = v.objectsData[v.currentObjectName].frames;
          return JSON.stringify({ object: v.currentObjectName, frames: frames.length });
        })()"""))
        print("download:", cdp.evaluate(
            ws, "!document.getElementById('downloads').hidden"))
    finally:
        proc.kill()
        httpd.shutdown()

    for failure in failures:
        print("FAIL    :", failure, file=sys.stderr)
    if failures:
        return 1
    print("ok      : the loop closed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
