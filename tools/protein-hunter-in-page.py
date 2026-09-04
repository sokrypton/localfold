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
                        help="comma-separated target entities; a bare sequence"
                             " is protein, or write dna=ATGC / rna=AUGC."
                             " Empty hallucinates a monomer.")
    parser.add_argument("--length", type=int, default=24,
                        help="the designed chain's length")
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--mode", default="flow", choices=["flow", "diffusion"],
                        help="which sampler the fold runs")
    parser.add_argument("--steps", type=int, default=16, help="AF3 sampler steps")
    parser.add_argument("--recycles", type=int, default=0)
    parser.add_argument("--percent-x", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--designer", default="auto",
                        help="auto, soluble, protein, ligand or na")
    parser.add_argument("--ligands", default="",
                        help="comma-separated CCD codes; each is fetched from"
                             " the PDB, which is a few KB per code")
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
        # ran, and `__hunterReady` is set by nothing, so waiting on the status
        # line to change is wrong too - it has not yet. What IS true once the
        # module has run: it has built the designer picker's options from the
        # registry, so that select has more than the one Auto option the HTML
        # ships with.
        cdp.wait_for(ws, "document.getElementById('results') !== null"
                         " && document.getElementById('designer').options.length > 1"
                         " && typeof window.py2Dmol !== 'undefined'",
                     what="the page to finish loading")
        time.sleep(0.5)

        # 🔴 THE TARGET IS ENTITY ROWS NOW, NOT A TEXTAREA, so it is set
        # through the list's own API rather than by writing a value. `set`
        # rebuilds the rows and notifies, which is what index.html's paste path
        # does too - poking a row's field instead would leave the model behind
        # the DOM, and the fold would run on whatever the model still held.
        if args.target or args.ligands:
            entities = []
            for field in [chunk.strip() for chunk in args.target.split(",")]:
                if not field:
                    continue
                kind, _, value = field.partition("=")
                if not value:
                    kind, value = "protein", kind
                entities.append({"type": kind.lower(), "value": value, "copies": 1})
            for code in [chunk.strip() for chunk in args.ligands.split(",")]:
                if code:
                    entities.append({"type": "ligand", "value": code, "copies": 1})
            written = cdp.evaluate(ws, """(() => {
              if (!window.__hunterTargets) return 'no entity list';
              window.__hunterTargets.set(%s);
              return JSON.stringify(window.__hunterTargets.read());
            })()""" % json.dumps(entities))
            print("entities:", written)

        controls = [
            ("start-sequence", ""),
            ("length", args.length),
            ("runs", args.runs),
            ("cycles", args.cycles),
            ("percent-x", args.percent_x),
            ("af3-mode", args.mode),
            ("steps", args.steps),
            ("recycles", args.recycles),
            ("seed", args.seed),
            ("designer", args.designer),
            ("alanine-bias", args.alanine_bias),
        ]
        problems = cdp.evaluate(ws, SET_CONTROLS % ", ".join(
            "[%s, %s]" % (json.dumps(k), json.dumps(v)) for k, v in controls))
        if json.loads(problems):
            print("controls:", problems, file=sys.stderr)
            return 1
        print("controls: ok ·", ", ".join("%s=%s" % (k, v) for k, v in controls))
        print("weights :", "./model-af3-int5/" if local else "the pinned remote")
        print("designer:", cdp.evaluate(
            ws, "(document.getElementById('designer-note')||{}).textContent"))

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
            # 🔴 "NOTHING IS BEST" IS A LEGITIMATE OUTCOME, NOT A FAILURE.
            # Every design cycle can be over the 20% alanine ceiling - a short
            # binder with the bias ramp off does it regularly - and the run is
            # then correctly reporting that it found nothing worth keeping.
            # What must never happen is TWO bests, or a best where an eligible
            # cycle was passed over.
            marked = sum(1 for row in rows if row["best"])
            eligible = [row for row in rows
                        if row["cycle"] != "0" and int(row["alanine"].rstrip("%")) <= 20]
            if marked > 1:
                failures.append("%d rows marked best, expected at most 1" % marked)
            elif marked == 0 and eligible:
                failures.append("nothing marked best though %d cycles were under"
                                " the alanine ceiling" % len(eligible))

        # 🔴 ONE OBJECT, ONE FRAME PER CYCLE, AND THE TARGET HOLDING STILL.
        # The frames are the replay: a viewer rebuilt per cycle shows one
        # structure and no play bar, and frames that were not superposed play
        # back as a structure being thrown around a room. So this counts them,
        # checks the transport controls appeared, and measures how far the
        # TARGET's alpha carbons moved between the first frame and the last -
        # which is what the fit holds at zero and what nothing else would.
        frames = cdp.evaluate(ws, """(() => {
          const reg = window.py2dmol_viewers || {};
          const key = Object.keys(reg)[0];
          const v = key && reg[key].renderer;
          if (!v) return JSON.stringify({ error: 'no viewer' });
          const frames = v.objectsData[v.currentObjectName].frames;
          // 🔴 A FRAME KEEPS ONE `coords` ENTRY PER RESIDUE - the alpha
          // carbon - beside a parallel `chains`. Reading it as a list of
          // ATOMS finds nothing, reports no chains, and the drift check
          // silently measures neither structure.
          const ca = (frame, chain) => frame.coords
            .filter((_, i) => frame.chains[i] === chain);
          const chains = [...new Set(frames[0].chains)].sort();
          const held = chains.filter((c) => c !== 'A');
          const on = held.length > 0 ? held[0] : 'A';
          // 🔴 THE CENTROIDS, NOT THE RMSD. A superposed pair is NOT expected
          // to have a small RMSD here: every cycle folds the target again from
          // scratch, so its CONFORMATION differs between frames and a 31-mer
          // peptide legitimately measured 3.9 A after a perfect fit. What a
          // Kabsch fit does guarantee is that it translates the two centroids
          // together. Measured both ways on this 31-mer target, superposition
          // on and then off:
          //
          //     on   centroid 0.00006 A   rmsd  3.86 A
          //     off  centroid 3.80    A   rmsd 13.60 A
          //
          // so the centroid separates them by five orders of magnitude and the
          // RMSD by a factor of 3.5 against a real conformational difference
          // of unknown size. The offset is not larger with superposition off
          // because fittedPdb in web/af3-model.js already fits each fold's own
          // frames to a per-fold reference, which removes most of the
          // augmentation's TRANSLATION - what it cannot remove is the
          // rotation, which is where the 13.6 comes from. The RMSD is reported
          // because it is worth reading, not because it is the check.
          const centre = (points) => points.reduce(
            (sum, p) => [sum[0] + p[0] / points.length,
                         sum[1] + p[1] / points.length,
                         sum[2] + p[2] / points.length], [0, 0, 0]);
          let drift = null;
          let offset = null;
          let extent = null;
          if (frames.length > 1) {
            const first = ca(frames[0], on);
            const last = ca(frames[frames.length - 1], on);
            if (first.length > 0 && first.length === last.length) {
              let total = 0;
              for (let i = 0; i < first.length; i += 1) {
                total += (first[i][0] - last[i][0]) ** 2
                       + (first[i][1] - last[i][1]) ** 2
                       + (first[i][2] - last[i][2]) ** 2;
              }
              drift = Math.sqrt(total / first.length);
              const a = centre(first);
              const b = centre(last);
              offset = Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
              // ...and how big the thing is, so "close" is relative to it.
              extent = Math.sqrt(first.reduce((sum, p) =>
                sum + ((p[0] - a[0]) ** 2 + (p[1] - a[1]) ** 2
                       + (p[2] - a[2]) ** 2) / first.length, 0));
            }
          }
          const bar = document.getElementById('playButton');
          return JSON.stringify({
            object: v.currentObjectName, frames: frames.length,
            labels: frames.map((f) => f.label),
            withConfidence: frames.filter((f) => f.confidence).length,
            chains, fittedOn: on, driftRmsd: drift,
            centroidOffset: offset, radius: extent,
            playBar: bar !== null && getComputedStyle(bar).display !== 'none',
          });
        })()""")
        print("frames  :", frames)
        shown = json.loads(frames)
        # 🔴 ONE FRAME PER CYCLE IS THE FLOOR, NOT THE COUNT. With the
        # diffusion steps kept - the page's default - a cycle contributes its
        # whole sampler trajectory plus the structure it settled to, so the
        # count is `cycles * steps`-ish. What must hold either way is that
        # every cycle is represented and nothing collapsed them.
        if shown.get("frames", 0) < expected:
            failures.append("%s frames in the viewer, expected at least %d"
                            % (shown.get("frames"), expected))
        settled = [text for text in (shown.get("labels") or []) if "step" not in text]
        if len(settled) != expected:
            failures.append("%d settled frames, expected %d" % (len(settled), expected))
        if expected > 1 and not shown.get("playBar"):
            failures.append("no play bar, so the cycles cannot be replayed")
        offset = shown.get("centroidOffset")
        if offset is None:
            failures.append("could not measure the fitted chain's drift")
        elif offset > 1.0:
            failures.append("the fitted chain's centroid moved %.2f A between the"
                            " first frame and the last (radius %.1f) - the frames"
                            " are not superposed"
                            % (offset, shown.get("radius") or 0))
        # AF3's confidence head runs once per fold, so only the settled frame
        # of each cycle has measured numbers - a step carrying the cycle's
        # would put a confident colour on a structure that has not earned it.
        if shown.get("withConfidence") != expected:
            failures.append("%s frames carry confidence, expected %d (one per cycle)"
                            % (shown.get("withConfidence"), expected))
        # 🔴 THE CARD MUST FOLLOW THE BAR, WHICH py2Dmol DOES NOT ANNOUNCE.
        # web/scores-card.js polls for it, and a poll that silently stopped
        # would leave the last cycle's numbers under whatever frame is drawn -
        # which looks like a working card. So this scrubs to a SAMPLER STEP,
        # which has no measured confidence, and checks the card empties.
        card = cdp.evaluate(ws, """(async () => {
          const reg = window.py2dmol_viewers || {};
          const v = reg[Object.keys(reg)[0]].renderer;
          const frames = v.objectsData[v.currentObjectName].frames;
          const read = () => ({
            shown: getComputedStyle(
              document.getElementById('predictionScoresBox')).display !== 'none',
            plddt: (document.getElementById('metricMeanPlddt') || {}).textContent,
          });
          const settled = frames.findLastIndex((f) => f.confidence);
          const step = frames.findIndex((f) => !f.confidence);
          const wait = () => new Promise((r) => setTimeout(r, 250));
          v.setFrame(settled); v.render('cycle'); await wait();
          const onSettled = read();
          v.setFrame(step); v.render('cycle'); await wait();
          const onStep = read();
          return JSON.stringify({ settledIndex: settled, stepIndex: step,
                                  onSettled, onStep });
        })()""")
        print("card    :", card)
        seen = json.loads(card)
        if not seen["onSettled"]["shown"] or seen["onSettled"]["plddt"] in ("-", ""):
            failures.append("the scores card shows no pLDDT on a settled frame")
        if seen["onStep"]["shown"]:
            failures.append("the scores card stayed up on a sampler step, which has"
                            " no measured confidence - the poll is not following")

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
