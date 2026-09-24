"""Pressing Fold must take the PREVIOUS job's alignment off the page.

    python3 tools/fold-clears-msa.py

🔴 THE HEADER AND THE ALIGNMENT ARE SIBLINGS, AND ONLY THE HEADER WAS HIDDEN.
`hideResults` clears everything below the status line when a fold starts, and
for the MSA it hid `#msa-buttons` - which holds the mode menu, the filters and
the chain picker and NOT the picture. `panels/msa.js` appends every
`.msa-canvas` to `viewEl.parentElement`, so the drawn alignment sits beside
that box. Reported as the previous object's MSA still displaying after pressing
Fold: its controls vanished and the alignment stayed.

It was hidden for a while because `openBlankFold` calls `MSA.clear()`, which
REMOVES the canvases - so the leftover only showed in the window between
pressing Fold and the trunk running, which on a complex is the search and the
weights download: the longest part of a fold and the whole reason the page
clears itself.

No weights and no GPU: the fixture is a twelve-residue poly-alanine and its own
alignment, loaded through `py2dmolLoadFiles`, and the fold is pressed with
nothing to fold - `hideResults` runs before anything reads the entity list. A
MutationObserver records what each element's inline display DOES from the
moment the button is clicked, because the interesting state is the one during
the run rather than the one after it.
"""
import json
import os
import socketserver
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import cdp                                                        # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 9673
DBG = 9233

PDB = "\n".join(
    "ATOM  %5d  CA  ALA A%4d    %8.3f%8.3f%8.3f  1.00 50.00           C"
    % (i + 1, i + 1, i * 3.8, 0.0, 0.0) for i in range(12)) + "\nEND\n"
A3M = ">query\n" + "A" * 12 + "\n>h1\n" + "A" * 11 + "G\n>h2\n" + "A" * 10 + "GG\n"


def serve():
    import http.server

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=ROOT, **kw)

        def log_message(self, *a):
            pass

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    return socketserver.ThreadingTCPServer(("127.0.0.1", PORT), Quiet)


MEASURE = """(async () => {
  const out = {};
  const load = (name, text) => window.py2dmolLoadFiles(
    [{ name, readAsync: () => Promise.resolve(text) }], false);
  await load('poly.pdb', %PDB%);
  await new Promise((r) => setTimeout(r, 700));
  await load('poly.a3m', %A3M%);
  await new Promise((r) => setTimeout(r, 1200));
  const canvases = () => Array.from(document.querySelectorAll('.msa-canvas'));
  out.canvasCount = canvases().length;
  out.panelBefore = getComputedStyle(document.getElementById('msa-buttons')).display;
  out.canvasBefore = canvases().map((c) => getComputedStyle(c).display);
  const seen = { panel: [], canvas: [] };
  const watch = (el, key) => new MutationObserver(
    () => seen[key].push(el.style.display || '(unset)'),
  ).observe(el, { attributes: true, attributeFilter: ['style'] });
  watch(document.getElementById('msa-buttons'), 'panel');
  canvases().forEach((c) => watch(c, 'canvas'));
  document.getElementById('predict').click();
  await new Promise((r) => setTimeout(r, 2000));
  out.seen = seen;
  out.panelAfter = getComputedStyle(document.getElementById('msa-buttons')).display;
  out.canvasAfter = canvases().map((c) => getComputedStyle(c).display);
  return JSON.stringify(out);
})()""".replace("%PDB%", json.dumps(PDB)).replace("%A3M%", json.dumps(A3M))


def main() -> int:
    httpd = serve()
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    proc, ws = cdp.launch(DBG, "/tmp/_cdp_fold_clears_msa")
    try:
        ws.call("Page.enable")
        ws.call("Page.navigate", url="http://127.0.0.1:%d/index.html" % PORT)
        cdp.wait_for(ws, "typeof window.py2dmolLoadFiles === 'function'"
                         " && !document.getElementById('predict').disabled",
                     timeout=90, what="the page to load")
        measured = json.loads(cdp.evaluate(ws, MEASURE))
    finally:
        proc.kill()
        httpd.shutdown()

    print("alignment canvases: %d, header %s, canvases %s"
          % (measured["canvasCount"], measured["panelBefore"],
             measured["canvasBefore"]))
    print("after pressing Fold: header %s, canvases %s"
          % (measured["panelAfter"], measured["canvasAfter"]))
    print("  displays written: %s" % json.dumps(measured["seen"]))

    # The control: a fixture that drew nothing has nothing to leave behind, and
    # would pass every check below.
    if measured["canvasCount"] == 0:
        print("FAIL: the fixture drew no alignment - nothing to measure")
        return 1
    if any(d == "none" for d in measured["canvasBefore"]):
        print("FAIL: the alignment was already hidden before Fold was pressed")
        return 1
    bad = []
    if measured["panelAfter"] != "none":
        bad.append("the MSA header stayed up after pressing Fold")
    if any(d != "none" for d in measured["canvasAfter"]):
        bad.append("the previous job's alignment is still drawn: %s"
                   % measured["canvasAfter"])
    for message in bad:
        print("FAIL:", message)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
