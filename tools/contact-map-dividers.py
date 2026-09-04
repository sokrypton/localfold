"""Does the contact map get its chain divider lines while a fold is running?

    python3 tools/contact-map-dividers.py

🔴 IT EXISTS BECAUSE THE MAP HAS TWO PATHS AND ONLY ONE OF THEM CARRIES CHAINS.
The heatmap rules a line wherever the chain changes, and it reads the chains off
the RENDERER - which fills them in when a structure is parsed. A finished fold
has one, so the lines are there. For the longest part of an AF3 fold there is
no structure at all: the trunk knows the contact map several recycles before the
sampler emits a first frame, and web/app.js pushes it straight at
`renderer.heatmapRenderer.setMaps()` for exactly that reason. On that path the
chains are empty and the lines are silently skipped.

It needs no weights and folds nothing: it drives the two states by hand - a map
with no structure behind it, and the same map with one - and reads the pixels
back. A divider is a two-pixel dark line, so the test is that the boundary
column is darker than its neighbours.
"""
import http.server
import os
import socketserver
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import cdp                                                   # noqa: E402

PORT, DBG = 9668, 9231
REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RESIDUES_PER_CHAIN = 6


def serve():
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=REPO, **kw)

        def log_message(self, *a):
            pass

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", PORT), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def two_chain_pdb():
    lines, serial = [], 1
    for chain, base in (("A", 0.0), ("B", 20.0)):
        for index in range(RESIDUES_PER_CHAIN):
            lines.append(
                "ATOM  %5d  CA  GLY %s%4d    %8.3f%8.3f%8.3f  1.00 80.00           C"
                % (serial, chain, index + 1, base + index * 3.8, 0.0, 0.0))
            serial += 1
    return "\n".join(lines) + "\nEND\n"


# The map web/app.js's contactMapFor() builds, and the push its onContacts does -
# optionally preceded by the chain layout that push now supplies.
PUSH_MAP = """(() => {
  const reg = window.py2dmol_viewers || {};
  const v = reg[Object.keys(reg)[0]] && reg[Object.keys(reg)[0]].renderer;
  if (!v || !v.heatmapRenderer) return 'no heatmap renderer';
  const n = %d;
  const withChains = %s;
  if (withChains && (v.chains || []).length === 0) {
    const ids = [];
    for (let c = 0; c < 2; c += 1) for (let r = 0; r < n / 2; r += 1) ids.push('AB'[c]);
    v.chains = ids;
  }
  const data = new Uint8Array(n * n);
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) data[i * n + j] = (i === j) ? 255 : 40;
  }
  v.heatmapRenderer.setMaps({ contact: { data, n, vmin: 0, vmax: 1 } });
  if (window.Heatmap && window.Heatmap.updateVisibility) window.Heatmap.updateVisibility(v);
  v.render('divider-probe');
  return 'pushed';
})()"""

# A divider is a 2 px dark line at the chain boundary. Read the column there and
# two columns that should be plain map, and compare their mean darkness.
READ_COLUMNS = """(() => {
  const reg = window.py2dmol_viewers || {};
  const v = reg[Object.keys(reg)[0]] && reg[Object.keys(reg)[0]].renderer;
  const h = v && v.heatmapRenderer;
  if (!h || !h.canvas) return { error: 'no canvas' };
  const ctx = h.canvas.getContext('2d');
  const size = h.size || h.canvas.width;
  const boundary = Math.floor(size / 2);
  const columnDarkness = (x) => {
    const d = ctx.getImageData(x, 0, 1, size).data;
    let sum = 0;
    for (let i = 0; i < size; i += 1) sum += (d[i * 4] + d[i * 4 + 1] + d[i * 4 + 2]) / 3;
    return 255 - sum / size;
  };
  // The line straddles the boundary, so take the darkest of the two pixels it
  // can land on rather than assuming which side rounding put it.
  const at = Math.max(columnDarkness(boundary - 1), columnDarkness(boundary));
  const away = (columnDarkness(boundary - 4) + columnDarkness(boundary + 4)) / 2;
  return {
    size, boundary, atBoundary: Number(at.toFixed(1)), awayFromIt: Number(away.toFixed(1)),
    chains: (v.chains || []).length,
    frames: ((v.objectsData || {})[v.currentObjectName] || {}).frames?.length ?? 0,
  };
})()"""


def main():
    httpd = serve()
    proc, ws = cdp.launch(DBG, "/tmp/_cdp_dividers_profile")
    failures = []
    try:
        ws.call("Page.enable")
        ws.call("Page.navigate", url="http://127.0.0.1:%d/index.html" % PORT)
        cdp.wait_for(ws, "typeof window.processFiles === 'function'"
                         " && !document.getElementById('predict').disabled",
                     what="the bundle and web/app.js to finish loading")

        n = RESIDUES_PER_CHAIN * 2

        # 1. What the trunk used to do: a map and nothing behind it.
        print("push  :", cdp.evaluate(ws, PUSH_MAP % (n, "false")))
        time.sleep(0.5)
        unruled = cdp.evaluate(ws, READ_COLUMNS)
        print("no structure, no chains (the bug):   ", unruled)

        # 2. What it does now: the same map with the chain layout supplied.
        print("push  :", cdp.evaluate(ws, PUSH_MAP % (n, "true")))
        time.sleep(0.5)
        before = cdp.evaluate(ws, READ_COLUMNS)
        print("no structure, chains supplied:       ", before)

        # 2. The finished state: the same map with a structure behind it.
        cdp.evaluate(ws, """
          window.__loaded = false;
          window.py2dmolLoadFiles([{ name: 'div.pdb',
            readAsync: () => Promise.resolve(`%s`) }])
            .then(() => { window.__loaded = true; });
        """ % two_chain_pdb().replace("\n", "\\n"))
        cdp.wait_for(ws, "window.__loaded === true", what="the structure to load")
        time.sleep(0.8)
        print("push  :", cdp.evaluate(ws, PUSH_MAP % (n, "false")))
        time.sleep(0.5)
        after = cdp.evaluate(ws, READ_COLUMNS)
        print("after inference (structure loaded):  ", after)

        # A divider is a hard dark line; without one the boundary column is
        # ordinary map. Ten levels of mean darkness is far more than the map's
        # own texture and far less than a 2 px black line contributes.
        def ruled(sample):
            return sample["atBoundary"] - sample["awayFromIt"] > 10

        if not ruled(after):
            failures.append("no divider with a structure loaded: %s" % after)
        if not ruled(before):
            failures.append("no divider during inference with chains supplied: %s" % before)
        # ...and the bug itself, kept as a live demonstration that the chains
        # are what the lines depend on rather than something else.
        if ruled(unruled):
            failures.append("a divider with no chains at all, so this probe is"
                            " not measuring what it thinks: %s" % unruled)
    finally:
        try:
            proc.terminate()
        except Exception:
            pass
        httpd.shutdown()

    for line in failures:
        print("FAIL", line)
    print("ok" if not failures else "%d failure(s)" % len(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
