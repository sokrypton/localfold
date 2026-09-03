"""Do the heatmap panel's tabs appear, and does OUR map format reach them?

    python3 tools/heatmap-panel.py

🔴 IT EXISTS BECAUSE THE PANEL IS VENDORED AND THE FORMAT IS OURS. py2Dmol
draws any map now, not just a PAE, and it reads them off `frame.maps` as
`{data: Uint8Array, n, vmin, vmax}` - a shape web/app.js writes by hand in
contactMapFor(). Nothing in the CPU suite can check that: the encoder is in a
DOM module and the reader is inside a 787 KB bundle. So this loads the real
page, hands it a structure the way a fold does, attaches the two maps the way
web/app.js does, and reads the tab strip back.

🔴 AND IT IS A VENDOR-BUMP TRIPWIRE. The panel was called the PAE panel until
recently and its markup ids moved; ours are the old names, which it still
accepts. A future bundle that drops them, or that changes the map format,
breaks the page silently - the panel simply stays hidden - and this is what
says so.

🔴 THE FILE READER IS `readAsync`, NOT `text`. web/app.js hands py2Dmol virtual
files as `{name, readAsync}`, and a file with the wrong reader loads without
error and produces no object at all: `objectsData` comes back empty and the
viewer looks idle. That cost a debugging round here.
"""
import http.server
import os
import socketserver
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import cdp                                                   # noqa: E402

PORT, DBG = 9666, 9229
REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


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
    """Two short chains, so a cross-chain map has something to be about."""
    lines, serial = [], 1
    for chain, base in (("A", 0.0), ("B", 20.0)):
        for index in range(6):
            lines.append(
                "ATOM  %5d  CA  GLY %s%4d    %8.3f%8.3f%8.3f  1.00 80.00           C"
                % (serial, chain, index + 1, base + index * 3.8, 0.0, 0.0))
            serial += 1
    return "\n".join(lines) + "\nEND\n"


def main():
    httpd = serve()
    proc, ws = cdp.launch(DBG, "/tmp/_cdp_heatmap_profile")
    failures = []
    try:
        ws.call("Page.enable")
        ws.call("Page.navigate", url="http://127.0.0.1:%d/index.html" % PORT)
        cdp.wait_for(ws, "typeof window.processFiles === 'function'"
                         " && !document.getElementById('predict').disabled",
                     what="the bundle and web/app.js to finish loading")

        cdp.evaluate(ws, """
          window.__loaded = false;
          window.py2dmolLoadFiles([{ name: 'heat.pdb',
            readAsync: () => Promise.resolve(`%s`) }])
            .then(() => { window.__loaded = true; });
        """ % two_chain_pdb().replace("\n", "\\n"))
        cdp.wait_for(ws, "window.__loaded === true", what="the structure to load")
        time.sleep(0.8)

        # Both maps, written exactly as web/app.js writes them: the contact map
        # as bytes under `frame.maps`, the PAE through the legacy field the
        # panel still folds in under its own key.
        attached = cdp.evaluate(ws, """(() => {
          const reg = window.py2dmol_viewers || {};
          const v = reg[Object.keys(reg)[0]] && reg[Object.keys(reg)[0]].renderer;
          if (!v) return 'no viewer';
          const name = v.currentObjectName;
          const frames = v.objectsData[name].frames;
          const n = 12;
          const contact = new Uint8Array(n * n);
          const pae = new Uint8Array(n * n);
          for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
            contact[i * n + j] = Math.round(255 * (Math.abs(i - j) < 3 ? 0.9 : 0.05));
            pae[i * n + j] = Math.min(255, Math.abs(i - j) * 8);
          }
          frames[0].maps = { contact: { data: contact, n, vmin: 0, vmax: 1 } };
          frames[0].pae = pae;
          frames[0].pae_n = n;
          v.setFrame(0);
          v.render('heatmap-panel-check');
          return name + ' frames=' + frames.length;
        })()""")
        if not isinstance(attached, str) or attached.startswith("no "):
            failures.append("could not attach maps: %s" % attached)
        time.sleep(0.8)

        state = cdp.evaluate(ws, """(() => {
          const c = document.getElementById('paeContainer');
          if (!c) return JSON.stringify({ error: 'no #paeContainer' });
          const tabs = [...c.querySelectorAll('[role="tab"], button')]
            .map((t) => t.textContent.trim()).filter(Boolean);
          return JSON.stringify({ visible: c.style.display !== 'none', tabs });
        })()""")
        print("attached:", attached)
        print("panel   :", state)

        if '"visible": true' not in state.replace('":', '": '):
            failures.append("the panel stayed hidden: %s" % state)
        for wanted in ("Contact", "PAE"):
            if wanted not in state:
                failures.append("no %s tab: %s" % (wanted, state))
    finally:
        proc.kill()
        httpd.shutdown()

    for line in failures:
        print("FAIL:", line)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
