"""Do the heatmap panel's tabs appear, and does OUR map format reach them?

    python3 tools/heatmap-panel.py

🔴 IT EXISTS BECAUSE THE PANEL IS VENDORED AND THE FORMAT IS OURS. py2Dmol
draws any map now, not just a PAE, and it reads them off `frame.maps` as
`{data: Uint8Array, n, vmin, vmax}` - a shape web/app.js writes by hand in
contactMapFor(). Nothing in the CPU suite can check that: the encoder is in a
DOM module and the reader is inside a 787 KB bundle. So this loads the real
page, hands it a structure the way a fold does, attaches the two maps the way
web/app.js does, and reads the tab strip back.

🔴 AND IT IS A VENDOR-BUMP TRIPWIRE, WHICH ALREADY CAUGHT ONE. The panel was
called the PAE panel until recently and its markup ids moved. Keeping the old
names looked fine - the JS still accepts them - but the vendored STYLESHEET
has no rule for them, so the container lost `position: relative` and the
absolutely positioned tab strip was drawn against the viewer instead of the
panel. Accepted by the code and unstyled by the CSS is the shape of bug that
looks like it works, so this asserts on GEOMETRY as well as on the tabs.

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
          const c = document.getElementById('heatmapContainer');
          if (!c) return JSON.stringify({ error: 'no #heatmapContainer' });
          const tabs = [...c.querySelectorAll('[role="tab"], button')]
            .map((t) => t.textContent.trim()).filter(Boolean);
          const cs = getComputedStyle(c);
          const box = c.getBoundingClientRect();
          // 🔴 THE STRIP MUST LAND INSIDE THE PANEL. It is absolutely
          // positioned, so this is the one thing that says the container is
          // its positioning context rather than something further up.
          const strip = c.querySelector('[role="tablist"]')
            || (c.querySelector('[role="tab"], button') || {}).parentElement;
          const s = strip ? strip.getBoundingClientRect() : null;
          return JSON.stringify({
            visible: cs.display !== 'none',
            position: cs.position,
            square: box.width > 0 && Math.abs(box.width - box.height) < 2,
            stripInside: !!s && s.left >= box.left - 1 && s.right <= box.right + 1
              && s.top >= box.top - 1 && s.bottom <= box.bottom + 1,
            tabs,
          });
        })()""")
        print("attached:", attached)
        print("panel   :", state)

        # 🔴 AND CLICKING A TAB MUST NOT MOVE ANYTHING. `pae` captions both
        # axes and `contact` captions neither, so a panel that reserves the
        # axis bands for the map ON SCREEN resizes and re-centres the plot on
        # every switch - the plot jumps and the strip moves with it. The
        # reservation is over the SET of maps; this is what says so.
        geom = cdp.evaluate(ws, """(() => {
          const c = document.getElementById('heatmapContainer');
          const canvas = document.getElementById('heatmapCanvas');
          const tabs = [...c.querySelectorAll('[role="tab"]')];
          const rect = () => {
            const b = canvas.getBoundingClientRect();
            return [Math.round(b.left), Math.round(b.top),
                    Math.round(b.width), Math.round(b.height)];
          };
          const seen = {};
          for (const t of tabs) { t.click(); seen[t.dataset.mapKey] = rect(); }
          const keys = Object.keys(seen);
          const same = keys.every((k) =>
            seen[k].join() === seen[keys[0]].join());
          return JSON.stringify({ same, seen });
        })()""")
        print("on click:", geom)
        if '"same":true' not in geom:
            failures.append("the plot moves when a tab is clicked: %s" % geom)

        # 🔴 AND THE PLOT MUST NOT MOVE WHEN THE SECOND MAP ARRIVES. A fold
        # adds them at different times - the contact map comes off the trunk
        # before there is a structure, the PAE only after the confidence head -
        # so a panel that reserves its axis bands for the maps it currently
        # HOLDS resizes mid-fold. Both scales caption their axes now, so the
        # reservation is the same with one map or two.
        staged = cdp.evaluate(ws, """(() => {
          const reg = window.py2dmol_viewers || {};
          const v = reg[Object.keys(reg)[0]] && reg[Object.keys(reg)[0]].renderer;
          const name = v.currentObjectName;
          const object = v.objectsData[name];
          const frame = object.frames[0];
          const n = frame.pae_n;
          const canvas = document.getElementById('heatmapCanvas');
          const rect = () => {
            const b = canvas.getBoundingClientRect();
            return [Math.round(b.left), Math.round(b.top),
                    Math.round(b.width), Math.round(b.height)];
          };
          const keptPae = frame.pae;
          // Contact alone, as it is during a fold...
          delete frame.pae;
          frame.maps = { contact: frame.maps.contact };
          window.Heatmap.updateFrame(v, object, 0);
          const contactOnly = rect();
          // ...then the PAE lands.
          frame.pae = keptPae;
          window.Heatmap.updateFrame(v, object, 0);
          const withPae = rect();
          return JSON.stringify({ contactOnly, withPae,
            same: contactOnly.join() === withPae.join() });
        })()""")
        print("staged  :", staged)
        if '"same":true' not in staged:
            failures.append("the plot moves when the PAE arrives after the"
                            " contact map: %s" % staged)

        # 🔴 A MAP ATTACHED AFTER THE FRAME WAS ADDED MUST STILL APPEAR, and a
        # plain render() is NOT what makes it. py2Dmol drives the panel from
        # setFrame and from its loader; render redraws the 3D scene without
        # re-resolving which maps the frame has. Both contact maps here are
        # computed off the critical path - AF2's per recycle, AF3's at
        # trunk-done - so they always arrive after their frame, and for a
        # while none of them ever showed.
        deferred = cdp.evaluate(ws, """(() => {
          const reg = window.py2dmol_viewers || {};
          const v = reg[Object.keys(reg)[0]] && reg[Object.keys(reg)[0]].renderer;
          const name = v.currentObjectName;
          // objectsData, because `v.objects` does not exist - see
          // refreshHeatmap in web/app.js.
          const object = v.objectsData[name];
          const frame = v.objectsData[name].frames[0];
          const n = frame.pae_n;
          const late = new Uint8Array(n * n).fill(200);
          frame.maps = Object.assign({}, frame.maps, {
            late: { data: late, n, vmin: 0, vmax: 1 },
          });
          const tabs = () => [...document.querySelectorAll(
            '#heatmapContainer [role="tab"]')].map((t) => t.dataset.mapKey);
          v.render('render-only');
          const afterRender = tabs();
          window.Heatmap.updateFrame(v, object, v.currentFrame || 0);
          v.render('after-update');
          return JSON.stringify({ afterRender, afterUpdate: tabs() });
        })()""")
        print("deferred:", deferred)
        if '"afterUpdate"' in deferred and "late" not in deferred.split('"afterUpdate"')[1]:
            failures.append("a map attached after the frame never appears: %s" % deferred)

        if '"visible":true' not in state:
            failures.append("the panel stayed hidden: %s" % state)
        if '"position":"relative"' not in state:
            failures.append("the container is not a positioning context, so the"
                            " tab strip escapes it: %s" % state)
        if '"square":true' not in state:
            failures.append("the panel is not square, so the vendored CSS is not"
                            " reaching it: %s" % state)
        if '"stripInside":true' not in state:
            failures.append("the tab strip is drawn outside the panel: %s" % state)
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
