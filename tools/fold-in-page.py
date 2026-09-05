"""Drive a REAL fold in the real page, and report what the frames carry.

    python3 tools/fold-in-page.py                       # AF2 monomer
    python3 tools/fold-in-page.py --model af3 --steps 4
    python3 tools/fold-in-page.py --sequence GWSTELEK... --recycles 1

🔴 IT EXISTS BECAUSE EVERY OTHER CHECK HERE MISSES THE PATH THAT MATTERS.
tools/heatmap-panel.py loads a structure and attaches maps by hand, so it
proves the PANEL works with our format; tools/gpu/probe-af2-contacts.js runs
the head against a fold, so it proves the ARITHMETIC. Neither runs web/app.js's
own wiring, and a contact map failed to appear three times in a row with both
of those passing - once because the panel is not told by render(), once
because `viewer.objects` does not exist, once because the shard was a 404.

So this presses the page's own Fold button and reads the frames back: what
maps each frame has, how many frames there are, and what the heatmap panel is
showing. If it says a frame has no `contact`, the wiring is wrong; if it says
the frame has one and the panel does not list it, the panel is not being told.

🔴 IT NEEDS THE WEIGHTS, so it is slow and it is not part of any suite. AF2
pulls its bundle from the pinned remote unless a local model/ directory is
served, which it is here - the server's root is the repo.
"""
import argparse
import http.server
import re
import json
import os
import socketserver
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cdp                                                   # noqa: E402

PORT, DBG = 9667, 9230
REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DEFAULT = "GWSTELEKHREELKEFLKKEGITLGFTNAEKQEQAQKLGLGKKVSPELLIKAFAILKK"


MANIFESTS = "/src/reference/manifests/index.js"
REMOTE_LINE = re.compile(rb'^\s*remote:\s*"[^"]*",\s*$', re.MULTILINE)


def serve(local_weights=True):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=REPO, **kw)

        def log_message(self, *a):
            pass

        def end_headers(self):
            # 🔴 SOURCE IS NEVER CACHED, because a cached ES module looks exactly
            # like a broken feature - CLAUDE.md's own trap, from the server's
            # side. Not the weights: they are hundreds of megabytes of shards
            # and re-reading them turns a one-minute run into ten.
            if self.path.split("?")[0].endswith((".js", ".html", ".css")):
                self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def do_GET(self):
            # 🔴 AND THE AF3 BUNDLE COMES OFF THE DISK BY DEFAULT. Its manifest
            # names a `remote`, which a deployed page is right to use and a
            # check that runs on demand is not: 150 MB per run. Rewriting the
            # one module as it is served changes nothing on disk.
            if local_weights and self.path.split("?")[0] == MANIFESTS:
                source = open(os.path.join(REPO, MANIFESTS.lstrip("/")), "rb").read()
                body = REMOTE_LINE.sub(b"", source)
                self.send_response(200)
                self.send_header("Content-Type", "text/javascript")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", PORT), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


# The page's own status line, polled by `cdp.wait_for` so a long fold narrates
# itself rather than ending in a timeout that names no stage.
STATUS_LINE = "(document.getElementById('status-message')||{}).textContent"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default=DEFAULT)
    parser.add_argument("--model", default="monomer",
                        help="the value of the #model select: monomer, multimer or af3")
    parser.add_argument("--recycles", default="1")
    parser.add_argument("--steps", default="4", help="AF3 sampler steps")
    parser.add_argument("--url", default=None,
                        help="drive a DEPLOYED page instead of this checkout,"
                             " e.g. https://localfold.org/index.html. The"
                             " weights then come from the bundle's pinned"
                             " remote, which is a ~97 MB download per run"
                             " because each run starts a fresh profile with an"
                             " empty cache.")
    parser.add_argument("--timeout", type=int, default=900)
    # 🔴 SINGLE SEQUENCE BY DEFAULT, because this tool is a wiring check and a
    # search is a minute of somebody else's server. `--msa-mode search` is
    # needed for `--template auto`, which has nothing to draw on without one.
    parser.add_argument("--msa-mode", default="none", choices=["none", "search"])
    parser.add_argument("--remote-weights", action="store_true",
                        help="fetch the AF3 bundle from its pinned remote"
                             " (~150 MB) instead of ./model-af3-int5/")
    parser.add_argument("--template", default="",
                        help="a PDB entry (1abc, 1abc_A) or UniProt accession"
                             " to show the first protein entity as a template,"
                             " or `auto` to use what the MSA search finds."
                             " Goes to the network either way.")
    parser.add_argument("--then-sequence", default=None,
                        help="fold a SECOND time on this sequence, which is a"
                             " fresh fold rather than a continuation")
    parser.add_argument("--then-recycles", default=None,
                        help="fold a SECOND time at this recycle count, which is"
                             " what the rewind-and-continue path does")
    args = parser.parse_args()

    httpd = serve(local_weights=args.url is None and not args.remote_weights)
    proc, ws = cdp.launch(DBG, "/tmp/_cdp_fold_profile")
    try:
        ws.call("Page.enable")
        ws.call("Page.navigate",
                url=args.url or ("http://127.0.0.1:%d/index.html" % PORT))
        cdp.wait_for(ws, "typeof window.processFiles === 'function'"
                         " && !document.getElementById('predict').disabled",
                     what="the page to finish loading")

        # 🔴 THE SEQUENCE FIELD IS CONTENTEDITABLE, NOT AN INPUT, and the list
        # reads it on `input` - so setting textContent alone leaves the entity
        # empty and Fold does nothing.
        cdp.evaluate(ws, """(() => {
          const field = document.querySelector('.entity-field [contenteditable],'
            + ' .entity-field textarea, .entity-field input');
          if (!field) return 'no field';
          if ('value' in field && field.tagName !== 'DIV') field.value = %s;
          else field.textContent = %s;
          field.dispatchEvent(new Event('input', { bubbles: true }));
          return field.tagName;
        })()""" % (json.dumps(args.sequence), json.dumps(args.sequence)))

        # 🔴 THE TEMPLATE GOES ON THE ENTITY, NOT ON A CONTROL. It lives behind
        # the row's ⋮ beside the modified residues, in the entity model that
        # web/entities.js expands - so it is set through the list's own API,
        # the way a paste would set it, rather than by poking at the popup.
        if args.template:
            print("template:", cdp.evaluate(ws, """(() => {
              const list = window.__entityList;
              if (!list) return 'no entity list';
              const entities = list.read();
              const protein = entities.find((e) => e.type === 'protein');
              if (!protein) return 'no protein entity';
              protein.template = %s;
              list.set(entities);
              return JSON.stringify(list.read().map((e) => e.template || null));
            })()""" % json.dumps({"auto": True} if args.template == "auto"
                                  else {"source": args.template})))

        cdp.evaluate(ws, """(() => {
          const set = (id, value) => {
            const el = document.getElementById(id);
            if (!el) return;
            el.value = value;
            el.dispatchEvent(new Event('change', { bubbles: true }));
          };
          set('model-family', %s);
          set('recycles', %s);
          set('af3-count', %s);
          set('msa-mode', %s);
        })()""" % (json.dumps(args.model), json.dumps(args.recycles),
                   json.dumps(args.steps), json.dumps(args.msa_mode)))
        time.sleep(0.5)
        print("controls:", cdp.evaluate(ws, """(() => {
          const v = (id) => (document.getElementById(id) || {}).value;
          return JSON.stringify({ model: v('model-family'), recycles: v('recycles'),
            msa: v('msa-mode'), af3count: v('af3-count') });
        })()"""))

        # 🔴 THE STATUS LINE IS SAMPLED, NOT GLANCED AT. A line that alternates
        # between two sentences reads as flicker, and neither a screenshot nor
        # a reading after the fold can see it - only the sequence of values it
        # took while the fold ran. Every distinct value is recorded, with the
        # numbers blanked, so the SHAPES it took can be counted.
        cdp.evaluate(ws, """(() => {
          window.__statusLog = [];
          const el = document.getElementById('status-message');
          const shape = (s) => s.replace(/[0-9]+(\\.[0-9]+)?/g, '#');
          new MutationObserver(() => {
            const now = shape(el.textContent || '');
            const log = window.__statusLog;
            if (log.length === 0 || log[log.length - 1] !== now) log.push(now);
          }).observe(el, { childList: true, characterData: true, subtree: true });
        })()""")
        cdp.evaluate(ws, "document.getElementById('predict').click()")
        cdp.wait_for(ws, """(() => {
          const s = document.getElementById('status-message');
          const text = s ? s.textContent : '';
          return /done|complete|finished|s\\b/i.test(text)
            && document.getElementById('downloads').style.display !== 'none';
        })()""", timeout=args.timeout, what="the fold to finish",
                     progress=STATUS_LINE)
        time.sleep(1.5)

        print("frames:", cdp.evaluate(ws, """(() => {
          const reg = window.py2dmol_viewers || {};
          const v = reg[Object.keys(reg)[0]] && reg[Object.keys(reg)[0]].renderer;
          if (!v) return 'no viewer';
          const name = v.currentObjectName;
          const frames = v.objectsData[name].frames;
          return JSON.stringify({
            object: name,
            frames: frames.length,
            perFrame: frames.map((f) => ({
              name: f.name,
              maps: f.maps ? Object.keys(f.maps) : [],
              pae: f.pae ? f.pae.length : 0,
            })),
          });
        })()"""))
        # 🔴 THE B-FACTOR IS THE COLOUR, so read it rather than the frame count.
        # A frame nothing has measured must carry zero - the pLDDT ramp paints
        # that red - and only the finished structure may carry real values.
        print("bfactor:", cdp.evaluate(ws, """(() => {
          const reg = window.py2dmol_viewers || {};
          const v = reg[Object.keys(reg)[0]] && reg[Object.keys(reg)[0]].renderer;
          if (!v) return 'no viewer';
          const frames = v.objectsData[v.currentObjectName].frames;
          // 🔴 A FRAME KEEPS A FLAT `plddts`, NOT PER-ATOM OBJECTS. Reading
          // it wrong reports zero for everything, which is exactly what an
          // uncoloured frame looks like - so the check would have passed
          // whatever the page did.
          return JSON.stringify(frames.map((f) => {
            const b = Array.from(f.plddts || []);
            if (b.length === 0) return { name: f.name, plddts: 'missing' };
            return { name: f.name, n: b.length, min: Math.min(...b).toFixed(1),
              max: Math.max(...b).toFixed(1) };
          }));
        })()"""))
        print("panel :", cdp.evaluate(ws, """(() => {
          const c = document.getElementById('heatmapContainer');
          return JSON.stringify({
            visible: c && getComputedStyle(c).display !== 'none',
            tabs: [...document.querySelectorAll('#heatmapContainer [role="tab"]')]
              .map((t) => t.dataset.mapKey),
          });
        })()"""))
        print("status:", cdp.evaluate(ws,
            "(document.getElementById('status-message')||{}).textContent"))
        shapes = cdp.evaluate(ws, """(() => {
          const log = window.__statusLog || [];
          const seen = [];
          for (const s of log) if (seen.indexOf(s) < 0) seen.push(s);
          return JSON.stringify({ changes: log.length, shapes: seen });
        })()""")
        print("statusln:", shapes)
        print("map1   :", cdp.evaluate(ws, """(() => {
              const reg = window.py2dmol_viewers || {};
              const v = reg[Object.keys(reg)[0]].renderer;
              const h = v.heatmapRenderer;
              const maps = h && h.maps;
              const sum = (m) => {
                if (!m || !m.data) return null;
                let a = 0;
                for (let i = 0; i < m.data.length; i += 1) a = (a + m.data[i] * (i % 7 + 1)) % 1000000007;
                return a;
              };
              const c = document.getElementById('heatmapContainer');
              return JSON.stringify({
                visible: !!(c && getComputedStyle(c).display !== 'none'),
                keys: maps ? Object.keys(maps) : [],
                contact: maps ? sum(maps.contact) : null,
              });
            })()"""))
        # 🔴 THE CAMERA IS PART OF THE ANSWER. addFrame recentres viewerState on
        # the centroid of every frame the object holds, so a rewind that clears
        # the frames and re-adds them walks the camera - which is what "the view
        # jumps" is. Read before and after, and compare.
        camera_before = cdp.evaluate(ws, """(() => {
              const reg = window.py2dmol_viewers || {};
              const v = reg[Object.keys(reg)[0]].renderer;
              const s = v.viewerState || {};
              const o = v.objectsData[v.currentObjectName] || {};
              const r = (x) => x === null || x === undefined ? null
                : (typeof x === 'number' ? Number(x.toFixed(3)) : x);
              return JSON.stringify({
                zoom: r(s.zoom), focal: r(s.focalLength),
                center: s.center ? [r(s.center.x), r(s.center.y), r(s.center.z)] : null,
                objCenter: (o.center || []).map(r),
                extent: r(o.maxExtent),
                rot0: (s.rotation && s.rotation[0] || []).map(r),
              });
            })()""")
        print("camera1:", camera_before)
        first_object = json.loads(cdp.evaluate(ws, """(() => {
          const reg = window.py2dmol_viewers || {};
          return JSON.stringify(reg[Object.keys(reg)[0]].renderer.currentObjectName);
        })()"""))

        # 🔴 THE SECOND FOLD IS THE ONE THAT REWINDS. Asking for more recycles
        # with everything else unchanged should keep the frames the earlier
        # passes already produced and append to them - not start an object over.
        if args.then_recycles is not None or args.then_sequence is not None:
            if args.then_sequence is not None:
                cdp.evaluate(ws, """(() => {
                  const field = document.querySelector('.entity-field [contenteditable],'
                    + ' .entity-field textarea, .entity-field input');
                  if ('value' in field && field.tagName !== 'DIV') field.value = %s;
                  else field.textContent = %s;
                  field.dispatchEvent(new Event('input', { bubbles: true }));
                })()""" % (json.dumps(args.then_sequence), json.dumps(args.then_sequence)))
                time.sleep(0.5)
            cdp.evaluate(ws, """(() => {
              const el = document.getElementById('recycles');
              if (%s !== null) {
                el.value = %s;
                el.dispatchEvent(new Event('change', { bubbles: true }));
              }
              document.getElementById('predict').click();
            })()""" % (json.dumps(args.then_recycles), json.dumps(args.then_recycles)))
            # 🔴 THE PANEL IS SAMPLED WHILE THE NEW FOLD IS STILL IN ITS TRUNK.
            # A map left over from the PREVIOUS fold is invisible afterwards -
            # by then the new one has replaced it - so the only moment it can
            # be caught is between the click and the first recycle's contacts.
            for _ in range(6):
                print("early  :", cdp.evaluate(ws, """(() => {
              const reg = window.py2dmol_viewers || {};
              const v = reg[Object.keys(reg)[0]].renderer;
              const h = v.heatmapRenderer;
              const maps = h && h.maps;
              const sum = (m) => {
                if (!m || !m.data) return null;
                let a = 0;
                for (let i = 0; i < m.data.length; i += 1) a = (a + m.data[i] * (i % 7 + 1)) % 1000000007;
                return a;
              };
              const c = document.getElementById('heatmapContainer');
              return JSON.stringify({
                visible: !!(c && getComputedStyle(c).display !== 'none'),
                keys: maps ? Object.keys(maps) : [],
                contact: maps ? sum(maps.contact) : null,
              });
            })()"""))
                time.sleep(0.4)
            cdp.wait_for(ws, """(() => {
              const s = document.getElementById('status-message');
              return /done|complete|finished|s\\b/i.test(s ? s.textContent : '')
                && document.getElementById('predict').disabled === false;
            })()""", timeout=args.timeout, what="the second fold to finish",
                     progress=STATUS_LINE)
            time.sleep(1.5)
            second = cdp.evaluate(ws, """(() => {
              const reg = window.py2dmol_viewers || {};
              const v = reg[Object.keys(reg)[0]].renderer;
              const names = Object.keys(v.objectsData);
              return JSON.stringify({
                objects: names,
                current: v.currentObjectName,
                frames: names.map((n) => [n, v.objectsData[n].frames.length,
                  v.objectsData[n].frames.map((f) => f.name)]),
              });
            })()""")
            print("2nd    :", second)
            camera_after = cdp.evaluate(ws, """(() => {
              const reg = window.py2dmol_viewers || {};
              const v = reg[Object.keys(reg)[0]].renderer;
              const s = v.viewerState || {};
              const o = v.objectsData[v.currentObjectName] || {};
              const r = (x) => x === null || x === undefined ? null
                : (typeof x === 'number' ? Number(x.toFixed(3)) : x);
              return JSON.stringify({
                zoom: r(s.zoom), focal: r(s.focalLength),
                center: s.center ? [r(s.center.x), r(s.center.y), r(s.center.z)] : null,
                objCenter: (o.center || []).map(r),
                extent: r(o.maxExtent),
                rot0: (s.rotation && s.rotation[0] || []).map(r),
              });
            })()""")
            print("camera2:", camera_after)
            # 🔴 A REWIND MUST NOT MOVE THE CAMERA. It continues on the object
            # it already had, so the reader is looking at a view they set - and
            # _switchToObject restores a viewerState that is only ever SAVED
            # when switching AWAY from an object, so asking for the object
            # already current restored its default and reset the rotation to
            # the identity. A fresh fold is a different object and is expected
            # to orient itself, so this only applies when the second fold
            # stayed on the first one's object.
            if json.loads(second).get("current") == first_object:
                was = json.loads(camera_before)
                now = json.loads(camera_after)
                # 🔴 ORIENTATION AND ZOOM ONLY. The centre and the focal
                # length are derived from the structure - a continuation that
                # re-samples really does land somewhere else, and following it
                # is right - but the ROTATION is the reader's, and moving it is
                # what reads as the view jumping.
                for field in ("rot0", "zoom"):
                    if was[field] != now[field]:
                        print("FAIL: a rewind moved the camera's %s: %r -> %r"
                              % (field, was[field], now[field]))
                for field in ("center", "focal"):
                    if was[field] != now[field]:
                        print("note: %s followed the structure: %r -> %r"
                              % (field, was[field], now[field]))
            print("2status:", cdp.evaluate(ws,
                "(document.getElementById('status-message')||{}).textContent"))
    finally:
        proc.kill()
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
