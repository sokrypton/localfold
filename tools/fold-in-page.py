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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default=DEFAULT)
    parser.add_argument("--model", default="monomer",
                        help="the value of the #model select: monomer, multimer or af3")
    parser.add_argument("--recycles", default="1")
    parser.add_argument("--steps", default="4", help="AF3 sampler steps")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    httpd = serve()
    proc, ws = cdp.launch(DBG, "/tmp/_cdp_fold_profile")
    try:
        ws.call("Page.enable")
        ws.call("Page.navigate", url="http://127.0.0.1:%d/index.html" % PORT)
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
          set('msa-mode', 'none');
        })()""" % (json.dumps(args.model), json.dumps(args.recycles),
                   json.dumps(args.steps)))
        time.sleep(0.5)
        print("controls:", cdp.evaluate(ws, """(() => {
          const v = (id) => (document.getElementById(id) || {}).value;
          return JSON.stringify({ model: v('model-family'), recycles: v('recycles'),
            msa: v('msa-mode'), af3count: v('af3-count') });
        })()"""))

        cdp.evaluate(ws, "document.getElementById('predict').click()")
        cdp.wait_for(ws, """(() => {
          const s = document.getElementById('status-message');
          const text = s ? s.textContent : '';
          return /done|complete|finished|s\\b/i.test(text)
            && document.getElementById('downloads').style.display !== 'none';
        })()""", timeout=args.timeout, what="the fold to finish")
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
    finally:
        proc.kill()
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
