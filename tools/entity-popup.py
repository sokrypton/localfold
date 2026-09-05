"""Measure the entity popup as the template source is changed.

    python3 tools/entity-popup.py

🔴 THE POPUP CHANGED SIZE UNDER THE CURSOR. Its template section renders a
different number of rows per source - a text box for a PDB entry or an
AlphaFold accession, a file picker and a chain box for an upload, nothing at all
for "no template" or "from the MSA search" - and the help text under them runs
from one line to three. So choosing from the menu moved the menu, and on the
last item it moved out from under the pointer.

This measures the popup's height and width for every source, which is the only
way to say whether it is steady: a layout that "looks fine" is one nobody
changed the setting on. Like tools/mobile-layout.py, it drives a real page with
CDP rather than reasoning about the CSS.
"""
import json
import os
import socketserver
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import cdp                                                        # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 9671
DBG = 9231


def serve():
    import http.server

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=ROOT, **kw)

        def log_message(self, *a):
            pass

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    return socketserver.ThreadingTCPServer(("127.0.0.1", PORT), Quiet)


# Open the first entity's menu, switch the template source, and measure.
MEASURE = """(() => {
  const row = document.querySelector('.entity-options')?.closest('div');
  if (row === null || row === undefined) return JSON.stringify({ error: 'no entity row' });
  const menu = row.querySelector('.entity-options');
  const out = [];
  const popup = () => document.querySelector('.entity-popup');
  if (popup() === null && menu !== null) menu.click();
  if (popup() === null) return JSON.stringify({ error: 'no popup opened' });
  for (const kind of ['none', 'pdb', 'afdb', 'search', 'upload']) {
    const select = popup().querySelector('.entity-template-kind');
    if (select === null) return JSON.stringify({ error: 'no template select' });
    select.value = kind;
    select.dispatchEvent(new Event('change', { bubbles: true }));
    const box = popup().getBoundingClientRect();
    const sections = popup().querySelectorAll('.entity-popup-section');
    const section = sections[sections.length - 1];
    out.push({ kind, height: Math.round(box.height), width: Math.round(box.width),
               section: section === undefined ? 0 : Math.round(section.getBoundingClientRect().height) });
  }
  // 🔴 AND CAN A PDB ID ACTUALLY BE TYPED. The source box redraws the popup on
  // every keystroke, which would destroy the input it is typing into: the
  // symptom is a field that takes one character and loses focus, and the
  // measurement is whether the box still holds the caret after an input event.
  const select = popup().querySelector('.entity-template-kind');
  select.value = 'pdb';
  select.dispatchEvent(new Event('change', { bubbles: true }));
  const box = popup().querySelector('.entity-template-source');
  box.focus();
  box.value = '1';
  box.dispatchEvent(new Event('input', { bubbles: true }));
  const after = document.querySelector('.entity-template-source');
  out.push({ kind: 'typing',
             focused: document.activeElement === after,
             sameNode: after === box,
             value: after === null ? null : after.value });
  return JSON.stringify(out);
})()"""


def main() -> int:
    httpd = serve()
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    proc, ws = cdp.launch(DBG, "/tmp/_cdp_entity_popup")
    try:
        ws.call("Page.enable")
        ws.call("Page.navigate", url="http://127.0.0.1:%d/index.html" % PORT)
        cdp.wait_for(ws, "typeof window.processFiles === 'function'"
                         " && !document.getElementById('predict').disabled",
                     timeout=90, what="the page to load")
        raw = cdp.evaluate(ws, MEASURE)
        measured = json.loads(raw)
        if isinstance(measured, dict):
            print("could not measure:", measured.get("error"))
            return 1
        typing = [row for row in measured if row.get("kind") == "typing"]
        measured = [row for row in measured if row.get("kind") != "typing"]
        heights = [row["height"] for row in measured]
        widths = [row["width"] for row in measured]
        for row in measured:
            print(f"  {row['kind']:<8} popup {row['height']:>4} x {row['width']:>4}"
                  f"   template section {row['section']:>4}")
        for row in typing:
            print(f"  typing   keeps the caret: {row['focused']}"
                  f"   same input node: {row['sameNode']}   value {row['value']!r}")
        spread = max(heights) - min(heights)
        print(f"height spread {spread}px, width spread {max(widths) - min(widths)}px")
        # 🔴 A FEW PIXELS IS A FONT, NOT A JUMP. The bar is that changing the
        # source must not move the popup enough to move what is under the
        # pointer; 8px is about half a row and well under the 28px a whole one
        # costs.
        failed = False
        if spread > 8:
            print(f"FAIL: the popup moves {spread}px between sources")
            failed = True
        for row in typing:
            if not row["focused"]:
                print("FAIL: the source box loses the caret as it is typed into")
                failed = True
        if failed:
            return 1
        print("OK: the popup holds its size across every template source")
        return 0
    finally:
        proc.kill()
        httpd.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
