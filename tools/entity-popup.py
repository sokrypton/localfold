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
  // 🔴 AND THE TWO MENUS ARE THE SAME CONTROL, MEASURED AS SUCH. The
  // alignment's is the same kind of choice as the template's - one setting,
  // one chain, what gets folded - and it shipped with a class no rule named,
  // so it was a browser-default select beside a styled one. "The same style"
  // is a claim about pixels, so it is read off the page rather than asserted
  // in a comment.
  {
    const template = popup().querySelector('.entity-template-kind');
    const alignment = popup().querySelector('.entity-msa-kind');
    const styleOf = (el) => {
      if (el === null) return null;
      const s = getComputedStyle(el);
      const box = el.getBoundingClientRect();
      return { h: Math.round(box.height), w: Math.round(box.width),
               font: s.fontSize, pad: s.paddingLeft };
    };
    out.push({ kind: 'menus', template: styleOf(template),
               alignment: styleOf(alignment) });
  }
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
  // 🔴 AND DO THE PER-CHAIN SETTINGS SURVIVE THE FOLD PATH. `setChains` is
  // the alignment-query-wins step - it runs on every fold of a single-chain
  // job that has an alignment - and it rebuilt every protein row as
  // `{type, value, copies}`, throwing away the alignment override, the
  // template and the modifications, which live nowhere else. The setting
  // vanished from the row before the fold finished and the NEXT fold had
  // nothing to honour, so it folded with the alignment it had been told not
  // to use. Reported as both halves of that.
  //
  // Driven through `setChains` rather than through a fold: the destructive
  // step is this one, and a fold needs weights and a GPU.
  {
    if (popup() === null) document.querySelector('.entity-options').click();
    const msa = popup()?.querySelector('.entity-msa-kind');
    if (msa === null || msa === undefined) {
      out.push({ kind: 'survives', set: null, kept: null, sameSequence: false });
      return JSON.stringify(out);
    }
    msa.value = 'none';
    msa.dispatchEvent(new Event('change', { bubbles: true }));
    const before = window.__entityList.read()[0];
    window.__entityList.setChains([before.value]);
    const after = window.__entityList.read()[0];
    out.push({ kind: 'survives', set: before.msa ?? null, kept: after.msa ?? null,
               sameSequence: before.value === after.value });
    // ...and a genuinely different query gets a BARE row, because those
    // settings belonged to the chain being replaced.
    window.__entityList.setChains(['MKVLAAGIVGLNLGGK']);
    const replaced = window.__entityList.read()[0];
    out.push({ kind: 'replaced', msa: replaced.msa ?? null,
               value: replaced.value });
    // ...and the fixture put back for the checks below, which need the
    // popup open on a row that still exists: `render()` replaced the one
    // this function captured.
    window.__entityList.set([before]);
  }
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
        menus = [row for row in measured if row.get("kind") == "menus"]
        survives = [row for row in measured if row.get("kind") == "survives"]
        replaced = [row for row in measured if row.get("kind") == "replaced"]
        measured = [row for row in measured
                    if row.get("kind") not in ("typing", "menus",
                                               "survives", "replaced")]
        heights = [row["height"] for row in measured]
        widths = [row["width"] for row in measured]
        for row in measured:
            print(f"  {row['kind']:<8} popup {row['height']:>4} x {row['width']:>4}"
                  f"   template section {row['section']:>4}")
        for row in typing:
            print(f"  typing   keeps the caret: {row['focused']}"
                  f"   same input node: {row['sameNode']}   value {row['value']!r}")
        for row in menus:
            print(f"  menus    template {row['template']}")
            print(f"           alignment {row['alignment']}")
        for row in survives:
            print(f"  setChains  set {row['set']!r} -> kept {row['kept']!r}"
                  f"   (same sequence: {row['sameSequence']})")
        for row in replaced:
            print(f"  a different query: msa {row['msa']!r}")
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
        for row in menus:
            one, two = row["template"], row["alignment"]
            if one is None or two is None:
                print(f"FAIL: a menu is missing from the popup: {row}")
                failed = True
            elif (one["h"], one["w"], one["font"], one["pad"]) != \
                 (two["h"], two["w"], two["font"], two["pad"]):
                print(f"FAIL: the alignment menu is not the template menu's"
                      f" shape: {two} against {one} - they are the same kind"
                      " of control and the popup reads as one only if they"
                      " look it")
                failed = True
        for row in survives:
            if row["set"] != "none":
                print("FAIL: the alignment menu did not record the setting")
                failed = True
            elif row["kept"] != "none":
                print("FAIL: setChains threw away the chain's alignment"
                      " setting - the row is rebuilt on every fold that has"
                      " an alignment, so the next fold has nothing to honour")
                failed = True
        for row in replaced:
            if row["msa"] is not None:
                print("FAIL: a row for a DIFFERENT sequence kept the old"
                      " chain's setting")
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
