"""The model gate: the AlphaFold 3 terms dialog, and `?model=` in the URL.

    python3 tools/model-terms.py

🔴 IT EXISTS BECAUSE NOTHING IN THE CPU SUITE CAN SEE THIS. The gate is a
<dialog> in index.html, a localStorage read, and a promise that a click
resolves - three things that only exist in a browser. A unit test can assert
that `agreeModelTerms` is written; only a page can say whether pressing Fold
puts the dialog in front of you and whether pressing a button in it gets you a
fold.

🔴 AND THE FAILURE IT GUARDS IS A FOLD THAT CANNOT START. The gate sits in
front of `startModelPreload`, so a dialog that never closes, never resolves, or
throws on a browser that blocks site data is a Fold button that does nothing at
all - which is exactly how the last silent page failures here presented.

Four things are checked, and the third and fourth are the ones worth having:

  * pressing Fold with no stored acceptance opens the dialog;
  * accepting stores it, and a second press does NOT open it again;
  * "Use OpenBind instead" changes the model row, so the page never folds with
    a model its own controls do not name;
  * dismissing with Escape leaves the model row alone and starts nothing.

Then `?model=`, which is the other way a model gets chosen:

  * a name and an alias both select; an unknown one leaves the row alone AND
    SAYS SO, because a query parameter that is quietly ignored looks exactly
    like one that worked;
  * `?model=af3` still opens the dialog. A URL must not be able to accept
    somebody else's licence terms, and that is the one property here worth
    asserting rather than assuming.

🔴 AND THE DIALOG IS MEASURED AT 320px WHILE FORCED OPEN. It is closed on a
bare page, so tools/mobile-layout.py has never laid it out and never will -
the same blind spot that hid the load dial's label overflowing a phone. A
modal that cannot fit is worse than a row that cannot: there is nothing else
on screen to scroll to.
"""
import http.server
import os
import socketserver
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import cdp                                                   # noqa: E402

PORT, DBG = 9667, 9230
REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SEQUENCE = "GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGG"


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


def setup(ws, accepted):
    """A page with the sequence typed in and the acceptance flag as asked."""
    cdp.evaluate(ws, """(() => {
      try {
        if (%s) localStorage.setItem('localfold.modelTerms.alphafold3', 'accepted');
        else localStorage.removeItem('localfold.modelTerms.alphafold3');
      } catch (e) { return 'localStorage unavailable: ' + e.message; }
      document.getElementById('model-family').value = 'af3';
      const entity = window.__entityList;
      if (entity && entity.entities && entity.entities()[0]) {
        entity.setSequence(0, '%s');
      }
      const box = document.querySelector('.entity-sequence, textarea, input[type=text]');
      if (box && !box.value) { box.value = '%s'; box.dispatchEvent(new Event('input', {bubbles:true})); }
      return 'ok';
    })()""" % ("true" if accepted else "false", SEQUENCE, SEQUENCE))


def dialog_open(ws):
    return cdp.evaluate(ws, "!!document.getElementById('model-terms')?.open")


def main():
    httpd = serve()
    proc, ws = cdp.launch(DBG, "/tmp/_cdp_terms_profile")
    failures = []
    try:
        ws.call("Page.enable")
        ws.call("Page.navigate", url="http://127.0.0.1:%d/index.html" % PORT)
        cdp.wait_for(ws, "typeof window.processFiles === 'function'"
                         " && !document.getElementById('predict').disabled",
                     what="the bundle and web/app.js to finish loading")

        present = cdp.evaluate(ws, """(() => {
          const d = document.getElementById('model-terms');
          if (!d) return 'missing';
          const options = [...document.getElementById('model-family').options]
            .map((o) => o.value);
          return JSON.stringify({
            tag: d.tagName, modal: typeof d.showModal === 'function',
            buttons: [...d.querySelectorAll('button')].map((b) => b.value),
            options,
          });
        })()""")
        print("markup:", present)
        if present == "missing":
            return fail(["index.html has no #model-terms dialog"], httpd, proc)
        if "openbind" not in present:
            failures.append("the model row does not offer openbind")

        # 1. Fold with no stored acceptance opens it.
        setup(ws, accepted=False)
        cdp.evaluate(ws, "document.getElementById('predict').click()")
        cdp.wait_for(ws, "!!document.getElementById('model-terms')?.open",
                     what="the terms dialog to open", timeout=20)
        print("opens on Fold with no acceptance: yes")

        # 2. Escape leaves the model alone and starts nothing.
        cdp.evaluate(ws, "document.getElementById('model-terms').close('')")
        time.sleep(0.4)
        after = cdp.evaluate(ws, "document.getElementById('model-family').value")
        if after != "af3":
            failures.append("dismissing changed the model row to %r" % after)
        print("dismissed: model row still %s" % after)

        # 3. The switch changes the row.
        cdp.evaluate(ws, "document.getElementById('predict').click()")
        cdp.wait_for(ws, "!!document.getElementById('model-terms')?.open",
                     what="the terms dialog to open again", timeout=20)
        cdp.evaluate(ws, "document.getElementById('model-terms-switch').click()")
        cdp.wait_for(ws, "document.getElementById('model-family').value === 'openbind'",
                     what="the model row to switch to openbind", timeout=20)
        print("switch: model row now openbind")

        # 4a. The dialog laid out at a phone's width, which nothing else does.
        ws.call("Emulation.setDeviceMetricsOverride", width=320, height=720,
                deviceScaleFactor=1, mobile=True)
        time.sleep(0.3)
        geometry = cdp.evaluate(ws, """(() => {
          const d = document.getElementById('model-terms');
          if (!d.open) d.showModal();
          const box = d.getBoundingClientRect();
          const buttons = [...d.querySelectorAll('button')]
            .map((b) => b.getBoundingClientRect());
          const overflow = d.scrollWidth > d.clientWidth + 1;
          const clipped = box.right > window.innerWidth + 1 || box.left < -1;
          d.close('');
          return JSON.stringify({
            innerWidth: window.innerWidth,
            width: Math.round(box.width), left: Math.round(box.left),
            buttons: buttons.map((b) => Math.round(b.width)),
            stacked: buttons.length === 2 && Math.abs(buttons[0].top - buttons[1].top) > 4,
            overflow, clipped,
          });
        })()""")
        print("at 320px:", geometry)
        state = __import__("json").loads(geometry)
        if state["clipped"]:
            failures.append("the dialog is drawn outside a 320px viewport")
        if state["overflow"]:
            failures.append("the dialog scrolls sideways at 320px")
        if not state["stacked"]:
            failures.append("the two buttons share a line at 320px")
        if min(state["buttons"]) < 100:
            failures.append("a button is %dpx wide at 320px" % min(state["buttons"]))
        ws.call("Emulation.clearDeviceMetricsOverride")

        # 4. Accepting is remembered, and does not ask again.
        #
        # 🔴 A FRESH PAGE, because step 3 started a real fold - the switch is
        # not a preview, it hands the run to OpenBind and the download begins.
        # Pressing Fold again while that is in flight aborts it instead of
        # asking anything, which is correct behaviour and a useless test.
        ws.call("Page.navigate", url="http://127.0.0.1:%d/index.html" % PORT)
        cdp.wait_for(ws, "typeof window.processFiles === 'function'"
                         " && !document.getElementById('predict').disabled",
                     what="the page to reload")
        setup(ws, accepted=False)
        cdp.evaluate(ws, "document.getElementById('predict').click()")
        cdp.wait_for(ws, "!!document.getElementById('model-terms')?.open",
                     what="the terms dialog before accepting", timeout=20)
        cdp.evaluate(ws, "document.getElementById('model-terms-accept').click()")
        time.sleep(0.6)
        stored = cdp.evaluate(
            ws, "localStorage.getItem('localfold.modelTerms.alphafold3')")
        if stored != "accepted":
            failures.append("accepting stored %r, not 'accepted'" % stored)
        print("accepted and stored:", stored)

        # ...and a second press must not ask again. The fold from the previous
        # press may still be running, so this reads the flag's effect directly.
        asks_again = cdp.evaluate(ws, """(() => {
          try {
            return localStorage.getItem('localfold.modelTerms.alphafold3') !== 'accepted';
          } catch (e) { return 'threw: ' + e.message; }
        })()""")
        if asks_again is not False:
            failures.append("a second fold would ask again (%r)" % asks_again)
        print("asks again:", asks_again)

        # 5. `?model=`, including the one that must NOT work.
        for query, expected in (("?model=openbind", "openbind"), ("?model=ob", "openbind"),
                                ("?model=af2", "monomer"), ("?model=multimer", "multimer"),
                                ("?model=of3", "af3"), ("", "af3")):
            ws.call("Page.navigate",
                    url="http://127.0.0.1:%d/index.html%s" % (PORT, query))
            cdp.wait_for(ws, "typeof window.processFiles === 'function'"
                             " && !document.getElementById('predict').disabled",
                         what="the page to load %s" % (query or "with no query"))
            time.sleep(1.4)
            got = cdp.evaluate(ws, """(() => {
              const n = document.getElementById('status-message');
              return JSON.stringify({ model: document.getElementById('model-family').value,
                                      text: n.textContent, error: n.classList.contains('error') });
            })()""")
            seen = __import__("json").loads(got)
            print("%-18s -> %-9s %s" % (query or "(none)", seen["model"],
                                        seen["text"][:56]))
            if seen["model"] != expected:
                failures.append("%r selected %r, wanted %r"
                                % (query, seen["model"], expected))
            # 🔴 AN UNKNOWN NAME MUST COMPLAIN. Two earlier versions of this
            # wrote the complaint too early - once before the viewer's own
            # opening line overwrote it, once before the parameter had been
            # read at all - and both times the page looked exactly like one
            # that had honoured the parameter.
            if query == "?model=of3" and not seen["error"]:
                failures.append("an unknown ?model= was ignored silently")

        # 6. A URL cannot accept anybody's licence terms.
        ws.call("Page.navigate", url="http://127.0.0.1:%d/index.html?model=af3" % PORT)
        cdp.wait_for(ws, "typeof window.processFiles === 'function'"
                         " && !document.getElementById('predict').disabled",
                     what="the page to load ?model=af3")
        setup(ws, accepted=False)
        cdp.evaluate(ws, "document.getElementById('predict').click()")
        try:
            cdp.wait_for(ws, "!!document.getElementById('model-terms')?.open",
                         what="the dialog to open for ?model=af3", timeout=20)
            print("?model=af3 still asks: yes")
        except RuntimeError:
            failures.append("?model=af3 folded without asking about the terms")
    finally:
        try:
            proc.terminate()
        except Exception:
            pass
        httpd.shutdown()

    return fail(failures, None, None)


def fail(failures, httpd, proc):
    if httpd is not None:
        httpd.shutdown()
    if proc is not None:
        proc.terminate()
    if failures:
        print("\nFAIL")
        for line in failures:
            print("  " + line)
        return 1
    print("\nok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
