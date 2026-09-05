"""Does a complex's contact map carry the RIGHT chain boundaries while it folds?

    python3 tools/trunk-chains.py

🔴 IT EXISTS BECAUSE THE BUG IS TRANSIENT AND EVERY OTHER CHECK LOOKS AT THE
END. The heatmap panel rules a divider wherever the chain changes and reads the
chains off the RENDERER, which fills them in when a structure is parsed. During
an AF3 trunk there is no structure - that is the whole reason web/app.js pushes
the map at the renderer directly - so the page writes the layout itself. It used
to write it only while `renderer.chains` was empty, to avoid fighting the
parser; on a SECOND fold it is not empty, it holds the PREVIOUS fold's layout,
and the new fold's map was ruled with the old fold's lines for the whole trunk.

So this folds twice with different chain layouts and samples what the renderer
holds, 40 ms apart, through both. Each sample is `frames:shape`, where the shape
is the chain ids run-length encoded - `A37B26` is 37 residues of chain A then 26
of B. What the fix has to show is the second fold reaching A37B26 while frames
is still 0. Before it:

    fold 2  0:A37  1:A37B26  2:A37B26 ...   <- the whole trunk unruled
    after   0:A37  0:A37B26  1:A37B26 ...   <- ruled from the first push

Nothing here needs the network: the MSA is left at none.
"""
import json
import subprocess
import sys
import tempfile
import time
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cdp                                                   # noqa: E402

PORT, DBG = 8735, 9235
REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
A = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTS"
B = "GWSTELEKHREELKEFLKKEGITLGF"

# 🔴 SAMPLED, NOT READ AFTERWARDS. By the time a fold finishes the parser has
# overwritten `renderer.chains` with the right answer, so a reading at the end
# passes whether or not the trunk was ruled correctly.
SAMPLER = """(() => {
  window.__chainLog = [];
  const reg = window.py2dmol_viewers || {};
  const r = reg[Object.keys(reg)[0]]?.renderer;
  window.__chainTimer && clearInterval(window.__chainTimer);
  window.__chainTimer = setInterval(() => {
    if (!r) return;
    const frames = r.objectsData?.[r.currentObjectName]?.frames?.length ?? 0;
    const shape = (r.chains || []).join('').replace(/(.)\\1*/g, (m) => m[0] + m.length);
    const now = frames + ':' + shape;
    const log = window.__chainLog;
    if (log[log.length - 1] !== now) log.push(now);
  }, 40);
  document.getElementById('predict').click();
})()"""


def main():
    server = subprocess.Popen([sys.executable, "-m", "http.server", str(PORT)],
                              cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)
    proc, ws = cdp.launch(DBG, tempfile.mkdtemp(prefix="gpu-chrome-chains"))
    try:
        ws.call("Page.enable")
        ws.call("Runtime.enable")
        ws.call("Page.navigate", url="http://127.0.0.1:%d/index.html" % PORT)
        cdp.wait_for(ws, "typeof window.processFiles === 'function'"
                         " && !document.getElementById('predict').disabled",
                     what="the page to finish loading")

        def setup(chains):
            return cdp.evaluate(ws, """(() => {
              const list = window.__entityList;
              list.set(%s.map((value) => (
                { type: 'protein', copies: 1, value, modifications: [] })));
              const set = (id, v) => { const el = document.getElementById(id);
                el.value = v; el.dispatchEvent(new Event('change', { bubbles: true })); };
              set('model-family', 'af3'); set('recycles', '1'); set('msa-mode', 'none');
              return JSON.stringify(list.read().map((e) => e.value.length));
            })()""" % json.dumps(chains))

        def fold(tag):
            cdp.evaluate(ws, SAMPLER)
            cdp.wait_for(ws, """(() => { const s = document.getElementById('status-message');
              return /pLDDT/.test(s ? s.textContent : ''); })()""",
                         timeout=600, what="fold " + tag)
            time.sleep(0.5)
            return json.loads(cdp.evaluate(ws, "JSON.stringify(window.__chainLog)"))

        print("fold 1, one chain :", setup([A]))
        first = fold("1")
        print("  ", " ".join(first))
        print("fold 2, two chains:", setup([A, B]))
        second = fold("2")
        print("  ", " ".join(second))

        want = "A%dB%d" % (len(A), len(B))
        trunk = [s for s in second if s.startswith("0:")]
        if not any(s == "0:" + want for s in trunk):
            print("FAIL: the trunk never carried %s; it held %s"
                  % (want, ", ".join(trunk) or "nothing"))
            return 1
        print("ok: the trunk carries", want, "before the first frame lands")
        return 0
    finally:
        proc.kill()
        server.kill()


if __name__ == "__main__":
    raise SystemExit(main())
