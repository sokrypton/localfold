"""A result belongs to the model that made it, and the viewers say so.

    python3 tools/check-model-pending.py

🔴 SWITCHING THE MODEL ROW USED TO LEAVE THE PREVIOUS MODEL'S ANSWER ON SCREEN
under the new model's name. The structure, the contact map and the confidence
numbers look exactly the same whichever row is selected above them, so there
is nothing in the picture that says it is the other model's - reported as
wanting the viewers to show PENDING instead of the previous result.

WHAT IT CHECKS, on a real page with a real result in it:

  * a result ingested under AlphaFold 3 is NOT veiled while that row is set;
  * moving the row to another model veils the structure box and the map box,
    names the model the page is now set to, and takes the scores card away -
    a stale pLDDT reads as a measurement rather than as a leftover;
  * moving it BACK unveils, because the result really is that model's;
  * and the AF2 number row does it too, since `chosenFamily` reads that select
    as well and a veil hung on one row only is a model switched in silence.

🔴 THE VEIL IS MEASURED AS PIXELS, not as a class. A class name is set by the
page and says nothing about whether a stylesheet arrived; `.result-pending`
with no rule behind it is a viewer that still shows the wrong model's answer.
The structure box is screenshotted before and after.

🔴 AND THE RESULT GETS ONTO THE PAGE THROUGH tools/colab_backend.py, WHICH IS A
FIXTURE HERE AND NOT THE SUBJECT. A fold needs weights and a card; the broker
lets a structure be pushed to a reader's page through the code that ingests a
real one, which is the only way this machine can put a prediction on screen.
See tools/check-colab-bridge.py, whose arms this borrows.
"""
import base64
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cdp                                                   # noqa: E402

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PORT = int(os.environ.get("PENDING_PORT", "8793"))
CDP_PORT = int(os.environ.get("PENDING_CDP_PORT", "9393"))
READER_CDP_PORT = int(os.environ.get("PENDING_READER_CDP_PORT", "9394"))
TOKEN = "check-model-pending-token"
BASE = f"http://127.0.0.1:{PORT}"

bad = []


def call(route, body=None):
    url = f"{BASE}{route}" + ("&" if "?" in route else "?") + f"t={TOKEN}"
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        url, data=data, headers={"content-type": "application/json"},
        method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(request, timeout=20) as answer:
            return answer.status, json.loads(answer.read() or b"{}")
    except urllib.error.HTTPError as refused:
        return refused.code, json.loads(refused.read() or b"{}")


def tiny_pdb():
    rows = ["ATOM  %5d  CA  ALA A%4d    %8.3f%8.3f%8.3f  1.00 50.00           C"
            % (i + 1, i + 1, 3.8 * i, 0.0, 0.0) for i in range(6)]
    return "\n".join(rows) + "\nEND\n"


def box_shot(ws, selector="canvasContainer"):
    """The structure box as pixels, so the veil is measured where it is drawn."""
    at = cdp.evaluate(ws, """(() => {
      const box = document.getElementById(%s).getBoundingClientRect();
      return { x: Math.round(box.x), y: Math.round(box.y),
               w: Math.round(box.width), h: Math.round(box.height) };
    })()""" % json.dumps(selector))
    shot = ws.call("Page.captureScreenshot", format="png", captureBeyondViewport=False,
                   clip={"x": at["x"], "y": at["y"], "width": at["w"],
                         "height": at["h"], "scale": 0.25})
    return base64.b64decode(shot["data"])


def veil_of(ws, box_id):
    return cdp.evaluate(ws, """(() => {
      const box = document.getElementById(%s);
      if (box === null) return null;
      const after = getComputedStyle(box, '::after');
      return { marked: box.classList.contains('result-pending'),
               says: after.content,
               paint: after.backgroundColor,
               scores: getComputedStyle(
                 document.getElementById('predictionScoresBox')).display };
    })()""" % json.dumps(box_id))


def set_row(ws, row_id, value):
    cdp.evaluate(ws, """(() => {
      const row = document.getElementById(%s);
      row.value = %s;
      row.dispatchEvent(new Event('change', { bubbles: true }));
      return row.value;
    })()""" % (json.dumps(row_id), json.dumps(value)))
    time.sleep(0.4)


print(f"starting the broker on {PORT} (it is the fixture, not the subject)…")
backend = subprocess.Popen(
    [sys.executable, "tools/colab_backend.py", "--port", str(PORT),
     "--cdp-port", str(CDP_PORT), "--token", TOKEN,
     "--profile", "/tmp/localfold-pending-runtime"],
    cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
reader = None
try:
    ready, deadline = False, time.time() + 180
    while time.time() < deadline and not ready:
        line = backend.stdout.readline()
        if line == "" and backend.poll() is not None:
            break
        if line.startswith("BACKEND "):
            ready = True
        elif line.strip():
            print("  " + line.rstrip())
    if not ready:
        print("FAIL: the broker never started")
        raise SystemExit(1)

    runtime_ws = None
    for target in json.load(urllib.request.urlopen(
            f"http://127.0.0.1:{CDP_PORT}/json/list")):
        if target.get("type") == "page":
            runtime_ws = cdp.WS(target["webSocketDebuggerUrl"])
            break
    runtime_ws.call("Network.enable")
    # Nothing here may fold: the weights stay away for the whole run.
    runtime_ws.call("Network.setBlockedURLs", urls=["*huggingface.co*"])
    # ...and the fold is HELD by taking the runtime's commands away, which is
    # what makes `folding` true so the reader attaches. See check-colab-bridge.
    runtime_ws.call("Network.setBlockedURLs", urls=["*huggingface.co*", "*/out*"])
    time.sleep(1.0)
    call("/in", {"op": "fold", "payload": {
        "entities": [{"type": "protein", "value": "GWSTELEKHRSVQ", "copies": 1}]}})

    reader, reader_ws = cdp.launch(READER_CDP_PORT, "/tmp/localfold-pending-reader")
    reader_ws.call("Page.enable")
    reader_ws.call("Runtime.enable")
    reader_ws.call("Page.addScriptToEvaluateOnNewDocument", source="""
      window.__pageErrors = [];
      addEventListener('error', (e) => window.__pageErrors.push(String(e.message)));
    """)
    reader_ws.call("Page.navigate",
                   url=f"{BASE}/index.html?backend=colab&t={TOKEN}")
    cdp.wait_for(reader_ws, "!!window.__entityList", 120, "the reader's page")
    attached, deadline = "", time.time() + 30
    while time.time() < deadline:
        attached = cdp.evaluate(reader_ws,
            "document.getElementById('status-message')?.textContent ?? ''")
        if "already running" in attached:
            break
        time.sleep(0.5)
    if "already running" not in attached:
        bad.append("the reader never attached to the held fold, so no result"
                   " could be put in front of it")

    # THE RESULT, as a fold's own ingestion sees it: the structure AND the
    # prediction behind it, whose `model` is how the page knows whose it is.
    cdp.evaluate(runtime_ws, """(async () => {
      const bridge = await import('/web/colab-bridge.js');
      const pdb = %s;
      bridge.tapOut('result', { pdb, atoms: 6, status: 'AlphaFold 3 · 6 residues',
        scores: {}, confidence: { meanPlddt: 88.1, ptm: 0.71 },
        predJson: JSON.stringify({ model: 'AlphaFold 3', stem: 'pending_test',
                                   pdb, confidence: { meanPlddt: 88.1, ptm: 0.71 } }) });
      return true;
    })()""" % json.dumps(tiny_pdb()))
    got, deadline = {}, time.time() + 60
    while time.time() < deadline:
        got = cdp.evaluate(reader_ws, """(() => {
          const reg = window.py2dmol_viewers || {};
          const r = reg[Object.keys(reg)[0]] && reg[Object.keys(reg)[0]].renderer;
          const o = r && r.objectsData ? r.objectsData[r.currentObjectName] : null;
          return { positions: o && o.frames && o.frames[0]
                     ? (o.frames[0].coords || []).length : 0,
                   status: document.getElementById('status-message')?.textContent ?? '' };
        })()""")
        if got.get("positions", 0) >= 6:
            break
        time.sleep(0.5)
    print(f"  a result is on screen: {got.get('positions')} positions,"
          f" {got.get('status')!r}")
    if got.get("positions", 0) < 6:
        bad.append("no result reached the page, so the rest of this measures"
                   " an empty viewer")

    # 1 · its own model's row: no veil.
    rest = veil_of(reader_ws, "canvasContainer")
    before = box_shot(reader_ws)
    print(f"  under AlphaFold 3: marked={rest['marked']} says={rest['says']}")
    if rest["marked"]:
        bad.append("the result is veiled under the very model that made it")

    # 2 · another model: veiled, named, and the numbers gone.
    set_row(reader_ws, "model-family", "boltz2")
    moved = veil_of(reader_ws, "canvasContainer")
    map_veil = veil_of(reader_ws, "heatmapContainer")
    after = box_shot(reader_ws)
    print(f"  under Boltz-2:      marked={moved['marked']} says={moved['says']}"
          f" paint={moved['paint']} scores={moved['scores']}")
    if not moved["marked"]:
        bad.append("switching the model left the previous model's result"
                   " unmarked - the viewers claim it as this model's")
    if "Boltz-2" not in (moved["says"] or ""):
        bad.append(f"the veil says {moved['says']!r}, which does not name the"
                   " model the page is now set to")
    if moved["paint"] in ("rgba(0, 0, 0, 0)", "transparent"):
        bad.append("the veil has no paint of its own, so the structure it is"
                   " meant to cover shows through - the stylesheet did not"
                   " arrive, whatever the class says")
    if moved["scores"] != "none":
        bad.append("the confidence card is still up under another model -"
                   " a stale pLDDT reads as a measurement")
    if not map_veil["marked"]:
        bad.append("the contact map is not veiled, only the structure")
    if before == after:
        bad.append("the structure box is pixel-identical before and after the"
                   " switch - nothing was actually drawn over it")
    else:
        print(f"  the box redrew: {len(before)} -> {len(after)} bytes of png")

    # 3 · back again: the result really is AlphaFold 3's.
    set_row(reader_ws, "model-family", "af3")
    back = veil_of(reader_ws, "canvasContainer")
    print(f"  back to AlphaFold 3: marked={back['marked']} scores={back['scores']}")
    if back["marked"]:
        bad.append("switching back to the model that made the result left the"
                   " veil up - the answer on screen is that model's")
    if back["scores"] == "none":
        bad.append("the confidence card did not come back with its own model")

    # 4 · the AF2 number row moves the family too.
    set_row(reader_ws, "model-family", "monomer")
    set_row(reader_ws, "af2Model", "3")
    af2 = veil_of(reader_ws, "canvasContainer")
    print(f"  under AlphaFold 2 model 3: marked={af2['marked']} says={af2['says']}")
    if not af2["marked"]:
        bad.append("the AF2 number row changed the family without veiling -"
                   " chosenFamily reads that select too")

    errors = cdp.evaluate(reader_ws, "window.__pageErrors || []")
    if errors:
        bad.append(f"the page threw: {errors[:2]}")
finally:
    if reader is not None:
        reader.kill()
    backend.send_signal(signal.SIGINT)
    try:
        backend.wait(timeout=10)
    except subprocess.TimeoutExpired:
        backend.kill()

print()
for line in bad:
    print("FAIL: " + line)
print("model pending: " + ("FAILED" if bad else "ok"))
raise SystemExit(1 if bad else 0)
