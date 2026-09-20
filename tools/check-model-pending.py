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
    as well and a veil hung on one row only is a model switched in silence;
  * THE DOWNLOAD ROW OFFERS WHAT THERE IS TO DOWNLOAD: nothing before the
    first fold, the fold's own files once there is one, neither while another
    model is selected - and the SESSION button all the while, because that one
    is about what the viewer is showing rather than about what was predicted.
    A disabled button says why it is off, and takes its own title back when it
    comes on - and in Colab mode the PDB button is CLICKED, because a button
    offered over a prediction the runtime never filled looks identical from
    the outside;
  * A PREDICTION CROSSES WITH ITS TYPES: JSON has no typed arrays, and
    flattened to plain ones they look right until something slices one -
    `download-all` died on "values.subarray is not a function" with a perfect
    structure on screen;
  * THE DEV PANEL DESCRIBES THE MACHINE THAT FOLDED: its rows are the
    runtime's own, with the runtime's card memory in them, and its header says
    which machine folded and which one is showing it - where before the
    phases were timed on the reader's clock and headed with the reader's user
    agent, about a browser that did nothing but draw;
  * AND WHERE THE SEQUENCE GOES, which is the footer's one job: the privacy
    line tracks the alignment mode and names the SERVICE rather than linking a
    site, and the provenance links are not back in the row beside it;
  * THE PAGE SAYS WHERE FOLD RUNS: a badge naming the runtime and its card,
    its pulse going amber when that runtime stops answering - at the same
    twenty seconds after which a fold in flight gives up, because a badge that
    still says connected while the fold gives up is the page saying two things
    at once - and a Disconnect that STOPS the service, because the browser on
    that machine holds its GPU for as long as it lives. After it the page is a
    VIEWER: no folding, every control that shapes the next one disabled and
    saying why, and the structure, plots and downloads it already has still
    there. It does not RELOAD - the server it was served by is what stopped.

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
    """The structure box as pixels, so the veil is measured where it is drawn.

    🔴 IN PAGE COORDINATES, PAST THE FOLD. The clip used to be the viewport
    rectangle with `captureBeyondViewport` off, so anything that made the page
    head taller - the Colab badge taking its own row on a narrow window - moved
    the box below the fold and BOTH shots came back identical. The arm then
    reported that nothing had been drawn over the structure, which was a
    statement about the camera rather than about the veil.
    """
    at = cdp.evaluate(ws, """(() => {
      const box = document.getElementById(%s).getBoundingClientRect();
      return { x: Math.round(box.x + scrollX), y: Math.round(box.y + scrollY),
               w: Math.round(box.width), h: Math.round(box.height) };
    })()""" % json.dumps(selector))
    shot = ws.call("Page.captureScreenshot", format="png", captureBeyondViewport=True,
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


def downloads_of(ws):
    """What the row offers, as a reader finds it: on, off, and why."""
    return cdp.evaluate(ws, """(() => {
      const row = document.getElementById('downloads');
      const one = (id) => {
        const button = document.getElementById(id);
        return button === null ? null
          : { off: button.disabled, why: button.getAttribute('title') ?? '' };
      };
      return { shown: row === null ? null : getComputedStyle(row).display,
               pdb: one('download-pdb'), all: one('download-all'),
               session: one('saveStateButton') };
    })()""")


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

    # 0 · BEFORE ANYTHING IS FOLDED, there is nothing to download and the row
    #     says so. A button that writes nothing is worse than no button: both
    #     download handlers read `activePrediction()` and return in silence.
    empty = downloads_of(reader_ws)
    print(f"  with nothing folded: row {empty['shown']},"
          f" pdb off={empty['pdb']['off']} session off={empty['session']['off']}")
    if empty["shown"] != "none":
        bad.append("the download row is up before anything has been folded")
    if not empty["pdb"]["off"] or not empty["all"]["off"]:
        bad.append("the structure downloads are live with no structure to give")

    # THE RESULT, as a fold's own ingestion sees it: the structure AND the
    # prediction behind it, whose `model` is how the page knows whose it is.
    # 🔴 THE DEV ROWS GO BEFORE THE RESULT, BECAUSE THAT IS WHEN THEY ARRIVE:
    # a reader consumes events while it is FOLLOWING a fold, and pushed
    # afterwards it is following nothing. The first version of this arm pushed
    # them after the result, measured an empty panel, and blamed the feature.
    # They are the runtime's own log - its card's memory, its clock.
    cdp.evaluate(runtime_ws, """(async () => {
      const bridge = await import('/web/colab-bridge.js');
      bridge.tapOut('dev', { reset: 'fold · af3 · alignment none · 3 recycles' });
      bridge.tapOut('dev', { phase: 'Trunk', ms: 4120, atMs: 40,
                             resident: 512.5, peak: 901.25, rise: 388.75 });
      bridge.tapOut('dev', { phase: 'Folding 8/8', ms: 990, atMs: 4160,
                             resident: 128.5, peak: 901.25, rise: 0 });
      return true;
    })()""")
    cdp.evaluate(runtime_ws, """(async () => {
      const bridge = await import('/web/colab-bridge.js');
      const pdb = %s;
      bridge.tapOut('result', { pdb, atoms: 6, status: 'AlphaFold 3 · 6 residues',
        scores: {}, confidence: { meanPlddt: 88.1, ptm: 0.71 },
        predJson: JSON.stringify({ model: 'AlphaFold 3', stem: 'pending_test', pdb,
          confidence: { meanPlddt: 88.1, ptm: 0.71,
            // 🔴 THE SHAPE A TYPED ARRAY TRAVELS IN. JSON has none, so the
            // runtime tags each one with its kind; flattened to a plain array
            // instead, `download-all` died on "values.subarray is not a
            // function" while the picture beside it was perfect.
            plddt: { __typed: 'Float32Array', v: [88.1, 90.2, 71.0, 65.5, 80.0, 92.3] },
            predictedAlignedError: { __typed: 'Float32Array',
                                     v: Array.from({ length: 36 }, (unused, i) => i / 4) } } }) });
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

    # 1 · its own model's row: no veil, and the files are there.
    ready = downloads_of(reader_ws)
    print(f"  with a fold on screen: row {ready['shown']},"
          f" pdb off={ready['pdb']['off']} session off={ready['session']['off']}")
    if ready["shown"] == "none":
        bad.append("a fold landed and the download row stayed hidden")
    if ready["pdb"]["off"] or ready["all"]["off"] or ready["session"]["off"]:
        bad.append("a fold landed and its own downloads are still off")
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
    # ...and the files go with the picture, because offering the covered
    # model's structure is the same claim the veil exists to stop making.
    veiled = downloads_of(reader_ws)
    print(f"  under Boltz-2:      pdb off={veiled['pdb']['off']}"
          f" ({veiled['pdb']['why']!r}) session off={veiled['session']['off']}")
    if not veiled["pdb"]["off"] or not veiled["all"]["off"]:
        bad.append("the downloads still offer the other model's fold while"
                   " its picture is veiled - the same claim, one control along")
    if "another model" not in (veiled["pdb"]["why"] or ""):
        bad.append(f"the disabled download says {veiled['pdb']['why']!r},"
                   " which does not say why it is off")
    if veiled["session"]["off"]:
        bad.append("the session download went off with the fold's - it is"
                   " about what the VIEWER is showing, which has not changed")
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
    restored = downloads_of(reader_ws)
    if restored["pdb"]["off"] or restored["all"]["off"]:
        bad.append("the downloads did not come back with the model that made"
                   " the fold on screen")
    if restored["pdb"]["why"] != ready["pdb"]["why"]:
        bad.append(f"the button came back wearing the reason it was off:"
                   f" {restored['pdb']['why']!r}")

    # 4 · the AF2 number row moves the family too.
    set_row(reader_ws, "model-family", "monomer")
    set_row(reader_ws, "af2Model", "3")
    af2 = veil_of(reader_ws, "canvasContainer")
    print(f"  under AlphaFold 2 model 3: marked={af2['marked']} says={af2['says']}")
    if not af2["marked"]:
        bad.append("the AF2 number row changed the family without veiling -"
                   " chosenFamily reads that select too")

    # 5 · AND THE DOWNLOAD ACTUALLY WRITES SOMETHING, in Colab mode, which is
    #     the question a flag cannot answer: `download-pdb` reads the
    #     prediction the RUNTIME sent, and a page that is offering a button
    #     over an empty prediction looks identical from the outside.
    set_row(reader_ws, "model-family", "af3")
    wrote = cdp.evaluate(reader_ws, """(async () => {
      const blobs = [];
      const made = URL.createObjectURL;
      URL.createObjectURL = (b) => { blobs.push(b); return made.call(URL, b); };
      document.getElementById('download-pdb').click();
      for (let tick = 0; tick < 40 && blobs.length === 0; tick += 1) {
        await new Promise((done) => setTimeout(done, 100));
      }
      URL.createObjectURL = made;
      const text = blobs[0] ? await blobs[0].text() : '';
      return { bytes: text.length,
               atoms: (text.match(/^ATOM/gm) || []).length };
    })()""")
    print(f"  the PDB button wrote {wrote['bytes']} bytes,"
          f" {wrote['atoms']} atoms")
    if wrote.get("atoms", 0) < 6:
        bad.append(f"the PDB download wrote {wrote.get('atoms')} atoms in"
                   " Colab mode - the button is offered over a prediction the"
                   " runtime did not fill")

    # 5b · AND ITS TYPED ARRAYS ARE TYPED ARRAYS. A prediction crosses as JSON,
    #      which has none: flattened to plain arrays they look right until
    #      something slices one, and `download-all` died on
    #      "values.subarray is not a function" with the structure on screen.
    kinds = cdp.evaluate(reader_ws, """(() => {
      const pred = window.__lastPrediction ? window.__lastPrediction() : null;
      const c = pred?.confidence ?? {};
      return { plddt: c.plddt?.constructor?.name ?? 'missing',
               pae: c.predictedAlignedError?.constructor?.name ?? 'missing',
               sliceable: typeof c.predictedAlignedError?.subarray === 'function',
               plddtHead: c.plddt ? Array.from(c.plddt).slice(0, 2) : null };
    })()""")
    print(f"  the prediction's arrays: {kinds}")
    if kinds.get("plddt") != "Float32Array" or kinds.get("pae") != "Float32Array":
        bad.append(f"the runtime's typed arrays arrived as {kinds.get('plddt')}"
                   f"/{kinds.get('pae')} - anything that slices one throws")
    if not kinds.get("sliceable"):
        bad.append("the PAE cannot be sliced, which is what the download does")

    # 6 · THE DEV PANEL IS THE RUNTIME'S, NOT THIS MACHINE'S. Its rows were
    #     timed on the reader's clock and filed under the reader's (empty)
    #     device, under a header naming the reader's browser - a report
    #     about a machine that did nothing but draw.
    report, deadline = "", time.time() + 30
    while time.time() < deadline:
        report = cdp.evaluate(reader_ws, """(() => {
          const button = document.getElementById('dev-toggle');
          if (button === null) return '';
          const panel = document.getElementById('dev-panel');
          if (panel === null || panel.hidden) button.click();
          return document.querySelector('#dev-panel pre')?.textContent ?? '';
        })()""")
        if "Trunk" in report:
            break
        time.sleep(0.5)
    head = "\n".join(report.split("\n")[:3])
    print(f"  the dev report is headed:\n    " + head.replace("\n", "\n    "))
    if "folded on: the Colab runtime" not in report:
        bad.append("the dev report does not say which machine folded - its"
                   " header describes the browser that only drew")
    if "shown in:" not in report:
        bad.append("the dev report dropped this browser entirely; it names"
                   " both or it is guessing")
    if "901.2" not in report and "901.3" not in report:
        bad.append("the runtime's own memory is not in the report - the rows"
                   " were recorded here instead of there")
    if "already running" in report:
        bad.append("the reader recorded its own replay of the runtime's"
                   " status line as phases, so the timeline is this browser's")
    if "device memory: not measured" in report:
        bad.append("the report still claims this machine's device memory was"
                   " not measured, which is a statement about the wrong one")

    # 7 · THE BADGE: where Fold runs, and whether it is still there.
    badge = cdp.evaluate(reader_ws, """(() => {
      const box = document.getElementById('colab-status');
      if (box === null) return null;
      return { state: box.dataset.state ?? '',
               says: box.querySelector('.colab-said')?.textContent ?? '',
               dot: getComputedStyle(box.querySelector('.colab-dot')).backgroundColor,
               leave: box.querySelector('button')?.textContent ?? '' };
    })()""")
    print(f"  the badge: {badge}")
    if badge is None:
        bad.append("a page folding on a Colab runtime says nowhere that it is")
    else:
        if badge["state"] != "live":
            bad.append(f"the badge reads {badge['state']!r} with the runtime"
                       " answering three times a second")
        # 🔴 THE CARD'S NAME IS THE POINT OF THE BADGE, not decoration: 'nvidia
        # turing' is the GPU and 'google swiftshader' is the CPU wearing its
        # clothes, and a reader cannot tell a slow fold from a CPU fold without
        # it. Its VALUE is this machine's, so what is asserted is that one
        # arrived at all.
        named = badge["says"].split("·")[-1].strip() if "·" in badge["says"] else ""
        if named == "":
            bad.append(f"the badge says {badge['says']!r} and never names the"
                       " card - a CPU fallback reads exactly like a GPU here")
        if "Colab" not in badge["says"]:
            bad.append(f"the badge says {badge['says']!r}, which does not name"
                       " where the fold is running")
        if badge["dot"] in ("rgba(0, 0, 0, 0)", "rgb(156, 163, 175)"):
            bad.append("the badge's pulse has no colour of its own - the"
                       " stylesheet did not arrive")
        if "isconnect" not in badge["leave"]:
            bad.append("the badge offers no way back to folding here")

    # 8 · ...and it goes amber when the runtime does. Twenty seconds is the
    #     fold loop's own bound and the two must agree, so this waits it out.
    runtime_ws.call("Page.navigate", url="about:blank")
    gone, deadline = "", time.time() + 45
    while time.time() < deadline:
        gone = cdp.evaluate(reader_ws, """(() => {
          const box = document.getElementById('colab-status');
          return (box?.dataset.state ?? '') + '|' +
                 (box?.querySelector('.colab-said')?.textContent ?? '');
        })()""")
        if gone.startswith("gone|"):
            break
        time.sleep(1.0)
    print(f"  with the runtime away: {gone!r}")
    if not gone.startswith("gone|"):
        bad.append("the runtime went away and the badge still says it is"
                   " there - a fold pressed now waits out its own bound")
    runtime_ws.call("Page.navigate",
                    url=f"{BASE}/index.html?role=runtime&t={TOKEN}")

    # 9 · Disconnect leaves Colab mode, which is the whole of the way back.
    cdp.evaluate(reader_ws, """(() => {
      document.getElementById('colab-status').querySelector('button').click();
      return true;
    })()""")
    # 🔴 THE PAGE MUST NOT RELOAD, because the server it was served BY is what
    # just stopped - so this waits for the page's STATE to change and then asks
    # the same document what it has become.
    after, deadline = {}, time.time() + 40
    while time.time() < deadline:
        after = cdp.evaluate(reader_ws, """(() => {
          const badge = document.getElementById('colab-status');
          return {
            url: location.search,
            badge: badge?.querySelector('.colab-said')?.textContent ?? 'gone',
            button: badge?.querySelector('button') == null ? 'gone' : 'still here',
            fold: !!document.getElementById('predict')?.disabled,
            model: !!document.getElementById('model-family')?.disabled,
            why: document.getElementById('predict')?.getAttribute('title') ?? '',
            says: document.getElementById('status-message')?.textContent ?? '',
            alive: typeof window.__entityList === 'object',
            downloads: !document.getElementById('download-pdb')?.disabled,
          };
        })()""")
        if after.get("fold"):
            break
        time.sleep(0.5)
    print(f"  after Disconnect: {after}")
    if "backend=colab" in after.get("url", ""):
        bad.append(f"Disconnect left the page on {after.get('url')!r}, so it"
                   " would ask a stopped runtime for the next fold")
    if not after.get("alive"):
        bad.append("the page did not survive Disconnect - it reloaded from the"
                   " server it had just told to stop")
    if not after.get("fold") or not after.get("model"):
        bad.append(f"folding is still offered after Disconnect (fold disabled"
                   f" {after.get('fold')}, model row {after.get('model')}) -"
                   " the page can only show what it already has")
    if "notebook" not in (after.get("why") or ""):
        bad.append(f"the disabled Fold button says {after.get('why')!r}, which"
                   " does not say why it cannot be pressed")
    if "stopped" not in after.get("says", ""):
        bad.append(f"the page says {after.get('says')!r} after Disconnect,"
                   " which does not tell the reader what just happened")
    if "stopped" not in (after.get("badge") or ""):
        bad.append(f"the badge reads {after.get('badge')!r} rather than saying"
                   " the runtime has stopped")
    if after.get("button") != "gone":
        bad.append("the badge still offers Disconnect on a runtime that has"
                   " already been disconnected")
    # 🔴 AND WHAT IS ALREADY HERE STAYS, which is the whole of what the page is
    # for now.
    if not after.get("downloads"):
        bad.append("the downloads went with the runtime - the fold that was"
                   " already made is still this page's to give")
    # ...and the service really is gone: the GPU is freed when that process and
    # its browser are, and not before.
    stopped, deadline = False, time.time() + 30
    while time.time() < deadline:
        try:
            call("/health")
        except Exception:                                     # noqa: BLE001
            stopped = True
            break
        time.sleep(0.5)
    print(f"  the fold service still answers: {not stopped};"
          f" the broker exited: {backend.poll() is not None}")
    if not stopped:
        bad.append("the broker is still serving after Disconnect - the"
                   " runtime's browser is still holding its card")
    if backend.poll() is None:
        bad.append("the broker process is still running after Disconnect, so"
                   " the notebook cell never ends and the GPU stays taken")

    # 10 · WHERE THE SEQUENCE GOES, WHICH IS THE FOOTER'S ONE JOB. The line
    #      tracks the alignment mode - "everything runs locally" is false the
    #      moment a search is chosen - and it names the SERVICE rather than
    #      linking a site, because a link in that line is something to click
    #      in a sentence whose whole purpose is to state a fact.
    note = cdp.evaluate(reader_ws, """(() => {
      const row = document.getElementById('msa-mode');
      const before = document.getElementById('privacy-note')?.textContent ?? '';
      row.value = 'search';
      row.dispatchEvent(new Event('change', { bubbles: true }));
      const box = document.getElementById('privacy-note');
      return { local: before, search: box?.textContent ?? '',
               links: box?.querySelectorAll('a').length ?? 0,
               forks: document.getElementById('footer-links')?.textContent.trim() ?? '' };
    })()""")
    print(f"  the footer says {note['search']!r} with a search chosen,"
          f" {note['links']} link(s); the links row is {note['forks']!r}")
    if "MMseqs2 server" not in note["search"]:
        bad.append(f"the footer says {note['search']!r} with a search chosen,"
                   " which does not name where the sequence goes")
    if note["links"] != 0:
        bad.append("the privacy line carries a link again - it is a statement,"
                   " not navigation")
    if "locally" not in note["local"]:
        bad.append(f"with no alignment the footer says {note['local']!r},"
                   " which is the weaker claim over the stronger case")
    if "py2Dmol" in note["forks"] or "alphafold2-webgpu" in note["forks"]:
        bad.append("the provenance links are back in the footer; they are"
                   " attribution and they live in the README")

    errors = cdp.evaluate(reader_ws, "window.__pageErrors || []")
    if errors:
        bad.append(f"the page threw: {errors[:2]}")
finally:
    if reader is not None:
        reader.kill()
    # It may already have stopped itself - the Disconnect arm asks it to.
    try:
        if backend.poll() is None:
            backend.send_signal(signal.SIGINT)
        backend.wait(timeout=10)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        backend.kill()

# 🔴 AND A GATE THAT LEAVES A BROWSER BEHIND IS THE NEXT MEASUREMENT'S PROBLEM.
# Counted after a green run: eight Chrome processes still on the runtime's
# profile, because the broker's own browser outlives the SIGINT that stops it.
# The broker kills it now; this asks, because the check costs nothing and the
# symptom - somebody else's browser on the machine - is one the neighbouring
# suites have been wrong about before.
left = subprocess.run(["pgrep", "-f", "localfold-pending-runtime"],
                      capture_output=True, text=True).stdout.split()
if left:
    bad.append(f"{len(left)} browser process(es) are still on the runtime's"
               " profile after the broker stopped")

print()
for line in bad:
    print("FAIL: " + line)
print("model pending: " + ("FAILED" if bad else "ok"))
raise SystemExit(1 if bad else 0)
