"""One fold at a time, and a model row that starts a new session.

    python3 tools/check-model-pending.py

🔴 SWITCHING THE MODEL ROW USED TO LEAVE THE PREVIOUS MODEL'S ANSWER ON SCREEN
under the new model's name. The structure, the contact map and the confidence
numbers look exactly the same whichever row is selected above them, so there
is nothing in the picture that says it is the other model's.

🔴 IT WAS ANSWERED WITH A COVER FIRST, AND THAT WAS THE WRONG SHAPE OF ANSWER.
A "pending" panel went over the result, with a rule for when it lifted and a
list of which parts it hid - and it hid two boxes while the play strip, the
MSA and the session download went on describing the fold underneath, so the
page said pending and looked finished. Pressing Fold dropped it, handing the
old result straight back for the whole of the trunk. Every one of those is a
question that only exists because something stale was being kept.

WHAT THE PAGE DOES NOW: every fold is its own object, listed in the picker,
and every panel reads the object being edited. There is nothing to cover, mark
or lift, and nothing to throw away either: the model row chooses what the next
fold uses and says nothing about what is on screen.

WHAT IT CHECKS, on a real page with a real result in it:

  * NONE OF IT REACHES AN ORDINARY PAGE: no badge, no retired controls, no
    "folded on:" in the dev report and no download row, on the page that folds
    in the reader's own browser - every rule here is conditioned on
    `?backend=colab` and this is what says so;
  * a result that lands is ON SCREEN, and the first fold makes one object;
  * moving the model row changes NOTHING on screen - the fold, its panels and
    its downloads stay, because the row says what to fold next and the panels
    describe the object being edited;
  * and moving it back changes nothing either;
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
    at once - and a Disconnect that RELEASES THE MACHINE (`unassign` is a POST
    to Colab's own runtime service, which this stands a stub in for) and stops
    the service on it, because the browser on
    that machine holds its GPU for as long as it lives. After it the page is a
    VIEWER: no folding, every control that shapes the next one disabled and
    saying why, and the structure, plots and downloads it already has still
    there. It does not RELOAD - the server it was served by is what stopped.

🔴 AND IT IS MEASURED AS WHAT A READER SEES - computed `display`, the objects
in the renderer and the frames in them - rather than as a class or a flag. The
cover that came before this was measured as a class first, which says a page
set an attribute and nothing about whether a stylesheet arrived.

🔴 AND THE RESULT GETS ONTO THE PAGE THROUGH tools/colab_backend.py, WHICH IS A
FIXTURE HERE AND NOT THE SUBJECT. A fold needs weights and a card; the broker
lets a structure be pushed to a reader's page through the code that ingests a
real one, which is the only way this machine can put a prediction on screen.
See tools/check-colab-bridge.py, whose arms this borrows.
"""
import base64
import http.server
import json
import os
import signal
import socketserver
import subprocess
import sys
import threading
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

# 🔴 A STAND-IN FOR COLAB'S RUNTIME SERVICE, which is how the machine gets
# handed back for real. `google.colab.runtime.unassign()` is a POST to
# `http://$TBE_RUNTIME_ADDR/unassign` - a plain HTTP address in the
# environment, not a call over the kernel's channel - so the broker can make
# it itself, and so this can watch it happen with no Colab runtime anywhere.
UNASSIGN_PORT = int(os.environ.get("PENDING_UNASSIGN_PORT", "8794"))
UNASSIGNED = []


class _Runtime(http.server.BaseHTTPRequestHandler):
    def do_POST(self):                                        # noqa: N802
        UNASSIGNED.append(self.path)
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *a):
        pass


socketserver.TCPServer.allow_reuse_address = True
_runtime_stub = socketserver.TCPServer(("127.0.0.1", UNASSIGN_PORT), _Runtime)
threading.Thread(target=_runtime_stub.serve_forever, daemon=True).start()


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


def page_of(ws):
    """Is the page showing a fold, or is it empty the way a fresh one is?

    🔴 THERE WAS A COVER, AND NOW THERE IS NOTHING TO COVER. Moving the model
    row used to leave the previous model's answer on screen, so it was veiled
    - and the simpler answer is that a row move starts a new session, which
    is an EMPTY page. So what this asks is what a reader sees: is the viewer
    up, is there a structure in it, is the download row offering anything,
    is the scores card there.
    """
    return cdp.evaluate(ws, """(() => {
      const shown = (id) => {
        const el = document.getElementById(id);
        return el === null ? 'absent' : getComputedStyle(el).display;
      };
      const reg = window.py2dmol_viewers || {};
      const r = reg[Object.keys(reg)[0]] && reg[Object.keys(reg)[0]].renderer;
      const objects = Object.keys(r && r.objectsData ? r.objectsData : {});
      const drawn = objects.reduce((total, name) =>
        total + ((r.objectsData[name].frames || []).length), 0);
      return { viewer: shown('viewer-container'),
               strip: shown('sequence-viewer-container'),
               msa: shown('msa-buttons'),
               downloads: shown('downloads'),
               scores: shown('predictionScoresBox'),
               objects: objects.length,
               frames: drawn };
    })()""")


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
    cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    # ...and it is told it is on a Colab machine, which is the only difference
    # between this run and one in a notebook.
    env={**os.environ, "TBE_RUNTIME_ADDR": f"127.0.0.1:{UNASSIGN_PORT}"})
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
    # 🔴 FIRST, THE ORDINARY PAGE - the one this all has to stay out of. Every
    # rule below is conditioned on `?backend=colab`, and a badge, a disabled
    # Fold button or a "folded on:" header appearing on localfold.org would be
    # today's work leaking onto the page that folds in your own browser.
    reader_ws.call("Page.navigate", url=f"{BASE}/index.html")
    cdp.wait_for(reader_ws, "!!window.__entityList", 120, "the plain page")
    plain = cdp.evaluate(reader_ws, """(() => {
      const button = document.getElementById('dev-toggle');
      if (button !== null) button.click();
      return {
        badge: document.getElementById('colab-status') === null ? 'none' : 'there',
        fold: !!document.getElementById('predict')?.disabled,
        model: !!document.getElementById('model-family')?.disabled,
        downloads: getComputedStyle(document.getElementById('downloads')).display,
        note: document.getElementById('privacy-note')?.textContent ?? '',
        dev: document.querySelector('#dev-panel pre')?.textContent ?? '',
        errors: window.__pageErrors || [],
      };
    })()""")
    print(f"  the ordinary page: badge {plain['badge']}, fold disabled"
          f" {plain['fold']}, downloads {plain['downloads']},"
          f" note {plain['note']!r}")
    if plain["badge"] != "none":
        bad.append("a page that folds in this browser is wearing the Colab"
                   " badge")
    if plain["fold"] or plain["model"]:
        bad.append("the ordinary page came up with folding retired - the"
                   " runtime rules reached a page with no runtime")
    if plain["downloads"] != "none":
        bad.append("the download row is up on a page that has folded nothing")
    if "locally" not in plain["note"]:
        bad.append(f"the footer says {plain['note']!r} on a page at rest")
    if "folded on:" in plain["dev"]:
        bad.append("the dev report claims another machine folded on a page"
                   " that has no runtime")
    if plain["errors"]:
        bad.append(f"the ordinary page threw: {plain['errors'][:2]}")

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
        // 🔴 THE LABEL IS A LIE HERE, DELIBERATELY, AND THAT IS THE TEST.
        // The reader used to recover "whose result is this" by reverse-lookup
        // of `model` in MODEL_LABELS - a map that is neither total nor
        // one-to-one, so the AF2 path's own `AlphaFold 2 (monomer-3)` (the
        // table says `AlphaFold 2 (model 3)`) resolved to undefined and NO
        // AF2 FOLD EVER VEILED in Colab mode. It reads `family` now. With
        // the two disagreeing, legs 1 and 2 below tell the readings apart:
        // by family this is AlphaFold 3's, so it is bare under AF3 and
        // covered under Boltz-2 - and by label it would be exactly the other
        // way round. Every other leg is unaffected, because every one of
        // them is about the family.
        predJson: JSON.stringify({ model: 'Boltz-2', family: 'af3',
          stem: 'pending_test', pdb,
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
    here = page_of(reader_ws)
    print(f"  a fold on screen: viewer={here['viewer']} objects={here['objects']}"
          f" frames={here['frames']} scores={here['scores']}")
    if here["viewer"] == "none" or here["frames"] == 0:
        bad.append(f"the fold that just landed is not on screen: {here}")
    if here["objects"] != 1:
        bad.append(f"the page holds {here['objects']} objects - one fold at a"
                   " time is the whole of this page's model, and py2Dmol's"
                   " picker is hidden on the strength of it")

    # ...and a session written from it WHILE IT IS STILL HERE, for the
    # restore-then-fold leg further down. It cannot write its own: by the
    # time it runs, the model-switch leg has emptied the page, and
    # `buildViewerState` of an empty viewer is a session with no objects in
    # it - measured, the restore then put nothing on screen and the leg
    # passed having asked nothing.
    sessionWritten = cdp.evaluate(reader_ws, """(async () => {
      const store = await import('/web/fold-session.js');
      const state = window.buildViewerState?.();
      if (!state) return 'no buildViewerState';
      state.localfold = { stem: 'restored_fold', model: 'AlphaFold 3',
                          family: 'af3', savedAt: Date.now(),
                          sequence: 'GWSTELEKHRSVQ' };
      // ...under the name a reader would see it by, which is also the name
      // that must survive the fold below.
      for (const o of state.objects || []) o.name = 'restored_fold';
      if (state.viewer_state) state.viewer_state.current_object_name = 'restored_fold';
      await store.saveSession(state);
      return 'saved';
    })()""")
    print(f"  a session was written: {sessionWritten}")
    if sessionWritten != "saved":
        bad.append(f"could not write a session to restore ({sessionWritten}),"
                   " so the restore-then-fold leg was never driven")

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

    # 6b · 🔴 MOVING THE MODEL ROW STARTS A NEW SESSION, WHICH IS AN EMPTY
    #      PAGE. It used to leave the previous model's structure, map, scores
    #      and downloads on screen under the new model's name; then it veiled
    #      them; then a switch EMPTIED the page - one fold at a time, each its
    #      own session. Reported first as "switching models looks finished",
    #      then as a cover that showed a play button, an MSA and a session
    #      download through it.
    #
    #      🔴 AND NOW IT CHANGES NOTHING AT ALL, which is the third answer and
    #      the one that needed no mechanism. Folds accumulate - every fold is
    #      its own object in the picker - and every panel reads
    #      `activePrediction()`, keyed by the object being edited: the scores
    #      card and the downloads describe what you are LOOKING at, the row
    #      describes what you are about to MAKE. Reported as "starting new
    #      prediction deletes the previous prediction/object".
    set_row(reader_ws, "model-family", "boltz2")
    time.sleep(1.0)
    kept = page_of(reader_ws)
    print(f"  after switching to Boltz-2: viewer={kept['viewer']}"
          f" objects={kept['objects']} frames={kept['frames']}"
          f" downloads={kept['downloads']} scores={kept['scores']}")
    if kept["objects"] != 1 or kept["frames"] == 0:
        bad.append(f"switching the model took the fold away:"
                   f" {kept['objects']} object(s), {kept['frames']} frame(s)"
                   " - the row says what to fold NEXT and nothing about what"
                   " is on screen")
    for part in ("viewer", "downloads"):
        if kept[part] in ("none", "absent"):
            bad.append(f"the {part} went away on a model switch ({kept[part]})"
                       " - a fold keeps its object and its panels")
    live_downloads = downloads_of(reader_ws)
    if live_downloads["pdb"]["off"] or live_downloads["all"]["off"]:
        bad.append("the downloads went off on a model switch - they describe"
                   " the fold on screen, which is still there")

    # ...and back, which must also change nothing.
    set_row(reader_ws, "model-family", "af3")
    time.sleep(1.0)
    still = page_of(reader_ws)
    print(f"  ...and back to AlphaFold 3: objects={still['objects']}"
          f" frames={still['frames']}")
    if still["objects"] != kept["objects"] or still["frames"] != kept["frames"]:
        bad.append(f"moving the row back changed the page: {kept} -> {still}")

    # 🔴 AND NOTHING ELSE A READER TOUCHES TAKES THE RESULT AWAY EITHER.
    #      Asked for in one line: "the results from previous run should not
    #      disappear until user hits fold". Setting up the next run is
    #      typing in a sequence and moving five controls, and every one of
    #      them used to be a `change` event on a page that cleared itself -
    #      the model row was the loud one, and the rest are the same shape.
    #      What may change the picture is pressing Fold.
    before_edits = page_of(reader_ws)
    edits = [("recycles", "3"), ("random-seed", "7"), ("msa-mode", "none"),
             ("af3-mode", "diffusion")]
    for row_id, value in edits:
        # ...a SELECT takes only a value it has; a number box takes any.
        kind = cdp.evaluate(reader_ws,
            "(() => { const r = document.getElementById(%s);"
            " return r === null ? 'absent'"
            "   : (r.options ? ([...r.options].some((o) => o.value === %s)"
            "     ? 'has' : 'no') : 'input'); })()"
            % (json.dumps(row_id), json.dumps(value)))
        if kind in ("has", "input"):
            set_row(reader_ws, row_id, value)
    # ...and the sequence itself, through the row's own input event.
    cdp.evaluate(reader_ws, """(() => {
      const box = document.querySelector('.entity-sequence, #sequence, textarea');
      if (box === null) return false;
      box.value = 'GWSTELEKHRSVQMD';
      box.dispatchEvent(new Event('input', { bubbles: true }));
      box.dispatchEvent(new Event('change', { bubbles: true }));
      return true;
    })()""")
    time.sleep(1.0)
    after_edits = page_of(reader_ws)
    print(f"  after editing the inputs: objects={after_edits['objects']}"
          f" frames={after_edits['frames']} downloads={after_edits['downloads']}"
          f" scores={after_edits['scores']}")
    for field in ("objects", "frames", "viewer", "downloads", "scores"):
        if after_edits[field] != before_edits[field]:
            bad.append(f"editing the inputs changed {field}:"
                       f" {before_edits[field]} -> {after_edits[field]}."
                       " Setting up the next run is not running it")

    # 6c · 🔴 A RESTORED FOLD SURVIVES THE NEXT FOLD, WHICH IS THE WHOLE
    #      POINT OF RESTORING IT BESIDE. Reported three times - "past object
    #      still lost in object list" - and reasoned about twice before it
    #      was ever DRIVEN. The sequence is restore, then fold, and what a
    #      reader looks at is the picker.
    #
    #      🔴 THE SESSION IS WRITTEN THROUGH THE STORE'S OWN DOOR, because
    #      the Colab reader never writes one: `rememberSessionWhenSettled`
    #      is called by the three LOCAL fold paths and the reader takes a
    #      finished prediction instead. The first version of this waited 40
    #      seconds for a session row that could never appear.
    picker = lambda: cdp.evaluate(reader_ws,
        "[...(document.getElementById('objectSelect')?.options ?? [])]"
        ".map((o) => o.value)")
    # 🔴 AND A SKIP IS LOUD. This was `if wrote == "saved"`, and leg 5 above
    # assigns `wrote` too - the bytes its download button produced - so the
    # test was False, the whole leg was skipped in silence, and the run came
    # back green having asked nothing. The name is its own now and the else
    # says so.
    if sessionWritten != "saved":
        bad.append("the restore-then-fold leg did not run at all")
    else:
        reader_ws.call("Page.navigate",
                       url=f"{BASE}/index.html?backend=colab&t={TOKEN}")
        cdp.wait_for(reader_ws, "!!window.__entityList", 120, "the reader, for the restore")
        offered, deadline2 = False, time.time() + 30
        while time.time() < deadline2:
            offered = bool(cdp.evaluate(reader_ws,
                "!document.getElementById('session')?.hidden"))
            if offered:
                break
            time.sleep(0.5)
        print(f"  the session is offered: {offered}")
        if not offered:
            bad.append("the saved session is not offered on a fresh page")
        else:
            cdp.evaluate(reader_ws,
                         "document.getElementById('session-restore').click(), 1")
            two, deadline2 = [], time.time() + 40
            while time.time() < deadline2:
                two = picker() or []
                if two:
                    break
                time.sleep(0.5)
            print(f"  restored: {two}")
            if not two:
                bad.append("restoring put nothing in the object list")
            else:
                # 🔴 AND THE MODEL ROW MOVES FIRST, WHICH IS THE REPORT.
                # "run AlphaFold 3, close the window, reopen, restore - it is
                # listed - run Boltz-2, and the AlphaFold 3 object
                # disappears." It was not the fold that took it: choosing
                # which model to run NEXT called startNewSession, which
                # cleared every object. A new session ends the fold THIS PAGE
                # made, and after a reopen there is none - the one object on
                # screen is the reader's.
                set_row(reader_ws, "model-family", "boltz2")
                time.sleep(1.5)
                kept = picker() or []
                print(f"  ...after switching to Boltz-2: {kept}")
                gone = [name for name in two if name not in kept]
                if gone:
                    bad.append(f"choosing another model took {gone} out of"
                               f" the object list - {two} became {kept}. A"
                               " restored fold is the reader's; a new session"
                               " ends the one this page folded")
                # ...and now FOLD, which is where it was going missing. The
                # reader's own button, and a frame from the runtime, which is
                # what takes openBlankFold through remoteFrameDrawer.
                cdp.evaluate(reader_ws,
                             "document.getElementById('predict').click(), 1")
                running = False
                for _ in range(40):
                    running = bool(cdp.evaluate(reader_ws,
                        "!!window.__foldState?.running"))
                    if running:
                        break
                    time.sleep(0.5)
                print(f"  a fold is running: {running}")
                if not running:
                    bad.append("the reader did not start a fold, so the"
                               " object list below was never asked the"
                               " question this leg exists for")
                cdp.evaluate(runtime_ws, """(async () => {
                  const bridge = await import('/web/colab-bridge.js');
                  bridge.tapOut('frame', %s);
                  return true;
                })()""" % json.dumps(tiny_pdb()))
                after, deadline2 = two, time.time() + 40
                while time.time() < deadline2:
                    after = picker() or []
                    if len(after) > len(two):
                        break
                    time.sleep(0.5)
                print(f"  ...then folded: {two} -> {after}")
                lost = [name for name in two if name not in after]
                if lost:
                    bad.append(f"the fold took {lost} out of the object list -"
                               f" {two} became {after}. A fold recycles its"
                               " OWN object; anything restored was asked for"
                               " by hand and stays")
                # ...and the fold is left in flight deliberately: the
                # block below navigates, which discards it, where pressing
                # Stop leaves "Stopping prediction…" on the status line for
                # leg 9 to read.

    # ...and PUT ONE BACK, because the legs below need a fold on screen and
    # the two above have just spent it: leg 9 asks whether a page whose
    # runtime has gone can still hand over the fold it already made, and on
    # an empty page that question has no subject. Measured - without this it
    # reported "the downloads went with the runtime" against a page that had
    # nothing to give. The path is the fixture's own: raise `folding` on the
    # broker, open the reader again so it attaches, push the result.
    call("/in", {"op": "fold", "payload": {
        "entities": [{"type": "protein", "value": "GWSTELEKHRSVQ", "copies": 1}]}})
    reader_ws.call("Page.navigate",
                   url=f"{BASE}/index.html?backend=colab&t={TOKEN}")
    cdp.wait_for(reader_ws, "!!window.__entityList", 120, "the reader, again")
    again, deadline2 = "", time.time() + 30
    while time.time() < deadline2:
        again = cdp.evaluate(reader_ws,
            "document.getElementById('status-message')?.textContent ?? ''")
        if "already running" in again:
            break
        time.sleep(0.5)
    cdp.evaluate(runtime_ws, """(async () => {
      const bridge = await import('/web/colab-bridge.js');
      const pdb = %s;
      bridge.tapOut('result', { pdb, atoms: 6, status: 'AlphaFold 3 · 6 residues',
        scores: {}, confidence: { meanPlddt: 88.1, ptm: 0.71 },
        predJson: JSON.stringify({ model: 'AlphaFold 3', family: 'af3',
          stem: 'pending_test', pdb,
          confidence: { meanPlddt: 88.1, ptm: 0.71,
            plddt: { __typed: 'Float32Array', v: [88.1, 90.2, 71.0, 65.5, 80.0, 92.3] },
            predictedAlignedError: { __typed: 'Float32Array',
                                     v: Array.from({ length: 36 }, (unused, i) => i / 4) } } }) });
      return true;
    })()""" % json.dumps(tiny_pdb()))
    reland, deadline2 = {}, time.time() + 60
    while time.time() < deadline2:
        reland = page_of(reader_ws)
        if reland.get("frames", 0) > 0:
            break
        time.sleep(0.5)
    print(f"  a fold is back on screen: objects={reland.get('objects')}"
          f" frames={reland.get('frames')}")
    if reland.get("frames", 0) == 0:
        bad.append("the second result never landed, so every leg below is"
                   " about an empty page rather than about a fold")

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

    # 8 · A QUIET PAGE IS NOT A DEAD RUNTIME, AND THE BADGE KNOWS. The page's
    #     poll stops while it holds its main thread - which is what folding
    #     does - so the badge must NOT go amber for that; what turns it amber
    #     is the browser itself going, which is the runtime going.
    runtime_ws.call("Page.navigate", url="about:blank")
    time.sleep(8.0)
    quiet = cdp.evaluate(reader_ws, """(() => {
      const box = document.getElementById('colab-status');
      return (box?.dataset.state ?? '') + '|' +
             (box?.querySelector('.colab-said')?.textContent ?? '');
    })()""")
    print(f"  with the page quiet but the browser alive: {quiet!r}")
    if quiet.startswith("gone|"):
        bad.append("the badge went amber for a page that had merely stopped"
                   " polling - a fold holds that thread, and a reader would"
                   " be told their working runtime had died")
    runtime_ws.call("Page.navigate",
                    url=f"{BASE}/index.html?role=runtime&t={TOKEN}")
    time.sleep(2.0)

    # 🔴 EVERY ASK IS COUNTED FROM HERE, at the page's own `fetch`. A request
    # to a server that has gone is a network ERROR and Chrome files no
    # resource-timing entry for one, so counting them the other way reported
    # zero while a console filled with 403s and 500s - which is what a reader
    # actually sees.
    beats0 = cdp.evaluate(reader_ws, "window.__colabBeats ?? 0")

    # 🔴 AND THE COUNTER IS PROVED TO SEE THE PULSE BEFORE ITS SILENCE IS
    # BELIEVED. The first version asserted only that the count stopped
    # growing, and it read 0 before the runtime was even killed - so it would
    # have passed against a wrapper that saw nothing at all, which is this
    # file's own "a probe that cannot say yes cannot say no".
    time.sleep(7.0)
    beating = cdp.evaluate(reader_ws, "(window.__colabBeats ?? 0)") - beats0
    print(f"  the badge polled {beating} time(s) in seven seconds")
    if beating == 0:
        bad.append("the counter cannot see the badge's own polling, so"
                   " whatever it says about the silence afterwards is worth"
                   " nothing")

    # ...and now the runtime really goes.
    subprocess.run(["pkill", "-f", "Google Chrome.*localfold-pending-runtime"],
                   check=False)
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
    print(f"  with the runtime gone: {gone!r}")
    if not gone.startswith("gone|"):
        bad.append("the runtime's browser went and the badge still says it is"
                   " there - a fold pressed now waits out its own bound")
    # 🔴 AND IT KEEPS BEATING WHILE THERE IS SOMETHING TO ASK. The broker is
    # still answering here - only the page on it died - so the badge must go
    # on checking, which is how it would recover if that page came back. The
    # first version of this arm asserted the OPPOSITE and was wrong: silence
    # belongs to the case where nobody is behind the door, which is what the
    # Disconnect arm below measures.
    first = cdp.evaluate(reader_ws, "(window.__colabBeats ?? 0)")
    time.sleep(10.0)
    second = cdp.evaluate(reader_ws, "(window.__colabBeats ?? 0)")
    print(f"  the badge's pulse with the broker still up: {first} -> {second}")
    if second == first:
        bad.append("the badge stopped checking while the broker was still"
                   " answering - a runtime page that came back would never"
                   " be noticed")


    # 9 · Disconnect leaves Colab mode, which is the whole of the way back.
    # 🔴 COUNTED AT THE PAGE'S OWN `fetch`, NOT IN RESOURCE TIMINGS. A request
    # to a server that has gone produces a network ERROR, and Chrome files no
    # resource entry for one - so the first version of this counted zero with
    # the pulse deliberately left running, and would have called a broken fix
    # fixed. What is wanted is the ASKING, which is this side of the wire.
    cdp.evaluate(reader_ws, """(() => {
      window.__asked = [];
      const real = window.fetch;
      window.fetch = (...args) => {
        const url = String(args[0]);
        if (window.__counting && /\/(down|out|up|in|health)\b/.test(url)) {
          window.__asked.push(url.replace(location.origin, ''));
        }
        return real.apply(window, args);
      };
      window.__counting = true;
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
    if "released" not in after.get("says", ""):
        bad.append(f"the page says {after.get('says')!r} after Disconnect on a"
                   " runtime that CAN be handed back - it should say so")
    if "released" not in (after.get("badge") or ""):
        bad.append(f"the badge reads {after.get('badge')!r} rather than saying"
                   " the machine was released")
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
    # 🔴 AND THE PAGE STOPS KNOCKING, which is what a reader sees when it does
    # not. Reported from a real runtime: a console full of
    # `GET /down?t=&head=1 403` and then 500 after 500, because the badge's
    # pulse ran on for the life of the tab and `door()` re-read the token from
    # a URL Disconnect had just cleared. Requests are counted from the
    # browser's own resource timings, after the click.
    # 🔴 AND NOW THERE IS NOBODY BEHIND THE DOOR, SO THE KNOCKING STOPS. This
    # is the state a reader reported: a console full of
    # `GET /down?t=&head=1 403` and then 500 after 500, because the pulse ran
    # on for the life of the tab. Three samples: a few beats may be in flight
    # or spent finding out, and then it must be still.
    beats_a = cdp.evaluate(reader_ws, "(window.__colabBeats ?? 0)")
    time.sleep(16.0)
    beats_b = cdp.evaluate(reader_ws, "(window.__colabBeats ?? 0)")
    time.sleep(12.0)
    beats_c = cdp.evaluate(reader_ws, "(window.__colabBeats ?? 0)")
    print(f"  the pulse after Disconnect: {beats_a} -> {beats_b} -> {beats_c}")
    if beats_c != beats_b:
        bad.append(f"the page is still beating after Disconnect"
                   f" ({beats_b} -> {beats_c}) - the service is gone and every"
                   " one of those is an error in a console somebody is"
                   " reading to find out whether Disconnect worked")
    if beats_b - beats_a > 4:
        bad.append(f"the page beat {beats_b - beats_a} times before giving up"
                   " - it is meant to stop on being told, not only on failing")

    # 🔴 AND THE MACHINE IS HANDED BACK, which is what "disconnect" means to
    # somebody paying for a runtime. Stopping the service frees the card; this
    # frees the VM, and it is a different call to a different address.
    for _ in range(20):
        if UNASSIGNED:
            break
        time.sleep(0.5)
    print(f"  Colab's runtime service was asked: {UNASSIGNED}")
    if UNASSIGNED != ["/unassign"]:
        bad.append(f"Colab's runtime service was asked {UNASSIGNED} - the"
                   " machine is still assigned and still being paid for")

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
    _runtime_stub.shutdown()
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
