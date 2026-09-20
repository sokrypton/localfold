"""LocalFold's own fold, on somebody else's GPU, reached over one URL.

    python3 tools/colab_backend.py --port 8710            # serve and fold
    python3 tools/colab_backend.py --port 8710 --token t  # a token of your own

It serves this checkout, drives a headless Chrome on the machine it runs on,
and answers two requests:

    GET  /health            what the card is, and what WebGPU it offers
    POST /fold              {sequence, model, ...} -> {pdb, status, ms}

🔴 THE POINT IS THAT THERE IS NO SECOND IMPLEMENTATION. The fold that runs
here is web/app.js's own, in a real browser, from this checkout - the same
code a visitor's laptop runs, on a card the laptop does not have. A Python
re-implementation would be a second answer to every question this repository
has already answered once, and the two would part company on the first
modified residue.

🔴 AND IT IS THE PAGE THAT IS DRIVEN, NOT A MODULE. tools/fold-in-page.py
exists because "every other check here misses the path that matters" - the
entity list, the model row, the fold button and everything web/app.js wires to
them. This reuses that door: `window.__entityList` and `#predict`, which is
what a person clicking would touch, and it reads the structure back out of the
page's own download button rather than from an internal.

🔴 THE TUNNEL IS PUBLIC WHILE IT LIVES, SO THE TOKEN IS NOT OPTIONAL. Anything
that reaches the URL can spend the GPU behind it, so every request carries the
token and a mismatch is refused before the body is read. `hmac.compare_digest`
rather than `==`, which is the one line that stops the comparison leaking its
answer in its timing.

🔴 AND CORS IS WIDE ON PURPOSE, WHICH IS ONLY SAFE BECAUSE OF THE TOKEN. The
page asking for a fold is on another origin entirely - a laptop's LocalFold,
or the deployed site - and an allow-list of origins cannot be written for a
URL that changes every session. The token is the whole of the authority here.
"""
import argparse
import hmac
import http.server
import json
import os
import secrets
import socketserver
import sys
import threading
import time
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cdp                                                   # noqa: E402

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# 🔴 ONE BROWSER, ONE PAGE, ONE JOB AT A TIME. There is one GPU behind this and
# a fold saturates it; two at once is not twice the throughput, it is two folds
# that both take longer against a memory ceiling neither expected. The lock is
# what makes the queue honest, and the page is kept between jobs because
# loading it costs seconds and the weights it caches are hundreds of megabytes.
LOCK = threading.Lock()
# 🔴 AND A FOLD IS A JOB, NOT A REQUEST THAT TAKES A MINUTE. The first version
# answered POST /fold with the finished structure, which is correct and is
# also a page that sits blank for the whole fold: no bar, no status, no
# sampler frames. A job the client POLLS carries all three while they happen.
# `stream` is what asks for it, and notebooks/localfold.ipynb's page always
# does. The blocking form is the one-shot door left for a curl - no shipped
# caller uses it, and the two notebooks that did were retired.
JOBS = {}
JOBS_LOCK = threading.Lock()

ADAPTER_JS = """(async () => {
  if (!navigator.gpu) return { webgpu: false, why: 'no navigator.gpu' };
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return { webgpu: false, why: 'no adapter' };
  const info = adapter.info
    ?? (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
  return {
    webgpu: true,
    vendor: info.vendor ?? null, architecture: info.architecture ?? null,
    device: info.device ?? null, description: info.description ?? null,
    /* 🔴 THE TWO THAT DECIDE WHETHER A NUMBER FROM HERE IS COMPARABLE WITH
       docs/A100.md's. Without them the same fold is 1.95x slower, and a
       backend that does not say which it had is a bench nobody can read. */
    shaderF16: adapter.features.has('shader-f16'),
    subgroupMatrix: adapter.features.has('chromium-experimental-subgroup-matrix'),
    maxBufferSize: adapter.limits.maxBufferSize,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
  };
})()"""

# 🔴 THE STRUCTURE COMES OUT OF THE DOWNLOAD BUTTON, not out of an internal.
# `lastPrediction.pdb` and the file a reader saves are written by different
# code and have disagreed before, and it is the FILE this backend is asked
# for. Hooking createObjectURL is how tools/fold-in-page.py reads one without
# a filesystem.
READBACK_JS = """(async () => {
  const reg = window.py2dmol_viewers || {};
  const v = reg[Object.keys(reg)[0]] && reg[Object.keys(reg)[0]].renderer;
  const o = v && v.objectsData[v.currentObjectName];
  const last = o && o.frames ? o.frames[o.frames.length - 1] : null;
  const blobs = [];
  const made = URL.createObjectURL;
  URL.createObjectURL = (b) => { blobs.push(b); return made.call(URL, b); };
  document.getElementById('download-pdb')?.click();
  for (let tick = 0; tick < 40 && blobs.length === 0; tick += 1) {
    await new Promise((done) => setTimeout(done, 100));
  }
  URL.createObjectURL = made;
  const pdb = blobs[0] ? await blobs[0].text() : '';
  // 🔴 AND WHAT THE PAGE KNOWS BESIDES THE COORDINATES. A remote fold used to
  // arrive as a structure and nothing else - no alignment, no confidence, no
  // scores - because this readback only ever clicked the download button.
  // `window.__lastPrediction()` is the same object the local path hands to
  // loadIntoViewer. The alignment can be megabytes and the matrices are n^2,
  // so they travel only when they exist.
  const pred = (window.__lastPrediction && window.__lastPrediction()) || {};
  // 🔴 THE WHOLE OBJECT, AS A STRING, AND NOT FIELD BY FIELD. The archive the
  // download button writes reads `stem`, `model`, `settings`, `entities`,
  // `msas`, `msaOrigin`, `confidence`, `chainLengths` and more; naming them
  // here is the field-by-field rebuild this repository has been bitten by six
  // times, and the failure is always silent - the button writes a zip with a
  // piece missing. JSON.stringify with a replacer because CDP returns JSON:
  // a Float32Array crossing it becomes {"0":...,"1":...}, which every reader
  // downstream treats as an object with no length.
  const predJson = JSON.stringify(pred, (key, value) =>
    (ArrayBuffer.isView(value) && !(value instanceof DataView))
      ? Array.from(value) : value);
  return {
    predJson,
    a3m: pred.a3m ?? null,
    confidence: pred.confidence ?? null,
    scores: pred.scores ?? null,
    chains: pred.chains ?? null,
    status: document.getElementById('status-message')?.textContent ?? null,
    frames: o && o.frames ? o.frames.length : 0,
    positions: v && v.coords ? v.coords.length : 0,
    names: v && v.positionNames ? v.positionNames.join(',') : null,
    plddt: last && last.plddts ? Array.from(last.plddts) : null,
    pae: last && last.pae_n ? last.pae_n : null,
    pdb,
    atoms: (pdb.match(/^ATOM|^HETATM/gm) || []).length,
  };
})()"""


class Backend:
    """The headless Chrome this serves from, started once."""

    def __init__(self, port, cdp_port, profile):
        self.port = port
        self.cdp_port = cdp_port
        self.profile = profile
        self.proc = None
        self.ws = None

    def start(self):
        # cdp.py already carries the Linux flags this needs - Vulkan, the
        # sandbox off (a runtime is root in a container), and the f16 feature
        # docs/A100.md prices at 1.95x. Headless is opt-in there because the
        # A100 box runs headed; here there is no display at all.
        os.environ.setdefault("LOCALFOLD_HEADLESS", "1")
        # 🔴 `--disable-vulkan-surface`, WHICH THE SHARED FLAGS DO NOT CARRY.
        # A surface is a thing you present TO, and this container has no
        # display: Chrome's own Colab recipe passes it, and without it the
        # first measured runtime came back on **SwiftShader** - vendor
        # 'google', architecture 'swiftshader', no shader-f16, a 1 GiB buffer
        # ceiling - which is the CPU wearing the card's clothes.
        self.proc, self.ws = cdp.launch(self.cdp_port, self.profile,
                                        extra_args=["--disable-vulkan-surface"])
        self.ws.call("Page.enable")
        self.ws.call("Runtime.enable")
        self.ws.call("Page.navigate", url=f"http://127.0.0.1:{self.port}/index.html")
        cdp.wait_for(self.ws, "!!window.__entityList", 180, "the page")
        # 🔴 THE TERMS DIALOG WOULD OTHERWISE EAT THE CLICK. AlphaFold 3's
        # parameters are gated behind an acknowledgement that opens in FRONT of
        # `predict`, so without this the press opens a modal, nothing folds,
        # and the wait runs to its timeout with the status line never moving.
        # Whoever started this backend accepted them by starting it; the
        # deploy-side gate (LOCALFOLD_ACCEPT_MODEL_TERMS) is untouched.
        cdp.evaluate(self.ws, """(() => {
          for (const key of ['alphafold3', 'openbind0', 'opendde', 'boltz2',
                             'protenix2', 'intellifold2', 'rosettafold3']) {
            try { localStorage.setItem('localfold.modelTerms.' + key, 'accepted'); }
            catch (cause) { /* a runtime with no storage still folds AF2 */ }
          }
          return true;
        })()""")
        return self

    def adapter(self):
        """What the card is - the whole question this backend rests on."""
        return cdp.evaluate(self.ws, ADAPTER_JS)

    def fold(self, request, job=None):
        """One fold, through the page's own controls.

        `job`, when given, is filled AS THE FOLD RUNS with what the page said
        and drew - the same status writes, bar fractions and sampler frames a
        local fold makes, tapped in web/app.js. See remoteTap there.
        """
        entities = request.get("entities") or [{
            "type": "protein", "value": request.get("sequence", ""), "copies": 1,
            "modifications": request.get("modifications", []),
        }]
        controls = {
            "model-family": request.get("model", "af3"),
            "recycles": str(request.get("recycles", 3)),
            "af3-count": str(request.get("steps", 25)),
            "msa-mode": request.get("msa", "none"),
        }
        started = time.time()
        cdp.evaluate(self.ws, """(() => {
          window.__entityList.set(%s);
          const controls = %s;
          for (const [id, value] of Object.entries(controls)) {
            const el = document.getElementById(id);
            if (!el) continue;
            el.value = value;
            el.dispatchEvent(new Event('change', { bubbles: true }));
          }
          return true;
        })()""" % (json.dumps(entities), json.dumps(controls)))
        cdp.wait_for(self.ws, "!document.getElementById('predict').disabled", 120,
                     "the fold button")
        # 🔴 AND THE PAGE MAY ALREADY BE FOLDING, which the button does not
        # say (it stays enabled throughout - it is how you stop one). Clicking
        # on top of a running fold is how the hold above came about: the second
        # click did nothing, no new object appeared, and the watcher waited.
        # A percentage in the status line is the page saying it is busy.
        said = cdp.evaluate(self.ws,
                            "document.getElementById('status-message')?.textContent ?? ''")
        if "%" in (said or ""):
            return {"error": "the runtime is already folding", "status": said}
        # 🔴 THE LAST FOLD IS STILL ON THE PAGE, AND ITS DOWNLOAD BUTTON
        # STILL WORKS - so "a structure exists" is not "this fold made one".
        # Asking for one after pressing Fold answered with the PREVIOUS
        # structure in 776 ms, under a status line reading "Folding 3/25 ·
        # 43%": a plausible answer to the wrong question, which is the worst
        # kind. What is NOT the fix is clearing the viewer first -
        # `clearAllObjects` is the Clear button's verb and it takes the page
        # with it, measured: the status line went to "Prediction stopped" and
        # then "Paste a sequence and press Fold", and nothing folded at all.
        #
        # 🔴 AND IT IS NOT THE OBJECT NAMES EITHER, WHICH IS WHAT THIS USED TO
        # REMEMBER. "A name that was not here before" is true of a first fold
        # and false of a second: `openBlankFold` REUSES the stem and rewinds
        # it, so the page held one object, `af3_1`, after two folds - and the
        # watcher waited out its whole timeout on a fold whose own status line
        # already read "AlphaFold 3 · 13 residues · in 1 s (trunk reused)".
        # The page states it instead (`window.__foldState`, web/app.js), which
        # is not a proxy for anything.
        # 🔴 ARMED BEFORE THE CLICK AND NOT BEFORE THE SETUP, or the entity
        # writes above fill it with their own status lines and the client
        # replays the last fold's ending as this one's beginning.
        cdp.evaluate(self.ws, "(window.__remoteTap = [], true)")
        # 🔴 AND THE CLICK IS TIMED BY THE PAGE'S OWN CLOCK, not this one's:
        # what the watcher compares against is `__foldState.since`, which the
        # page writes with its own Date.now(). Two machines' clocks agreeing is
        # not something to rest a completion test on - and here they are the
        # same machine only by accident of this backend being local.
        pressed = cdp.evaluate(self.ws, """(() => {
          const at = Date.now();
          document.getElementById('predict').click();
          return at;
        })()""")
        # 🔴 AND THE FOLD IS WATCHED BY WHAT IT MAKES, NOT BY THE BUTTON. The
        # first version waited for `predict` to go disabled and then enabled
        # again, which cannot see a fold that takes less time than the poll:
        # warm, a 13-mer is **260 ms** on this machine and the button is down
        # and up between two 250 ms samples, so the wait timed out on a fold
        # that had already finished. The objects were cleared above, so "an
        # object with frames exists" is monotonic, is this fold's, and is true
        # exactly once the fold has produced something.
        # 🔴 THE HOLD IS BOUNDED, AND NOT BY THE CALLER ALONE. A fold that
        # never starts - because the page was already folding when the click
        # landed - leaves this loop waiting out its whole timeout with the
        # GPU's lock in its hand, and every request behind it answered 429.
        # Measured: two folds fired back to back left the runtime `busy: true`
        # for minutes with nothing running. Five minutes is longer than any
        # fold this backend has been asked for and short enough to forgive.
        deadline = time.time() + min(request.get("timeout", 300), 1800)
        while True:
            state = cdp.evaluate(self.ws, """(() => {
              const reg = window.py2dmol_viewers || {};
              const v = reg[Object.keys(reg)[0]] && reg[Object.keys(reg)[0]].renderer;
              const out = {};
              for (const name of Object.keys(v ? v.objectsData : {})) {
                out[name] = (v.objectsData[name].frames || []).length;
              }
              const button = document.getElementById('predict');
              const fold = window.__foldState || null;
              // ...and the tap, drained in the SAME round trip the watch was
              // already making: a second evaluate per 250 ms would double the
              // CDP traffic to learn the same thing.
              const tap = window.__remoteTap || [];
              return { objects: out, idle: !!(button && !button.disabled),
                       folding: fold ? fold.running : null, since: fold ? fold.since : 0,
                       tap: tap.splice(0, tap.length),
                       status: document.getElementById('status-message')?.textContent ?? '' };
            })()""")
            if job is not None:
                job["events"].extend(state.get("tap") or [])
                job["status"] = state.get("status") or job["status"]
            # 🔴 AND FRAMES ARE NOT AN ENDING. The sampler STREAMS them, so a
            # new object has frames a few hundred milliseconds in - measured,
            # this returned at 807 ms under "Folding 7/25 · 54%" and handed
            # back the PREVIOUS fold's file, because `lastPrediction` is
            # written at the end and the download button had nothing newer.
            #
            # What says "finished" is the status line having no PERCENTAGE in
            # it: every working state carries one ("Trunk 1/2 · 2%", "Folding
            # 7/25 · 54%", "Language model · 40%") and the summary that
            # replaces it does not.
            # 🔴 FINISHED IS THE PAGE SAYING SO, AND NOTHING ELSE. The two
            # rules this replaces were both proxies and both had a case they
            # could not see: a NEW object with frames is false for a repeat
            # fold, which reuses the stem, and a status line without a
            # percentage is true before the click has been acted on. What is
            # left of `before` is the one thing it was genuinely good for -
            # telling this fold's object from the last one's - which the
            # readback below uses.
            done = (state.get("folding") is False
                    and (state.get("since") or 0) > pressed)
            any_frames = any((state.get("objects") or {}).values())
            if done and any_frames:
                break
            if time.time() > deadline:
                return {"error": "timed out", "status": state.get("status")}
            # 🔴 AND "IDLE" IS NOT A STATE THIS PAGE HAS. The first version
            # read `predict.disabled` as "a fold is running" and gave up when
            # it was false - measured, the button stays ENABLED for the whole
            # fold (it is how you stop one), so that test fired five seconds
            # into every fold and reported "no structure" while one was being
            # made. Worse, returning left the fold RUNNING, and the next
            # request's click stopped it: the second request then read
            # "Prediction stopped" and blamed itself.
            #
            # What does say a fold ended badly is the status line, which is
            # also what a reader would be looking at.
            said = (state.get("status") or "").lower()
            if any(word in said for word in ("stopped", "failed", "error", "refus")):
                return {"error": "the fold did not finish",
                        "status": state.get("status")}
            time.sleep(0.25)
        # 🔴 AND THE TAP IS DRAINED AFTER THE LOOP, OR THE LAST FRAMES NEVER
        # TRAVEL. The watch loop drains as it polls and then BREAKS - so every
        # frame the sampler emitted between the final poll and the end of the
        # fold was collected by the page and thrown away here. Reported as
        # skipped frames during a remote fold, and it is the END of the
        # trajectory that goes, which is the part worth watching.
        if job is not None:
            tail = cdp.evaluate(self.ws, """(() => {
              const t = window.__remoteTap || [];
              return t.splice(0, t.length);
            })()""")
            job["events"].extend(tail or [])
        # 🔴 "READY" IS "IT CAN HAND ONE OVER", NOT "THE BUTTON CAME BACK".
        # `loadIntoViewer` CLEARS the object's frames and re-adds them, so the
        # moment after a fold ends is a window with a settled status line, an
        # EMPTY object and a download button that writes nothing. Measured: a
        # fold whose status read "13 residues, pLDDT 88.8" reported frames 0
        # and zero atoms, while the same read a minute later gave 25 frames and
        # an 8,802-byte PDB. Waiting on any one internal is a race against a
        # rebuild nobody here owns; asking for the ARTEFACT until it exists is
        # not - and the artefact is what was requested anyway.
        deadline = time.time() + 120
        while True:
            out = cdp.evaluate(self.ws, READBACK_JS)
            if out.get("atoms"):
                break
            if time.time() > deadline:
                out["error"] = "the page never produced a structure"
                break
            time.sleep(0.5)
        # ...and once more, because `loadIntoViewer` runs at the END of a local
        # fold and the readback above WAITS for it: the frames it re-adds, and
        # the status line it writes, are emitted inside that wait.
        if job is not None:
            last = cdp.evaluate(self.ws, """(() => {
              const t = window.__remoteTap || [];
              return t.splice(0, t.length);
            })()""")
            job["events"].extend(last or [])
        out["ms"] = int((time.time() - started) * 1000)
        return out


def serve(port, backend, token, host="127.0.0.1"):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=REPO, **kw)

        def log_message(self, *a):
            pass

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers",
                             "content-type, x-localfold-token")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def _json(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def _authorised(self):
            given = self.headers.get("X-LocalFold-Token", "")
            if not given:
                query = urllib.parse.urlparse(self.path).query
                given = urllib.parse.parse_qs(query).get("t", [""])[0]
            return hmac.compare_digest(given, token)

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self):
            route = urllib.parse.urlparse(self.path).path
            if route == "/job":
                if not self._authorised():
                    return self._json(403, {"error": "token"})
                asked = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                job = JOBS.get(asked.get("id", [""])[0])
                if job is None:
                    return self._json(404, {"error": "no such job"})
                # 🔴 A WATERMARK, NOT A QUEUE THE READER DRAINS. Two polls can
                # overlap and a client can be re-created by a reload, so the
                # server never removes anything: `since` is what the caller has
                # already applied, which makes a repeated poll idempotent.
                since = int(asked.get("since", ["0"])[0])
                events = job["events"][since:]
                return self._json(200, {
                    "status": job["status"], "done": job["done"],
                    "events": events, "n": since + len(events),
                    "result": job["result"] if job["done"] else None,
                })
            if route == "/health":
                if not self._authorised():
                    return self._json(403, {"error": "token"})
                return self._json(200, {"ok": True, "busy": LOCK.locked(),
                                        "gpu": backend.adapter()})
            return super().do_GET()

        def do_POST(self):
            route = urllib.parse.urlparse(self.path).path
            if route not in ("/fold", "/stop"):
                return self._json(404, {"error": "no such route"})
            if not self._authorised():
                return self._json(403, {"error": "token"})
            # 🔴 STOPPING IS THE SAME BUTTON. `predict` is a toggle - it is how
            # a reader stops a fold - so the page needs no stop control of its
            # own and this cannot drift from what a person would do.
            if route == "/stop":
                try:
                    cdp.evaluate(backend.ws,
                                 "(document.getElementById('predict').click(), true)")
                    return self._json(200, {"ok": True})
                except Exception as cause:                    # noqa: BLE001
                    return self._json(500, {"error": str(cause)})
            length = int(self.headers.get("Content-Length", "0"))
            try:
                request = json.loads(self.rfile.read(length) or b"{}")
            except ValueError as cause:
                return self._json(400, {"error": f"not JSON: {cause}"})
            if not LOCK.acquire(blocking=False):
                return self._json(429, {"error": "one GPU, one fold: try again"})
            # The blocking form: the answer IS the structure. Nothing in the
            # tree asks for it; it is the door a curl has.
            if not request.get("stream"):
                try:
                    return self._json(200, backend.fold(request))
                except Exception as cause:                    # noqa: BLE001
                    return self._json(500, {"error": str(cause)})
                finally:
                    LOCK.release()
            # ...and the streamed form, which the page uses: the answer is a
            # name to poll, and the lock is held by the THREAD rather than by
            # this request - so it is released where the fold ends, in every
            # way it can end, and not where the POST returns.
            name = secrets.token_urlsafe(9)
            job = {"events": [], "status": "", "done": False, "result": None}
            with JOBS_LOCK:
                JOBS[name] = job
                for old_name in list(JOBS)[:-8]:
                    JOBS.pop(old_name, None)

            def run():
                try:
                    job["result"] = backend.fold(request, job)
                except Exception as cause:                    # noqa: BLE001
                    job["result"] = {"error": str(cause)}
                finally:
                    job["done"] = True
                    LOCK.release()

            threading.Thread(target=run, daemon=True).start()
            return self._json(200, {"job": name})

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    # 🔴 LOOPBACK, NOT EVERY INTERFACE. The tunnel client runs on this same
    # machine and reaches the port over 127.0.0.1, so binding wider buys
    # nothing and offers the runtime's own network a GPU with an HTTP API on
    # it. `--host 0.0.0.0` is there for whoever has a reason; the default has
    # none.
    httpd = socketserver.ThreadingTCPServer((host, port), Handler)
    httpd.daemon_threads = True
    return httpd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8710)
    parser.add_argument("--cdp-port", type=int, default=9333)
    parser.add_argument("--token", default=None,
                        help="the shared secret; one is generated when absent")
    parser.add_argument("--profile", default="/tmp/localfold-backend")
    parser.add_argument("--host", default="127.0.0.1",
                        help="what to bind; the tunnel reaches loopback")
    arguments = parser.parse_args()

    token = arguments.token or secrets.token_urlsafe(24)
    backend = Backend(arguments.port, arguments.cdp_port, arguments.profile)
    httpd = serve(arguments.port, backend, token, arguments.host)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"serving {REPO} on {arguments.host}:{arguments.port}", flush=True)
    backend.start()
    # One line, machine-readable, for the notebook cell that prints the handle.
    print("BACKEND " + json.dumps({"token": token, "gpu": backend.adapter()}), flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
