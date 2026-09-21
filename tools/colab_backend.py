"""LocalFold's own fold, on somebody else's GPU, reached over one URL.

    python3 tools/colab_backend.py --port 8710            # serve and broker
    python3 tools/colab_backend.py --port 8710 --token t  # a token of your own

It serves this checkout, opens a headless Chrome on it, and brokers between
two copies of the same page: the one it opened - `index.html?role=runtime`,
which folds - and the one a reader opens, `index.html?backend=colab`, which
asks. Four routes carry that, all of them token-checked:

    GET  /health            what the card is, and whether a fold is running
    POST /in                the reader asks: {op: "fold"|"stop"|"ping", payload}
    GET  /out?since=N       the runtime page collects what has been asked
    POST /up                the runtime page pushes what it says and draws
    GET  /down?since=N      the reader receives it

🔴 THE POINT IS THAT THERE IS NO SECOND IMPLEMENTATION. The fold that runs
here is web/app.js's own, in a real browser, from this checkout - the same
code a visitor's laptop runs, on a card the laptop does not have. A Python
re-implementation would be a second answer to every question this repository
has already answered once, and the two would part company on the first
modified residue.

🔴 AND THIS PROCESS IS A POST OFFICE, NOT A DRIVER. It used to press the page's
Fold button over CDP, scrape `#status-message` for the words "failed" and
"stopped", read the structure out of the download button and collect the
page's commentary by evaluating a splice every 250 ms - so the reader saw the
fold as often as a busy page answered the debugger, which was reported as the
bar sitting at "embedder · 1%" for a whole fold and everything arriving at the
end. All of that is now web/colab-bridge.js, INSIDE the page, where an event
is sent in the same task that made it and a command is a control being set
rather than a mouse being imitated. What is left for CDP is the two things
only it can do: start the browser and say what card it got.

🔴 THE TOKEN IS NOT OPTIONAL. Colab's proxy makes the port reachable to
whoever holds the notebook's URL, and anything that reaches it can spend the
GPU behind it - so every route checks the token before it reads a body, with
`hmac.compare_digest` rather than `==`, which is the one line that stops the
comparison leaking its answer in its timing.

🔴 AND CORS STAYS WIDE, WHICH IS ONLY SAFE BECAUSE OF THE TOKEN. Both pages are
same-origin with this server today, so nothing here needs it; it is kept for
the page a reader might point at a runtime from their own laptop, and an
allow-list of origins cannot be written for a URL that changes every session.
"""
import argparse
import hmac
import http.server
import json
import os
import secrets
import socketserver
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cdp                                                   # noqa: E402

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# 🔴 TWO MAILBOXES AND ONE SEQUENCE EACH, WHICH IS THE WHOLE BROKER. `EVENTS`
# is what the runtime page has said - status writes, bar fractions, sampler
# frames, and the finished prediction - and `COMMANDS` is what readers have
# asked of it. Both are append-only and both are read by WATERMARK: a caller
# says what it has already applied and gets what came after, so a poll that
# overlaps another, or a page reloaded mid-fold, repeats itself rather than
# losing anything. Nothing is ever removed on read.
EVENTS = []
COMMANDS = []
MAIL_LOCK = threading.Lock()
# 🔴 AND THE OLDEST EVENTS DO GO, because a session is not one fold: 25 sampler
# frames at a few kilobytes each, several folds deep, is a process that grows
# for as long as the notebook is open. `EVENT_BASE` is how many have been
# dropped, so a watermark keeps meaning the same thing across the drop and a
# reader that has fallen 4,000 events behind is told where the stream now
# starts instead of being handed the wrong ones.
EVENT_BASE = 0
EVENT_CAP = 4000

# 🔴 ONE GPU, ONE FOLD. Two at once is not twice the throughput, it is two
# folds that both take longer against a memory ceiling neither expected - and
# the page cannot refuse for us, because a second Fold click on a running page
# is how a fold gets STOPPED. The flag is raised when a fold command is
# accepted and lowered by the runtime page's own `result`, which is the event
# that says it has finished in every way a fold can finish.
FOLDING = {"on": False}
# 🔴 AND WHEN THE RUNTIME PAGE LAST ASKED FOR ITS COMMANDS, which is the only
# sign of life there is. A Colab runtime is recycled when the notebook is
# closed or left idle, and a reader whose fold was mid-flight then polls a
# broker that will never have another event for it - forever, because
# `FOLDING` is raised by the broker and lowered by the page, so a page that
# has gone takes the flag with it. The page's own poll is the heartbeat; no
# second mechanism and nothing extra on the wire.
LAST_SEEN = {"at": 0.0}
# 🔴 AND A WAY TO END IT FROM THE PAGE. A reader who is done with the runtime
# wants its GPU back, and the only thing that frees it is this process going
# away: the browser it started holds the card for as long as it lives. The
# notebook cell is blocked on the wait below, so setting this ends the cell.
STOPPING = threading.Event()
# 🔴 AND THE MACHINE ITSELF CAN BE RELEASED, WHICH IS NOT THE SAME THING.
# Stopping this service frees the CARD; the Colab VM stays assigned until the
# notebook lets it go, and a reader who pressed Disconnect meant the session.
# `google.colab.runtime.unassign()` is that, and reading its source is what
# made it reachable from here: it is a POST to a plain HTTP address in the
# environment (`TBE_RUNTIME_ADDR`), not a call over the kernel's channel - so
# a subprocess of the cell can do it, which is what this is.
RUNTIME_ADDR = os.environ.get("TBE_RUNTIME_ADDR")


def unassign_runtime():
    """Hand the Colab machine back. False where there is no machine to hand."""
    if not RUNTIME_ADDR:
        return False
    try:
        request = urllib.request.Request(f"http://{RUNTIME_ADDR}/unassign",
                                         data=b"", method="POST")
        with urllib.request.urlopen(request, timeout=10) as answer:
            return answer.status == 200
    except Exception as cause:                                # noqa: BLE001
        print(f"the runtime refused to unassign: {cause}", flush=True)
        return False


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

class Backend:
    """The headless Chrome this serves from, started once."""

    def __init__(self, port, cdp_port, profile, token):
        self.port = port
        self.cdp_port = cdp_port
        self.profile = profile
        self.token = token
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
        # 🔴 `role=runtime` IS THE PAGE BEING TOLD WHICH HALF IT IS, and the
        # token rides beside it because every route this page calls checks
        # one. web/colab-bridge.js reads both out of its own URL.
        self.ws.call("Page.navigate", url=(
            f"http://127.0.0.1:{self.port}/index.html"
            f"?role=runtime&t={urllib.parse.quote(self.token)}"))
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

    def wait_for_bridge(self, seconds=60):
        """...and that the page can REACH us, which is the other half.

        🔴 A STARTUP CHECK THAT USES THE REAL ROUTE. The bridge announces
        itself by POSTing `runtime-ready` to /up, so waiting for that event to
        appear in the mailbox proves the page loaded, read its token and can
        push - the three things every later event depends on. Waiting on a
        page internal over CDP, which is what this used to do, proves only the
        first and is exactly the pull this arrangement was built to stop
        relying on.
        """
        deadline = time.time() + seconds
        while time.time() < deadline:
            with MAIL_LOCK:
                if any(e.get("kind") == "runtime-ready" for e in EVENTS):
                    return True
            time.sleep(0.25)
        return False

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

        def _seen(self):
            """Milliseconds since the runtime page last asked for commands."""
            with MAIL_LOCK:
                at = LAST_SEEN["at"]
            return None if at == 0 else int((time.time() - at) * 1000)

        def _since(self):
            asked = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            try:
                return max(0, int(asked.get("since", ["0"])[0]))
            except ValueError:
                return 0

        def do_GET(self):
            route = urllib.parse.urlparse(self.path).path
            if route in ("/down", "/out", "/health") and not self._authorised():
                return self._json(403, {"error": "token"})
            # 🔴 A WATERMARK, NOT A QUEUE THE READER DRAINS. Two polls can
            # overlap and a page can be re-created by a reload, so nothing is
            # ever removed on read: `since` is what the caller has already
            # applied, which makes a repeated poll idempotent.
            if route == "/down":
                # 🔴 `head=1` IS THE WATERMARK WITHOUT THE STREAM. A reader
                # opening a fold needs to know where the stream stands so it
                # can ignore the last fold's events - and asking for that with
                # `since=0` hands it every frame of the last fold to throw
                # away, which on a 25-step sampler is megabytes.
                asked = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                if asked.get("head", [""])[0] == "1":
                    # 🔴 `_seen()` TAKES THE SAME LOCK, AND IT IS NOT
                    # REENTRANT. Called from inside a `with MAIL_LOCK` this
                    # deadlocked the whole broker - the first `head=1` request
                    # never returned AND never released, so every later
                    # request hung behind it and the gate timed out three arms
                    # later, in a route that was innocent. Read it first.
                    seen = self._seen()
                    with MAIL_LOCK:
                        head = {"events": [], "n": EVENT_BASE + len(EVENTS),
                                "folding": FOLDING["on"], "runtimeSeen": seen}
                    return self._json(200, head)
                since = self._since()
                with MAIL_LOCK:
                    first = max(0, since - EVENT_BASE)
                    events = EVENTS[first:]
                    n = EVENT_BASE + len(EVENTS)
                    folding = FOLDING["on"]
                # `from` says where the answer actually starts, which is only
                # different from `since` for a caller that fell behind the cap.
                return self._json(200, {"events": events, "n": n,
                                        "from": EVENT_BASE + first,
                                        "folding": folding,
                                        "runtimeSeen": self._seen()})
            if route == "/out":
                # 🔴 `head=1` IS A RUNTIME PAGE SAYING IT HAS JUST STARTED, and
                # it exists because a reload replayed the SESSION. The page
                # polled from zero, so every command the notebook had ever
                # sent was obeyed again - measured: the heartbeat arm navigates
                # that page away and back, and it came back and re-ran a fold
                # from ten minutes earlier, 58 residues of it, weights and all.
                # A page that has just loaded is not owed the past.
                asked = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                head = asked.get("head", [""])[0] == "1"
                since = self._since()
                with MAIL_LOCK:
                    LAST_SEEN["at"] = time.time()
                    commands = [] if head else COMMANDS[since:]
                    n = len(COMMANDS)
                return self._json(200, {"commands": commands, "n": n})
            if route == "/health":
                with MAIL_LOCK:
                    folding = FOLDING["on"]
                return self._json(200, {"ok": True, "busy": folding,
                                        "runtimeSeen": self._seen(),
                                        # ...so the page can say whether
                                        # Disconnect releases the MACHINE or
                                        # only stops the service on it.
                                        "colabRuntime": bool(RUNTIME_ADDR),
                                        "gpu": backend.adapter()})
            return super().do_GET()

        def _body(self):
            length = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(length) or b"{}")

        def do_POST(self):
            route = urllib.parse.urlparse(self.path).path
            if route not in ("/up", "/in"):
                return self._json(404, {"error": "no such route"})
            if not self._authorised():
                return self._json(403, {"error": "token"})
            try:
                body = self._body()
            except ValueError as cause:
                return self._json(400, {"error": f"not JSON: {cause}"})

            # THE RUNTIME PAGE SPEAKING. Every event is stamped on arrival
            # beside the page's own `at`, so a reader can say whether a fold
            # was slow or the feed was - the question the CDP version could
            # not answer, and the one that made this rewrite worth doing.
            if route == "/up":
                got = int(time.time() * 1000)
                arrived = body.get("events") or []
                global EVENT_BASE
                with MAIL_LOCK:
                    for event in arrived:
                        if not isinstance(event, dict):
                            continue
                        event["got"] = got
                        EVENTS.append(event)
                        if event.get("kind") == "result":
                            FOLDING["on"] = False
                        # 🔴 AND A PAGE THAT HAS JUST LOADED IS NOT FOLDING.
                        # The flag is raised when a command is accepted and
                        # lowered by the page's own `result`, so a runtime
                        # that died mid-fold - or was reloaded - took the flag
                        # with it and every later fold was refused 429 for the
                        # rest of the session. An announcement is that page
                        # saying it has just started.
                        if event.get("kind") == "runtime-ready":
                            FOLDING["on"] = False
                    if len(EVENTS) > EVENT_CAP:
                        drop = len(EVENTS) - EVENT_CAP // 2
                        del EVENTS[:drop]
                        EVENT_BASE += drop
                    n = EVENT_BASE + len(EVENTS)
                return self._json(200, {"ok": True, "n": n})

            # ...AND THE READER ASKING. `fold` is the only op that can collide
            # with itself, and the refusal is here rather than in the page
            # because a second Fold click on a running page STOPS the fold -
            # the button is a toggle, so the page cannot tell us "busy" by
            # refusing a press.
            op = body.get("op")
            # 🔴 SHUTDOWN IS THE BROKER'S OWN, NOT THE PAGE'S. Every other op is
            # forwarded to the runtime page and obeyed there; this one ends the
            # service - the browser, the GPU it holds and this process - so it
            # is answered here, after the answer has been written. What it
            # cannot do is end the Colab RUNTIME: that machine belongs to the
            # notebook, and only the notebook's own Runtime menu releases it.
            if op == "shutdown":
                self._json(200, {"ok": True, "stopping": True,
                                 "unassign": bool(RUNTIME_ADDR)})
                threading.Thread(target=lambda: (time.sleep(0.3),
                                                 STOPPING.set()), daemon=True).start()
                return None
            if op not in ("fold", "stop", "ping"):
                return self._json(400, {"error": f'unknown op "{op}"'})
            with MAIL_LOCK:
                if op == "fold" and FOLDING["on"]:
                    return self._json(429, {"error": "one GPU, one fold: try again"})
                if op == "fold":
                    FOLDING["on"] = True
                COMMANDS.append({"seq": len(COMMANDS), "op": op,
                                 "payload": body.get("payload")})
                return self._json(200, {"ok": True, "seq": len(COMMANDS) - 1})

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
    backend = Backend(arguments.port, arguments.cdp_port, arguments.profile, token)
    httpd = serve(arguments.port, backend, token, arguments.host)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"serving {REPO} on {arguments.host}:{arguments.port}"
          + (" · Disconnect will release this Colab machine" if RUNTIME_ADDR
             else " · no Colab runtime here, Disconnect stops the service"),
          flush=True)
    # 🔴 THE SERVER FIRST, THE PAGE SECOND. The page announces itself to /up
    # the moment it loads, so a browser started before the socket is listening
    # announces into a refused connection and the wait below times out on a
    # runtime that is working perfectly.
    backend.start()
    if not backend.wait_for_bridge():
        print("the page never announced itself: the bridge cannot reach /up",
              flush=True)
    # One line, machine-readable, for the notebook cell that prints the handle.
    print("BACKEND " + json.dumps({"token": token, "gpu": backend.adapter()}), flush=True)
    try:
        # Woken by Ctrl-C, or by a reader pressing Disconnect - see STOPPING.
        while not STOPPING.wait(timeout=3600):
            pass
        print("stopped by the page", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        # 🔴 THE BROWSER IS THIS PROCESS'S, AND IT DOES NOT DIE WITH IT.
        # `cdp.launch` starts a headless Chrome that outlives a Ctrl-C and a
        # SIGINT from a gate - found by counting processes after a green run:
        # eight of them, on the profile this backend had just stopped using.
        # In a Colab runtime the container takes them; on a developer's machine
        # they are the "another browser on the machine" that makes the next
        # measurement somebody else's.
        if backend.proc is not None:
            backend.proc.terminate()
            try:
                backend.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                backend.proc.kill()
        # 🔴 THE BROWSER FIRST, THE MACHINE SECOND. Unassigning pulls the VM
        # out from under this process, so anything that has to happen on the
        # way out has to have happened already.
        if STOPPING.is_set() and unassign_runtime():
            print("the Colab runtime has been unassigned", flush=True)


if __name__ == "__main__":
    main()
