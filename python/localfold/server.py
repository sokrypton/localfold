"""LocalFold's page, folding on a native port - this machine's GPU, or a runtime's reached over one URL.

    localfold serve                                                  # (the wheel) this machine, the browser opened
    python3 python/localfold/server.py --native --local --open       # the same from a checkout
    python3 python/localfold/server.py --port 8710 --native --token t   # a token of your own (Colab's notebook)

It serves the page - the reader's page is `index.html?backend=native&t=<token>` on it (with no token on the reader's
own machine, --local) - and brokers between that
page and one worker process, python/localfold/worker.py, which folds with the native port this machine has:
metal/ on Apple silicon, cuda/ on an NVIDIA card (Colab's runtime is one). Instead of the page's WebGPU code: twice
the speed on an M2, every model, no binding ceiling and no weights in the browser. /health says which (`native`).
Three routes carry that, all of them token-checked:

    GET  /health            what the card is, and whether a fold is running
    POST /in                the reader asks: {op: "fold"|"stop"|"shutdown", payload}
    GET  /down?since=N      the reader receives what the worker says

🔴 THERE IS NO BROWSER ON THIS SIDE ANY MORE. This used to open a headless
Chrome on `index.html?role=runtime` and relay folds to the page's own WebGPU
code; once every reader's fold went to CUDA nothing reached that page, and it
was removed with its mailbox (`/up`, `/out`), its weights proxy and its warm-up
(see docs/WEB.md, 2026-10-08). What the card is comes from the driver.

🔴 AND THIS PROCESS IS A POST OFFICE, NOT A DRIVER. The worker prints one
bridge event per line - a status write, a bar fraction, a sampler frame, a
contact map, the finished result - and each is numbered, stamped and held here
until the reader asks for it. Nothing is decided about a molecule in Python:
the worker's exporters read the job with the page's own reader.

🔴 THE TOKEN IS NOT OPTIONAL. Colab's proxy makes the port reachable to
whoever holds the notebook's URL, and anything that reaches it can spend the
GPU behind it - so every route checks the token before it reads a body, with
`hmac.compare_digest` rather than `==`, which is the one line that stops the
comparison leaking its answer in its timing.

🔴 AND CORS STAYS WIDE, WHICH IS ONLY SAFE BECAUSE OF THE TOKEN. The page is
same-origin with this server today, so nothing here needs it; it is kept for a
page a reader might point at a runtime from their own laptop, and an allow-list
of origins cannot be written for a URL that changes every session.
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
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
# the checkout (python/localfold/ is two below it) - or, installed, whatever `localfold serve` says
REPO = os.environ.get("LOCALFOLD_REPO") or os.path.normpath(os.path.join(HERE, "..", ".."))
# ...and what is served: the checkout, or the wheel's built site
SITE = os.environ.get("LOCALFOLD_SITE_DIR") or REPO


NATIVE = "metal" if sys.platform == "darwin" else "cuda"


def gpu_info():
    """The first GPU as the driver names it - {name, memoryMiB} - or {} where there is none. On a Mac the chip and
    the machine's memory, which its GPU shares."""
    if NATIVE == "metal":
        try:
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                                  timeout=10).stdout.strip()
            memory = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True,
                                        timeout=10).stdout) >> 20
            return {"name": chip or "Apple GPU", "memoryMiB": memory}
        except (OSError, subprocess.SubprocessError, ValueError):
            return {"name": "Apple GPU"}
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return {}
    if not out:
        return {}
    name, _, memory = out[0].rpartition(",")
    try:
        return {"name": name.strip(), "memoryMiB": int(memory)}
    except ValueError:
        return {"name": out[0].strip()}


# 🔴 ONE MAILBOX AND ONE SEQUENCE, WHICH IS THE WHOLE BROKER. `EVENTS` is what
# the worker has said, append-only and read by WATERMARK: a caller says what it
# has already applied and gets what came after, so a poll that overlaps another,
# or a page reloaded mid-fold, repeats itself rather than losing anything.
# Nothing is ever removed on read.
EVENTS = []
MAIL_LOCK = threading.Lock()
# ...and notified on every arrival, so a reader's `/down?wait=` returns the moment there is something to
# read rather than on its next poll (a 300 ms poll was most of a warm CUDA fold's click-to-result)
MAIL = threading.Condition(MAIL_LOCK)
# 🔴 AND THE OLDEST EVENTS DO GO, because a session is not one fold: 25 sampler
# frames at a few kilobytes each, several folds deep, is a process that grows
# for as long as the notebook is open. `EVENT_BASE` is how many have been
# dropped, so a watermark keeps meaning the same thing across the drop and a
# reader that has fallen 4,000 events behind is told where the stream now
# starts instead of being handed the wrong ones.
EVENT_BASE = 0
EVENT_CAP = 4000

# 🔴 ONE GPU, ONE FOLD. Two at once is not twice the throughput, it is two
# folds that both take longer against a memory ceiling neither expected. The
# flag is raised when a fold is accepted and lowered by the worker's `result`,
# which is the event that says it has finished in every way a fold can finish -
# including the worker dying, which `Worker._read` turns into one.
FOLDING = {"on": False}
# 🔴 AND A WAY TO END IT FROM THE PAGE. A reader who is done with the runtime
# wants its GPU back, and the worker holds the card for as long as it lives.
# The notebook cell is blocked on the wait below, so setting this ends the cell.
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


def push_event(event):
    """One event into the mailbox, stamped with its arrival beside the worker's own `at`."""
    global EVENT_BASE
    event["got"] = int(time.time() * 1000)
    with MAIL_LOCK:
        EVENTS.append(event)
        if event.get("kind") == "result":
            FOLDING["on"] = False
        if len(EVENTS) > EVENT_CAP:
            drop = len(EVENTS) - EVENT_CAP // 2
            del EVENTS[:drop]
            EVENT_BASE += drop
        MAIL.notify_all()


class Worker:
    """python/localfold/worker.py, started on its first fold and kept for the next.

    🔴 ITS LINES ARE EVENTS, ITS SEQ IS ITS OWN. The worker prints one bridge
    event per line; they are numbered here, so the reader's sort-by-seq holds.
    `LOCALFOLD_WORKER` stands a stub in for it, which is how
    tools/check-native-bridge.py and tools/check-model-pending.py test this path
    with no card (tools/stub_worker.py).
    """

    def __init__(self):
        self.proc = None
        self.seq = 0
        self.lock = threading.Lock()

    def _start(self):
        worker = (os.environ.get("LOCALFOLD_WORKER") or os.environ.get("LOCALFOLD_CUDA_WORKER")
                  or os.path.join(HERE, "worker.py"))
        # (its own process group, so a Stop takes its featurisers and model servers with it: Linux's prctl does
        # that for each child, and macOS has none)
        self.proc = subprocess.Popen(
            [sys.executable, worker],
            cwd=REPO, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1, start_new_session=True)
        threading.Thread(target=self._read, args=(self.proc,), daemon=True).start()

    def _read(self, proc):
        for line in proc.stdout:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("kind") in ("ready", "cuda-ready"):
                continue
            with self.lock:
                event["seq"] = self.seq
                self.seq += 1
            push_event(event)
        # ...a worker that died mid-fold must still end the fold.
        with MAIL_LOCK:
            folding = FOLDING["on"]
        if folding and proc is self.proc:
            push_event({"kind": "result", "seq": self.seq, "at": int(time.time() * 1000),
                        "payload": {"error": "the native worker exited"}})

    def fold(self, payload):
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                self._start()
            self.proc.stdin.write(json.dumps(payload) + "\n")
            self.proc.stdin.flush()

    def stop(self):
        """A native binary cannot be interrupted from outside it: the worker goes, and
        the next fold starts a new one."""
        with self.lock:
            proc, self.proc = self.proc, None
        if proc is not None:
            kill_group(proc)
        push_event({"kind": "result", "seq": self.seq, "at": int(time.time() * 1000),
                    "payload": {"error": "stopped"}})


def kill_group(proc):
    """The worker and every process it started (its own session: see Worker._start)."""
    import signal
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        proc.kill()


def serve(port, token, host="127.0.0.1", worker=None, gpu=None, local=False):
    gpu = gpu or {}

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=SITE, **kw)

        # 🔴 NO-STORE ON THE PAGE'S OWN FILES, as tools/serve.py sends and for
        # the reason CLAUDE.md gives: SimpleHTTPRequestHandler sends no cache
        # headers, so Chrome caches every ES module heuristically, and a
        # reader's page holding last week's native-bridge.js talking to this
        # week's broker is a fold that fails on the wire format.
        def end_headers(self):
            self.send_header("Cache-Control", "no-store, must-revalidate")
            super().end_headers()

        def log_message(self, *a):
            pass

        def _cors(self):
            # (not for a local server: its page is same-origin, and the wildcard would let any other page read it)
            if local:
                return
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers",
                             "content-type, x-localfold-token")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def _json(self, code, payload):
            body = json.dumps(payload).encode()
            # 🔴 COMPRESSED, BECAUSE ON COLAB THIS CROSSES THE INTERNET. A
            # finished fold's result is megabytes of JSON and a trajectory frame
            # a PDB's worth of text; gzip takes them 2.5x and 4-5x at its
            # fastest level. Only where the client asked (a browser always does;
            # urllib does not) and only for a body worth it.
            gzipped = len(body) > 65536 and "gzip" in self.headers.get("Accept-Encoding", "")
            if gzipped:
                import gzip
                body = gzip.compress(body, compresslevel=1)
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            if gzipped:
                self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def _authorised(self):
            if token is None:           # (a local server started without one: the reader's own machine, on loopback)
                return True
            given = self.headers.get("X-LocalFold-Token", "")
            if not given:
                query = urllib.parse.urlparse(self.path).query
                given = urllib.parse.parse_qs(query).get("t", [""])[0]
            return hmac.compare_digest(given, token)

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.end_headers()

        def _since(self):
            asked = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            try:
                return max(0, int(asked.get("since", ["0"])[0]))
            except ValueError:
                return 0

        def do_GET(self):
            route = urllib.parse.urlparse(self.path).path
            if route in ("/down", "/health") and not self._authorised():
                return self._json(403, {"error": "token"})
            # 🔴 A WATERMARK, NOT A QUEUE THE READER DRAINS. Two polls can
            # overlap and a page can be re-created by a reload, so nothing is
            # ever removed on read: `since` is what the caller has already
            # applied, which makes a repeated poll idempotent.
            if route == "/down":
                asked = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                # 🔴 `head=1` IS THE WATERMARK WITHOUT THE STREAM. A reader
                # opening a fold needs to know where the stream stands so it
                # can ignore the last fold's events - and asking with `since=0`
                # hands it every frame of the last fold to throw away.
                if asked.get("head", [""])[0] == "1":
                    with MAIL_LOCK:
                        return self._json(200, {"events": [], "n": EVENT_BASE + len(EVENTS),
                                                "folding": FOLDING["on"]})
                since = self._since()
                # `wait=MS` (at most ten seconds): held until there is an event past `since`, so the reader
                # need not poll
                wait = asked.get("wait", ["0"])[0]
                wait = min(int(wait), 10000) / 1000 if wait.isdigit() else 0
                with MAIL_LOCK:
                    if wait:
                        MAIL.wait_for(lambda: EVENT_BASE + len(EVENTS) > since, timeout=wait)
                    first = max(0, since - EVENT_BASE)
                    events = EVENTS[first:]
                    n = EVENT_BASE + len(EVENTS)
                    folding = FOLDING["on"]
                # `from` says where the answer actually starts, which is only
                # different from `since` for a caller that fell behind the cap.
                return self._json(200, {"events": events, "n": n, "waits": True,
                                        "from": EVENT_BASE + first, "folding": folding})
            if route == "/health":
                with MAIL_LOCK:
                    folding = FOLDING["on"]
                return self._json(200, {"ok": True, "busy": folding,
                                        # ...so the page can say whether
                                        # Disconnect releases the MACHINE or
                                        # only stops the service on it.
                                        "colabRuntime": bool(RUNTIME_ADDR),
                                        # ...or the reader's own machine (`localfold serve`): the page then offers
                                        # no Disconnect - its Stop ends a fold, and Ctrl-C ends the server
                                        "local": local,
                                        "backends": ["native"] if worker is not None else [],
                                        # (which native port folds: cuda/ or metal/)
                                        "native": NATIVE if worker is not None else None,
                                        "gpu": gpu})
            return super().do_GET()

        def _body(self):
            length = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(length) or b"{}")

        def do_POST(self):
            route = urllib.parse.urlparse(self.path).path
            if route != "/in":
                return self._json(404, {"error": "no such route"})
            if not self._authorised():
                return self._json(403, {"error": "token"})
            try:
                body = self._body()
            except ValueError as cause:
                return self._json(400, {"error": f"not JSON: {cause}"})
            op = body.get("op")
            # 🔴 SHUTDOWN ENDS THE SERVICE - the worker, the card it holds and
            # this process - so it is answered here, after the answer has been
            # written, and the machine is handed back on the way out (main).
            if op == "shutdown":
                self._json(200, {"ok": True, "stopping": True,
                                 "unassign": bool(RUNTIME_ADDR)})
                threading.Thread(target=lambda: (time.sleep(0.3),
                                                 STOPPING.set()), daemon=True).start()
                return None
            if op == "stop":
                if worker is not None and FOLDING["on"]:
                    worker.stop()
                return self._json(200, {"ok": True})
            if op != "fold":
                return self._json(400, {"error": f'unknown op "{op}"'})
            if worker is None:
                return self._json(400, {"error": "this runtime has no native backend"})
            with MAIL_LOCK:
                if FOLDING["on"]:
                    return self._json(429, {"error": "one GPU, one fold: try again"})
                FOLDING["on"] = True
            worker.fold(body.get("payload") or {})
            return self._json(200, {"ok": True, "backend": NATIVE})

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
    parser.add_argument("--token", default=None,
                        help="the shared secret; one is generated when absent - except with --local, which needs none")
    parser.add_argument("--host", default="127.0.0.1",
                        help="what to bind; the tunnel reaches loopback")
    parser.add_argument("--native", "--cuda", dest="native", action="store_true",
                        help="offer this machine's native backend: metal/ on a Mac, cuda/ elsewhere (built: "
                             "metal/build.sh, cuda/build.sh); --cuda is its old name")
    parser.add_argument("--open", action="store_true", help="open the page in the default browser")
    parser.add_argument("--local", action="store_true",
                        help="this is the reader's own machine (localfold serve): no token unless --token names one,"
                             " and the page offers no Disconnect")
    arguments = parser.parse_args()

    # 🔴 A LOCAL SERVER TAKES NO TOKEN, BY CHOICE: on the reader's own machine the link stays plain
    # (http://127.0.0.1:8710/index.html?backend=native). Anything else on this machine that can reach loopback - a
    # page open in the browser included - can then ask it for a fold; Colab's port is reachable from outside, so a
    # runtime keeps its token.
    token = arguments.token or (None if arguments.local else secrets.token_urlsafe(24))
    worker = Worker() if arguments.native else None
    gpu = gpu_info()
    httpd = serve(arguments.port, token, arguments.host, worker, gpu, local=arguments.local)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"serving {SITE} on {arguments.host}:{arguments.port}"
          + (" · Disconnect will release this Colab machine" if RUNTIME_ADDR
             else " · Ctrl-C stops it" if arguments.local
             else " · no Colab runtime here, Disconnect stops the service"),
          flush=True)
    # One line, machine-readable, for the notebook cell that prints the handle.
    print("BACKEND " + json.dumps({"token": token, "gpu": gpu}), flush=True)
    url = (f"http://{'127.0.0.1' if arguments.host in ('0.0.0.0', '') else arguments.host}:{arguments.port}"
           f"/index.html?backend=native" + (f"&t={token}" if token else ""))
    print(f"open {url}", flush=True)
    if arguments.open:
        import webbrowser
        webbrowser.open(url)
    try:
        # Woken by Ctrl-C, or by a reader pressing Disconnect - see STOPPING.
        while not STOPPING.wait(timeout=3600):
            pass
        print("stopped by the page", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        if worker is not None and worker.proc is not None:
            kill_group(worker.proc)
        # 🔴 THE WORKER FIRST, THE MACHINE SECOND. Unassigning pulls the VM
        # out from under this process, so anything that has to happen on the
        # way out has to have happened already.
        if STOPPING.is_set() and unassign_runtime():
            print("the Colab runtime has been unassigned", flush=True)


if __name__ == "__main__":
    main()
