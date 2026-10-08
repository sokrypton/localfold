"""The Colab bridge carries a CUDA fold both ways, and the feed is LIVE: npm run test:colab.

    python3 tools/check-colab-bridge.py

NO GPU AND NO WEIGHTS. It starts tools/colab_backend.py with
tools/colab_stub_worker.py standing in for cuda/worker.py - the stub emits what
this gate appends to its feed and holds its fold until it has sent a result -
and drives every route from a reader's side, over plain HTTP and through a real
reader's page (`index.html?backend=colab`) in a second browser. What it proves:

  * /health says what the card is (the driver's name) and that CUDA is offered;
  * one GPU, one fold - a second `fold` while one holds is refused 429, and the
    refusal lifts when the worker's `result` arrives;
  * every event is numbered (`seq`) and carries both clocks - `at` from the
    worker, `got` from the broker - so a slow feed and a slow fold are two
    numbers; the watermark is idempotent and `head=1` is the stream's head;
  * THE FEED IS LIVE: a reader holding `/down?wait=` is answered within a
    second of the worker speaking, not on its next poll;
  * what the broker drops past its cap is not dropped in silence (`from`);
  * Stop ends a held fold by ending the worker, and the next fold starts a new
    one; a worker that dies mid-fold ends the fold with an error;
  * THE READER'S OWN PAGE: no backend to choose, the Live box shown, the badge
    naming the card; pressing Fold hands the worker a job carrying every
    control the reader set; the worker's status, frames and result reach that
    page's screen, its prediction's typed arrays are typed arrays, and the
    reader did no model work of its own;
  * a reader that opens the page MID-FOLD attaches to it;
  * a runtime started without `--cuda` says so on the badge and refuses Fold;
  * and no route answers anything without the token.

🔴 WHAT IT CANNOT COVER is a fold: the model never runs here, and what the
reader ingests is a structure this gate wrote down. `npm run test:cuda` folds
for real through the same broker and page.
"""
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cdp                                                   # noqa: E402

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PORT = int(os.environ.get("BRIDGE_PORT", "8791"))
READER_CDP_PORT = int(os.environ.get("BRIDGE_READER_CDP_PORT", "9392"))
TOKEN = "check-colab-bridge-token"
BASE = f"http://127.0.0.1:{PORT}"
WORK = tempfile.mkdtemp(prefix="localfold-bridge-check-")
FEED = os.path.join(WORK, "feed.jsonl")
JOBS = os.path.join(WORK, "jobs.jsonl")
STUB = os.path.join(REPO, "tools", "colab_stub_worker.py")
SEQUENCE = "GWSTELEKHREELKEFLKKEGITLGFTNAEKQEQAQKLGLGKKVSPELLIKAFAILKK"

bad = []


def tiny_pdb(shift=0.0):
    """Four alpha carbons - a structure to py2Dmol, small enough to write down."""
    rows = ["ATOM  %5d  CA  ALA A%4d    %8.3f%8.3f%8.3f  1.00 50.00           C"
            % (i + 1, i + 1, 3.8 * i + shift, 0.0, 0.0) for i in range(4)]
    return "\n".join(rows) + "\nEND\n"


def cuda_result(pdb, status="AlphaFold 3 on CUDA (stub) · done"):
    """A result in cuda/worker.py's own shape, which is what the page ingests."""
    n = 4
    return {"cuda": True, "model": "cuda af3", "family": "af3", "pdb": pdb,
            "confidence": {"plddt": [50.0] * n, "meanPlddt": 50.0, "ptm": 0.5,
                           "predictedAlignedError": [1.0] * (n * n), "contactProbs": [0.5] * (n * n)},
            "tokens": {"chainIds": ["A"] * n, "resIds": list(range(1, n + 1))},
            "chains": ["AAAA"], "msas": {}, "atoms": n, "status": status}


def feed(*events):
    with open(FEED, "a") as handle:
        for kind, payload in events:
            handle.write(json.dumps({"kind": kind, "payload": payload}) + "\n")


def jobs():
    if not os.path.exists(JOBS):
        return []
    with open(JOBS) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def call(route, body=None, token=TOKEN, timeout=20, base=BASE):
    url = f"{base}{route}"
    url += ("&" if "?" in route else "?") + f"t={token}" if token is not None else ""
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, headers={"content-type": "application/json"},
                                     method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:
            return answer.status, json.loads(answer.read() or b"{}")
    except urllib.error.HTTPError as refused:
        return refused.code, json.loads(refused.read() or b"{}")


def head():
    return call("/down?head=1")[1]


def wait_for(kind, since, seconds=20):
    """The reader's own loop: /down until `kind` turns up. Returns (event, n, everything seen)."""
    deadline, seen = time.time() + seconds, []
    while time.time() < deadline:
        code, said = call(f"/down?since={since}&wait=2000")
        if code != 200:
            return None, since, seen
        since = said.get("n", since)
        for event in said.get("events") or []:
            seen.append(event)
            if event.get("kind") == kind:
                return event, since, seen
    return None, since, seen


def start_broker(port, cuda=True, extra_env=None):
    proc = subprocess.Popen(
        [sys.executable, "tools/colab_backend.py", "--port", str(port), "--token", TOKEN,
         *(["--cuda"] if cuda else [])],
        cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        env=dict(os.environ, LOCALFOLD_CUDA_WORKER=STUB, LOCALFOLD_STUB_FEED=FEED,
                 LOCALFOLD_STUB_JOBS=JOBS, **(extra_env or {})))
    deadline = time.time() + 60
    while time.time() < deadline:
        line = proc.stdout.readline()
        if line == "" and proc.poll() is not None:
            break
        if line.startswith("BACKEND "):
            return proc, json.loads(line[len("BACKEND "):])
    raise SystemExit(f"FAIL: the broker on {port} never printed its BACKEND line")


def stop_broker(proc):
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def open_reader(ws, port=PORT):
    ws.call("Page.navigate", url=f"http://127.0.0.1:{port}/index.html?backend=colab&t={TOKEN}")
    cdp.wait_for(ws, "!!window.__entityList && !!document.querySelector('.colab-said')", 120, "the reader's page")
    # 🔴 A FRESH PROFILE HAS ACCEPTED NO MODEL TERMS, and the dialog eats the click.
    cdp.evaluate(ws, """(() => {
      for (const key of ['alphafold3', 'openbind0', 'opendde', 'boltz2', 'protenix2', 'intellifold2', 'rosettafold3'])
        try { localStorage.setItem('localfold.modelTerms.' + key, 'accepted'); } catch (cause) {}
      return true;
    })()""")


print(f"starting the broker on {PORT} with the stub worker…")
backend, announced = start_broker(PORT)
reader = None
try:
    # 1 · what the card is, and what is offered.
    code, health = call("/health")
    print(f"  /health: backends {health.get('backends')}, gpu {health.get('gpu')}, colabRuntime {health.get('colabRuntime')}")
    if code != 200 or health.get("backends") != ["cuda"]:
        bad.append(f"/health answered {code} offering {health.get('backends')}, not ['cuda']")
    if health.get("gpu") != announced.get("gpu"):
        bad.append("/health and the BACKEND line name different cards")

    # 2 · one GPU, one fold; numbered, stamped events; the watermark.
    start = head().get("n", 0)
    code, said = call("/in", {"op": "fold", "payload": {"backend": "cuda", "job": "{}"}})
    if code != 200:
        bad.append(f"a fold was refused {code}: {said}")
    deadline = time.time() + 10
    while time.time() < deadline and not jobs():
        time.sleep(0.05)
    if len(jobs()) != 1:
        bad.append(f"the worker was handed {len(jobs())} job(s), not one")
    code, busy = call("/in", {"op": "fold", "payload": {"backend": "cuda", "job": "{}"}})
    if code != 429:
        bad.append(f"a second fold while one holds answered {code}, not 429")
    if not head().get("folding"):
        bad.append("head=1 does not say a fold is running while one holds")
    feed(("status", "stub · trunk"), ("progress", 0.5), ("frame", tiny_pdb()), ("result", cuda_result(tiny_pdb())))
    result, n, seen = wait_for("result", start)
    kinds = [event.get("kind") for event in seen]
    print(f"  a fold: {kinds}, seq {[event.get('seq') for event in seen]}")
    if kinds != ["status", "progress", "frame", "result"]:
        bad.append(f"the fold's events arrived as {kinds}")
    seqs = [event.get("seq") for event in seen]
    if seqs != sorted(seqs) or len(set(seqs)) != len(seqs):
        bad.append(f"the events are not numbered in order: {seqs}")
    if any("at" not in event or "got" not in event for event in seen):
        bad.append("an event is missing one of its two clocks (at, got)")
    again = call(f"/down?since={n}")[1]
    if again.get("events"):
        bad.append("the watermark is not idempotent: a poll at the head returned events")
    if head().get("folding"):
        bad.append("the result did not lower the busy flag")

    # 3 · THE FEED IS LIVE: a held ask is answered when the worker speaks.
    call("/in", {"op": "fold", "payload": {"backend": "cuda", "job": "{}"}})
    time.sleep(0.5)
    n = head().get("n", 0)
    lags = []
    for i in range(5):
        answer = {}
        reader_thread = threading.Thread(target=lambda: answer.update(call(f"/down?since={n}&wait=8000")[1]))
        reader_thread.start()
        time.sleep(0.3)
        sent = time.time()
        feed(("status", f"live {i}"))
        reader_thread.join(10)
        lags.append(time.time() - sent)
        n = answer.get("n", n)
    feed(("result", cuda_result(tiny_pdb())))
    wait_for("result", n)
    print(f"  the feed: answered {', '.join(f'{lag * 1000:.0f}' for lag in lags)} ms after the worker spoke")
    if max(lags) > 1.0:
        bad.append(f"a held /down?wait= answered {max(lags):.2f} s after the worker spoke - the feed is not live")

    # 4 · what the broker drops is not dropped in silence.
    call("/in", {"op": "fold", "payload": {"backend": "cuda", "job": "{}"}})
    time.sleep(0.3)
    feed(*[("status", f"flood {i}") for i in range(4100)], ("result", cuda_result(tiny_pdb())))
    deadline = time.time() + 30
    while time.time() < deadline and head().get("folding"):
        time.sleep(0.1)
    flooded = call("/down?since=0")[1]
    print(f"  after 4,100 events: a reader at 0 is told the stream starts at {flooded.get('from')} of {flooded.get('n')}")
    if not flooded.get("from"):
        bad.append("past the cap, a reader that fell behind is not told where the stream now starts")

    # 5 · Stop ends a held fold, and the next fold is a new worker; a dead worker ends its fold.
    n = head().get("n", 0)
    call("/in", {"op": "fold", "payload": {"backend": "cuda", "job": "{}"}})
    time.sleep(0.3)
    call("/in", {"op": "stop"})
    stopped, n, _ = wait_for("result", n)
    print(f"  Stop: {(stopped or {}).get('payload')}, folding {head().get('folding')}")
    if (stopped or {}).get("payload", {}).get("error") != "stopped" or head().get("folding"):
        bad.append("Stop did not end the held fold with 'stopped' and lower the flag")
    before = len(jobs())
    call("/in", {"op": "fold", "payload": {"backend": "cuda", "job": "{}"}})
    time.sleep(0.5)
    if len(jobs()) != before + 1:
        bad.append("the fold after a Stop never reached a worker")
    feed(("__exit", None))
    died, n, _ = wait_for("result", n)
    print(f"  a worker that dies mid-fold: {(died or {}).get('payload')}")
    if "exited" not in str((died or {}).get("payload", {}).get("error")) or head().get("folding"):
        bad.append("a worker that died mid-fold did not end the fold with an error")

    # 6 · THE READER'S OWN PAGE.
    print("  opening the reader's page…")
    reader, reader_ws = cdp.launch(READER_CDP_PORT, "/tmp/localfold-bridge-reader")
    reader_ws.call("Page.enable")
    reader_ws.call("Runtime.enable")
    # 🔴 THE COLLECTOR GOES IN BEFORE THE PAGE DOES, and so does the count of model work: a fold that
    # runs somewhere else must not ask this browser for a GPU or pull a weight shard.
    reader_ws.call("Page.addScriptToEvaluateOnNewDocument", source="""
      window.__pageErrors = [];
      addEventListener('error', (e) => window.__pageErrors.push(String(e.message)));
      addEventListener('unhandledrejection', (e) => window.__pageErrors.push('unhandled: ' + String(e.reason)));
      window.__readerGpu = 0;
      if (navigator.gpu) {
        const ask = navigator.gpu.requestAdapter.bind(navigator.gpu);
        navigator.gpu.requestAdapter = (...a) => { window.__readerGpu += 1; return ask(...a); };
      }
      try { performance.setResourceTimingBufferSize(100000); } catch {}
    """)
    open_reader(reader_ws)
    time.sleep(1.5)
    badge = cdp.evaluate(reader_ws, """(() => ({
      said: document.querySelector('.colab-said')?.textContent ?? '',
      select: !!document.querySelector('#colab-status select'),
      live: !!document.querySelector('.colab-live input'),
    }))()""")
    print(f"  the badge: {badge}")
    if badge["select"]:
        bad.append("the badge offers a backend select - CUDA is not a choice")
    if not badge["live"]:
        bad.append("the badge offers no Live box on a runtime with CUDA")
    card = (health.get("gpu") or {}).get("name")
    if card and card not in badge["said"]:
        bad.append(f"the badge reads {badge['said']!r}, not naming the card {card!r}")
    # 🔴 A FORM THAT IS NOT THE DEFAULTS in every control with a choice, through the page's own change
    # handlers, because the question is whether the reader's settings reach the worker.
    chosen = cdp.evaluate(reader_ws, """(() => {
      const g = (id) => document.getElementById(id);
      window.__entityList.set([{ type: 'protein', value: %s, copies: 1, modifications: [] }]);
      for (const [id, value] of [['msa-mode', 'none'], ['af3-mode', 'flow']]) {
        g(id).value = value;
        g(id).dispatchEvent(new Event('change', { bubbles: true }));
      }
      const counts = [...g('af3-count').options].map((o) => o.value);
      const count = counts.find((v) => v !== g('af3-count').value) ?? counts[0];
      const want = { 'af3-count': count, 'recycles': '1', 'random-seed': '7' };
      for (const [id, value] of Object.entries(want)) g(id).value = value;
      return { ...want, 'af3-mode': 'flow', 'msa-mode': 'none' };
    })()""" % json.dumps(SEQUENCE))
    before = len(jobs())
    cdp.wait_for(reader_ws, "!document.getElementById('predict').disabled", 60, "the reader's fold button")
    cdp.evaluate(reader_ws, "(document.getElementById('predict').click(), true)")
    deadline = time.time() + 30
    while time.time() < deadline and len(jobs()) == before:
        time.sleep(0.1)
    asked = jobs()[-1] if len(jobs()) > before else None
    if asked is None:
        bad.append("pressing Fold on the reader's page handed the CUDA worker no job - foldOnBackend never asked")
    else:
        sent = asked.get("controls") or {}
        print(f"  the reader asked CUDA: {len(asked.get('entities') or [])} entity, {len(sent)} control(s),"
              f" model {asked.get('family')}, job {bool(asked.get('job'))}, frames {asked.get('frames')}")
        if asked.get("backend") != "cuda":
            bad.append(f"the reader's fold went to backend {asked.get('backend')!r}, not cuda")
        # 🔴 EVERY CONTROL THE READER SET, NOT THE FIVE SOMEBODY LISTED - the allow-list trap.
        missing = {k: (v, sent.get(k)) for k, v in chosen.items() if sent.get(k) != v}
        if missing:
            bad.append(f"the reader's form did not travel (wanted, carried): {missing}")
        if not asked.get("job") or not asked.get("family"):
            bad.append("the CUDA job carried no AlphaFold 3 JSON or no resolved family")
    feed(("status", "stub · diffusion"), ("frame", tiny_pdb(0)), ("frame", tiny_pdb(2)),
         ("result", cuda_result(tiny_pdb(4), "AlphaFold 3 on CUDA (stub) · done in 0.1 s")))
    seen = {}
    deadline = time.time() + 30
    while time.time() < deadline:
        seen = cdp.evaluate(reader_ws, """(() => {
          const p = window.__lastPrediction ? window.__lastPrediction() : null;
          return {
            status: document.getElementById('status-message')?.textContent ?? '',
            lag: (window.__remoteLag || []).length,
            pae: p?.confidence?.predictedAlignedError?.constructor?.name ?? null,
            plddt: p?.confidence?.plddt?.constructor?.name ?? null,
            model: p?.model ?? null,
            errors: window.__pageErrors || [],
          };
        })()""")
        if "done in 0.1 s" in seen.get("status", ""):
            break
        time.sleep(0.3)
    print(f"  the reader reads {seen.get('status')!r} after {seen.get('lag')} event(s);"
          f" its prediction: {seen.get('model')}, PAE {seen.get('pae')}, pLDDT {seen.get('plddt')}")
    if "done in 0.1 s" not in seen.get("status", ""):
        bad.append("the worker's result never reached the reader's status line")
    # 🔴 A PREDICTION CROSSES AS JSON, WHICH HAS NO TYPED ARRAYS: `download-all` slices the PAE with
    # `subarray`, and a plain array of the right numbers passes everything else.
    if seen.get("pae") != "Float32Array" or seen.get("plddt") != "Float32Array":
        bad.append(f"the reader's prediction holds PAE {seen.get('pae')} and pLDDT {seen.get('plddt')}, not Float32Array")
    if seen.get("errors"):
        bad.append(f"the reader's page threw: {seen['errors'][:2]}")
    work = cdp.evaluate(reader_ws, """(() => ({
      gpu: window.__readerGpu,
      shards: performance.getEntriesByType('resource').map((r) => r.name)
        .filter((name) => /huggingface\\.co|\\.bin(\\?|$)/.test(name)),
    }))()""")
    if work["gpu"] or work["shards"]:
        bad.append(f"the reader did model work of its own: {work['gpu']} adapter request(s), {work['shards'][:3]}")

    # 7 · a reader that opens the page MID-FOLD attaches to it.
    call("/in", {"op": "fold", "payload": {"backend": "cuda", "job": "{}"}})
    time.sleep(0.3)
    open_reader(reader_ws)
    attached = ""
    deadline = time.time() + 20
    while time.time() < deadline and "already running" not in attached:
        attached = cdp.evaluate(reader_ws, "document.getElementById('status-message')?.textContent ?? ''")
        time.sleep(0.3)
    feed(("frame", tiny_pdb(1)), ("result", cuda_result(tiny_pdb(3), "attached fold · done")))
    landed = ""
    deadline = time.time() + 20
    while time.time() < deadline and "attached fold" not in landed:
        landed = cdp.evaluate(reader_ws, "document.getElementById('status-message')?.textContent ?? ''")
        time.sleep(0.3)
    print(f"  a reader arriving mid-fold: {attached!r}, then {landed!r}")
    if "already running" not in attached or "attached fold" not in landed:
        bad.append("a reader that opened mid-fold did not attach to it and receive its result")

    # 8 · a runtime with no CUDA says so and refuses.
    bare, _ = start_broker(PORT + 2, cuda=False)
    try:
        refused = call("/in", {"op": "fold", "payload": {"backend": "cuda", "job": "{}"}}, base=f"http://127.0.0.1:{PORT + 2}")
        open_reader(reader_ws, PORT + 2)
        time.sleep(4)
        said = cdp.evaluate(reader_ws, "document.querySelector('.colab-said')?.textContent ?? ''")
        live = cdp.evaluate(reader_ws, "!!document.querySelector('.colab-live')")
        print(f"  no --cuda: fold {refused}, badge {said!r}, Live box {live}")
        if refused[0] != 400 or "no CUDA backend" not in refused[1].get("error", ""):
            bad.append(f"a runtime with no CUDA answered a fold {refused}")
        if "no CUDA backend" not in said or live:
            bad.append(f"a runtime with no CUDA reads {said!r} with a Live box {live}")
    finally:
        stop_broker(bare)

    # 9 · and nothing answers without the token.
    for route, body in (("/down?since=0", None), ("/health", None), ("/in", {"op": "stop"})):
        code, _ = call(route, body, token=None)
        if code != 403:
            bad.append(f"{route} answered {code} with no token, not 403")
    print("  every route refuses a missing token")
finally:
    if reader is not None:
        reader.kill()
    stop_broker(backend)

print()
for line in bad:
    print("FAIL: " + line)
print("colab bridge: " + ("FAILED" if bad else "ok"))
raise SystemExit(1 if bad else 0)
