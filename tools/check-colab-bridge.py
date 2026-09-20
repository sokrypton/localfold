"""Two-way between the reader's page and the runtime that folds for it.

    python3 tools/check-colab-bridge.py

🔴 THE TRANSPORT IS WHAT THIS PROVES, AND IT NEEDS NO GPU AND NO WEIGHTS -
which is the point: the thing that broke in Colab was never the fold, it was
the feed. tools/colab_backend.py brokers between two copies of index.html -
`?role=runtime`, opened headlessly here, and `?backend=colab`, which a reader
opens - and every route is exercised from the reader's side over plain HTTP.

WHAT IT CHECKS, in the order a session does them:

  * the runtime page ANNOUNCES ITSELF by pushing `runtime-ready` to /up, which
    is the page having loaded, read its token, and reached the broker;
  * a command travels reader -> broker -> page and an answer travels back:
    `ping` is the op that needs no card, and a `pong` carrying the page's own
    clock is the round trip;
  * EVERY EVENT CARRIES BOTH CLOCKS - the page's `at` and the broker's `got` -
    so a slow feed can be told from a slow fold rather than argued about;
  * the watermark is idempotent: the same `since` twice is the same answer,
    and nothing is dropped between two polls;
  * one GPU, one fold - a second `fold` while one is running is refused 429,
    and the refusal lifts when the page reports a result;
  * and no route answers anything without the token.

🔴 WHAT IT CANNOT COVER is a fold: no weights on a developer's machine and no
card in CI. The fold command IS driven, with an empty entity list, so the
command path and the result event are real - what the page reports is an
error, which is the correct answer to that request and arrives by the same
route a structure would.
"""
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PORT = int(os.environ.get("BRIDGE_PORT", "8791"))
CDP_PORT = int(os.environ.get("BRIDGE_CDP_PORT", "9391"))
TOKEN = "check-colab-bridge-token"
BASE = f"http://127.0.0.1:{PORT}"

bad = []


def call(route, body=None, token=TOKEN, timeout=20):
    url = f"{BASE}{route}"
    url += ("&" if "?" in route else "?") + f"t={token}" if token is not None else ""
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        url, data=data, headers={"content-type": "application/json"},
        method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:
            return answer.status, json.loads(answer.read() or b"{}")
    except urllib.error.HTTPError as refused:
        return refused.code, json.loads(refused.read() or b"{}")


def wait_for_event(kind, since=0, seconds=30):
    """The reader's own loop: poll /down until `kind` turns up."""
    deadline = time.time() + seconds
    seen = []
    while time.time() < deadline:
        code, said = call(f"/down?since={since}")
        if code != 200:
            return None, since, seen
        since = said.get("n", since)
        for event in said.get("events") or []:
            seen.append(event)
            if event.get("kind") == kind:
                return event, since, seen
        time.sleep(0.2)
    return None, since, seen


print(f"starting the broker on {PORT} (a headless Chrome comes with it)…")
backend = subprocess.Popen(
    [sys.executable, "tools/colab_backend.py", "--port", str(PORT),
     "--cdp-port", str(CDP_PORT), "--token", TOKEN,
     "--profile", "/tmp/localfold-bridge-check"],
    cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
try:
    ready, adapter = False, None
    deadline = time.time() + 180
    while time.time() < deadline and not ready:
        line = backend.stdout.readline()
        if line == "" and backend.poll() is not None:
            break
        if line.startswith("BACKEND "):
            adapter = json.loads(line[len("BACKEND "):]).get("gpu")
            ready = True
        elif line.strip():
            print("  " + line.rstrip())
    if not ready:
        print("FAIL: the backend never printed its BACKEND line")
        raise SystemExit(1)
    print(f"  adapter: {adapter.get('vendor')} {adapter.get('architecture')}"
          f" webgpu={adapter.get('webgpu')}")

    # 1 · the page announced itself, which is the push direction at startup.
    code, said = call("/down?since=0")
    kinds = [event.get("kind") for event in said.get("events") or []]
    print(f"  /down: {code}, {len(kinds)} event(s): {kinds[:4]}")
    if "runtime-ready" not in kinds:
        bad.append("the runtime page never pushed `runtime-ready` to /up -"
                   " the page loaded but cannot reach the broker")
    since = said.get("n", 0)

    # 2 · a command out and an answer back, on the one op that needs no card.
    sent = time.time()
    code, said = call("/in", {"op": "ping"})
    if code != 200:
        bad.append(f"/in refused a ping: {code} {said}")
    pong, since, _ = wait_for_event("pong", since, 20)
    if pong is None:
        bad.append("no `pong` came back: a command does not reach the runtime"
                   " page, or its events do not reach the reader")
    else:
        print(f"  ping -> pong in {round((time.time() - sent) * 1000)} ms")
        # 3 · both clocks, which is what makes a late feed measurable.
        if pong.get("at") is None or pong.get("got") is None:
            bad.append(f"an event carries only one clock: {sorted(pong)}")
        else:
            print(f"  clocks: page {pong['at']} broker {pong['got']}"
                  f" (feed {pong['got'] - pong['at']} ms)")

    # 4 · the watermark is idempotent and loses nothing.
    code, first = call(f"/down?since={since}")
    code, again = call(f"/down?since={since}")
    if first.get("events") != again.get("events") or first.get("n") != again.get("n"):
        bad.append("two polls at the same watermark gave different answers")
    code, ahead = call(f"/down?since={first.get('n', since)}")
    if ahead.get("n", 0) < first.get("n", 0):
        bad.append("the watermark went backwards")
    print(f"  watermark: idempotent at {since}, stream at {ahead.get('n')}")

    # 5 · one GPU, one fold. The fold is driven with an EMPTY entity list, so
    #     the page refuses it in its own words rather than loading weights.
    code, said = call("/in", {"op": "fold", "payload": {"entities": [], "msa": "none"}})
    if code != 200:
        bad.append(f"/in refused the first fold: {code} {said}")
    code, busy = call("/in", {"op": "fold", "payload": {"entities": []}})
    if code != 429:
        bad.append(f"a second fold while one is running answered {code},"
                   " not 429 - one GPU cannot serve two")
    else:
        print(f"  second fold refused: {busy.get('error')}")
    result, since, seen = wait_for_event("result", since, 60)
    if result is None:
        bad.append("the fold command never produced a `result` event - the"
                   " runtime page took the command and said nothing back")
    else:
        payload = result.get("payload") or {}
        print(f"  fold with no sequence -> {json.dumps(payload)[:120]}")
        if not payload.get("error") and not payload.get("status"):
            bad.append("the result of an impossible fold says neither an error"
                       " nor a status, so a reader is told nothing")
    code, after = call("/health")
    if after.get("busy") is not False:
        bad.append("the runtime is still marked busy after a result -"
                   " every later fold would be refused 429")

    # 6 · and nothing answers without the token.
    for route, body in (("/down?since=0", None), ("/out?since=0", None),
                        ("/health", None), ("/up", {"events": []}),
                        ("/in", {"op": "ping"})):
        code, _ = call(route, body, token=None)
        if code != 403:
            bad.append(f"{route} answered {code} with no token, not 403")
    print("  every route refuses a missing token")
finally:
    backend.send_signal(signal.SIGINT)
    try:
        backend.wait(timeout=10)
    except subprocess.TimeoutExpired:
        backend.kill()

print()
for line in bad:
    print("FAIL: " + line)
print("colab bridge: " + ("FAILED" if bad else "ok"))
raise SystemExit(1 if bad else 0)
