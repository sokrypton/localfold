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
  * THE READER'S OWN PAGE does all of that for real: a second browser opens
    `index.html?backend=colab`, is handed a sequence and clicked, and what it
    ends up showing must be what the runtime said. The weights are blocked on
    the runtime page first, so the fold fails in seconds instead of pulling
    hundreds of megabytes - the transport is what is being measured, and a
    failure travels the same way a structure does;
  * THE FEED HOLDS WHILE THE PAGE IS BUSY - twenty events pushed from the
    runtime page across six seconds of 300 ms blocking tasks, which is what a
    fold does to a main thread. This is the regression guard for the fault the
    bridge was written for;
  * WHAT THE BROKER DROPS IS NOT DROPPED IN SILENCE: its mailbox is capped,
    and a reader that has fallen behind the cap is told rather than left with
    a gap in the trajectory it cannot see;
  * A READER THAT ARRIVES MID-FOLD ATTACHES TO IT rather than sitting idle,
    which is what a reload, a second window or the notebook's link opened
    twice all are;
  * A RUNTIME THAT GOES AWAY IS VISIBLE, AND A BUSY ONE IS NOT MISTAKEN FOR
    IT: the runtime page's own command poll is the heartbeat, and it STOPS
    while that page holds its main thread - which is what a fold does - so
    silence alone would abort a fold that was working. `browserAlive`, the
    DevTools endpoint the browser PROCESS serves, is what separates them, and
    both states are driven here: the page navigated away (quiet, still alive)
    and the browser killed (gone);
  * and no route answers anything without the token.

🔴 WHAT IT CANNOT COVER is a fold: no weights on a developer's machine and no
card in CI. The fold command IS driven, with an empty entity list, so the
command path and the result event are real - what the page reports is an
error, which is the correct answer to that request and arrives by the same
route a structure would.
"""
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
PORT = int(os.environ.get("BRIDGE_PORT", "8791"))
CDP_PORT = int(os.environ.get("BRIDGE_CDP_PORT", "9391"))
READER_CDP_PORT = int(os.environ.get("BRIDGE_READER_CDP_PORT", "9392"))
# 🔴 THE WEIGHTS ARE BLOCKED ON THE RUNTIME PAGE, which is what makes a REAL
# fold safe to drive from a developer's machine: the fold starts, the page
# reports that it cannot load the model, and everything that report is made of
# travels the way a fold's would. Without this the arm downloads hundreds of
# megabytes from huggingface to prove a transport.
WEIGHTS = "*huggingface.co*"
# 🔴 A STRUCTURE SMALL ENOUGH TO WRITE DOWN, because the frame path and the
# ingestion at the end of a remote fold are the reader's own code and nothing
# else here reaches them: no weights means no real frames, and a fold is the
# one thing this gate cannot drive. Four alpha carbons is a structure to
# py2Dmol - it draws a tube through them - and the third frame moves them, so
# "the frames arrived" cannot be satisfied by one frame drawn three times.
def tiny_pdb(shift):
    rows = []
    for i in range(4):
        rows.append(
            "ATOM  %5d  CA  ALA A%4d    %8.3f%8.3f%8.3f  1.00 50.00           C"
            % (i + 1, i + 1, 3.8 * i + shift, 0.0, 0.0))
    return "\n".join(rows) + "\nEND\n"
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

    # 6 · THE READER'S OWN PAGE, which no amount of curl can stand in for.
    #     Everything above drives the wire; this drives web/app.js's
    #     `foldOnBackend` in a real browser - the half that was REWRITTEN and
    #     that a wire test would pass with in pieces.
    print("  opening the reader's page…")
    runtime_ws = None
    reader = None
    try:
        # The runtime's own browser, joined as a second debugger client, only
        # to take the weights away. Everything else about that page is left
        # exactly as the backend set it up.
        for target in json.load(urllib.request.urlopen(
                f"http://127.0.0.1:{CDP_PORT}/json/list")):
            if target.get("type") == "page":
                runtime_ws = cdp.WS(target["webSocketDebuggerUrl"])
                break
        runtime_ws.call("Network.enable")
        # 🔴 AND IT STAYS BLOCKED FOR THE REST OF THE RUN. Unblocking at the
        # end of this arm let a REPLAYED fold command - the reload bug the
        # heartbeat arm exposed - fetch 681 MB of weights and actually fold,
        # which then filled the viewer the arms below were reading and made a
        # mutation pass. Nothing in this gate may ever fold.
        runtime_ws.call("Network.setBlockedURLs", urls=[WEIGHTS])

        reader, reader_ws = cdp.launch(READER_CDP_PORT, "/tmp/localfold-bridge-reader")
        reader_ws.call("Page.enable")
        reader_ws.call("Runtime.enable")
        # 🔴 THE COLLECTOR GOES IN BEFORE THE PAGE DOES. A throw inside the
        # rewritten transport would otherwise be a fold that quietly does
        # nothing, which is the failure this arm exists to catch.
        reader_ws.call("Page.addScriptToEvaluateOnNewDocument", source="""
          window.__pageErrors = [];
          addEventListener('error', (e) => window.__pageErrors.push(String(e.message)));
          addEventListener('unhandledrejection',
            (e) => window.__pageErrors.push('unhandled: ' + String(e.reason)));
        """)
        reader_ws.call("Page.navigate", url=(
            f"http://127.0.0.1:{PORT}/index.html?backend=colab&t={TOKEN}"))
        cdp.wait_for(reader_ws, "!!window.__entityList", 120, "the reader's page")
        # 🔴 THE TERMS DIALOG EATS THE CLICK ON A FRESH PROFILE, and it did:
        # the first run of this arm reported a page that had been handed a
        # sequence, had an enabled Fold button, was clicked, and then sat at
        # "Ready. Paste a sequence and press Fold." with no command sent. The
        # backend accepts them for the RUNTIME page; the reader's browser is a
        # different profile and had accepted nothing.
        cdp.evaluate(reader_ws, """(() => {
          for (const key of ['alphafold3', 'openbind0', 'opendde', 'boltz2',
                             'protenix2', 'intellifold2', 'rosettafold3']) {
            try { localStorage.setItem('localfold.modelTerms.' + key, 'accepted'); }
            catch (cause) { /* nothing to do */ }
          }
          return true;
        })()""")
        # A sequence and a press, which is all a person does.
        cdp.evaluate(reader_ws, """(() => {
          window.__entityList.set([{ type: 'protein',
            value: 'GWSTELEKHREELKEFLKKEGITLGFTNAEKQEQAQKLGLGKKVSPELLIKAFAILKK',
            copies: 1, modifications: [] }]);
          document.getElementById('msa-mode').value = 'none';
          document.getElementById('msa-mode').dispatchEvent(
            new Event('change', { bubbles: true }));
          return true;
        })()""")
        cdp.wait_for(reader_ws, "!document.getElementById('predict').disabled", 60,
                     "the reader's fold button")
        cdp.evaluate(reader_ws, "(document.getElementById('predict').click(), true)")

        # The command reaches the broker... and it is the one this click made,
        # not the empty fold the arm above sent over curl.
        asked, deadline = None, time.time() + 30
        while time.time() < deadline and asked is None:
            code, said = call("/out?since=0")
            for command in said.get("commands") or []:
                if command.get("op") != "fold":
                    continue
                if (command.get("payload") or {}).get("entities"):
                    asked = command
            time.sleep(0.25)
        if asked is None:
            bad.append("pressing Fold on the reader's page put no `fold`"
                       " command in the broker - foldOnBackend never asked")
        else:
            entities = (asked.get("payload") or {}).get("entities") or []
            print(f"  the reader asked: {asked['op']},"
                  f" {len(entities)} entity, model {(asked.get('payload') or {}).get('model')}")

        # ...and the runtime's answer reaches the reader's own screen. The
        # fold cannot succeed with the weights blocked; what is asserted is
        # that its commentary ARRIVED and was applied, which is the bug.
        seen, deadline = {}, time.time() + 120
        while time.time() < deadline:
            seen = cdp.evaluate(reader_ws, """(() => ({
              lag: (window.__remoteLag || []).length,
              worst: Math.max(0, ...(window.__remoteLag || [0])),
              status: document.getElementById('status-message')?.textContent ?? '',
              errors: window.__pageErrors || [],
            }))()""")
            if seen.get("lag", 0) > 0 and "folding on the runtime" not in seen.get("status", ""):
                break
            time.sleep(0.5)
        print(f"  the reader applied {seen.get('lag')} event(s), worst feed"
              f" {seen.get('worst')} ms, and reads: {seen.get('status')!r}")
        if seen.get("lag", 0) == 0:
            bad.append("the reader's page applied no events at all: the"
                       " runtime spoke and nothing reached the screen")
        if "folding on the runtime" in seen.get("status", ""):
            bad.append("the reader's status line never moved off its own"
                       " opening line - the runtime's words did not arrive")
        if seen.get("errors"):
            bad.append(f"the reader's page threw: {seen['errors'][:2]}")
    finally:
        if reader is not None:
            reader.kill()

    # 7 · AND THE FEED HOLDS UP WHILE THE PAGE IS BUSY, which is the whole
    #     complaint. The runtime page is made to block its main thread in
    #     300 ms chunks - what a fold does to it - with an event pushed before
    #     each one. Every event must still arrive promptly, because it leaves
    #     in the task that made it; a collected-and-drained feed cannot, which
    #     is what "embedder · 1%" looked like from the reader's chair.
    runtime_ws = None
    try:
        for target in json.load(urllib.request.urlopen(
                f"http://127.0.0.1:{CDP_PORT}/json/list")):
            if target.get("type") == "page":
                runtime_ws = cdp.WS(target["webSocketDebuggerUrl"])
                break
        # 🔴 THE PAGE'S OWN MODULE INSTANCE, not a second copy: an ES module is
        # cached by URL, so importing it here is the object web/app.js imports.
        # Anything else would measure a transport nothing uses.
        code, head = call("/down?head=1")
        before = head.get("n", 0)
        cdp.evaluate(runtime_ws, """(async () => {
          const bridge = await import('/web/colab-bridge.js');
          const spin = (ms) => { const end = performance.now() + ms;
                                 while (performance.now() < end); };
          (async () => {
            for (let i = 0; i < 20; i += 1) {
              bridge.tapOut('status', 'load ' + i);
              spin(300);
            }
          })();
          return true;
        })()""", await_promise=True)
        lags, arrived, deadline = [], [], time.time() + 40
        seen = before
        while time.time() < deadline and len(lags) < 20:
            code, said = call(f"/down?since={seen}")
            seen = said.get("n", seen)
            for event in said.get("events") or []:
                if str(event.get("payload", "")).startswith("load "):
                    lags.append(event["got"] - event["at"])
                    arrived.append(event.get("seq"))
            time.sleep(0.2)
        if len(lags) < 20:
            bad.append(f"only {len(lags)} of 20 events arrived from a busy"
                       " page - the feed stops when the fold gets going")
        else:
            worst = max(lags)
            lags.sort()
            print(f"  busy page (20 events across 6 s of 300 ms tasks):"
                  f" feed p50 {lags[10]} ms, worst {worst} ms")
            # A pushed event leaves before the block that follows it, so the
            # bound is about the send and not about the page's tasks. A second
            # is twenty times what this measures and still catches a feed that
            # has gone back to being collected.
            if worst > 1000:
                bad.append(f"the worst event took {worst} ms to reach the"
                           " broker from a busy page - the feed is being"
                           " collected rather than pushed")
            # 🔴 AND EVERY ONE OF THEM IS THERE, EXACTLY ONCE. Several sends
            # are in flight at once, so the broker's order is the network's:
            # what must hold is that nothing was dropped or doubled, and that
            # the page's own `seq` is on each one - which is what the reader
            # sorts by. Arrivals out of order are REPORTED rather than
            # asserted: on loopback there are usually none, and the sort in
            # web/app.js is for the Colab proxy, which is not this.
            if sorted(arrived) != list(range(min(arrived), min(arrived) + 20)):
                bad.append(f"the 20 events came back as seqs {sorted(arrived)}"
                           " - one was dropped, doubled, or carries no seq")
            out_of_order = sum(1 for a, b in zip(arrived, arrived[1:]) if b < a)
            print(f"  seq: 20 distinct, {out_of_order} arrived out of order")
    finally:
        pass

    # 8 · A RUNTIME THAT GOES AWAY IS VISIBLE - AND A BUSY ONE IS NOT MISTAKEN
    #     FOR IT. Colab's ordinary ending is the notebook being closed or the
    #     runtime recycled, and a reader mid-fold would otherwise poll a
    #     broker that can never answer. The page's own command poll is the
    #     heartbeat; the DevTools endpoint, served by the browser PROCESS, is
    #     what separates "this page is busy" from "this machine is gone".
    runtime_ws = None
    try:
        for target in json.load(urllib.request.urlopen(
                f"http://127.0.0.1:{CDP_PORT}/json/list")):
            if target.get("type") == "page":
                runtime_ws = cdp.WS(target["webSocketDebuggerUrl"])
                break
        # 🔴 AFTER A SETTLE, because the arm above deliberately blocks that
        # page's main thread for six seconds and its poll stops with it. The
        # first version read the heartbeat straight afterwards and reported
        # 6.2 s of silence as a heartbeat that was not beating - the arm
        # measuring the arm before it.
        time.sleep(1.5)
        code, said = call("/health")
        fresh = said.get("runtimeSeen")
        if fresh is None or fresh > 3000:
            bad.append(f"a live runtime page reads {fresh} ms since its last"
                       " command poll - the heartbeat is not beating")
        if said.get("browserAlive") not in (None, True):
            bad.append(f"the browser reads {said.get('browserAlive')!r} while"
                       " its page is answering")

        # 🔴 THE BASELINE IS TAKEN HERE, not carried down from an arm above:
        # the replay check below counted folds from earlier arms and reported
        # the fix as broken. An arm's baseline is the state immediately
        # before it - this file's own rule, relearned twice now.
        code, head = call("/down?head=1")
        since = head.get("n", since)

        # (a) THE PAGE STOPS ANSWERING AND THE BROWSER DOES NOT. A page with
        #     no `role` runs no bridge, which is what a page busy in a fold
        #     looks like from here - and a reader must NOT give up on it.
        runtime_ws.call("Page.navigate", url="about:blank")
        quiet, alive, deadline = 0, None, time.time() + 20
        while time.time() < deadline:
            code, said = call("/health")
            quiet, alive = said.get("runtimeSeen") or 0, said.get("browserAlive")
            if quiet > 6000:
                break
            time.sleep(0.5)
        print(f"  page quiet: {quiet} ms, browser alive: {alive}")
        if quiet <= 6000:
            bad.append("the heartbeat did not age while the page was away, so"
                       " a reader cannot tell a dead runtime from a slow fold")
        if alive is not True:
            bad.append(f"the browser reads {alive!r} while it is running - a"
                       " reader would abandon a fold whose page is merely"
                       " busy, which is what a fold looks like")

        # ...and back, which is also the bridge restarting by itself.
        runtime_ws.call("Page.navigate", url=(
            f"http://127.0.0.1:{PORT}/index.html?role=runtime&t={TOKEN}"))
        back, deadline = None, time.time() + 60
        while time.time() < deadline:
            code, said = call("/health")
            back = said.get("runtimeSeen")
            if back is not None and back < 2000:
                break
            time.sleep(0.5)
        print(f"  heartbeat: {fresh} ms fresh, {back} ms once the page is back")
        if back is None or back > 2000:
            bad.append(f"the heartbeat did not come back ({back} ms) after the"
                       " runtime page reloaded - the bridge does not restart")
        # 🔴 AND A RELOADED PAGE DOES NOT REPLAY THE SESSION. It polled from
        # zero, so coming back it obeyed every command the notebook had ever
        # sent: measured as a fold from an arm ten minutes earlier being run
        # again, weights and all.
        code, said = call(f"/down?since={since}")
        since = said.get("n", since)
        replayed = [e for e in (said.get("events") or [])
                    if e.get("kind") in ("fold-begin", "result")]
        if replayed:
            bad.append(f"the reloaded runtime page replayed {len(replayed)}"
                       " command event(s) - it starts from zero rather than"
                       " from where the queue stands")

    finally:
        pass


    # 9 · A READER THAT ARRIVES MID-FOLD ATTACHES TO IT. A Colab fold is
    #     minutes long and the page in front of it is an ordinary tab, so
    #     "only the page that pressed Fold is watching" is a reader who
    #     reloads and sees nothing at all.
    #
    #     🔴 THE FOLD IS HELD BY TAKING /out AWAY FROM THE RUNTIME PAGE, not
    #     by folding something slow: the broker raises `folding` when it
    #     ACCEPTS the command, and a page that cannot collect its commands
    #     never finishes it. That makes the state deterministic instead of a
    #     race against a real fold's first seconds.
    held = None
    try:
        for target in json.load(urllib.request.urlopen(
                f"http://127.0.0.1:{CDP_PORT}/json/list")):
            if target.get("type") == "page":
                held = cdp.WS(target["webSocketDebuggerUrl"])
                break
        held.call("Network.enable")
        held.call("Network.setBlockedURLs", urls=["*/out*"])
        time.sleep(1.0)
        code, said = call("/in", {"op": "fold", "payload": {
            "entities": [{"type": "protein", "value": "GWSTELEKHRSVQ", "copies": 1}]}})
        code, head = call("/down?head=1")
        if not head.get("folding"):
            bad.append("the broker does not say it is folding after accepting"
                       " a fold command, so no reader can attach to one")

        late, late_ws = cdp.launch(READER_CDP_PORT + 1, "/tmp/localfold-bridge-late")
        try:
            late_ws.call("Page.enable")
            late_ws.call("Page.addScriptToEvaluateOnNewDocument", source="""
              window.__pageErrors = [];
              addEventListener('error', (e) => window.__pageErrors.push(String(e.message)));
              addEventListener('unhandledrejection',
                (e) => window.__pageErrors.push('unhandled: ' + String(e.reason)));
            """)
            late_ws.call("Page.navigate", url=(
                f"http://127.0.0.1:{PORT}/index.html?backend=colab&t={TOKEN}"))
            cdp.wait_for(late_ws, "!!window.__entityList", 120, "the late reader")
            attached, deadline = "", time.time() + 30
            while time.time() < deadline:
                attached = cdp.evaluate(late_ws,
                    "document.getElementById('status-message')?.textContent ?? ''")
                if "already running" in attached:
                    break
                time.sleep(0.5)
            print(f"  a page opened mid-fold reads: {attached!r}")
            if "already running" not in attached:
                bad.append("a page opened while the runtime was folding sat"
                           " idle - it did not attach to the fold")

            # ...AND WHAT IT DOES WITH WHAT ARRIVES. Frames pushed from the
            # runtime page must be DRAWN by the reader, and the `result` must
            # be INGESTED - `loadIntoViewer`, the sequence strip, the download
            # buttons - which is the reader's own code at the end of every
            # remote fold and is otherwise reached only by folding.
            for index in range(3):
                cdp.evaluate(held, """(async () => {
                  const bridge = await import('/web/colab-bridge.js');
                  bridge.tapOut('status', 'sampler %d of 3');
                  bridge.tapOut('progress', %f);
                  bridge.tapOut('frame', %s);
                  return true;
                })()""" % (index + 1, (index + 1) / 3.0,
                           json.dumps(tiny_pdb(index * 2.0))))
                time.sleep(0.4)
            drew, deadline = {}, time.time() + 30
            while time.time() < deadline:
                drew = cdp.evaluate(late_ws, """(() => {
                  const reg = window.py2dmol_viewers || {};
                  const r = reg[Object.keys(reg)[0]] &&
                            reg[Object.keys(reg)[0]].renderer;
                  const o = r && r.objectsData ? r.objectsData[r.currentObjectName] : null;
                  return { frames: o && o.frames ? o.frames.length : 0,
                           positions: o && o.frames && o.frames[0]
                             ? (o.frames[0].coords || []).length : 0,
                           status: document.getElementById('status-message')?.textContent ?? '',
                           errors: window.__pageErrors || [] };
                })()""")
                if drew.get("frames", 0) >= 3 and drew.get("positions") == 4:
                    break
                time.sleep(0.5)
            print(f"  it drew {drew.get('frames')} sampler frame(s) of"
                  f" {drew.get('positions')} positions and reads:"
                  f" {drew.get('status')!r}")
            # 🔴 THE POSITION COUNT IS WHAT MAKES THIS ABOUT THE PUSHED
            # FRAMES. "Three or more frames" was satisfied by a 58-residue
            # fold that a replayed command had started, so the mutation it was
            # written for walked straight through it.
            if drew.get("frames", 0) < 3 or drew.get("positions") != 4:
                bad.append(f"the attached reader drew {drew.get('frames')} of"
                           f" 3 pushed frames at {drew.get('positions')}"
                           " positions, not 4 - the sampler's walk does not"
                           " reach the screen")

            # 🔴 AND WHAT THE BROKER DROPS MUST NOT BE DROPPED IN SILENCE.
            # A session is not one fold, so the mailbox is capped at 4,000 and
            # loses its oldest - and a reader that has fallen behind that
            # point has lost events it never applied. `from` says where the
            # stream now starts; the flood below is what makes it move.
            code, before_gap = call("/down?head=1")
            code, flooded = call("/up", {"events": [
                {"kind": "status", "payload": "flood %d" % i, "at": 0, "seq": 10000 + i}
                for i in range(4100)]})
            gap, deadline = 0, time.time() + 30
            while time.time() < deadline:
                gap = cdp.evaluate(late_ws, "window.__remoteGap || 0")
                if gap > 0:
                    break
                time.sleep(0.5)
            code, after_gap = call("/down?head=1")
            print(f"  cap: {before_gap.get('n')} -> {after_gap.get('n')} events,"
                  f" the attached reader noticed {gap} dropped")
            if gap <= 0:
                bad.append("the broker dropped its oldest events and the"
                           " attached reader carried on as though nothing had"
                           " happened - a gap in the trajectory, in silence")

            cdp.evaluate(held, """(async () => {
              const bridge = await import('/web/colab-bridge.js');
              bridge.tapOut('result', { pdb: %s, atoms: 4, status: 'folded here',
                                        scores: {}, predJson: null });
              return true;
            })()""" % json.dumps(tiny_pdb(4.0)))
            ingested, deadline = {}, time.time() + 40
            while time.time() < deadline:
                ingested = cdp.evaluate(late_ws, """(() => {
                  const reg = window.py2dmol_viewers || {};
                  const r = reg[Object.keys(reg)[0]] &&
                            reg[Object.keys(reg)[0]].renderer;
                  const o = r && r.objectsData ? r.objectsData[r.currentObjectName] : null;
                  return { status: document.getElementById('status-message')?.textContent ?? '',
                           positions: o && o.frames && o.frames[0]
                             ? (o.frames[0].coords || []).length : 0,
                           errors: window.__pageErrors || [] };
                })()""")
                if "folded here" in ingested.get("status", ""):
                    break
                time.sleep(0.5)
            print(f"  and ingested it: {ingested.get('status')!r},"
                  f" {ingested.get('positions')} positions drawn")
            if "folded here" not in ingested.get("status", ""):
                bad.append("the runtime's finished fold never landed on the"
                           " reader's page - `result` is not ingested")
            if ingested.get("positions", 0) < 4:
                bad.append(f"the ingested structure has"
                           f" {ingested.get('positions')} positions, not 4 -"
                           " loadIntoViewer got something it could not read")
            if ingested.get("errors"):
                bad.append(f"the attached reader threw: {ingested['errors'][:2]}")
        finally:
            late.kill()
    finally:
        if held is not None:
            try:
                held.call("Network.setBlockedURLs", urls=[])
            except Exception:                                 # noqa: BLE001
                pass
    # The injected `result` above is what ended the held fold, so the broker
    # must be idle again - which is also the rule that a session can fold
    # twice.
    code, after = call("/health")
    if after.get("busy") is not False:
        bad.append("the broker is still busy after the fold ended, so every"
                   " later fold would be refused")

    # 10 · THE BROWSER GOES, WHICH IS THE RUNTIME GOING. This is the state
    #     a reader gives up on, and the only one.
    # 🔴 THE BROWSER ONLY. `pkill -f localfold-bridge-check` also matches the
    # BROKER, whose own command line carries that profile in `--profile` - so
    # the first version killed the post office as well and the next request
    # died on "connection refused", which is a different fault wearing this
    # one's clothes.
    #
    # 🔴 AND THE BINARY'S NAME IS NOT THE WAY TO SEPARATE THEM. This read
    # "Google Chrome.*" - the macOS path - so on Linux, where the binary is
    # `google-chrome` or `chromium`, it matched NOTHING: the browser lived,
    # `browserAlive` stayed None, and the arm failed saying a reader cannot
    # tell a gone runtime from a busy one. It could; nothing had gone. What
    # separates the two on BOTH platforms is the flag - the browser is given
    # `--user-data-dir=<profile>` by cdp.launch and the broker takes
    # `--profile <path>` - so the `=` does the work the binary name was doing.
    subprocess.run(["pkill", "-f", "user-data-dir=/tmp/localfold-bridge-check"],
                   check=False)
    dead, deadline = None, time.time() + 30
    while time.time() < deadline:
        try:
            code, said = call("/health")
        except Exception as cause:                            # noqa: BLE001
            bad.append(f"the broker went with the browser ({cause}) - the"
                       " post office is not the thing being killed here")
            break
        dead = said.get("browserAlive")
        if dead is False:
            break
        time.sleep(0.5)
    print(f"  with the browser gone: browserAlive {dead}")
    if dead is not False:
        bad.append(f"the browser reads {dead!r} after it was killed - a"
                   " reader cannot tell a gone runtime from a busy one")

    # 11 · and nothing answers without the token.
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
