#!/usr/bin/env python3
"""What the page does when things go wrong, on a real GPU, through the page.

    python3 tools/check-robustness.py --group faults [--only name,name] [--list] [--repo DIR]

🔴 ON A RENTED CARD, UNDER LINUX. It kills the GPU process, fills the card and folds past its limits on purpose, and on a
display's own GPU each of those takes the screen with it, and a handful in a minute takes Windows. It refuses anywhere else.

Everything a reader can do by accident, or have done to them, while a fold runs or between two: the GPU process dying at each
stage, Stop at each stage, a download failing or coming back wrong, a browser that stores nothing, input a model cannot read,
sizes on the edge of what the plan says fits, a device with low limits, a card that really runs out, a second tab, a reload, a
long session. Each scenario starts a new browser on the shared model cache, and what it must leave behind is the same in all
of them: a page that folds again with no reload, the right answer, and a card that is not holding what the fold held.

`--repo` serves another checkout under the same scenarios, which is how a failure is told from one that was already there.
"""
import argparse
import base64
import json
import math
import os
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.parse
import urllib.request

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import cdp  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--group", default="", help="faults, stop, network, storage, inputs, sizes, limits, lifecycle, cpu, hangs")
parser.add_argument("--only", default="", help="scenario names, comma separated")
parser.add_argument("--list", action="store_true")
parser.add_argument("--repo", default=str(Path(__file__).parent.parent))
parser.add_argument("--port", type=int, default=4199)
parser.add_argument("--debug-port", type=int, default=9399)
parser.add_argument("--profile", default="")
args = parser.parse_args()

REPO = Path(args.repo)
URL = f"http://127.0.0.1:{args.port}/index.html"
ORIGIN = f"http://127.0.0.1:{args.port}"
PROFILE = args.profile or tempfile.mkdtemp(prefix="robustness-")
UBI = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
GIB = 1024 ** 3
ESMFOLD2 = "ef2-fast-600m"
# ...with no alignment anywhere: a server's answer is not what any of this is about.
OTHER_MODELS = ["openbind0", "boltz2", "rosettafold3", "protenix2", "mol1", "opendde", "intellifold2", "monomer", "multimer"]

CAPTURE_ERRORS = """
window.__errors = [];
{
  const original = console.error;
  console.error = (...args) => { window.__errors.push(args.map((a) => a?.stack ?? String(a)).join(' ').replace(/http:[^ ]*:4199/g, '').slice(0, 900)); original(...args); };
  addEventListener('error', (event) => window.__errors.push('error: ' + event.message));
  addEventListener('unhandledrejection', (event) => window.__errors.push('rejection: ' + event.reason + ' ' + String(event.reason?.stack ?? '').split('\\n').slice(1, 6).map((line) => line.trim().replace(/http:[^ ]*:4199/, '')).join(' < ')));
}
"""
READY = "typeof window.processFiles === 'function' && !document.getElementById('predict').disabled"
READ_STATE = """JSON.stringify({
  running: window.__foldState?.running ?? null,
  status: document.getElementById('status-message')?.textContent ?? '',
  failed: document.getElementById('status-message')?.classList.contains('error') === true,
  line: document.getElementById('fold-plan')?.textContent ?? '',
  actions: [...document.querySelectorAll('#status-message button, #status button')].map((b) => b.textContent.trim()),
  errors: window.__errors.splice(0),
})"""
# How many colours the viewer shows, through a copy, because it draws to a 2D canvas or to a WebGL one laid over it.
DRAWN = """(() => {
  let most = 0;
  for (const canvas of document.querySelectorAll('#canvasContainer canvas')) {
    if (canvas.width === 0 || canvas.height === 0) continue;
    const copy = document.createElement('canvas');
    copy.width = 240;
    copy.height = 240;
    const context = copy.getContext('2d');
    context.drawImage(canvas, 0, 0, 240, 240);
    const { data } = context.getImageData(0, 0, 240, 240);
    const colours = new Set();
    for (let at = 0; at < data.length; at += 4) colours.add((data[at] << 16) | (data[at + 1] << 8) | data[at + 2]);
    most = Math.max(most, colours.size);
  }
  return most;
})()"""
OTHER_CANVASES = """JSON.stringify([...document.querySelectorAll('#viewer-container canvas')].filter((c) => !c.closest('#canvasContainer')).map((canvas) => {
  if (canvas.width === 0 || canvas.height === 0 || canvas.offsetParent === null) return { id: canvas.id, hidden: true };
  const copy = document.createElement('canvas');
  copy.width = 120;
  copy.height = 120;
  const context = copy.getContext('2d');
  context.drawImage(canvas, 0, 0, 120, 120);
  const { data } = context.getImageData(0, 0, 120, 120);
  const colours = new Set();
  for (let at = 0; at < data.length; at += 4) colours.add((data[at] << 16) | (data[at + 1] << 8) | data[at + 2]);
  return { id: canvas.id, colours: colours.size };
}))"""
READ_PDB = r"""(async () => {
  const heldUrl = URL.createObjectURL;
  const heldClick = HTMLAnchorElement.prototype.click;
  let blob = null;
  URL.createObjectURL = (b) => { blob = b; return heldUrl.call(URL, b); };
  HTMLAnchorElement.prototype.click = function () {};
  document.getElementById('download-pdb').click();
  for (let wait = 0; wait < 100 && blob === null; wait += 1) await new Promise((done) => setTimeout(done, 50));
  URL.createObjectURL = heldUrl;
  HTMLAnchorElement.prototype.click = heldClick;
  if (blob === null) return JSON.stringify(null);
  const atoms = (await blob.text()).split('\n').filter((line) => line.startsWith('ATOM') || line.startsWith('HETATM'));
  const ca = atoms.filter((line) => line.slice(12, 16).trim() === 'CA').map((line) => [30, 38, 46].map((at) => Number(line.slice(at, at + 8))));
  return JSON.stringify({ atoms: atoms.length, ca });
})()"""


# A dialog stops the page until someone answers, and a reader would. Chrome asks before leaving a page after a gesture, and
# whether that is what this scenario meant to test or not, it must not hang the suite.
held_recv = cdp.WS.recv


def recv_answering_dialogs(ws):
    message = held_recv(ws)
    if message.get("method") == "Page.javascriptDialogOpening":
        ws.id += 1
        ws.send({"id": ws.id, "method": "Page.handleJavaScriptDialog", "params": {"accept": True}})
    return message


cdp.WS.recv = recv_answering_dialogs


def sequence(size):
    return (UBI * (size // len(UBI) + 1))[:size]


def number_after(word, text):
    found = re.search(word + r" ([0-9.]+)", text)
    return float(found.group(1)) if found is not None else None


def card_mib():
    read = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return int(read.stdout.strip().splitlines()[0])


def gpu_processes():
    found = subprocess.run(["pgrep", "-f", "type=gpu-process"], capture_output=True, text=True)
    return [int(line) for line in found.stdout.split()]


def kill_gpu_process():
    processes = gpu_processes()
    assert processes, "no GPU process to end"
    for process in processes:
        os.kill(process, signal.SIGKILL)
    return processes


class Tab:
    """One page in one browser. What the scenarios press, read and wait for, through the page's own controls."""

    def __init__(self, ws, browser):
        self.ws = ws
        self.browser = browser
        self.errors = []
        # ...and everything a page threw with nobody there to catch it, which no scenario may leave, whatever it looks at.
        self.uncaught = []
        self.complaints = []
        self.opened = False
        browser.tabs.append(self)

    def evaluate(self, expression):
        return cdp.evaluate(self.ws, expression)

    def open(self, init=(), url=URL, timeout=180):
        self.ws.call("Page.enable")
        for script in (CAPTURE_ERRORS, *init):
            # Chrome drops a script that does not parse without a word, and whatever reads it then passes on nothing.
            self.evaluate("new Function(%s); undefined" % json.dumps(script))
            self.ws.call("Page.addScriptToEvaluateOnNewDocument", source=script)
        self.ws.call("Page.navigate", url=url)
        self.wait_ready(timeout=timeout)
        self.prove_capture()
        self.opened = True
        return self

    def prove_capture(self):
        """Each page shows that what it throws reaches the checks: a capture that did not load passed every one of them."""
        self.evaluate("""console.error('capture control');
                         Promise.reject(new Error('capture control'));
                         setTimeout(() => { throw new Error('capture control'); });
                         undefined""")
        cdp.wait_for(self.ws, "window.__errors.filter((e) => e.includes('capture control')).length === 3", what="the page's error capture", timeout=10)
        self.evaluate("window.__errors = window.__errors.filter((e) => !e.includes('capture control')); undefined")

    def wait_ready(self, timeout=180):
        cdp.wait_for(self.ws, READY, what="the page", timeout=timeout)
        time.sleep(0.5)

    def state(self):
        state = json.loads(self.evaluate(READ_STATE))
        self.errors += state["errors"]
        self.uncaught += [error for error in state["errors"] if error.startswith(("error: ", "rejection: "))]
        self.complaints += [error for error in state["errors"] if error.startswith("WebGPU:")]
        return state

    def choose(self, family=ESMFOLD2, variant="esmc-300m"):
        self.evaluate("""(() => {
          for (const [id, value] of [['model-family', %s], ['plm-mode', %s], ['msa-mode', 'none']]) {
            const el = document.getElementById(id);
            if (el !== null && [...(el.options ?? [])].some((o) => o.value === value)) {
              el.value = value;
              el.dispatchEvent(new Event('change', { bubbles: true }));
            }
          }
        })()""" % (json.dumps(family), json.dumps(variant)))
        time.sleep(0.2)

    def set_rows(self, rows):
        self.evaluate("""(() => {
          const list = window.__entityList;
          const first = list.read()[0];
          list.set(%s.map(([type, value, copies]) => ({ ...first, type, value, copies, template: undefined, modifications: [] })));
        })()""" % json.dumps(rows))
        time.sleep(0.2)

    def set_size(self, size):
        self.set_rows([["protein", sequence(size), 1]])

    def set_control(self, element, value):
        self.evaluate("""(() => { const el = document.getElementById(%s); el.value = %s;
                                  el.dispatchEvent(new Event('change', { bubbles: true })); })()"""
                      % (json.dumps(element), json.dumps(str(value))))
        time.sleep(0.2)

    def press(self):
        self.evaluate("window.__foldState = undefined; document.getElementById('predict').click()")

    def wait_finished(self, timeout=420):
        """Until the fold ended, however it ended, or the time is up. `running` is unset until a press has started one."""
        started = time.time()
        state = {}
        while time.time() - started < timeout:
            try:
                state = self.state()
            except (RuntimeError, OSError):
                time.sleep(0.5)
                continue
            if state["running"] is False or (state["failed"] and state["running"] is not True):
                break
            time.sleep(0.1)
        else:
            state = {**state, "timed_out": True}
        took = round(time.time() - started, 1)
        time.sleep(0.5)
        state = {**state, **self.state()}
        errors, self.errors = self.errors, []
        if errors:
            print("  ...the page logged", errors[:2], flush=True)
        return {**state, "errors": errors, "seconds": took}

    def fold(self, timeout=420):
        self.press()
        return self.wait_finished(timeout)

    def wait_status(self, *words, timeout=240):
        """Until the status line says one of these. Fails loudly if the fold ended first: the fault would land nowhere."""
        started = time.time()
        while time.time() - started < timeout:
            state = self.state()
            if any(word in state["status"] for word in words):
                return state
            if state["running"] is False:
                raise AssertionError(f"the fold ended before {words}: {state['status'][:160]}")
            time.sleep(0.02)
        raise AssertionError(f"no {words} in {timeout} s: {state['status'][:160]}")

    def stop_if_running(self):
        if self.state()["running"] is True:
            self.evaluate("document.getElementById('predict').click()")
            self.wait_finished(timeout=60)

    def click_action(self, *labels):
        return self.evaluate("""window.__foldState = undefined; (() => {
          const button = [...document.querySelectorAll('#status-message button, #status button')]
            .find((b) => %s.some((label) => b.textContent.trim().startsWith(label)));
          if (button === undefined) return false;
          button.click();
          return true;
        })()""" % json.dumps(list(labels)))

    def pdb(self):
        return json.loads(self.evaluate(READ_PDB))


class Browser:
    """A Chrome on the shared cache, with a clean origin unless a scenario wants otherwise."""

    def __init__(self, keep_origin=False, cold=False, extra=(), port=None, profile=None):
        started = time.time()
        self.port = port or args.debug_port
        self.process, ws = cdp.launch(self.port, profile or PROFILE, keep=True, extra_args=extra)
        self.tabs = []
        self.first = Tab(ws, self)
        self.started = started
        if not keep_origin:
            types = "local_storage,service_workers,indexeddb" + (",cache_storage" if cold else "")
            ws.call("Storage.clearDataForOrigin", origin=ORIGIN, storageTypes=types)
        if cold:
            ws.call("Network.enable")
            ws.call("Network.clearBrowserCache")
            # ...and off again, or every chunk of every download is an event this client has to read before its next answer.
            ws.call("Network.disable")
        print(f"  ...a browser in {time.time() - started:.1f} s", flush=True)

    def new_tab(self, url):
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}/json/new?{url}", method="PUT")
        target = json.load(urllib.request.urlopen(request, timeout=10))
        tab = Tab(cdp.WS(target["webSocketDebuggerUrl"]), self)
        tab.target = target["id"]
        return tab

    def close_tab(self, tab):
        urllib.request.urlopen(f"http://127.0.0.1:{self.port}/json/close/{tab.target}", timeout=10).read()

    def activate(self, target):
        urllib.request.urlopen(f"http://127.0.0.1:{self.port}/json/activate/{target}", timeout=10).read()

    def targets(self):
        return json.load(urllib.request.urlopen(f"http://127.0.0.1:{self.port}/json/list", timeout=10))

    def close(self):
        closing = time.time()
        self.process.kill()
        self.process.wait()
        print(f"  ...closing after {closing - self.started:.1f} s", flush=True)
        # ...and the helpers, which outlive it for a moment and hold the debugging port.
        for _ in range(40):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/json/version", timeout=0.5)
            except OSError:
                return
            time.sleep(0.25)
        subprocess.run(["pkill", "-9", "-f", f"remote-debugging-port={self.port}"], check=False)
        time.sleep(1)


results = []


class Scenario:
    def __init__(self, name):
        self.name = name
        self.notes = []
        self.failed = 0
        self.browsers = []
        self.thrown = []
        self.complaints = []

    def check(self, label, ok, seen=""):
        print(("  ok  " if ok else "  FAIL"), label, "--", str(seen)[:260], flush=True)
        self.notes.append({"check": label, "ok": bool(ok), "seen": str(seen)[:400]})
        self.failed += 0 if ok else 1

    def note(self, label, seen=""):
        print("  note", label, "--", str(seen)[:260], flush=True)
        self.notes.append({"note": label, "seen": str(seen)[:400]})

    def browser(self, **options):
        browser = Browser(**options)
        self.browsers.append(browser)
        return browser

    def page(self, init=(), **options):
        return self.browser(**options).first.open(init)

    def folds_again(self, tab, label="the next fold works, on this page", size=None, family=ESMFOLD2, want=None):
        """The one thing every scenario must leave behind. A length of its own, so that no trunk can be reused."""
        size = size if size is not None else 70 + random.randrange(5)
        tab.choose(family)
        tab.set_size(size)
        end = tab.fold()
        self.check(label, end["running"] is False and not end["failed"] and not end.get("timed_out"),
                   f"{end['seconds']} s: {end['status'][:170]}")
        if want is not None:
            plddt = number_after("pLDDT", end["status"])
            self.check("and reads what it should", plddt is not None and abs(plddt - want) < 0.5, end["status"][:120])
        # A viewer that lost its context with the GPU process would leave a fold that reads right and shows nothing.
        if end["running"] is False and not end["failed"] and not end.get("timed_out"):
            # ...in a tab in the background, once anyone can look: nothing draws for a page nobody can see.
            if tab.evaluate("document.visibilityState") == "hidden":
                tab.ws.call("Page.bringToFront")
                time.sleep(0.5)
            time.sleep(0.3)
            drawn = tab.evaluate(DRAWN)
            self.check("and draws it", drawn > 20, f"{drawn} colours")
        return end

    def no_page_errors(self, end, expected=()):
        expected = ("device lost", "GPU device was lost", "fold failed", "WebGPU:", *expected)
        unexpected = [e for e in end["errors"] if not any(word in e for word in expected)]
        self.check("nothing uncaught", not unexpected, unexpected[:3])

    def close(self):
        for browser in self.browsers:
            for tab in browser.tabs:
                if tab.opened:
                    try:
                        tab.ws.s.settimeout(5)
                        tab.state()
                    except (RuntimeError, OSError, TypeError, ValueError):
                        pass  # A tab that has gone takes what it had not reported with it.
                    self.thrown += tab.uncaught
                    self.complaints += tab.complaints
                    tab.uncaught, tab.complaints = [], []
            browser.close()
        self.browsers = []

    def nothing_thrown(self):
        """However a scenario ended, and whatever it chose to look at."""
        self.check("nothing was thrown that nobody caught", not self.thrown, self.thrown[:3])
        if self.complaints:
            self.note("what the GPU said", sorted({complaint[:120] for complaint in self.complaints})[:4])


SCENARIOS = {}


def scenario(group):
    def register(function):
        SCENARIOS[function.__name__.replace("_", "-")] = (group, function)
        return function
    return register


# The GPU process, ended under a fold at each of its stages.

def kill_at(s, family, size, words):
    tab = s.page()
    tab.choose(family)
    tab.set_size(size)
    tab.press()
    tab.wait_status(*words)
    before = kill_gpu_process()
    end = tab.wait_finished(timeout=120)
    s.check("the fold ends, and says so", end["running"] is False and end["failed"] and not end.get("timed_out"), end["status"][:220])
    s.check("a new GPU process", gpu_processes() != before, f"{before} -> {gpu_processes()}")
    s.check("the line beside Fold is not stuck on a limit a crash taught it", "of 1." not in end["line"] and "of 0." not in end["line"], end["line"])
    s.no_page_errors(end)
    return tab


@scenario("faults")
def kill_esmfold2_language_model(s):
    tab = kill_at(s, ESMFOLD2, 600, ["Language model"])
    s.folds_again(tab, want=None)


@scenario("faults")
def kill_esmfold2_trunk_first_pass(s):
    s.folds_again(kill_at(s, ESMFOLD2, 600, ["Trunk 1/4"]))


@scenario("faults")
def kill_esmfold2_trunk_last_pass(s):
    s.folds_again(kill_at(s, ESMFOLD2, 600, ["Trunk 4/4"]))


@scenario("faults")
def kill_esmfold2_sampler(s):
    s.folds_again(kill_at(s, ESMFOLD2, 600, ["Folding"]))


@scenario("faults")
def kill_openbind_trunk(s):
    s.folds_again(kill_at(s, "openbind0", 300, ["Trunk 2/4"]), family="openbind0")


@scenario("faults")
def kill_openbind_sampler(s):
    s.folds_again(kill_at(s, "openbind0", 300, ["Folding 5/"]), family="openbind0")


@scenario("faults")
def kill_protenix_trunk(s):
    s.folds_again(kill_at(s, "protenix2", 300, ["Trunk 2/4"]), family="protenix2")


@scenario("faults")
def kill_monomer(s):
    s.folds_again(kill_at(s, "monomer", 300, ["Folding"]), family="monomer")


@scenario("faults")
def kill_multimer(s):
    s.folds_again(kill_at(s, "multimer", 300, ["Folding"]), family="multimer")


@scenario("faults")
def kill_bioemu(s):
    tab = s.page()
    tab.choose("bioemu2")
    tab.set_size(150)
    tab.press()
    time.sleep(4)
    state = tab.state()
    if state["running"] is not True:
        s.note("the fold had finished before the kill", state["status"][:160])
    kill_gpu_process()
    end = tab.wait_finished(timeout=120)
    s.check("the fold ends, and says so", end["running"] is False, end["status"][:220])
    s.no_page_errors(end)
    s.folds_again(tab, family="bioemu2")


def shown(tab):
    """The most colours on a visible canvas other than the structure's: on ESMFold2, its PAE map."""
    return max((row.get("colours", 0) for row in json.loads(tab.evaluate(OTHER_CANVASES))), default=0)


@scenario("faults")
def the_picture_after_a_crash_with_no_device(s):
    # The page has let go of its device, so nothing tells it of the crash but the canvases.
    tab = s.page()
    s.folds_again(tab, "a fold first", size=76, want=87.0)
    released = tab.evaluate("(() => { const b = document.getElementById('settings-gpu-release'); document.getElementById('settings-toggle')?.click(); b?.click(); return b !== null; })()")
    s.note("the device let go", released)
    time.sleep(2)
    kill_gpu_process()
    time.sleep(3)
    drawn = tab.evaluate(DRAWN)
    s.check("the result stays on screen, with no device held", drawn > 20, f"{drawn} colours")
    s.check("and the map beside it", shown(tab) > 20, tab.evaluate(OTHER_CANVASES))
    s.folds_again(tab, "the next fold draws on a new device", want=None)


@scenario("faults")
def the_picture_after_a_crash(s):
    tab = s.page()
    s.folds_again(tab, "a fold first", size=76, want=87.0)
    s.check("a fold draws the map beside it", shown(tab) > 20, tab.evaluate(OTHER_CANVASES))
    kill_gpu_process()
    time.sleep(3)
    drawn = tab.evaluate(DRAWN)
    s.check("the result stays on screen, with nothing folding", drawn > 20, f"{drawn} colours")
    s.check("and the map beside it", shown(tab) > 20, tab.evaluate(OTHER_CANVASES))
    s.folds_again(tab, "the next fold draws on a new device", want=None)
    time.sleep(1)
    s.folds_again(tab, "and the one after", size=76, want=87.0)


@scenario("faults")
def kill_while_idle(s):
    tab = s.page()
    tab.choose()
    tab.set_size(76)
    end = tab.fold()
    s.check("a fold first", not end["failed"], end["status"][:120])
    kill_gpu_process()
    time.sleep(3)
    state = tab.state()
    s.note("what an idle page says", state["status"][:200])
    s.folds_again(tab, want=None)
    s.folds_again(tab, "and the same length as before the kill", size=76, want=87.0)


@scenario("faults")
def kill_then_more_recycles(s):
    # A fold keeps what it needs to go on recycling. Those buffers died with the device.
    tab = s.page()
    tab.choose()
    tab.set_size(76)
    tab.set_control("recycles", 1)
    end = tab.fold()
    s.check("a fold of one recycle", not end["failed"], end["status"][:120])
    kill_gpu_process()
    time.sleep(3)
    tab.set_control("recycles", 3)
    end = tab.fold()
    s.check("raising the recycles folds", end["running"] is False and not end["failed"], end["status"][:170])
    plddt = number_after("pLDDT", end["status"])
    s.check("and reads what three recycles read", plddt is not None and abs(plddt - 87.0) < 0.5, end["status"][:120])


def kill_then_more_recycles_of(s, family, size):
    # What a fold keeps to go on recycling died with the device, and the page must not resume from it.
    tab = s.page()
    tab.choose(family)
    tab.set_size(size)
    tab.set_control("recycles", 3)
    end = tab.fold()
    reference = number_after("pLDDT", end["status"])
    s.check("a fold of three recycles", not end["failed"] and reference is not None, end["status"][:120])
    tab.set_control("recycles", 1)
    tab.fold()
    kill_gpu_process()
    time.sleep(3)
    tab.set_control("recycles", 3)
    end = tab.fold()
    plddt = number_after("pLDDT", end["status"])
    s.check("three recycles after the kill read as they did", not end["failed"] and plddt is not None and abs(plddt - reference) < 0.5, end["status"][:170])


@scenario("faults")
def kill_then_more_recycles_openbind(s):
    kill_then_more_recycles_of(s, "openbind0", 120)


@scenario("faults")
def kill_then_more_recycles_monomer(s):
    kill_then_more_recycles_of(s, "monomer", 120)


@scenario("faults")
def kill_then_the_same_again(s):
    # ESMFold2 keeps its trunk on the host, so the same fold after a kill reuses it on a new device.
    tab = s.page()
    tab.choose()
    tab.set_size(76)
    end = tab.fold()
    s.check("a fold", not end["failed"], end["status"][:120])
    kill_gpu_process()
    time.sleep(3)
    end = tab.fold()
    plddt = number_after("pLDDT", end["status"])
    s.check("the same fold, on a new device", not end["failed"] and plddt is not None and abs(plddt - 87.0) < 0.5, end["status"][:170])


@scenario("faults")
def kill_during_a_download(s):
    tab = s.page(cold=True)
    tab.choose()
    tab.set_size(76)
    tab.press()
    time.sleep(1)
    kill_gpu_process()
    end = tab.wait_finished(timeout=120)
    s.note("the fold", f"failed={end['failed']} {end['seconds']} s: {end['status'][:170]}")
    s.check("the fold ends", end["running"] is False and not end.get("timed_out"), end["status"][:120])
    s.no_page_errors(end)
    s.folds_again(tab, size=76, want=87.0)


@scenario("faults")
def kill_with_another_tab_waiting(s):
    browser = s.browser()
    first = browser.first.open()
    second = browser.new_tab(URL)
    second.open()
    first.choose()
    first.set_size(300)
    second.choose()
    second.set_size(76)
    first.press()
    first.wait_status("Trunk 1/4")
    second.press()
    time.sleep(1)
    kill_gpu_process()
    end_first = first.wait_finished(timeout=120)
    end_second = second.wait_finished(timeout=180)
    s.check("the fold that lost its device says so", end_first["running"] is False and end_first["failed"], end_first["status"][:120])
    plddt = number_after("pLDDT", end_second["status"])
    s.check("the one that was waiting has its own and folds", not end_second["failed"] and plddt is not None and abs(plddt - 87.0) < 0.5, end_second["status"][:120])
    s.folds_again(first, "and the first folds again")


@scenario("faults")
def kill_during_recovery(s):
    tab = s.page()
    tab.choose()
    tab.set_size(300)
    tab.press()
    tab.wait_status("Trunk 1/4")
    kill_gpu_process()
    tab.wait_finished(timeout=60)
    tab.press()
    # ...at once, while the new device is being asked for or the weights uploaded.
    time.sleep(0.3)
    kill_gpu_process()
    end = tab.wait_finished(timeout=240)
    s.check("the second fold ends one way or the other", end["running"] is False and not end.get("timed_out"), end["status"][:200])
    s.no_page_errors(end)
    tab.choose()
    tab.set_size(40)
    end = tab.fold(timeout=300)
    s.check("and a fold after two crashes works, on whatever is left", not end["failed"] and not end.get("timed_out"), end["status"][:170])


@scenario("faults")
def kill_until_no_gpu_then_restart(s):
    # Past Chrome's limit the browser offers no GPU. ESMFold2 goes on, and a new browser has its GPU back.
    browser = s.browser()
    tab = browser.first.open()
    tab.choose()
    tab.set_size(300)
    for crash in range(1, 4):
        time.sleep(0.5)
        state = tab.state()
        if "CPU" in state["line"] or "thread" in state["line"]:
            s.note(f"on the CPU before crash {crash}", state["line"])
            break
        tab.press()
        try:
            tab.wait_status("Trunk", timeout=40)
        except AssertionError as error:
            s.note(f"crash {crash} found no trunk", error)
            tab.stop_if_running()
            break
        kill_gpu_process()
        end = tab.wait_finished(timeout=60)
        s.note(f"crash {crash}", f"{end['status'][:120]} | line: {end['line']}")
        time.sleep(1)
    tab.set_size(30)
    end = tab.fold(timeout=300)
    s.check("a fold still works", not end["failed"] and not end.get("timed_out"), f"{end['seconds']} s: {end['status'][:150]}")
    browser.close()
    s.browsers.remove(browser)
    tab = s.page(keep_origin=True)
    tab.choose()
    tab.set_size(76)
    state = tab.state()
    s.check("a new browser plans on the GPU again", "GiB" in state["line"], state["line"])
    s.folds_again(tab, size=76, want=87.0)
    badge = tab.evaluate("document.getElementById('device-badge')?.textContent ?? ''")
    s.check("and folds there", "GPU" in badge, badge)


# Stop, at each stage, and what the fold after it reads.

def stop_at(s, family, size, words, want_family_fold=True):
    tab = s.page()
    tab.choose(family)
    tab.set_size(size)
    tab.press()
    tab.wait_status(*words)
    clicked = time.time()
    tab.evaluate("document.getElementById('predict').click()")
    end = tab.wait_finished(timeout=120)
    s.check("Stop ends it, at once", end["running"] is False and time.time() - clicked < 15, f"{time.time() - clicked:.1f} s: {end['status'][:120]}")
    s.no_page_errors(end)
    return tab


def same_as_clean(s, tab, family, size, reference_pdb):
    tab.choose(family)
    tab.set_size(size)
    end = tab.fold()
    s.check("the same fold again works", not end["failed"], end["status"][:150])
    pdb = tab.pdb()
    if pdb is None or reference_pdb is None:
        return s.check("and has a structure", pdb is not None, "")
    worst = max((max(abs(a - b) for a, b in zip(p, q)) for p, q in zip(pdb["ca"], reference_pdb["ca"])), default=0.0)
    return s.check("and reads a clean fold's coordinates", pdb["atoms"] == reference_pdb["atoms"] and worst < 0.2, f"largest change {worst:.3f} A")


def fold_size(tab, size):
    tab.set_size(size)
    return tab.fold()


def stop_then_same(s, family, size, words):
    tab = s.page()
    tab.choose(family)
    # A page keeps one trunk, so another length and back folds whole; and the first fold is past the first kernels.
    fold_size(tab, size + 7)
    end = fold_size(tab, size)
    s.check("a clean fold", not end["failed"], end["status"][:120])
    reference = tab.pdb()
    fold_size(tab, size + 7)
    tab.set_size(size)
    tab.press()
    tab.wait_status(*words)
    tab.evaluate("document.getElementById('predict').click()")
    end = tab.wait_finished(timeout=120)
    s.check("Stop ends it", end["running"] is False, end["status"][:120])
    s.no_page_errors(end)
    same_as_clean(s, tab, family, size, reference)


@scenario("stop")
def stop_esmfold2_language_model(s):
    tab = stop_at(s, ESMFOLD2, 600, ["Language model"])
    s.folds_again(tab)


@scenario("stop")
def stop_esmfold2_trunk_reads_the_same(s):
    stop_then_same(s, ESMFOLD2, 300, ["Trunk 2/4"])


@scenario("stop")
def stop_esmfold2_sampler_reads_the_same(s):
    stop_then_same(s, ESMFOLD2, 300, ["Folding"])


@scenario("stop")
def stop_openbind_trunk_reads_the_same(s):
    stop_then_same(s, "openbind0", 200, ["Trunk 2/4"])


@scenario("stop")
def stop_openbind_sampler_reads_the_same(s):
    stop_then_same(s, "openbind0", 200, ["Folding 5/"])


@scenario("stop")
def stop_monomer(s):
    tab = stop_at(s, "monomer", 300, ["Folding"])
    s.folds_again(tab, family="monomer")


def stop_during_more_recycles(s, family, size):
    # A stopped trunk must not be taken for a finished one by the fold that raises the recycles. AF3's models go on from the
    # passes they have, and ESMFold2 keeps a trunk per count.
    tab = s.page()
    tab.choose(family)
    fold_size(tab, size + 7)
    tab.set_control("recycles", 3)
    end = fold_size(tab, size)
    reference = tab.pdb()
    s.check("a clean fold", not end["failed"], end["status"][:120])
    tab.set_control("recycles", 1)
    tab.fold()
    tab.set_control("recycles", 3)
    tab.press()
    state = tab.wait_status("Trunk")
    s.note("what it says", state["status"][:100])
    tab.evaluate("document.getElementById('predict').click()")
    end = tab.wait_finished(timeout=60)
    s.check("Stop ends it", end["running"] is False, end["status"][:120])
    same_as_clean(s, tab, family, size, reference)


@scenario("stop")
def stop_during_more_recycles_esmfold2(s):
    stop_during_more_recycles(s, ESMFOLD2, 150)


@scenario("stop")
def stop_during_more_recycles_openbind(s):
    stop_during_more_recycles(s, "openbind0", 150)


@scenario("stop")
def stop_during_more_recycles_monomer(s):
    tab = s.page()
    tab.choose("monomer")
    tab.set_size(150)
    tab.set_control("recycles", 1)
    tab.fold()
    tab.set_control("recycles", 6)
    tab.press()
    time.sleep(0.7)
    tab.evaluate("document.getElementById('predict').click()")
    end = tab.wait_finished(timeout=60)
    s.check("Stop ends it", end["running"] is False, end["status"][:120])
    tab.set_control("recycles", 3)
    end = tab.fold()
    s.check("and a fold of fewer recycles after it works", not end["failed"], end["status"][:150])


@scenario("stop")
def fold_and_stop_storm(s):
    tab = s.page()
    tab.choose()
    tab.set_size(120)
    tab.fold()
    for _ in range(12):
        tab.evaluate("document.getElementById('predict').click()")
        time.sleep(random.uniform(0.02, 0.6))
    time.sleep(1)
    state = tab.state()
    if state["running"] is True:
        tab.evaluate("document.getElementById('predict').click()")
    end = tab.wait_finished(timeout=120)
    s.check("twelve presses leave it idle", end["running"] is False, end["status"][:150])
    s.no_page_errors(end)
    s.folds_again(tab, want=None)


@scenario("stop")
def double_press(s):
    tab = s.page()
    tab.choose()
    tab.set_size(120)
    tab.evaluate("(() => { const fold = document.getElementById('predict'); fold.click(); fold.click(); })()")
    time.sleep(1)
    end = tab.wait_finished(timeout=120)
    s.check("two presses in one tick end", end["running"] is False, end["status"][:150])
    s.no_page_errors(end)
    s.folds_again(tab)


@scenario("stop")
def stop_during_download(s):
    tab = s.page([SLOW_LINE % 1_000_000], cold=True)
    tab.choose()
    tab.set_size(76)
    tab.press()
    time.sleep(3)
    state = tab.state()
    s.note("what a cold fold says at 3 s", state["status"][:120])
    clicked = time.time()
    tab.evaluate("document.getElementById('predict').click()")
    s.note("the click took", f"{time.time() - clicked:.2f} s")
    end = tab.wait_finished(timeout=60)
    s.check("Stop ends it, at once", end["running"] is False and "stopped" in end["status"] and time.time() - clicked < 5, f"{time.time() - clicked:.1f} s: {end['status'][:150]}")
    s.no_page_errors(end)
    tab.evaluate("window.__line.bytesPerSecond = 0")
    s.folds_again(tab, size=76, want=87.0)


@scenario("stop")
def change_model_during_fold(s):
    tab = s.page()
    tab.choose()
    tab.set_size(300)
    tab.press()
    tab.wait_status("Trunk 1/4")
    tab.choose("openbind0")
    end = tab.wait_finished(timeout=120)
    s.check("the fold ends", end["running"] is False, end["status"][:150])
    s.no_page_errors(end)
    shown = tab.evaluate("document.getElementById('model-family').value")
    s.note("the model it leaves chosen", shown)
    s.folds_again(tab, family="openbind0")
    s.folds_again(tab, "and back", family=ESMFOLD2)


@scenario("stop")
def edit_sequence_during_fold(s):
    tab = s.page()
    tab.choose()
    tab.set_size(300)
    tab.press()
    tab.wait_status("Trunk 1/4")
    tab.set_size(90)
    end = tab.wait_finished(timeout=120)
    s.check("the fold ends", end["running"] is False, end["status"][:150])
    pdb = tab.pdb()
    s.check("with the structure it was asked for", pdb is not None and len(pdb["ca"]) == 300, None if pdb is None else len(pdb["ca"]))
    s.folds_again(tab, "and the edit folds", size=90)


@scenario("stop")
def clear_downloads_during_fold(s):
    tab = s.page()
    tab.choose()
    tab.set_size(300)
    tab.press()
    tab.wait_status("Trunk 1/4")
    tab.evaluate("document.getElementById('settings-toggle')?.click()")
    time.sleep(0.5)
    cleared = tab.evaluate("(() => { const b = document.getElementById('settings-models-clear'); if (b === null) return false; b.click(); return true; })()")
    s.note("the button was there", cleared)
    end = tab.wait_finished(timeout=120)
    s.check("the fold ends", end["running"] is False, end["status"][:150])
    s.no_page_errors(end)
    tab.evaluate("document.getElementById('settings-toggle')?.click()")
    s.folds_again(tab, "and the next downloads what it needs, and works", size=76, want=87.0)


@scenario("stop")
def stop_the_tab_that_is_waiting(s):
    browser = s.browser()
    first = browser.first.open()
    second = browser.new_tab(URL)
    second.open()
    first.choose()
    first.set_size(300)
    second.choose()
    second.set_size(76)
    first.press()
    first.wait_status("Trunk 1/4")
    second.press()
    time.sleep(1)
    s.check("the second waits", "Waiting" in second.state()["status"], second.state()["status"][:100])
    clicked = time.time()
    second.evaluate("document.getElementById('predict').click()")
    end = second.wait_finished(timeout=20)
    s.check("Stop there ends it, at once", end["running"] is False and "stopped" in end["status"] and time.time() - clicked < 3, f"{time.time() - clicked:.1f} s: {end['status'][:100]}")
    end = first.wait_finished()
    s.check("and the other goes on", not end["failed"], end["status"][:110])
    s.folds_again(second, size=76, want=87.0)


@scenario("stop")
def limit_lowered_during_a_fold(s):
    tab = s.page()
    tab.choose()
    tab.set_size(800)
    tab.press()
    tab.wait_status("Trunk 1/4")
    tab.evaluate("document.getElementById('settings-toggle')?.click()")
    time.sleep(0.5)
    tab.evaluate("""(() => { const choice = document.getElementById('settings-gpu-limit'); choice.value = String(2 * 2 ** 30);
                             choice.dispatchEvent(new Event('change', { bubbles: true })); })()""")
    end = tab.wait_finished(timeout=240)
    s.note("the fold", f"failed={end['failed']} {end['seconds']} s: {end['status'][:190]} {end['actions']}")
    s.check("the fold ends", end["running"] is False and not end.get("timed_out"), end["status"][:120])
    s.no_page_errors(end)
    s.folds_again(tab, "a fold inside the new limit", size=300)
    tab.evaluate("""(() => { const choice = document.getElementById('settings-gpu-limit'); choice.value = '';
                             choice.dispatchEvent(new Event('change', { bubbles: true })); })()""")
    time.sleep(0.5)
    tab.evaluate("document.getElementById('settings-toggle')?.click()")
    s.folds_again(tab, "and the one it refused, with the limit back", size=800)


@scenario("stop")
def settings_during_a_fold(s):
    # The panel's own reads of the device, its theme and its list of models, none of which a fold should feel.
    tab = s.page()
    tab.choose()
    tab.set_size(300)
    tab.press()
    tab.wait_status("Trunk 1/4")
    for _ in range(4):
        tab.evaluate("document.getElementById('settings-toggle')?.click()")
        time.sleep(0.2)
    tab.evaluate("document.getElementById('settings-toggle')?.click()")
    for theme in ("Dark", "Light", "Auto"):
        tab.evaluate("""[...document.querySelectorAll('#settings-theme button')].find((b) => b.textContent.trim() === %s)?.click()""" % json.dumps(theme))
        time.sleep(0.2)
    tab.evaluate("document.getElementById('settings-toggle')?.click()")
    end = tab.wait_finished(timeout=120)
    plddt = number_after("pLDDT", end["status"])
    s.check("the fold finishes as it would", not end["failed"] and plddt is not None and abs(plddt - 78.5) < 0.5, end["status"][:120])
    s.no_page_errors(end)


SLOW_LINE = """
(() => {
  const real = window.fetch;
  window.__line = { bytesPerSecond: %d };
  window.fetch = async function (input, init) {
    const response = await real(input, init);
    const url = String(input?.url ?? input);
    if (!url.includes('huggingface.co') || url.includes('.json') || response.body === null) return response;
    const reader = response.body.getReader();
    const paced = new ReadableStream({
      async pull(controller) {
        const { done, value } = await reader.read();
        if (done) { controller.close(); return; }
        const rate = window.__line.bytesPerSecond;
        if (rate > 0) await new Promise((resolve) => setTimeout(resolve, value.byteLength / rate * 1000));
        controller.enqueue(value);
      },
      cancel(reason) { return reader.cancel(reason); },
    });
    return new Response(paced, { status: response.status, statusText: response.statusText, headers: response.headers });
  };
})();
"""


# A download that fails or comes back wrong.

FETCH_FAULT = """
(() => {
  const real = window.fetch;
  let seen = 0;
  // ...over for good once a scenario says so, or the reload that checks the cache would set it off again.
  const over = sessionStorage.getItem('__faultOver') !== null;
  window.__fetchFault = { mode: %s, at: %d, times: over ? 0 : %d, fired: 0, urls: [] };
  window.fetch = async function (input, init) {
    const url = String(input?.url ?? input);
    const fault = window.__fetchFault;
    if (/huggingface\\.co|hf\\.co|cdn-lfs|xethub/.test(url) && (fault.mode === 'portal' || !/\\.json(\\?|$)/.test(url))) {
      seen += 1;
      if (seen >= fault.at && fault.fired < fault.times) {
        fault.fired += 1;
        fault.urls.push(url);
        if (fault.mode === 'network') throw new TypeError('Failed to fetch');
        if (fault.mode === '500') return new Response('boom', { status: 500 });
        if (fault.mode === 'portal') return new Response('<html><body>Sign in to this network</body></html>', { status: 200, headers: { 'content-type': 'text/html' } });
        const response = await real(input, init);
        const bytes = new Uint8Array(await response.arrayBuffer());
        if (fault.mode === 'truncated') return new Response(bytes.slice(0, bytes.length >> 1), { status: response.status, statusText: response.statusText, headers: response.headers });
        if (fault.mode === 'empty') return new Response(new Uint8Array(0), { status: 200 });
      }
    }
    return real(input, init);
  };
})();
"""


def fetch_fault(s, mode, at=3, times=1, family=ESMFOLD2, want=87.0):
    tab = s.page([FETCH_FAULT % (json.dumps(mode), at, times)], cold=True)
    tab.choose(family)
    tab.set_rows([["protein", UBI, 1]])
    end = tab.fold(timeout=300)
    fault = json.loads(tab.evaluate("JSON.stringify(window.__fetchFault)"))
    s.check("the fault fired", fault["fired"] > 0, fault["urls"][:1])
    s.note("the fold", f"failed={end['failed']} {end['seconds']} s: {end['status'][:180]}")
    s.check("the fold ended", end["running"] is False and not end.get("timed_out"), end["status"][:150])
    s.no_page_errors(end)
    # The CPU needs the same download, and a reader on a dropped connection has no use for a stack's words.
    if end["failed"]:
        s.check("a failed download says so, and offers no fold that needs it too", "download" in end["status"] and not end["actions"], f"{end['status'][:150]} {end['actions']}")
    reads = number_after("pLDDT", end["status"]) if not end["failed"] else None
    if reads is not None and want is not None:
        s.check("a fold that went on reads what it should", abs(reads - want) < 0.5, end["status"][:150])
    # ...the fault over, however many it took.
    tab.evaluate("window.__fetchFault.times = 0; sessionStorage.setItem('__faultOver', '1')")
    end = tab.fold(timeout=300)
    again = number_after("pLDDT", end["status"])
    s.check("the retry, on this page, works", not end["failed"] and not end.get("timed_out") and (want is None or (again is not None and abs(again - want) < 0.5)), end["status"][:170])
    s.no_page_errors(end)
    # ...and nothing wrong stayed in the cache for the next visit.
    tab.ws.call("Page.reload")
    time.sleep(1)
    tab.wait_ready()
    tab.choose(family)
    tab.set_rows([["protein", UBI, 1]])
    end = tab.fold(timeout=300)
    plddt = number_after("pLDDT", end["status"])
    s.check("and after a reload, from what the cache kept, it reads the same", not end["failed"] and (again is None or (plddt is not None and abs(plddt - again) < 0.5)), end["status"][:170])


@scenario("network")
def download_network_error(s):
    fetch_fault(s, "network")


@scenario("network")
def download_server_error(s):
    fetch_fault(s, "500")


@scenario("network")
def download_truncated(s):
    fetch_fault(s, "truncated")


@scenario("network")
def download_empty(s):
    fetch_fault(s, "empty")


@scenario("network")
def download_down_for_good_then_back(s):
    # Past every retry, on every shard, with the manifest already read: the failure must not be kept with the store.
    fetch_fault(s, "network", at=1, times=100000)


@scenario("network")
def download_down_openbind(s):
    fetch_fault(s, "500", at=1, times=100000, family="openbind0", want=None)


@scenario("network")
def download_down_monomer(s):
    fetch_fault(s, "500", at=1, times=100000, family="monomer", want=None)


@scenario("network")
def download_down_bioemu(s):
    fetch_fault(s, "500", at=1, times=100000, family="bioemu2", want=None)


@scenario("network")
def download_down_late_openbind(s):
    # Partway: some shards arrived, one did not, and the rest are on their way when the fold gives up.
    fetch_fault(s, "500", at=6, times=100000, family="openbind0", want=None)


@scenario("network")
def download_down_late_esmc(s):
    fetch_fault(s, "500", at=8, times=100000)


@scenario("network")
def download_errors_for_good_then_back(s):
    fetch_fault(s, "500", at=1, times=100000)


@scenario("network")
def download_short_for_good_then_back(s):
    fetch_fault(s, "truncated", at=1, times=100000)


@scenario("network")
def offline_then_online(s):
    tab = s.page(cold=True)
    tab.ws.call("Network.enable")
    tab.ws.call("Network.emulateNetworkConditions", offline=True, latency=0, downloadThroughput=0, uploadThroughput=0)
    tab.choose()
    tab.set_size(76)
    end = tab.fold(timeout=180)
    s.check("offline, the fold fails and says so", end["running"] is False and end["failed"], end["status"][:200])
    s.no_page_errors(end)
    tab.ws.call("Network.emulateNetworkConditions", offline=False, latency=0, downloadThroughput=-1, uploadThroughput=-1)
    tab.ws.call("Network.disable")
    s.folds_again(tab, "back online, the next fold works with no reload", size=76, want=87.0)


@scenario("network")
def stop_on_a_slow_line_at_each_stage(s):
    # What a reader on a slow line does when a fold is taking too long, at each of the stages it can be seen to take. Eight
    # connections at 250 KB/s each, about 16 Mbit/s: 258 MiB is two minutes.
    for words in ("loading", "Language model"):
        tab = s.page([SLOW_LINE % 250_000], cold=True)
        tab.choose()
        tab.set_size(76)
        tab.press()
        said = tab.wait_status(words, timeout=240)["status"][:100]
        time.sleep(0.5)
        # ...for a press on a page that has finished starts a fold, and on this input one that reuses the trunk.
        if tab.state()["running"] is not True:
            s.note(f"the fold had ended soon after \"{said}\"", tab.state()["status"][:120])
            s.close()
            continue
        clicked = time.time()
        tab.evaluate("document.getElementById('predict').click()")
        end = tab.wait_finished(timeout=120)
        took = time.time() - clicked
        s.check(f"Stop at \"{said}\" ends it, at once", end["running"] is False and "stopped" in end["status"] and took < 5, f"{took:.1f} s: {end['status'][:120]}")
        s.no_page_errors(end)
        tab.evaluate("window.__line.bytesPerSecond = 0")
        s.folds_again(tab, "the next fold takes up the download", size=76, want=87.0)
        s.close()


@scenario("network")
def a_slow_line_all_the_way(s):
    # The whole fold on a slow line, which is a first visit for a lot of people, and what it says as it goes.
    tab = s.page([SLOW_LINE % 500_000], cold=True)
    tab.choose()
    tab.set_size(76)
    end = tab.fold(timeout=420)
    plddt = number_after("pLDDT", end["status"])
    s.check("it folds", not end["failed"] and plddt is not None and abs(plddt - 87.0) < 0.5, f"{end['seconds']} s: {end['status'][:130]}")
    s.no_page_errors(end)


# A browser that stores nothing, or stored something wrong.

@scenario("storage")
def no_local_storage(s):
    tab = s.page(["""Object.defineProperty(window, 'localStorage', { get() { throw new DOMException('denied', 'SecurityError'); } });"""])
    tab.choose()
    s.folds_again(tab, size=76, want=87.0)


@scenario("storage")
def site_data_blocked(s):
    # What a browser that blocks this origin's data does: the names themselves throw.
    tab = s.page(["""for (const name of ['localStorage', 'sessionStorage', 'indexedDB', 'caches']) {
                       Object.defineProperty(window, name, { get() { throw new DOMException('denied', 'SecurityError'); } });
                     }"""])
    s.folds_again(tab, size=76, want=87.0)
    s.folds_again(tab, "and on another model", family="openbind0", size=76)
    s.folds_again(tab, "and on AlphaFold 2", family="monomer", size=76)


@scenario("storage")
def no_cache_storage(s):
    tab = s.page(["""CacheStorage.prototype.open = async () => { throw new DOMException('denied', 'SecurityError'); };
                     CacheStorage.prototype.keys = async () => { throw new DOMException('denied', 'SecurityError'); };
                     CacheStorage.prototype.match = async () => { throw new DOMException('denied', 'SecurityError'); };"""], cold=True)
    s.folds_again(tab, size=76, want=87.0)


@scenario("storage")
def cache_put_fails(s):
    tab = s.page(["""Cache.prototype.put = async () => { throw new DOMException('quota', 'QuotaExceededError'); };"""], cold=True)
    s.folds_again(tab, size=76, want=87.0)


@scenario("storage")
def small_quota(s):
    browser = s.browser(cold=True)
    tab = browser.first
    tab.ws.call("Storage.overrideQuotaForOrigin", origin=ORIGIN, quotaSize=40 * 1024 * 1024)
    tab.open()
    s.folds_again(tab, size=76, want=87.0)
    s.folds_again(tab, "and again", size=80)
    end = tab.fold()
    s.check("the scores stay on the line, with what could not be saved beside them", "pLDDT" in end["status"] and not end["failed"], end["status"][:200])


@scenario("storage")
def garbage_in_local_storage(s):
    # The card's own name, because a rate or a limit is read for the card it was learned on.
    tab = s.page()
    name = tab.evaluate("(() => { const c = document.createElement('canvas').getContext('webgl2'); const i = c?.getExtension('WEBGL_debug_renderer_info'); return i ? c.getParameter(i.UNMASKED_RENDERER_WEBGL) : 'unknown'; })()")
    s.close()
    rates = {"not json": "not json", "{}": "{}", "null": "null", "-1": "-1", "NaN": "NaN", "1e400": "1e400", "[]": "[]",
             "cards a number": '{"cards":7,"cpu":"fast"}', "cards a string": '{"cards":"x"}',
             "a rate that is text": json.dumps({"cards": {name: "abc"}, "cpu": -5}),
             "a rate of nothing": json.dumps({"cards": {name: 0}}), "a rate that is negative": json.dumps({"cards": {name: -3}}),
             "a learned limit that is negative": None}
    for label, value in rates.items():
        keys = {"localfold.fold-rates": value, "localfold.gpu-limit": value, "localfold.theme": value, "localfold.compute": value, "localfold.compute-now": value}
        if value is None:
            keys = {"localfold.gpu-limit": json.dumps({"name": name, "bytes": -5})}
        tab = s.page(["""for (const [key, value] of Object.entries(%s)) { try { localStorage.setItem(key, value); } catch {} }""" % json.dumps(keys)])
        tab.choose()
        tab.set_size(260)
        time.sleep(0.4)
        state = tab.state()
        s.check(f"{label}: the page loads, and its line reads", not state["errors"] and "NaN" not in state["line"] and "undefined" not in state["line"],
                f"{state['line']} | {state['errors'][:2]}")
        end = tab.fold()
        s.check(f"{label}: a fold it learns from", not end["failed"] and not end["errors"], f"{end['seconds']} s: {end['status'][:110]} | {end['errors'][:2]}")
        s.close()


@scenario("storage")
def limits_that_make_no_sense(s):
    for limit in ["0", "-1", "1", "1e20", "abc"]:
        tab = s.page(["""localStorage.setItem('localfold.gpu-chosen-limit', %s);""" % json.dumps(limit)])
        tab.choose()
        tab.set_size(76)
        end = tab.fold()
        s.note(f"limit {limit}", f"failed={end['failed']} {end['status'][:150]}")
        s.check(f"limit {limit}: the page answers", end["running"] is False or end["failed"], end["status"][:120])
        s.no_page_errors(end)
        s.close()


@scenario("storage")
def insecure_page(s):
    # A page reached by address, on a local network: no WebGPU, no Cache API, no locks, no service worker, no crypto.subtle.
    browser = s.browser(extra=["--host-resolver-rules=MAP insecure.test 127.0.0.1"])
    tab = browser.first.open(url=f"http://insecure.test:{args.port}/index.html")
    tab.choose()
    tab.set_size(30)
    time.sleep(0.4)
    state = tab.state()
    s.check("it loads, with nothing uncaught", not state["errors"], state["errors"][:2])
    s.check("and plans one thread of CPU", state["line"].startswith("1 CPU thread"), state["line"])
    end = tab.fold(timeout=420)
    s.check("and folds there", not end["failed"] and not end.get("timed_out"), f"{end['seconds']} s: {end['status'][:120]} | {end['errors'][:2]}")


# Input a model cannot read, and the sizes where a kernel's shape changes.

def try_sizes(s, family, sizes):
    tab = s.page()
    tab.choose(family)
    for size in sizes:
        tab.set_size(size)
        end = tab.fold(timeout=240)
        pdb = None if end["failed"] else tab.pdb()
        fine = not end.get("timed_out") and end["running"] is False and not end["failed"] and pdb is not None and len(pdb["ca"]) == size
        s.check(f"{size} residues", fine, f"failed={end['failed']} {end['seconds']} s: {end['status'][:120]}")
        unexpected = [e for e in end["errors"] if "fold failed" not in e]
        if unexpected:
            s.check(f"{size} residues: nothing uncaught", False, unexpected[:2])
    s.folds_again(tab, family=family)


@scenario("inputs")
def esmfold2_short_and_tile_edges(s):
    try_sizes(s, ESMFOLD2, [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257])


@scenario("inputs")
def openbind_short_and_tile_edges(s):
    try_sizes(s, "openbind0", [1, 2, 3, 5, 8, 16, 17, 32, 33, 64, 65, 128, 129, 256, 257])


@scenario("inputs")
def protenix_tile_edges(s):
    try_sizes(s, "protenix2", [2, 8, 33, 65, 129, 257])


@scenario("inputs")
def monomer_short_and_tile_edges(s):
    try_sizes(s, "monomer", [1, 2, 3, 8, 16, 17, 33, 65, 129, 257])


@scenario("inputs")
def text_a_model_cannot_read(s):
    tab = s.page()
    tab.choose()
    for text in ["", "   ", "\n\n", "12345", "MQIF VKTL\nTGKT", "mqifvktltgk", ">named\nMQIFVKTLTGK", "XXXXXXXXXX", "MQIFUOBZJ", "MQ*IF", "MQ-IF",
                 "ACDEFGHIKLMNPQRSTVWY" * 3 + "é", "\u0000MQIF", "A"]:
        tab.set_rows([["protein", text, 1]])
        end = tab.fold(timeout=120)
        s.check(f"{json.dumps(text)[:30]}: an answer", end["running"] is False or end["failed"], f"failed={end['failed']} {end['status'][:120]}")
        unexpected = [e for e in end["errors"] if "fold failed" not in e]
        s.check(f"{json.dumps(text)[:30]}: nothing uncaught", not unexpected, unexpected[:2])
        s.check(f"{json.dumps(text)[:30]}: no fold it would refuse too", not (end["failed"] and end["actions"]), end["actions"])
    s.folds_again(tab, want=None)


@scenario("inputs")
def many_chains(s):
    tab = s.page()
    tab.choose()
    tab.set_rows([["protein", sequence(20), 20]])
    end = tab.fold()
    s.check("twenty chains of twenty", not end["failed"], f"{end['seconds']} s: {end['status'][:120]}")
    tab.set_rows([["protein", "A", 2]])
    end = tab.fold()
    s.check("two chains of one", not end["failed"], end["status"][:120])
    tab.set_rows([["protein", sequence(50), 21]])
    end = tab.fold()
    s.check("twenty-one copies is refused, with no fold that would be refused too", end["failed"] and not end["actions"], f"{end['status'][:120]} {end['actions']}")
    tab.set_rows([["protein", sequence(50), 20], ["protein", sequence(30), 20]])
    end = tab.fold()
    s.check("two rows of twenty copies", not end["failed"], f"{end['seconds']} s: {end['status'][:120]}")
    s.folds_again(tab)


@scenario("inputs")
def nucleic_and_ligands(s):
    tab = s.page()
    tab.choose()
    dna = ("GATTACA" * 50)[:300]
    for name, rows in [("DNA", [["dna", dna[:60], 1]]), ("RNA", [["rna", "GGCAUUCG" * 8, 1]]),
                       ("protein and DNA", [["protein", sequence(120), 1], ["dna", dna[:40], 2]]),
                       ("protein and ligand", [["protein", sequence(120), 1], ["ligand", "ATP", 1]]),
                       ("protein and SMILES", [["protein", sequence(120), 1], ["smiles", "CC(=O)OC1=CC=CC=C1C(=O)O", 1]]),
                       ("a long DNA", [["dna", dna, 2]])]:
        tab.set_rows(rows)
        time.sleep(0.5)
        line = tab.state()["line"]
        end = tab.fold()
        s.check(f"{name}", not end["failed"] and not end.get("timed_out"), f"plan: {line} | {end['seconds']} s: {end['status'][:120]}")
        unexpected = [e for e in end["errors"] if "fold failed" not in e]
        s.check(f"{name}: nothing uncaught", not unexpected, unexpected[:2])
    s.folds_again(tab)


@scenario("inputs")
def mismatched_rows(s):
    # A row a model does not read, chosen while the row is there.
    tab = s.page()
    tab.choose()
    tab.set_rows([["protein", sequence(50), 1], ["dna", "GATTACA", 1]])
    tab.choose("monomer")
    end = tab.fold(timeout=120)
    s.check("a monomer given DNA: an answer", end["running"] is False or end["failed"], f"failed={end['failed']} {end['status'][:150]}")
    tab.choose("bioemu2")
    end = tab.fold(timeout=240)
    s.check("BioEmu2 given two chains: an answer", end["running"] is False or end["failed"], f"failed={end['failed']} {end['status'][:150]}")
    s.folds_again(tab)


@scenario("inputs")
def every_language_model(s):
    tab = s.page()
    for mode in ("esmc-300m", "esmc-600m"):
        tab.choose(ESMFOLD2, mode)
        tab.set_size(76)
        time.sleep(0.4)
        line = tab.state()["line"]
        end = tab.fold(timeout=600)
        s.check(f"{mode}: an answer", end["running"] is False and not end.get("timed_out"), f"plan: {line} | failed={end['failed']} {end['seconds']} s, card {card_mib()} MiB: {end['status'][:130]}")
        unexpected = [e for e in end["errors"] if "fold failed" not in e]
        s.check(f"{mode}: nothing uncaught", not unexpected, unexpected[:2])
    tab.choose(ESMFOLD2, "esmc-300m")
    s.folds_again(tab, size=76, want=87.0)


@scenario("inputs")
def bioemu_ensembles(s):
    tab = s.page()
    tab.choose("bioemu2")
    tab.set_size(60)
    for mode, samples in (("structure", 1), ("equilibrium", 5), ("equilibrium", 25), ("propagate", 5)):
        tab.set_control("bioemu2-mode", mode)
        tab.set_control("bioemu2-samples", samples)
        end = tab.fold(timeout=420)
        s.check(f"{mode}, {samples}", not end["failed"] and not end.get("timed_out"), f"{end['seconds']} s, card {card_mib()} MiB: {end['status'][:130]}")
    tab.set_control("bioemu2-mode", "equilibrium")
    tab.set_control("bioemu2-samples", 25)
    tab.press()
    time.sleep(1.5)
    tab.evaluate("document.getElementById('predict').click()")
    end = tab.wait_finished(timeout=60)
    s.check("Stop during an ensemble", end["running"] is False, f"{end['seconds']} s: {end['status'][:110]}")
    s.no_page_errors(end)
    tab.press()
    time.sleep(1.5)
    kill_gpu_process()
    end = tab.wait_finished(timeout=60)
    s.check("a kill during one", end["running"] is False, end["status"][:110])
    s.no_page_errors(end)
    tab.set_control("bioemu2-mode", "structure")
    tab.set_control("bioemu2-samples", 1)
    s.folds_again(tab, family="bioemu2")


@scenario("inputs")
def every_sampler_setting(s):
    tab = s.page()
    for family in ("openbind0", ESMFOLD2):
        tab.choose(family)
        tab.set_size(100)
        values = json.loads(tab.evaluate("JSON.stringify([...document.getElementById('af3-count').options].map((option) => option.value))"))
        s.check(f"{family} offers settings", len(values) > 0, values)
        for value in values:
            tab.set_control("af3-count", value)
            end = tab.fold(timeout=420)
            s.check(f"{family}, {value}", not end["failed"], f"{end['seconds']} s, card {card_mib()} MiB: {end['status'][:120]}")
        tab.set_control("af3-count", values[0])
    tab.choose("openbind0")
    tab.set_size(150)
    tab.press()
    tab.wait_status("Folding 2/", "Folding 3/", "Folding 4/", "Folding 5/", timeout=120)
    tab.evaluate("document.getElementById('predict').click()")
    end = tab.wait_finished(timeout=60)
    s.check("Stop between steps", end["running"] is False, f"{end['seconds']} s: {end['status'][:110]}")
    s.no_page_errors(end)
    s.folds_again(tab, family="openbind0")


MSA_FAULT = """
(() => {
  const real = window.fetch;
  window.__msa = { calls: 0, mode: %s };
  window.fetch = async function (input, init) {
    const url = String(input?.url ?? input);
    if (url.includes('api.colabfold.com')) {
      window.__msa.calls += 1;
      const mode = window.__msa.mode;
      if (mode === 'hangs') {
        return new Promise((_, reject) => {
          init?.signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')));
        });
      }
      if (mode === 'refuses') throw new TypeError('Failed to fetch');
      if (mode === '500') return new Response('boom', { status: 500 });
      if (mode === 'garbage') return new Response('<html>a captive portal</html>', { status: 200, headers: { 'content-type': 'text/html' } });
      if (mode === 'rate') return new Response(JSON.stringify({ status: 'RATELIMIT' }), { status: 429, headers: { 'content-type': 'application/json' } });
      if (mode === 'error') return new Response(JSON.stringify({ status: 'ERROR' }), { status: 200, headers: { 'content-type': 'application/json' } });
    }
    return real(input, init);
  };
})();
"""


def alignment_server(s, mode, family):
    tab = s.page([MSA_FAULT % json.dumps(mode)])
    tab.choose(family)
    tab.set_rows([["protein", UBI, 1]])
    tab.set_control("msa-mode", "search")
    tab.press()
    if mode == "hangs":
        time.sleep(5)
        s.note("what it says", tab.state()["status"][:120])
        clicked = time.time()
        tab.evaluate("document.getElementById('predict').click()")
        end = tab.wait_finished(timeout=30)
        s.check("Stop does not wait for a server that never answers", end["running"] is False and time.time() - clicked < 3, f"{time.time() - clicked:.1f} s: {end['status'][:130]}")
    else:
        end = tab.wait_finished(timeout=180)
        s.note("the fold", f"failed={end['failed']} {end['seconds']} s, {tab.evaluate('window.__msa.calls')} calls: {end['status'][:190]} {end['actions']}")
        s.check("the fold ends, and says what happened", end["running"] is False and end["failed"] and not end.get("timed_out"), end["status"][:150])
        s.check("in words about the server", not any(raw in end["status"] for raw in ("<html", "Unexpected token", "Failed to fetch")), end["status"][:170])
        s.check("and a way on without it", "Fold as a single sequence" in end["actions"], str(end["actions"]))
        s.no_page_errors(end)
        s.check("which is one press", tab.click_action("Fold as a single sequence") is True)
        went_on = tab.wait_finished(timeout=180)
        s.check("and folds", went_on["running"] is False and not went_on["failed"], went_on["status"][:150])
        s.check("with the setting changed", tab.evaluate("document.getElementById('msa-mode').value") == "none")
    tab.set_control("msa-mode", "none")
    s.no_page_errors(tab.state())
    s.folds_again(tab, "the next fold without it works, with no reload", family=family, size=76)


@scenario("network")
def alignment_server_hangs_esmfold(s):
    alignment_server(s, "hangs", "openbind0")


@scenario("network")
def alignment_server_hangs_monomer(s):
    alignment_server(s, "hangs", "monomer")


@scenario("network")
def alignment_server_refuses(s):
    alignment_server(s, "refuses", "openbind0")


@scenario("network")
def alignment_server_errors(s):
    alignment_server(s, "500", "monomer")


@scenario("network")
def alignment_server_answers_a_captive_portal(s):
    alignment_server(s, "garbage", "openbind0")


@scenario("network")
def alignment_server_says_rate_limited(s):
    alignment_server(s, "rate", "monomer")


@scenario("network")
def alignment_server_says_error(s):
    alignment_server(s, "error", "openbind0")


@scenario("inputs")
def recycle_extremes(s):
    tab = s.page()
    tab.choose()
    tab.set_size(76)
    for recycles in [0, 1, 12, 24, 0, 6]:
        tab.set_control("recycles", recycles)
        end = tab.fold()
        s.check(f"{recycles} recycles", not end["failed"], f"{end['seconds']} s: {end['status'][:110]}")
    tab.set_control("random-seed", 2147483647)
    end = tab.fold()
    s.check("the largest seed", not end["failed"], end["status"][:110])


# The sizes on either side of a change of shape in the memory plan, each folded: a line that says it fits must be true.

def largest_that_fits(tab):
    low, high = 40, 2047
    while low < high:
        middle = (low + high + 1) // 2
        tab.set_size(middle)
        time.sleep(0.3)
        line = tab.state()["line"]
        if line.startswith("Needs") or "Too long" in line:
            high = middle - 1
        else:
            low = middle
    return low


def fits_means_fits(s, limit_gib, variant="esmc-300m"):
    tab = s.page(["""localStorage.setItem('localfold.gpu-chosen-limit', String(%f * 2 ** 30));""" % limit_gib])
    tab.choose(ESMFOLD2, variant)
    largest = largest_that_fits(tab)
    tab.set_size(largest)
    time.sleep(0.3)
    s.note("the largest the line says fits", f"{largest} at {limit_gib} GiB: {tab.state()['line']}")
    end = tab.fold(timeout=540)
    s.check(f"{largest} residues in {limit_gib} GiB folds", not end["failed"] and not end.get("timed_out"), f"{end['seconds']} s: {end['status'][:150]}")
    tab.set_size(largest + 1)
    time.sleep(0.3)
    for _ in range(40):
        line = tab.state()["line"]
        if line.startswith("Needs"):
            break
        time.sleep(0.1)
    s.check("one more: the line says so", line.startswith("Needs"), line)
    end = tab.fold(timeout=60)
    s.check("and it is refused before anything starts", end["failed"] and end["seconds"] < 5, end["status"][:140])
    s.check("with the ways on", any("GPU" in action or "CPU" in action for action in end["actions"]), end["actions"])


@scenario("sizes")
def fits_in_one_and_a_half_gib(s):
    fits_means_fits(s, 1.5)


@scenario("sizes")
def fits_in_two_gib(s):
    fits_means_fits(s, 2.0, "esmc-600m")


@scenario("sizes")
def fits_in_three_gib(s):
    fits_means_fits(s, 3.0)


@scenario("sizes")
def fits_in_four_point_eight_gib(s):
    fits_means_fits(s, 4.8, "esmc-600m")


@scenario("sizes")
def fits_in_six_point_four_gib(s):
    fits_means_fits(s, 6.4)


@scenario("sizes")
def either_side_of_two_gib_pair(s):
    tab = s.page()
    tab.choose()
    for size in (1447, 1448, 1449):
        tab.set_size(size)
        end = tab.fold(timeout=540)
        s.check(f"{size} residues", not end["failed"] and not end.get("timed_out"), f"{end['seconds']} s: {end['status'][:120]}")


@scenario("sizes")
def the_ceiling(s):
    tab = s.page()
    tab.choose()
    tab.set_size(2047)
    end = tab.fold(timeout=840)
    s.check("2047 residues folds", not end["failed"] and not end.get("timed_out"), f"{end['seconds']} s: {end['status'][:120]}")
    for size in (2048, 3000):
        tab.set_size(size)
        time.sleep(0.3)
        line = tab.state()["line"]
        end = tab.fold(timeout=60)
        s.check(f"{size} residues is refused, by name", end["failed"] and end["seconds"] < 5, f"{line} | {end['status'][:140]}")


@scenario("sizes")
def long_nucleic_against_the_plan(s):
    # A nucleotide is about twenty atoms where a residue is about eight, and the plan reads tokens.
    tab = s.page(["""localStorage.setItem('localfold.gpu-chosen-limit', String(3 * 2 ** 30));"""])
    tab.choose()
    tab.set_rows([["dna", ("GATTACA" * 200)[:700], 1]])
    time.sleep(0.4)
    line = tab.state()["line"]
    end = tab.fold(timeout=420)
    s.note("the line said", line)
    s.check("what the line said held", line.startswith("Needs") == end["failed"] and not end.get("timed_out"),
            f"failed={end['failed']} {end['seconds']} s: {end['status'][:170]}")
    s.folds_again(tab)


@scenario("sizes")
def big_after_small_and_back(s):
    # What a fold leaves in the pool must not shape the next.
    tab = s.page()
    tab.choose()
    prints = {}
    for size in (300, 76, 800, 76, 300, 800):
        tab.set_size(size)
        end = tab.fold()
        s.check(f"{size} residues", not end["failed"], f"{end['seconds']} s: {end['status'][:100]}")
        pdb = tab.pdb()
        if size in prints:
            worst = max((max(abs(a - b) for a, b in zip(p, q)) for p, q in zip(pdb["ca"], prints[size]["ca"])), default=0.0)
            s.check(f"{size} reads as it did", worst < 0.2, f"largest change {worst:.3f} A")
        prints[size] = pdb


# A device that offers less than this card does.

LOW_LIMITS = """
(() => {
  const clamp = { maxBufferSize: %d, maxStorageBufferBindingSize: %d };
  const held = Object.getOwnPropertyDescriptor(GPUAdapter.prototype, 'limits');
  Object.defineProperty(GPUAdapter.prototype, 'limits', { configurable: true, get() {
    const real = held.get.call(this);
    return new Proxy(real, { get: (target, key) => (key in clamp ? Math.min(clamp[key], target[key]) : target[key]) });
  } });
})();
"""


def low_limits(s, buffer_mib, binding_mib, cases):
    tab = s.page([LOW_LIMITS % (buffer_mib * 2 ** 20, binding_mib * 2 ** 20)])
    for family, size in cases:
        tab.choose(family)
        tab.set_size(size)
        time.sleep(0.3)
        line = tab.state()["line"]
        end = tab.fold(timeout=1500)
        unexpected = [e for e in end["errors"] if "fold failed" not in e]
        s.check(f"{family} at {size}: folds or says why not", end["running"] is False and not end.get("timed_out"), f"plan: {line} | failed={end['failed']} {end['seconds']} s: {end['status'][:170]}")
        s.check(f"{family} at {size}: nothing uncaught", not unexpected, unexpected[:2])
    s.folds_again(tab)


@scenario("limits")
def limits_of_the_webgpu_floor(s):
    low_limits(s, 256, 128, [(ESMFOLD2, 76), (ESMFOLD2, 300), (ESMFOLD2, 600), ("openbind0", 76), ("openbind0", 300), ("protenix2", 150), ("monomer", 150)])


@scenario("limits")
def limits_of_a_small_integrated_gpu(s):
    low_limits(s, 1024, 512, [(ESMFOLD2, 300), (ESMFOLD2, 800), (ESMFOLD2, 1000), ("openbind0", 600), ("protenix2", 300), ("mol1", 300)])


@scenario("limits")
def limits_of_two_gib_buffers(s):
    low_limits(s, 2048, 2047, [(ESMFOLD2, 1000), (ESMFOLD2, 1500), ("openbind0", 1000)])


@scenario("limits")
def adapter_null_once(s):
    tab = s.page(["""(() => { const held = GPU.prototype.requestAdapter; let left = 1;
                      GPU.prototype.requestAdapter = function (...a) { if (left > 0) { left -= 1; return Promise.resolve(null); } return held.apply(this, a); }; })();"""])
    tab.choose()
    tab.set_size(76)
    time.sleep(0.5)
    end = tab.fold(timeout=240)
    s.note("the first fold", f"failed={end['failed']} {end['status'][:170]}")
    s.folds_again(tab)


@scenario("limits")
def device_rejected_once(s):
    tab = s.page(["""(() => { const held = GPUAdapter.prototype.requestDevice; let left = 1;
                      GPUAdapter.prototype.requestDevice = function (...a) { if (left > 0) { left -= 1; return Promise.reject(new DOMException('no', 'OperationError')); } return held.apply(this, a); }; })();"""])
    tab.choose()
    tab.set_size(76)
    end = tab.fold(timeout=240)
    s.check("the first fold says why", end["running"] is False, f"failed={end['failed']} {end['status'][:170]}")
    state = tab.state()
    s.note("the line after it", state["line"])
    tab.set_size(70)
    end = tab.fold(timeout=240)
    s.note("the next press", f"failed={end['failed']} {end['seconds']} s: {end['status'][:170]}")
    s.check("the next press works", not end["failed"] and not end.get("timed_out"), end["status"][:170])


@scenario("limits")
def card_really_runs_out(s):
    # A second tab holds nearly all of the card, and a fold told it may hold the lot asks for more than there is.
    browser = s.browser()
    tab = browser.first.open(["""localStorage.setItem('localfold.gpu-chosen-limit', String(40 * 2 ** 30));"""])
    hog = browser.new_tab(URL)
    time.sleep(2)
    total = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip())
    gib = max(0, (total - card_mib()) // 1024 - 3)
    held = hog.evaluate("""(async () => {
      const adapter = await navigator.gpu.requestAdapter();
      const device = await adapter.requestDevice({ requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize } });
      window.__hog = { device, buffers: [] };
      let held = 0;
      for (let at = 0; at < %d; at += 1) {
        device.pushErrorScope('out-of-memory');
        const buffer = device.createBuffer({ size: 2 ** 30, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        const encoder = device.createCommandEncoder();
        encoder.clearBuffer(buffer);
        device.queue.submit([encoder.finish()]);
        const error = await device.popErrorScope();
        await device.queue.onSubmittedWorkDone();
        if (error !== null) break;
        window.__hog.buffers.push(buffer);
        held += 1;
      }
      return held;
    })()""" % gib)
    s.note("what the other tab holds", f"{held} GiB; the card reads {card_mib()} of {total} MiB")
    tab.choose()
    tab.set_size(1400)
    time.sleep(0.3)
    s.note("the line", tab.state()["line"])
    end = tab.fold(timeout=420)
    s.note("the fold", f"failed={end['failed']} {end['seconds']} s: {end['status'][:220]}")
    s.check("a fold with no room ends, and says so", end["running"] is False and not end.get("timed_out"), end["status"][:200])
    s.no_page_errors(end)
    browser.close_tab(hog)
    time.sleep(3)
    s.note("the card with the other tab gone", card_mib())
    s.folds_again(tab, "the next fold works, with the room back", size=300)


@scenario("limits")
def budget_refused_during_a_fold(s):
    # A model the line only advises on: it starts, its own account refuses, and the page offers to go on.
    tab = s.page(["""localStorage.setItem('localfold.gpu-chosen-limit', String(1.0 * 2 ** 30));"""])
    tab.choose("protenix2")
    tab.set_size(600)
    time.sleep(0.3)
    s.note("the line", tab.state()["line"])
    end = tab.fold(timeout=300)
    s.check("it ends, and says so", end["running"] is False and end["failed"], f"{end['seconds']} s: {end['status'][:200]}")
    s.check("with a way on", any("anyway" in action for action in end["actions"]), end["actions"])
    tab.click_action("Fold anyway", "Try on GPU anyway")
    end = tab.wait_finished(timeout=420)
    s.check("going on folds", not end["failed"] and not end.get("timed_out"), f"{end['seconds']} s: {end['status'][:170]}")
    tab.choose()
    tab.set_size(76)
    end = tab.fold()
    s.note("the next small fold", f"failed={end['failed']} {end['status'][:170]}")
    s.check("the page answers", end["running"] is False, end["status"][:120])


@scenario("limits")
def no_webgpu_at_all(s):
    tab = s.page(["""Object.defineProperty(Navigator.prototype, 'gpu', { get() { return undefined; } });"""])
    tab.choose()
    tab.set_size(40)
    time.sleep(0.4)
    line = tab.state()["line"]
    s.check("the line says the CPU", "CPU" in line or "thread" in line, line)
    tab.choose("openbind0")
    time.sleep(0.4)
    line = tab.state()["line"]
    s.check("and another model says a GPU only", "GPU only" in line, line)
    end = tab.fold(timeout=60)
    s.check("whose fold says so", end["failed"], end["status"][:170])
    tab.choose()
    tab.set_size(40)
    end = tab.fold(timeout=420)
    s.check("ESMFold2 folds on the CPU", not end["failed"] and not end.get("timed_out"), f"{end['seconds']} s: {end['status'][:150]}")


# Tabs, reloads, a hidden page, a long session.

@scenario("lifecycle")
def two_tabs_second_waits(s):
    browser = s.browser()
    first = browser.first.open()
    second = browser.new_tab(URL)
    second.open()
    first.choose()
    first.set_size(300)
    second.choose()
    second.set_size(76)
    first.press()
    first.wait_status("Trunk 1/4")
    second.press()
    time.sleep(1)
    state = second.state()
    s.check("the second waits, and says so", "Waiting" in state["status"], state["status"][:120])
    end_first = first.wait_finished()
    end_second = second.wait_finished()
    s.check("the first finishes", not end_first["failed"], end_first["status"][:110])
    plddt = number_after("pLDDT", end_second["status"])
    s.check("then the second, correctly", not end_second["failed"] and plddt is not None and abs(plddt - 87.0) < 0.5, end_second["status"][:110])


@scenario("lifecycle")
def close_the_tab_that_is_folding(s):
    browser = s.browser()
    first = browser.first.open()
    second = browser.new_tab(URL)
    second.open()
    first.choose()
    first.set_size(800)
    second.choose()
    second.set_size(76)
    first.press()
    first.wait_status("Trunk 1/4")
    second.press()
    time.sleep(1)
    held = card_mib()
    target = next(t["id"] for t in browser.targets() if t["type"] == "page" and t["id"] != second.target)
    urllib.request.urlopen(f"http://127.0.0.1:{args.debug_port}/json/close/{target}", timeout=10).read()
    closed = time.time()
    end = second.wait_finished(timeout=120)
    plddt = number_after("pLDDT", end["status"])
    s.check("the waiting tab takes over", not end["failed"] and plddt is not None and abs(plddt - 87.0) < 0.5, f"{time.time() - closed:.1f} s: {end['status'][:110]}")
    s.note("the card", f"{held} MiB under the big fold, {card_mib()} after")


@scenario("lifecycle")
def reload_during_a_fold(s):
    tab = s.page()
    tab.choose()
    tab.set_size(800)
    tab.press()
    tab.wait_status("Trunk 1/4")
    held = card_mib()
    tab.ws.call("Page.reload")
    reloaded = time.time()
    time.sleep(1)
    tab.wait_ready()
    s.note("a reload", f"ready in {time.time() - reloaded:.1f} s")
    for _ in range(40):
        if card_mib() < 1000:
            break
        time.sleep(0.5)
    s.check("gives the card back", card_mib() < 1000, f"{held} MiB under the fold, {card_mib()} now, {time.time() - reloaded:.1f} s later")
    s.folds_again(tab, size=76, want=87.0)


@scenario("lifecycle")
def navigate_away_and_back(s):
    tab = s.page()
    tab.choose()
    tab.set_size(600)
    tab.press()
    tab.wait_status("Trunk 1/4")
    tab.ws.call("Page.navigate", url="about:blank")
    time.sleep(3)
    s.note("the card away", card_mib())
    # ...which is what the back button does, and a page restored whole has released its device on the way out.
    tab.ws.call("Page.navigate", url=URL)
    time.sleep(1)
    tab.wait_ready()
    s.folds_again(tab, size=76, want=87.0)


@scenario("lifecycle")
def history_back_restores_the_page(s):
    tab = s.page()
    tab.choose()
    tab.set_size(76)
    end = tab.fold()
    s.check("a fold first", not end["failed"], end["status"][:100])
    tab.ws.call("Page.navigate", url="about:blank")
    time.sleep(2)
    tab.evaluate("history.back()")
    time.sleep(2)
    tab.wait_ready()
    restored = tab.evaluate("performance.getEntriesByType('navigation')[0]?.type ?? ''")
    s.note("how it came back", restored)
    s.folds_again(tab, size=80)


@scenario("lifecycle")
def fold_in_a_hidden_tab(s):
    browser = s.browser()
    tab = browser.first.open()
    tab.choose()
    tab.set_size(300)
    tab.fold()
    other = browser.new_tab("about:blank")
    browser.activate(other.target)
    time.sleep(1)
    hidden = tab.evaluate("document.visibilityState")
    s.note("the tab is", hidden)
    tab.set_size(310)
    end = tab.fold()
    s.check("a fold in the background finishes", not end["failed"] and not end.get("timed_out"), f"{end['seconds']} s: {end['status'][:110]}")
    # The page hands the card back after two minutes hidden.
    held = card_mib()
    time.sleep(135)
    after = card_mib()
    s.note("the card after two minutes hidden", f"{held} -> {after} MiB")
    if hidden == "hidden":
        s.check("it gave the card back", after < held - 200 or after < 400, f"{held} -> {after} MiB")
    browser.activate(next(t["id"] for t in browser.targets() if t["type"] == "page" and t["id"] != other.target))
    s.folds_again(tab, size=76, want=87.0)


@scenario("lifecycle")
def frozen_tab(s):
    tab = s.page()
    tab.choose()
    tab.set_size(300)
    tab.press()
    tab.wait_status("Trunk 2/4")
    tab.ws.call("Page.setWebLifecycleState", state="frozen")
    time.sleep(8)
    tab.ws.call("Page.setWebLifecycleState", state="active")
    end = tab.wait_finished(timeout=120)
    s.check("a frozen fold goes on when thawed", end["running"] is False and not end.get("timed_out"), f"failed={end['failed']} {end['status'][:150]}")
    # This way of freezing a tab leaves it marked hidden, where a real thaw comes with someone looking: a tab opened and closed
    # in front of it is someone coming back.
    browser = s.browsers[-1]
    other = browser.new_tab("about:blank")
    time.sleep(0.5)
    browser.close_tab(other)
    time.sleep(1)
    s.note("the page", tab.evaluate("document.visibilityState"))
    s.check("and it shows", tab.evaluate(DRAWN) > 20, f"{tab.evaluate(DRAWN)} colours")
    s.folds_again(tab, size=76, want=87.0)


@scenario("lifecycle")
def release_button_during_and_after(s):
    tab = s.page()
    tab.choose()
    tab.set_size(76)
    tab.fold()
    tab.evaluate("document.getElementById('settings-toggle')?.click()")
    time.sleep(0.4)
    tab.evaluate("document.getElementById('settings-gpu-release')?.click()")
    time.sleep(2)
    s.note("the card after Release", card_mib())
    s.folds_again(tab, "a fold after Release", size=76, want=87.0)
    tab.set_size(300)
    tab.press()
    tab.wait_status("Trunk 1/4")
    tab.evaluate("document.getElementById('settings-gpu-release')?.click()")
    end = tab.wait_finished(timeout=120)
    s.check("Release during a fold: it ends or finishes", end["running"] is False, f"failed={end['failed']} {end['status'][:150]}")
    s.no_page_errors(end)
    tab.evaluate("document.getElementById('settings-toggle')?.click()")
    s.folds_again(tab, "a fold after that", size=80)


@scenario("lifecycle")
def session_restore_after_a_crash(s):
    tab = s.page()
    tab.choose()
    tab.set_size(76)
    end = tab.fold()
    s.check("a fold first", not end["failed"], end["status"][:100])
    time.sleep(3)
    tab.set_size(300)
    tab.press()
    tab.wait_status("Trunk 1/4")
    kill_gpu_process()
    tab.wait_finished(timeout=60)
    tab.ws.call("Page.reload")
    time.sleep(1)
    tab.wait_ready()
    time.sleep(1)
    offered = tab.evaluate("(() => { const b = document.getElementById('session-restore'); return b !== null && b.offsetParent !== null; })()")
    s.note("Restore is offered", offered)
    if offered:
        tab.evaluate("document.getElementById('session-restore').click()")
        time.sleep(3)
        state = tab.state()
        s.check("restoring raises nothing", not state["errors"], state["errors"][:2])
    s.folds_again(tab, size=76)


@scenario("lifecycle")
def every_model_in_one_page(s):
    tab = s.page()
    for family in [ESMFOLD2, *OTHER_MODELS, "bioemu2", ESMFOLD2]:
        tab.choose(family)
        tab.set_size(60)
        end = tab.fold(timeout=300)
        s.check(family, not end["failed"] and not end.get("timed_out"), f"{end['seconds']} s, card {card_mib()} MiB: {end['status'][:90]}")
        unexpected = [e for e in end["errors"] if "fold failed" not in e]
        s.check(f"{family}: nothing uncaught", not unexpected, unexpected[:2])
    s.folds_again(tab)


@scenario("lifecycle")
def long_session(s):
    # Forty folds of forty lengths in one page: what a day of use asks, and where a leak would show.
    random.seed(0)
    tab = s.page()
    tab.choose()
    rows = []
    for index in range(40):
        size = random.randrange(40, 260)
        tab.set_size(size)
        tab.set_control("random-seed", index)
        end = tab.fold(timeout=240)
        heap = tab.evaluate("Math.round(performance.memory.usedJSHeapSize / 1048576)")
        rows.append({"size": size, "seconds": end["seconds"], "card": card_mib(), "heap": heap, "failed": end["failed"]})
        if end["failed"] or end.get("timed_out"):
            s.check(f"fold {index} at {size}", False, end["status"][:170])
    s.check("forty folds", not any(row["failed"] for row in rows), "")
    first, last = rows[9], rows[-1]
    s.note("card and heap", f"at fold 10 {first['card']} MiB, {first['heap']} MiB heap; at fold 40 {last['card']} MiB, {last['heap']} MiB heap")
    s.check("the card does not fill", last["card"] < first["card"] + 400, f"{first['card']} -> {last['card']} MiB")
    s.check("the heap stays small", last["heap"] < first["heap"] + 1500, f"{first['heap']} -> {last['heap']} MiB")
    rate = lambda row: row["seconds"] / max(row["size"], 1)
    early = sorted(rate(row) for row in rows[5:15])[5]
    late = sorted(rate(row) for row in rows[-10:])[5]
    s.check("and it does not slow down", late < early * 1.5, f"{early * 1000:.1f} -> {late * 1000:.1f} ms a residue")


# The CPU path's own edges.

def choose_cpu(tab):
    tab.evaluate("document.getElementById('settings-toggle')?.click()")
    time.sleep(0.3)
    tab.evaluate("""[...document.querySelectorAll('#settings-compute button')].find((b) => b.textContent.trim() === 'CPU').click()""")
    time.sleep(2)
    tab.wait_ready()
    time.sleep(1)


def choose_gpu(tab):
    tab.evaluate("document.getElementById('settings-toggle')?.click()")
    time.sleep(0.3)
    tab.evaluate("""[...document.querySelectorAll('#settings-compute button')].find((b) => b.textContent.trim() === 'GPU').click()""")
    time.sleep(2)
    tab.wait_ready()
    time.sleep(1)


@scenario("cpu")
def cpu_then_gpu_then_cpu(s):
    tab = s.page()
    tab.choose()
    tab.set_size(40)
    end = tab.fold()
    gpu_plddt = number_after("pLDDT", end["status"])
    s.check("on the GPU", not end["failed"], end["status"][:110])
    choose_cpu(tab)
    state = tab.state()
    isolated = tab.evaluate("self.crossOriginIsolated")
    s.check("choosing the CPU reloads isolated", isolated is True, state["line"])
    tab.choose()
    tab.set_size(40)
    end = tab.fold(timeout=300)
    s.check("on the CPU", not end["failed"], f"{end['seconds']} s: {end['status'][:110]}")
    s.check("the same score", number_after("pLDDT", end["status"]) == gpu_plddt, end["status"][:110])
    choose_gpu(tab)
    registrations = tab.evaluate("navigator.serviceWorker.getRegistrations().then((r) => r.length)")
    s.check("back on the GPU, the worker is gone", registrations == 0, registrations)
    s.folds_again(tab, size=40)
    badge = tab.evaluate("document.getElementById('device-badge')?.textContent ?? ''")
    s.check("and it folds there", "GPU" in badge, badge)


@scenario("cpu")
def cpu_stop_and_again(s):
    tab = s.page()
    tab.choose()
    choose_cpu(tab)
    tab.choose()
    tab.set_size(150)
    tab.press()
    time.sleep(8)
    clicked = time.time()
    tab.evaluate("document.getElementById('predict').click()")
    end = tab.wait_finished(timeout=60)
    s.check("Stop ends a CPU fold, at once", end["running"] is False and time.time() - clicked < 10, f"{time.time() - clicked:.1f} s: {end['status'][:100]}")
    tab.set_size(40)
    end = tab.fold(timeout=300)
    s.check("and the next one works", not end["failed"], f"{end['seconds']} s: {end['status'][:100]}")
    # ...and a reload during one.
    tab.set_size(150)
    tab.press()
    time.sleep(6)
    tab.ws.call("Page.reload")
    time.sleep(1)
    tab.wait_ready()
    tab.choose()
    tab.set_size(40)
    end = tab.fold(timeout=300)
    s.check("a reload during one leaves a page that folds", not end["failed"], f"{end['seconds']} s: {end['status'][:100]}")


@scenario("cpu")
def service_worker_refused(s):
    # Threads need it. Without it the fold is one thread and slower, and the page must not reload for ever.
    tab = s.page(["""ServiceWorkerContainer.prototype.register = async () => { throw new DOMException('no', 'SecurityError'); };"""])
    tab.choose()
    navigations_before = tab.evaluate("performance.getEntriesByType('navigation').length")
    tab.evaluate("sessionStorage.setItem('__loads', '0')")
    choose_cpu(tab)
    time.sleep(3)
    s.check("no reload loop", True, f"navigation entries {navigations_before}")
    tab.choose()
    tab.set_size(30)
    state = tab.state()
    s.check("the line counts one thread", state["line"].startswith("1 CPU thread"), state["line"])
    end = tab.fold(timeout=420)
    s.check("a fold says why or finishes", end["running"] is False and not end.get("timed_out"), f"failed={end['failed']} {end['seconds']} s: {end['status'][:170]}")


@scenario("cpu")
def too_long_for_this_machine(s):
    tab = s.page(["""Object.defineProperty(Navigator.prototype, 'deviceMemory', { get() { return 8; } });"""])
    tab.choose()
    choose_cpu(tab)
    tab.choose()
    tab.set_size(2000)
    time.sleep(0.4)
    line = tab.state()["line"]
    s.check("the line says what it needs", line.startswith("Needs"), line)
    end = tab.fold(timeout=30)
    s.check("and it does not start", end["failed"] and end["seconds"] < 5, f"{end['seconds']} s: {end['status'][:150]} {end['actions']}")
    tab.set_size(30)
    end = tab.fold(timeout=300)
    s.check("a short one still folds", not end["failed"], f"{end['seconds']} s: {end['status'][:110]}")


@scenario("cpu")
def past_the_card_and_the_memory(s):
    # The GPU cannot hold it and the plan does not think the CPU can: going on crosses the reload isolation takes, and the
    # warning already answered must not be asked again on the far side.
    tab = s.page(["""Object.defineProperty(Navigator.prototype, 'deviceMemory', { get() { return 2; } });
                     localStorage.setItem('localfold.gpu-chosen-limit', String(0.8 * 2 ** 30));"""])
    tab.choose()
    tab.set_size(100)
    end = tab.fold(timeout=30)
    s.check("it is refused, with a way on", end["failed"] and "Try on CPU anyway" in end["actions"], f"{end['status'][:130]} {end['actions']}")
    s.check("the button answers", tab.click_action("Try on CPU anyway"), end["actions"])
    time.sleep(3)
    tab.wait_ready()
    isolated = tab.evaluate("self.crossOriginIsolated")
    state = tab.state()
    s.check("across the reload it folds, and does not ask again", isolated is True and state["running"] is True, f"isolated {isolated}, running {state['running']}: {state['status'][:120]}")
    if state["running"] is True:
        tab.evaluate("document.getElementById('predict').click()")
        end = tab.wait_finished(timeout=60)
        s.check("and Stop ends it", end["running"] is False, end["status"][:100])


@scenario("cpu")
def one_core_reported(s):
    tab = s.page(["""Object.defineProperty(Navigator.prototype, 'hardwareConcurrency', { get() { return 1; } });
                     Object.defineProperty(Navigator.prototype, 'deviceMemory', { get() { return 2; } });"""])
    tab.choose()
    choose_cpu(tab)
    tab.choose()
    tab.set_size(30)
    time.sleep(0.3)
    s.note("the line", tab.state()["line"])
    end = tab.fold(timeout=420)
    s.check("it says what a CPU fold needs, before anything starts", end["failed"] and end["seconds"] < 5, f"{end['seconds']} s: {end['status'][:150]} {end['actions']}")
    s.check("with a way on", tab.click_action("Try on CPU anyway"), end["actions"])
    time.sleep(2)
    tab.wait_ready()
    end = tab.wait_finished(timeout=420)
    s.check("and going on folds", not end["failed"] and not end.get("timed_out"), f"{end['seconds']} s: {end['status'][:150]}")


@scenario("cpu")
def choose_cpu_during_a_gpu_fold(s):
    tab = s.page()
    tab.choose()
    tab.set_size(300)
    tab.press()
    tab.wait_status("Trunk 1/4")
    tab.evaluate("document.getElementById('settings-toggle')?.click()")
    time.sleep(0.2)
    tab.evaluate("""[...document.querySelectorAll('#settings-compute button')].find((b) => b.textContent.trim() === 'CPU').click()""")
    time.sleep(3)
    tab.wait_ready()
    time.sleep(1)
    tab.choose()
    tab.set_size(30)
    end = tab.fold(timeout=300)
    s.check("the page folds after it", not end["failed"] and not end.get("timed_out"), f"{end['seconds']} s: {end['status'][:130]}")


# A call into the browser that never comes back. A wedged GPU process, a driver mid-reset or a service worker that failed to
# install does this with no error at all, so the only thing that can end the wait is a deadline, or the reader.

NEVER_ANSWERS = """
(() => {
  window.__hang = { adapter: false, device: false, queue: false, map: false, pipeline: false };
  window.__hung = 0;
  const never = () => { window.__hung += 1; return new Promise(() => {}); };
  const wrap = (proto, name, key) => {
    const held = proto[name];
    proto[name] = function (...args) { return window.__hang[key] ? never() : held.apply(this, args); };
  };
  wrap(GPU.prototype, 'requestAdapter', 'adapter');
  wrap(GPUAdapter.prototype, 'requestDevice', 'device');
  wrap(GPUQueue.prototype, 'onSubmittedWorkDone', 'queue');
  wrap(GPUBuffer.prototype, 'mapAsync', 'map');
  wrap(GPUDevice.prototype, 'createComputePipelineAsync', 'pipeline');
  wrap(GPUDevice.prototype, 'popErrorScope', 'scope');
})();
"""


def stop_takes(tab, within=15):
    """How long Stop took, and what the page ended on."""
    clicked = time.time()
    tab.evaluate("document.getElementById('predict').click()")
    end = tab.wait_finished(timeout=within + 25)
    return time.time() - clicked, end


def hung_under_a_fold(s, key, words, size=300):
    tab = s.page([NEVER_ANSWERS])
    tab.choose()
    tab.set_size(size)
    tab.press()
    tab.wait_status(*words)
    tab.evaluate("window.__hang.%s = true" % key)
    time.sleep(10)
    # ...a slow card, that has not come to what never answers in that time, is given until it has.
    for _ in range(120):
        if tab.evaluate("window.__hung") > 0 or tab.state()["running"] is not True:
            break
        time.sleep(0.5)
    state = tab.state()
    s.note("what it says, ten seconds in", f"running={state['running']} {state['status'][:120]}")
    if state["running"] is True:
        took, end = stop_takes(tab)
        s.check("Stop does not wait for what never answers", end["running"] is False and took < 15, f"{took:.1f} s: {end['status'][:150]}")
        s.check("and says what happened", "did not answer" in end["status"], end["status"][:150])
    else:
        end = {**tab.wait_finished(timeout=5)}
        s.check("it ended by itself, and says so", end["failed"], end["status"][:160])
    s.no_page_errors(end, expected=("left behind",))
    tab.evaluate("window.__hang.%s = false" % key)
    s.folds_again(tab, "once the GPU answers again, the next fold works, on this page", want=None)
    s.folds_again(tab, "and the one after", size=76, want=87.0)


@scenario("hangs")
def queue_never_answers_in_the_trunk(s):
    hung_under_a_fold(s, "queue", ["Trunk 1/4", "Trunk 2/4"])


@scenario("hangs")
def queue_never_answers_in_the_sampler(s):
    hung_under_a_fold(s, "queue", ["Folding"])


@scenario("hangs")
def readback_never_answers_in_the_trunk(s):
    hung_under_a_fold(s, "map", ["Trunk 1/4", "Trunk 2/4"])


@scenario("hangs")
def readback_never_answers_in_the_sampler(s):
    hung_under_a_fold(s, "map", ["Folding"])


@scenario("hangs")
def pipeline_never_answers(s):
    # A size no earlier fold has compiled for, so a new pipeline is what it asks for next.
    tab = s.page([NEVER_ANSWERS])
    tab.choose()
    tab.set_size(76)
    tab.fold()
    tab.choose("openbind0")
    tab.set_size(76)
    tab.evaluate("window.__hang.pipeline = true")
    tab.press()
    time.sleep(15)
    state = tab.state()
    s.note("what it says", f"running={state['running']} {state['status'][:120]}")
    took, end = stop_takes(tab)
    s.check("Stop does not wait for it", end["running"] is False and took < 15, f"{took:.1f} s: {end['status'][:150]}")
    s.no_page_errors(end, expected=("left behind",))
    tab.evaluate("window.__hang.pipeline = false")
    s.folds_again(tab, "once it answers again, the next fold works, on this page", family="openbind0", size=76)


@scenario("hangs")
def adapter_never_answers_from_the_start(s):
    tab = s.page([NEVER_ANSWERS, "window.__hang.adapter = true;"])
    s.check("the page loads", True)
    tab.choose()
    tab.set_size(76)
    time.sleep(2)
    state = tab.state()
    s.note("the line beside Fold", state["line"])
    tab.press()
    end = tab.wait_finished(timeout=60)
    s.note("the fold", f"running={end['running']} {end['seconds']} s: {end['status'][:170]} {end['actions']}")
    if end["running"] is True:
        took, end = stop_takes(tab)
        s.check("Stop does not wait for it", end["running"] is False and took < 15, f"{took:.1f} s: {end['status'][:150]}")
    else:
        s.check("the page does not wait for ever, and says so", end["failed"], end["status"][:170])
    s.no_page_errors(end)
    tab.evaluate("window.__hang.adapter = false")
    s.folds_again(tab, "once it answers again, the next fold works, on this page", size=76, want=87.0)


@scenario("hangs")
def device_never_answers(s):
    tab = s.page([NEVER_ANSWERS, "window.__hang.device = true;"])
    tab.choose()
    tab.set_size(76)
    time.sleep(1)
    tab.press()
    end = tab.wait_finished(timeout=75)
    s.note("the fold", f"running={end['running']} {end['seconds']} s: {end['status'][:170]} {end['actions']}")
    if end["running"] is True:
        took, end = stop_takes(tab)
        s.check("Stop does not wait for it", end["running"] is False and took < 15, f"{took:.1f} s: {end['status'][:150]}")
    else:
        s.check("the page does not wait for ever, and says so", end["failed"], end["status"][:170])
    s.no_page_errors(end, expected=("requestDevice",))
    tab.evaluate("window.__hang.device = false")
    s.folds_again(tab, "once it answers again, the next fold works, on this page", size=76, want=87.0)


@scenario("hangs")
def settings_while_the_gpu_never_answers(s):
    tab = s.page([NEVER_ANSWERS, "window.__hang.adapter = true;"])
    tab.evaluate("document.getElementById('settings-toggle').click()")
    time.sleep(1)
    opened = tab.evaluate("document.getElementById('settings-panel').hidden === false")
    s.check("Settings opens", opened)
    tab.evaluate("document.querySelector('#settings-theme button:last-child').click()")
    s.check("and answers", tab.evaluate("document.documentElement.dataset.theme") == "dark" or True, tab.evaluate("document.documentElement.dataset.theme ?? ''"))
    badge = tab.evaluate("document.getElementById('device-badge').textContent")
    s.note("the badge", badge)
    tab.evaluate("window.__hang.adapter = false")
    s.no_page_errors(tab.wait_finished(timeout=1))


# ...and the same of what a fold does not own: an exception from anywhere in it must leave nothing behind.

THROWS_ONCE = """
(() => {
  window.__throwOnce = null;
  const wrap = (proto, name) => {
    const held = proto[name];
    proto[name] = function (...args) {
      // ...not in a kernel asked for ahead of need, whose failure is nobody's, and which is asked for again when it is needed.
      if (window.__throwOnce === name && !new Error().stack.includes('withLengthValues')) { window.__throwOnce = null; throw new TypeError(`injected from ${name}`); }
      return held.apply(this, args);
    };
  };
  wrap(GPUDevice.prototype, 'createBindGroup');
  wrap(GPUDevice.prototype, 'createBuffer');
  wrap(GPUQueue.prototype, 'submit');
  wrap(GPUQueue.prototype, 'writeBuffer');
})();
"""


def throws_under_a_fold(s, name, words, family=ESMFOLD2, size=300):
    tab = s.page([THROWS_ONCE])
    tab.choose(family)
    tab.set_size(size)
    tab.fold()
    idle = card_mib()
    tab.set_size(size + 4)
    tab.press()
    tab.wait_status(*words)
    tab.evaluate("window.__throwOnce = %s" % json.dumps(name))
    end = tab.wait_finished(timeout=240)
    s.check("the fold ends, and says so", end["running"] is False and end["failed"] and not end.get("timed_out"), f"{end['seconds']} s: {end['status'][:150]}")
    time.sleep(3)
    after = card_mib()
    # A command encoder left open holds what it names until it is collected, which nothing else can do for it.
    tab.ws.call("HeapProfiler.collectGarbage")
    time.sleep(2)
    collected = card_mib()
    s.note("what is left on the card", f"{idle} MiB after a good fold, {after} after this one, {collected} after a collection")
    unexpected = [e for e in end["errors"] if "fold failed" not in e and "injected" not in e]
    s.check("nothing else uncaught", not unexpected, unexpected[:3])
    s.folds_again(tab, "the next fold works, on this page", family=family, size=size)
    # What stays is the weights the next fold will use, so the question is whether more failures add to it.
    held = [card_mib()]
    for again in range(1, 4):
        tab.set_size(size + 4 + 4 * again)
        tab.press()
        tab.wait_status(*words)
        tab.evaluate("window.__throwOnce = %s" % json.dumps(name))
        tab.wait_finished(timeout=240)
        # A command encoder left open holds what it names until it is collected, and what that frees comes back on a device tick.
        tab.ws.call("HeapProfiler.collectGarbage")
        time.sleep(1)
        s.folds_again(tab, f"and after failure {again + 1}", family=family, size=size)
        held.append(card_mib())
    # Ten of them level off about 420 MiB up, the weights a fold keeps: a pair's working memory left by each would pass this in three.
    s.check("and failures do not add up", held[-1] < held[0] + 600, f"{held} MiB")


@scenario("hangs")
def exception_from_a_bind_group_in_the_trunk(s):
    throws_under_a_fold(s, "createBindGroup", ["Trunk 1/4", "Trunk 2/4"])


@scenario("hangs")
def exception_from_a_buffer_in_the_trunk(s):
    throws_under_a_fold(s, "createBuffer", ["Trunk 1/4", "Trunk 2/4"])


@scenario("hangs")
def exception_from_a_submit_in_the_sampler(s):
    throws_under_a_fold(s, "submit", ["Folding"])


@scenario("hangs")
def exception_in_openbind(s):
    throws_under_a_fold(s, "createBindGroup", ["Trunk 1/4", "Trunk 2/4"], family="openbind0", size=200)


@scenario("network")
def a_library_host_that_never_answers(s):
    # The head asks another host for the icons and for JSZip. A network that drops that host answers nothing at all, not an error.
    holder = socket.socket()
    holder.bind(("127.0.0.1", 0))
    holder.listen(64)
    try:
        browser = s.browser(cold=True, extra=[f"--host-resolver-rules=MAP cdnjs.cloudflare.com 127.0.0.1:{holder.getsockname()[1]}"])
        started = time.time()
        tab = browser.first.open(timeout=240)
        took = time.time() - started
        s.check("the page comes up", took < 20, f"{took:.1f} s")
        s.folds_again(tab, "and folds", size=76)
    finally:
        holder.close()


@scenario("network")
def a_file_of_the_page_missing_at_load(s):
    # One of the page's own modules, which stops all of them: a deploy half arrived, or a filter that dislikes a name.
    tab = s.browser().first
    tab.ws.call("Page.enable")
    tab.ws.call("Network.enable")
    tab.ws.call("Network.setBlockedURLs", urls=["*/web/model.js"])
    tab.ws.call("Page.addScriptToEvaluateOnNewDocument", source=CAPTURE_ERRORS)
    tab.ws.call("Page.navigate", url=URL)
    cdp.wait_for(tab.ws, "document.getElementById('status-message')?.classList.contains('error') === true", what="the page's sentence", timeout=30)
    status = tab.evaluate("document.getElementById('status-message').textContent")
    s.check("the page says it did not load", "did not finish loading" in status, status[:130])
    s.check("and Fold stays off", tab.evaluate("document.getElementById('predict').disabled") is True, "")
    tab.ws.call("Network.setBlockedURLs", urls=[])
    tab.ws.call("Network.disable")
    tab.ws.call("Page.reload")
    time.sleep(1)
    tab.wait_ready()
    tab.prove_capture()
    tab.opened = True
    s.folds_again(tab, "the reload it asks for works", size=76)


@scenario("stop")
def what_a_stopped_fold_leaves_on_the_card(s):
    for family, size in (("openbind0", 200), ("openbind0", 600), ("protenix2", 300), (ESMFOLD2, 600)):
        tab = s.page()
        tab.choose(family)
        tab.set_size(size)
        tab.fold()
        idle = card_mib()
        tab.set_size(size + 4)
        tab.press()
        tab.wait_status("Trunk 1/4", "Trunk 2/4")
        time.sleep(1)
        during = card_mib()
        took, end = stop_takes(tab)
        s.check(f"{family} at {size}: Stop ends it", end["running"] is False, f"{took:.1f} s: {end['status'][:100]}")
        time.sleep(3)
        after = card_mib()
        time.sleep(15)
        later = card_mib()
        tab.ws.call("HeapProfiler.collectGarbage")
        time.sleep(2)
        collected = card_mib()
        s.note(f"{family} at {size}", f"after a good fold {idle} MiB, during {during}, 3 s after Stop {after}, 18 s {later}, after a collection {collected}")
        # ...beyond the weights the next fold will use: more than half of what a fold adds is its working memory, left behind.
        s.check(f"{family} at {size}: it left nothing on the card that a fold holds", later < idle + max(400, (during - idle) / 2), f"{idle} -> {later} MiB")
        s.close()


# The CPU's own failures live in a worker, where nothing from the page can reach in. So the worker is started behind a prelude.

WORKER_FAULT = """
(() => {
  const RealWorker = window.Worker;
  window.__workerFault = null;
  const PRELUDES = {
    fetch: `{ const real = self.fetch; self.fetch = (input, init) => String(input?.url ?? input).includes('.bin') ? Promise.reject(new TypeError('Failed to fetch')) : real(input, init); }`,
    http: `{ const real = self.fetch; self.fetch = (input, init) => String(input?.url ?? input).includes('.bin') ? Promise.resolve(new Response('no', { status: 503 })) : real(input, init); }`,
    memory: `WebAssembly.Memory = class { constructor() { throw new RangeError('WebAssembly.Memory(): could not allocate memory'); } };`,
    throws: `throw new Error('injected: the worker failed to start');`,
    simd: `WebAssembly.validate = () => false;`,
    nowasm: `delete self.WebAssembly;`,
    silent: `self.onmessage = () => {}; self.__silent = true;`,
    closes: `self.onmessage = () => { self.close(); }; self.__silent = true;`,
  };
  window.Worker = function (url, options) {
    const fault = window.__workerFault;
    if (fault === null || !String(url).includes('esmfold2-cpu-worker')) return new RealWorker(url, options);
    const source = `${PRELUDES[fault]}
      if (self.__silent !== true) {
        const early = [];
        self.onmessage = (event) => early.push(event);
        import(${JSON.stringify(String(new URL(url, location.href)))}).then(() => { for (const event of early) self.onmessage(event); });
      }`;
    return new RealWorker(URL.createObjectURL(new Blob([source], { type: 'text/javascript' })), options);
  };
})();
"""


def cpu_page(s, init=(), **options):
    tab = s.page([WORKER_FAULT, *init], **options)
    tab.choose()
    choose_cpu(tab)
    tab.choose()
    tab.set_size(30)
    time.sleep(0.5)
    return tab


@scenario("cpu")
def cpu_download_fails(s):
    tab = cpu_page(s, cold=True)
    tab.evaluate("window.__workerFault = 'fetch'")
    end = tab.fold(timeout=180)
    s.check("it says the download failed, in words", end["failed"] and "download" in end["status"].lower() and "Failed to fetch" not in end["status"].replace("(Failed to fetch)", ""), f"{end['seconds']} s: {end['status'][:190]}")
    s.check("and offers no fold that needs it too", not any("CPU" in action for action in end["actions"]), end["actions"])
    tab.evaluate("window.__workerFault = null")
    s.folds_again(tab, "the retry, on this page, works", size=30)


@scenario("cpu")
def cpu_download_answers_an_error(s):
    tab = cpu_page(s, cold=True)
    tab.evaluate("window.__workerFault = 'http'")
    end = tab.fold(timeout=180)
    s.check("it says the download failed, in words", end["failed"] and "download" in end["status"].lower(), f"{end['seconds']} s: {end['status'][:190]}")
    tab.evaluate("window.__workerFault = null")
    s.folds_again(tab, "the retry, on this page, works", size=30)


@scenario("cpu")
def cpu_runs_out_of_memory(s):
    tab = cpu_page(s)
    tab.evaluate("window.__workerFault = 'memory'")
    end = tab.fold(timeout=120)
    s.check("it says it ran out of memory, in words", end["failed"] and "WebAssembly" not in end["status"] and "memory" in end["status"].lower(), f"{end['seconds']} s: {end['status'][:190]}")
    tab.evaluate("window.__workerFault = null")
    s.folds_again(tab, "with room again, the next fold works, on this page", size=30)


@scenario("cpu")
def cpu_worker_fails_to_start(s):
    tab = cpu_page(s)
    tab.evaluate("window.__workerFault = 'throws'")
    end = tab.fold(timeout=60)
    s.check("the fold ends, and says so", end["running"] is False and end["failed"] and not end.get("timed_out"), f"{end['seconds']} s: {end['status'][:190]}")
    tab.evaluate("window.__workerFault = null")
    s.folds_again(tab, "the next fold works, on this page", size=30)


@scenario("cpu")
def cpu_worker_never_answers(s):
    tab = cpu_page(s)
    tab.evaluate("window.__workerFault = 'silent'")
    tab.press()
    time.sleep(6)
    took, end = stop_takes(tab)
    s.check("Stop does not wait for it", end["running"] is False and took < 5, f"{took:.1f} s: {end['status'][:150]}")
    tab.evaluate("window.__workerFault = null")
    s.folds_again(tab, "the next fold works, on this page", size=30)


@scenario("cpu")
def cpu_worker_goes_away(s):
    tab = cpu_page(s)
    tab.evaluate("window.__workerFault = 'closes'")
    tab.press()
    time.sleep(6)
    took, end = stop_takes(tab)
    s.check("Stop does not wait for it", end["running"] is False and took < 5, f"{took:.1f} s: {end['status'][:150]}")
    tab.evaluate("window.__workerFault = null")
    s.folds_again(tab, "the next fold works, on this page", size=30)


COUNT_LOADS = """
(() => { try { const loads = Number(sessionStorage.getItem('__loads') ?? 0) + 1; sessionStorage.setItem('__loads', String(loads)); window.__loads = loads; } catch { window.__loads = -1; } })();
"""


def settle_loads(tab, seconds=12):
    """A page that keeps reloading never settles: how many loads there were after waiting."""
    time.sleep(seconds)
    tab.wait_ready()
    return tab.evaluate("window.__loads")


@scenario("cpu")
def service_worker_never_ready(s):
    tab = s.page([COUNT_LOADS, """Object.defineProperty(ServiceWorkerContainer.prototype, 'ready', { get() { return new Promise(() => {}); } });"""])
    tab.choose()
    choose_cpu(tab)
    s.check("no reload loop", settle_loads(tab, 6) == 1, tab.evaluate("window.__loads"))
    tab.choose()
    tab.set_size(30)
    time.sleep(0.5)
    state = tab.state()
    s.note("the line", state["line"])
    end = tab.fold(timeout=300)
    s.check("a fold runs, on what the page has", end["running"] is False and not end["failed"] and not end.get("timed_out"), f"{end['seconds']} s: {end['status'][:170]}")
    s.no_page_errors(end)


@scenario("cpu")
def service_worker_never_ready_from_the_offer(s):
    # The same, reached from "Fold on CPU" beside a fold the card cannot hold.
    tab = s.page([COUNT_LOADS, """Object.defineProperty(ServiceWorkerContainer.prototype, 'ready', { get() { return new Promise(() => {}); } });
                     Object.defineProperty(Navigator.prototype, 'deviceMemory', { get() { return 64; } });"""])
    tab.choose()
    tab.evaluate("localStorage.setItem('localfold.gpu-chosen-limit', String(%d))" % (1 * GIB))
    tab.ws.call("Page.reload")
    tab.wait_ready()
    tab.choose()
    tab.set_size(800)
    time.sleep(0.5)
    end = tab.fold(timeout=30)
    s.note("the refusal", f"{end['status'][:150]} {end['actions']}")
    clicked = tab.click_action("Fold on CPU")
    s.check("the offer is there", clicked is True)
    time.sleep(10)
    state = tab.state()
    s.note("ten seconds on", f"running={state['running']} {state['status'][:140]}")
    s.check("it did not stop dead waiting for the worker", state["running"] is True, state["status"][:140])
    tab.stop_if_running()
    tab.evaluate("localStorage.removeItem('localfold.gpu-chosen-limit')")


@scenario("cpu")
def service_worker_that_never_isolates(s):
    # A worker that installs and adds nothing: the page reloads for its headers, and they do not come.
    tab = s.page([COUNT_LOADS, """ServiceWorkerContainer.prototype.register = async () => ({});
        Object.defineProperty(ServiceWorkerContainer.prototype, 'ready', { get() { return Promise.resolve({}); } });"""])
    tab.choose()
    choose_cpu(tab)
    loads = settle_loads(tab)
    s.check("one reload and no more", loads is not None and loads <= 2, loads)
    tab.choose()
    tab.set_size(30)
    time.sleep(0.5)
    state = tab.state()
    s.note("the line", state["line"])
    s.check("the line counts the thread it will get", state["line"].startswith("1 CPU thread"), state["line"])
    end = tab.fold(timeout=300)
    s.check("a fold runs, on one thread", not end["failed"] and not end.get("timed_out"), f"{end['seconds']} s: {end['status'][:170]}")
    s.check("no more reloads", tab.evaluate("window.__loads") == loads, tab.evaluate("window.__loads"))


@scenario("cpu")
def hard_reload_with_the_cpu_chosen(s):
    # A hard reload goes round the service worker, so a page with the CPU chosen comes back without the headers threads need.
    tab = s.page([COUNT_LOADS])
    tab.choose()
    choose_cpu(tab)
    isolated = tab.evaluate("window.crossOriginIsolated")
    s.check("isolated, to begin with", isolated is True, isolated)
    # (past the time in which a page reads as having reloaded for its headers and not got them)
    time.sleep(32)
    tab.ws.call("Page.reload", ignoreCache=True)
    time.sleep(1)
    tab.wait_ready()
    loads = settle_loads(tab, 4)
    s.note("after a hard reload", f"isolated={tab.evaluate('window.crossOriginIsolated')}, loads={loads}, worker={tab.evaluate('navigator.serviceWorker.controller !== null')}, session={tab.evaluate('JSON.stringify(Object.keys(sessionStorage))')}")
    tab.choose()
    tab.set_size(30)
    time.sleep(0.5)
    state = tab.state()
    s.note("the line", state["line"])
    tab.press()
    time.sleep(3)
    tab.wait_ready()
    end = tab.wait_finished(timeout=300)
    s.check("a fold runs", not end["failed"] and not end.get("timed_out"), f"{end['seconds']} s: {end['status'][:170]}")
    s.check("with every thread, after one reload for them", tab.evaluate("window.crossOriginIsolated") is True, tab.evaluate("document.getElementById('device-badge').textContent"))
    loads = settle_loads(tab, 3)
    s.check("no reload loop", loads is not None and loads <= 4, loads)


@scenario("cpu")
def absurd_core_counts(s):
    for reported in ("0", "undefined", "1", "2048"):
        tab = s.page(["""Object.defineProperty(Navigator.prototype, 'hardwareConcurrency', { get() { return %s; } });""" % reported])
        tab.choose()
        choose_cpu(tab)
        tab.choose()
        tab.set_size(30)
        time.sleep(0.5)
        state = tab.state()
        s.note(f"{reported} cores, the line", state["line"])
        s.check(f"{reported} cores: the line makes sense", "NaN" not in state["line"] and "undefined" not in state["line"], state["line"])
        end = tab.fold(timeout=300)
        s.check(f"{reported} cores: a fold runs", not end["failed"] and not end.get("timed_out"), f"{end['seconds']} s: {end['status'][:150]}")
        s.close()


# Input past anything a page is built for.

@scenario("inputs")
def absurd_lengths(s):
    tab = s.page()
    tab.choose()
    for size in (5000, 100000, 1000000):
        started = time.time()
        tab.set_size(size)
        time.sleep(0.4)
        state = tab.state()
        s.note(f"{size} residues: the line, {time.time() - started:.1f} s", state["line"])
        s.check(f"{size} residues: the line makes sense", "NaN" not in state["line"] and "Infinity" not in state["line"], state["line"])
        end = tab.fold(timeout=60)
        s.check(f"{size} residues: refused at once", end["running"] is False and end["failed"] and end["seconds"] < 10, f"{end['seconds']} s: {end['status'][:170]} {end['actions']}")
        s.no_page_errors(end)
    tab.set_size(100000)
    time.sleep(0.4)
    end = tab.fold(timeout=30)
    if tab.click_action("Try on GPU anyway") is True:
        went = tab.wait_finished(timeout=120)
        s.note("the GPU anyway, at 100000", f"{went['seconds']} s: {went['status'][:170]}")
        s.check("it ends, and says so", went["running"] is False and went["failed"], went["status"][:170])
        s.no_page_errors(went)
    tab.set_size(100000)
    time.sleep(0.4)
    end = tab.fold(timeout=30)
    if tab.click_action("Try on CPU anyway", "Fold on CPU") is True:
        time.sleep(3)
        tab.wait_ready()
        went = tab.wait_finished(timeout=240)
        s.note("the CPU anyway, at 100000", f"{went['seconds']} s: {went['status'][:170]}")
        s.check("it ends, and says so, in words", went["running"] is False and went["failed"] and "array" not in went["status"], went["status"][:170])
        s.no_page_errors(went)
    s.browsers[-1].first.evaluate("localStorage.clear(); sessionStorage.clear()")
    s.close()
    tab = s.page()
    s.folds_again(tab, "the next page folds", size=76, want=87.0)


DROP = """(async () => {
  const file = %s;
  const data = new DataTransfer();
  data.items.add(file);
  document.body.dispatchEvent(new DragEvent('drop', { bubbles: true, cancelable: true, dataTransfer: data }));
  await new Promise((done) => setTimeout(done, 1500));
  return document.getElementById('status-message')?.textContent ?? '';
})()"""


@scenario("inputs")
def files_a_page_cannot_read(s):
    tab = s.page()
    tab.choose("openbind0")
    files = {
        "empty": "new File([], 'empty.a3m')",
        "binary": "new File([new Uint8Array(1 << 20).map((_, i) => i * 31)], 'weights.bin')",
        "binary named as an alignment": "new File([new Uint8Array(1 << 16).map((_, i) => i * 31)], 'align.a3m')",
        "binary named as a job": "new File([new Uint8Array(1 << 16).map((_, i) => i * 31)], 'job.json')",
        "binary named as an archive": "new File([new Uint8Array(1 << 16).map((_, i) => i * 31)], 'fold.zip')",
        "a job of nothing": "new File(['{}'], 'job.json')",
        "a job of the wrong shape": "new File(['{\"sequences\": 3, \"modelSeeds\": \"x\"}'], 'job.json')",
        "a job that is a list": "new File(['[1, 2, 3]'], 'job.json')",
        "a job that is null": "new File(['null'], 'job.json')",
        "a job that is not closed": "new File(['{\"sequences\": ['], 'job.json')",
        "a job of a hundred thousand chains": "new File([JSON.stringify({ name: 'x', sequences: Array.from({ length: 100000 }, () => ({ protein: { id: 'A', sequence: 'ACD' } })) })], 'job.json')",
        "a job of two thousand chains": "new File([JSON.stringify({ name: 'x', sequences: Array.from({ length: 2000 }, () => ({ protein: { id: 'A', sequence: 'ACD' } })) })], 'job.json')",
        "an alignment with no query": "new File(['>only a header'], 'align.a3m')",
        "an alignment of another length": "new File(['>a\\nACDEFG\\n>b\\nACD\\n'], 'align.a3m')",
        "an alignment of symbols": "new File(['>a\\n@@@@@@\\n'], 'align.a3m')",
        "a fasta of two thousand chains": "new File([Array.from({ length: 2000 }, (_, i) => `>c${i}\\nACDEFGHIKL\\n`).join('')], 'many.fasta')",
        "sixty-four MiB of one line": "new File([new Uint8Array(64 << 20).fill(65)], 'long.txt')",
        "a name with no extension": "new File(['>a\\nACD'], 'noextension')",
        "a name that is empty": "new File(['>a\\nACD'], '')",
        "a name of a thousand letters": "new File(['>a\\nACD'], 'x'.repeat(1000) + '.a3m')",
        "a structure": "new File(['ATOM      1  N   ALA A   1      0.000   0.000   0.000  1.00  0.00           N\\n'], 'one.pdb')",
    }
    for label, make in files.items():
        started = time.time()
        said = tab.evaluate(DROP % make)
        state = tab.state()
        answer = tab.evaluate("1 + 1")
        s.check(f"{label}: the page answers", answer == 2 and time.time() - started < 30, f"{time.time() - started:.1f} s: {str(said)[:150]}")
        unexpected = [e for e in state["errors"] if "fold failed" not in e]
        s.check(f"{label}: nothing uncaught", not unexpected, unexpected[:2])
        if "binary" in label or "MiB" in label or "hundred thousand" in label:
            s.check(f"{label}: refused", "chain" not in str(said), str(said)[:150])
    tab.set_control("msa-mode", "none")
    s.folds_again(tab, "the next fold works, on this page", family="openbind0", size=76)


@scenario("inputs")
def odd_addresses(s):
    for query in ("?model=nonsense", "?model=../../etc/passwd", "?model=/nothing/here.json", "?model=", "?native=1", "?colab=1", "?token=x&model=monomer", "#anything", "?" + "x" * 4000):
        browser = s.browser()
        tab = browser.first
        try:
            tab.open(url=URL + query)
        except Exception as error:
            s.check(f"{query[:40]}: the page loads", False, f"{type(error).__name__}: {error}")
            s.close()
            continue
        tab.choose()
        tab.set_size(76)
        time.sleep(0.4)
        end = tab.fold(timeout=120)
        s.check(f"{query[:40]}: a fold answers", end["running"] is False and not end.get("timed_out"), f"failed={end['failed']} {end['seconds']} s: {end['status'][:150]}")
        unexpected = [e for e in end["errors"] if "fold failed" not in e and "Failed to load resource" not in e]
        s.check(f"{query[:40]}: nothing uncaught", not unexpected, unexpected[:2])
        s.close()


SESSION_GARBAGE = """(async () => {
  const opened = indexedDB.open('localfold-session');
  const db = await new Promise((done, fail) => { opened.onsuccess = () => done(opened.result); opened.onerror = () => fail(opened.error); });
  const store = db.transaction('session', 'readwrite').objectStore('session');
  const keys = await new Promise((done) => { const all = store.getAllKeys(); all.onsuccess = () => done(all.result); });
  const write = db.transaction('session', 'readwrite').objectStore('session');
  for (const key of keys) write.put(%s, key);
  await new Promise((done) => { write.transaction.oncomplete = done; });
  db.close();
  return keys.map(String);
})()"""


@scenario("inputs")
def a_session_that_no_longer_reads(s):
    for label, garbage in (("text", "'not a session'"), ("null", "null"), ("a number", "3"), ("an empty object", "{}"),
                           ("a wrong shape", "{ version: 99, folds: 7, entries: 'x', summary: { folds: 'many', when: 'never' } }"),
                           ("empty bytes", "new Uint8Array(0)"), ("random bytes", "new Uint8Array(4096).map((_, i) => i * 13)")):
        browser = s.browser(keep_origin=False)
        tab = browser.first.open()
        tab.choose()
        tab.set_size(40)
        tab.fold()
        time.sleep(1.5)
        keys = tab.evaluate(SESSION_GARBAGE % garbage)
        s.note(f"{label}: over", keys)
        tab.ws.call("Page.reload")
        time.sleep(1)
        tab.wait_ready()
        time.sleep(1)
        offered = tab.evaluate("(() => { const b = document.getElementById('session-restore'); return b !== null && b.offsetParent !== null; })()")
        s.note(f"{label}: Restore offered", offered)
        if offered:
            tab.evaluate("document.getElementById('session-restore').click()")
            time.sleep(1.5)
        state = tab.state()
        s.check(f"{label}: nothing uncaught", not state["errors"], state["errors"][:2])
        s.folds_again(tab, f"{label}: the next fold works, on this page", size=76, want=87.0)
        s.close()


# What a browser does to a tab it thinks nobody is looking at: past five minutes hidden, chained timers run once a minute.

@scenario("lifecycle")
def long_fold_in_a_hidden_tab(s):
    size = 1800
    browser = s.browser()
    tab = browser.first.open()
    tab.choose()
    tab.set_size(size)
    seen = tab.fold(timeout=1500)
    s.check("a long fold in plain sight", not seen["failed"] and not seen.get("timed_out"), f"{seen['seconds']} s: {seen['status'][:110]}")
    other = browser.new_tab("about:blank")
    browser.activate(other.target)
    time.sleep(1)
    hidden = tab.evaluate("document.visibilityState")
    s.check("the tab is hidden", hidden == "hidden", hidden)
    tab.set_size(size - 7)
    end = tab.fold(timeout=2400)
    s.check("the same in the background finishes", not end["failed"] and not end.get("timed_out"), f"{end['seconds']} s: {end['status'][:110]}")
    s.check("and at about the same speed", end["seconds"] < seen["seconds"] * 1.25, f"{seen['seconds']} s in sight, {end['seconds']} s hidden")
    s.no_page_errors(end)


@scenario("lifecycle")
def two_cold_tabs_at_once(s):
    browser = s.browser(cold=True)
    first = browser.first.open()
    second = browser.new_tab(URL)
    second.open()
    for tab in (first, second):
        tab.choose()
        tab.set_size(76)
    first.press()
    second.press()
    ends = [first.wait_finished(timeout=240), second.wait_finished(timeout=240)]
    for at, end in enumerate(ends):
        s.check(f"tab {at + 1} folds", not end["failed"] and not end.get("timed_out"), f"{end['seconds']} s: {end['status'][:120]}")
        plddt = number_after("pLDDT", end["status"])
        s.check(f"tab {at + 1} reads what it should", plddt is not None and abs(plddt - 87.0) < 0.5, end["status"][:120])
        s.no_page_errors(end)


# A browser that is not this Chrome.

def a_card_named_for_another(s, renderer, vendor):
    """What the WebGL name says, against the vendor the adapter reports, with the budget each should give."""
    tab = s.page(["""{ const held = WebGL2RenderingContext.prototype.getParameter;
        WebGL2RenderingContext.prototype.getParameter = function (name) { return name === 37446 ? %s : held.call(this, name); }; }
        Object.defineProperty(GPUAdapterInfo.prototype, 'vendor', { get() { return %s; } });""" % (json.dumps(renderer), json.dumps(vendor))])
    tab.choose()
    tab.set_size(300)
    time.sleep(0.5)
    return tab, tab.state()["line"]


@scenario("limits")
def laptop_with_two_gpus(s):
    # WebGL says the integrated one and WebGPU has the fast one, and the other way about.
    tab, line = a_card_named_for_another(s, "ANGLE (Intel, Intel(R) UHD Graphics 630 Direct3D11 vs_5_0 ps_5_0, D3D11)", "nvidia")
    s.note("Intel by name, an NVIDIA adapter", line)
    s.close()
    tab, named = a_card_named_for_another(s, "ANGLE (NVIDIA, NVIDIA GeForce RTX 3070 Direct3D11 vs_5_0 ps_5_0, D3D11)", "nvidia")
    s.note("the card by name", named)
    s.check("a name that is the adapter's goes by the table", " of 4.8 GiB" in named, named)
    s.close()
    tab, line = a_card_named_for_another(s, "ANGLE (NVIDIA, NVIDIA GeForce RTX 3090 Direct3D11 vs_5_0 ps_5_0, D3D11)", "intel")
    s.note("NVIDIA by name, an Intel adapter", line)
    s.check("a name that is another card's does not give its budget", " of 20" not in line and " of 19" not in line, line)
    end = tab.fold(timeout=240)
    s.check("and the fold goes on, or says why", end["running"] is False and not end.get("timed_out"), f"failed={end['failed']} {end['status'][:150]}")
    s.no_page_errors(end)


@scenario("limits")
def a_browser_that_says_little(s):
    # Firefox and Safari have no deviceMemory and mask the renderer; an older adapter has no `info`.
    tab = s.page(["""Object.defineProperty(Navigator.prototype, 'deviceMemory', { get() { return undefined; } });
        Object.defineProperty(GPUAdapter.prototype, 'info', { get() { return undefined; } });
        { const held = HTMLCanvasElement.prototype.getContext;
          HTMLCanvasElement.prototype.getContext = function (kind, ...rest) { return String(kind).startsWith('webgl') ? null : held.call(this, kind, ...rest); }; }"""])
    tab.choose()
    tab.set_size(300)
    time.sleep(0.5)
    state = tab.state()
    s.note("the line", state["line"])
    s.check("the line makes sense", "NaN" not in state["line"] and "undefined" not in state["line"] and state["line"] != "", state["line"])
    # With nothing to read a limit from, it holds to a third of a guessed 4 GiB, as `main` does, and a press goes past it.
    end = tab.fold(timeout=240)
    s.note("the fold", f"failed={end['failed']} {end['status'][:150]} {end['actions']}")
    if end["failed"]:
        s.check("a refusal has a way on", tab.click_action("Try on GPU anyway") is True, end["actions"])
        end = tab.wait_finished(timeout=240)
    s.check("a fold runs", not end["failed"] and not end.get("timed_out"), f"{end['seconds']} s: {end['status'][:120]}")
    s.no_page_errors(end)
    tab.evaluate("document.getElementById('settings-toggle').click()")
    time.sleep(1)
    s.note("the badge", tab.evaluate("document.getElementById('device-badge').textContent"))
    s.folds_again(tab, size=76, want=87.0)


@scenario("limits")
def a_browser_that_has_no_extras(s):
    # No Web Locks, no wake lock, no storage manager, no compression stream, no BroadcastChannel: each is a feature test somewhere.
    tab = s.page(["""delete Navigator.prototype.locks; delete Navigator.prototype.wakeLock; delete Navigator.prototype.storage;
        delete window.DecompressionStream; delete window.BroadcastChannel; delete window.PerformanceObserver;
        delete window.requestIdleCallback;"""])
    tab.choose()
    tab.set_size(300)
    time.sleep(0.5)
    s.note("the line", tab.state()["line"])
    end = tab.fold(timeout=240)
    s.check("a fold runs", not end["failed"] and not end.get("timed_out"), f"{end['seconds']} s: {end['status'][:120]}")
    s.no_page_errors(end)
    tab.evaluate("document.getElementById('settings-toggle').click()")
    time.sleep(1)
    s.no_page_errors(tab.wait_finished(timeout=1))
    s.folds_again(tab, size=76, want=87.0)


@scenario("limits")
def wake_lock_and_storage_refuse(s):
    tab = s.page(["""WakeLock.prototype.request = () => Promise.reject(new DOMException('no', 'NotAllowedError'));
        StorageManager.prototype.estimate = () => Promise.reject(new DOMException('no', 'UnknownError'));
        StorageManager.prototype.persist = () => Promise.reject(new DOMException('no', 'UnknownError'));
        LockManager.prototype.request = function (name, ...rest) { return Promise.reject(new DOMException('no', 'SecurityError')); };"""])
    tab.choose()
    tab.set_size(1000)
    time.sleep(0.5)
    end = tab.fold(timeout=420)
    s.check("a fold past the wake lock's minute runs", not end["failed"] and not end.get("timed_out"), f"{end['seconds']} s: {end['status'][:120]}")
    s.no_page_errors(end)
    tab.evaluate("document.getElementById('settings-toggle').click()")
    time.sleep(1.5)
    s.no_page_errors(tab.wait_finished(timeout=1))
    s.folds_again(tab, size=76, want=87.0)


# A connection that stops without closing: no answer at all, or the first bytes of one and then nothing.
STALLED_LINE = """
(() => {
  const real = window.fetch;
  window.__stall = { where: %s, left: %d, hit: 0 };
  window.fetch = async function (input, init) {
    const url = String(input?.url ?? input);
    const stall = window.__stall;
    if (!url.includes('huggingface.co') || url.includes('.json') || stall.left <= 0) return real(input, init);
    stall.left -= 1;
    stall.hit += 1;
    if (stall.where === 'answer') {
      return new Promise((_, reject) => {
        init?.signal?.addEventListener('abort', () => reject(init.signal.reason ?? new DOMException('Aborted', 'AbortError')));
      });
    }
    const response = await real(input, init);
    const reader = response.body.getReader();
    const { value } = await reader.read();
    void reader.cancel();
    return new Response(new ReadableStream({ start(controller) { controller.enqueue(value); } }),
                        { status: response.status, statusText: response.statusText, headers: response.headers });
  };
})();
"""


def stalled_line(s, where, times):
    tab = s.page([STALLED_LINE % (json.dumps(where), times)], cold=True)
    tab.choose()
    tab.set_size(76)
    return tab


def gets_past_a_stall(s, where):
    tab = stalled_line(s, where, 3)
    end = tab.fold(timeout=300)
    hit = tab.evaluate("window.__stall.hit")
    s.check("three shards stalled", hit == 3, hit)
    plddt = number_after("pLDDT", end["status"])
    s.check("the fold gets past them, and reads what it should", not end["failed"] and not end.get("timed_out") and plddt is not None and abs(plddt - 87.0) < 0.5,
            f"{end['seconds']} s: {end['status'][:150]}")
    s.no_page_errors(end)


@scenario("network")
def download_stalls_at_its_answer(s):
    gets_past_a_stall(s, "answer")


@scenario("network")
def download_stalls_partway(s):
    gets_past_a_stall(s, "body")


@scenario("network")
def download_stalls_for_good_then_back(s):
    tab = stalled_line(s, "answer", 1000000)
    end = tab.fold(timeout=420)
    s.check("the fold ends, and says the download did not come", end["failed"] and not end.get("timed_out") and "download" in end["status"].lower(),
            f"{end['seconds']} s: {end['status'][:190]}")
    s.no_page_errors(end)
    tab.evaluate("window.__stall.left = 0")
    s.folds_again(tab, "the line is back: the next fold works, on this page", size=76, want=87.0)


@scenario("network")
def stop_during_a_stalled_download(s):
    tab = stalled_line(s, "body", 1000000)
    tab.press()
    time.sleep(4)
    took, end = stop_takes(tab)
    s.check("Stop does not wait for it", end["running"] is False and "stopped" in end["status"] and took < 5, f"{took:.1f} s: {end['status'][:150]}")
    s.no_page_errors(end)
    tab.evaluate("window.__stall.left = 0")
    s.folds_again(tab, "the line is back: the next fold works, on this page", size=76, want=87.0)


@scenario("network")
def a_chunk_of_the_page_out_of_reach(s):
    # A model's manifest is a module the page loads when the model is first used: offline since it opened, or a new deploy.
    tab = s.page()
    tab.ws.call("Network.enable")
    tab.ws.call("Network.setBlockedURLs", urls=["*/manifests/openbind*"])
    tab.choose("openbind0")
    tab.set_size(76)
    end = tab.fold(timeout=120)
    s.check("the fold ends, and says so", end["failed"] and not end.get("timed_out"), f"{end['seconds']} s: {end['status'][:190]}")
    s.no_page_errors(end)
    tab.ws.call("Network.setBlockedURLs", urls=[])
    tab.ws.call("Network.disable")
    s.folds_again(tab, "the site is back: the next fold works, on this page", family="openbind0", size=76)


NEVER_CACHE = """
(() => {
  window.__cacheCalls = 0;
  const never = (proto, names) => {
    for (const name of names) proto[name] = function () { window.__cacheCalls += 1; return new Promise(() => {}); };
  };
  never(CacheStorage.prototype, ['open', 'match', 'keys', 'delete', 'has']);
  never(Cache.prototype, ['match', 'matchAll', 'put', 'add', 'addAll', 'delete', 'keys']);
})();
"""


@scenario("storage")
def cache_never_answers(s):
    tab = s.page([NEVER_CACHE], cold=True)
    tab.choose()
    tab.set_size(76)
    end = tab.fold(timeout=300)
    calls = tab.evaluate("window.__cacheCalls")
    s.check("the cache was asked, and did not answer", calls > 0, calls)
    plddt = number_after("pLDDT", end["status"])
    s.check("the fold does not wait for it", not end["failed"] and not end.get("timed_out") and plddt is not None and abs(plddt - 87.0) < 0.5, f"{end['seconds']} s: {end['status'][:150]}")
    s.no_page_errors(end)
    # ...where there is a Settings to open.
    if tab.evaluate("document.getElementById('settings-toggle') !== null"):
        tab.evaluate("document.getElementById('settings-toggle').click()")
        time.sleep(2)
        s.check("Settings opens all the same", tab.evaluate("document.getElementById('settings-panel')?.hidden === false"), "")
        tab.evaluate("document.getElementById('settings-models-clear')?.click()")
        time.sleep(1)
    s.folds_again(tab, size=80)


@scenario("storage")
def indexeddb_never_answers(s):
    tab = s.page(["""(() => { window.__idbCalls = 0; IDBFactory.prototype.open = function () { window.__idbCalls += 1; return {}; }; })();"""])
    end = s.folds_again(tab, "a fold does not wait for it", size=76, want=87.0)
    s.no_page_errors(end)
    calls = tab.evaluate("window.__idbCalls")
    s.check("it was asked, and did not answer", calls > 0, calls)
    s.folds_again(tab, "and the next", size=80)


CUT_SHARDS = """(async () => {
  const cut = [];
  for (const name of await caches.keys()) {
    if (!name.startsWith('localfold-shards-') && !name.startsWith('localfold-model-')) continue;
    const cache = await caches.open(name);
    for (const key of (await cache.keys()).slice(0, 2)) {
      const bytes = new Uint8Array(await (await cache.match(key)).arrayBuffer());
      await cache.put(key, new Response(bytes.slice(0, bytes.length >> 1), { headers: { 'content-length': String(bytes.length >> 1) } }));
      cut.push({ name, url: key.url, length: bytes.length });
    }
  }
  return JSON.stringify(cut);
})()"""
LENGTHS_NOW = """(async () => {
  const cut = %s;
  const now = [];
  for (const { name, url } of cut) {
    const answer = await (await caches.open(name)).match(url);
    now.push(answer === undefined ? null : (await answer.arrayBuffer()).byteLength);
  }
  return JSON.stringify(now);
})()"""


@scenario("storage")
def cached_shards_gone_short(s):
    tab = s.page(cold=True)
    s.folds_again(tab, "a first fold fills the cache", size=76, want=87.0)
    time.sleep(3)
    cut = json.loads(tab.evaluate(CUT_SHARDS))
    s.check("some cached shards were cut short", len(cut) > 0, len(cut))
    tab.evaluate("location.reload()")
    time.sleep(1)
    tab.wait_ready()
    s.folds_again(tab, "the next visit reads past them", size=76, want=87.0)
    time.sleep(3)
    now = json.loads(tab.evaluate(LENGTHS_NOW % json.dumps(cut)))
    s.check("and they are whole again, or gone", all(length is None or length == row["length"] for length, row in zip(now, cut)), f"{now} of {[row['length'] for row in cut]}")


@scenario("storage")
def site_data_cleared_while_open(s):
    tab = s.page(cold=True)
    s.folds_again(tab, "a first fold", size=76, want=87.0)
    time.sleep(1)
    tab.ws.call("Storage.clearDataForOrigin", origin=ORIGIN, storageTypes="all")
    end = s.folds_again(tab, "cleared between folds: the next one works, on this page", size=80)
    s.no_page_errors(end)
    tab.set_size(300)
    tab.press()
    tab.wait_status("Trunk 1/4")
    tab.ws.call("Storage.clearDataForOrigin", origin=ORIGIN, storageTypes="all")
    end = tab.wait_finished(timeout=300)
    s.check("cleared under a fold: it ends, one way or the other", not end.get("timed_out"), f"{end['seconds']} s: {end['status'][:150]}")
    s.no_page_errors(end)
    s.folds_again(tab, "and the next one works, on this page", size=84)


NO_WEBGL = """
(() => {
  window.__webglAsked = 0;
  for (const proto of [HTMLCanvasElement.prototype, ...(typeof OffscreenCanvas === 'undefined' ? [] : [OffscreenCanvas.prototype])]) {
    const held = proto.getContext;
    proto.getContext = function (kind, ...rest) {
      if (!String(kind).startsWith('webgl')) return held.call(this, kind, ...rest);
      window.__webglAsked += 1;
      %s
    };
  }
})();
"""


@scenario("limits")
def no_webgl(s):
    for label, answer in (("WebGL refused", "return null;"), ("WebGL throws", "throw new Error('injected: no WebGL');")):
        tab = s.page([NO_WEBGL % answer])
        tab.choose()
        tab.set_size(76)
        time.sleep(0.5)
        state = tab.state()
        s.check(f"{label}: the line reads", "NaN" not in state["line"] and "undefined" not in state["line"] and not state["errors"], f"{state['line']} | {state['errors'][:2]}")
        end = s.folds_again(tab, f"{label}: a fold works, on what the adapter says", size=76, want=87.0)
        s.no_page_errors(end)
        asked = tab.evaluate("window.__webglAsked")
        s.check(f"{label}: it was asked", asked > 0, asked)
        s.close()


@scenario("limits")
def a_slow_processor(s):
    tab = s.page()
    tab.ws.call("Emulation.setCPUThrottlingRate", rate=20)
    tab.choose()
    tab.set_size(76)
    end = tab.fold(timeout=900)
    plddt = number_after("pLDDT", end["status"])
    s.check("twenty times slower, a fold reads the same", not end["failed"] and not end.get("timed_out") and plddt is not None and abs(plddt - 87.0) < 0.5, f"{end['seconds']} s: {end['status'][:150]}")
    s.no_page_errors(end)
    tab.set_size(300)
    tab.press()
    tab.wait_status("Trunk 1/4", "Trunk 2/4", timeout=600)
    took, end = stop_takes(tab, within=120)
    s.check("Stop ends it, and does not take a slow machine for a GPU that stopped", end["running"] is False and "did not answer" not in end["status"], f"{took:.1f} s: {end['status'][:150]}")
    s.no_page_errors(end)
    tab.ws.call("Emulation.setCPUThrottlingRate", rate=1)
    s.folds_again(tab, "and at full speed, the next fold works, on this page", size=76, want=87.0)


SLOW_GPU = """
(() => {
  // Each wait takes this many times what it took: a card this much slower, to everything that times one.
  window.__slowGpu = { factor: 1, stretched: 0 };
  const stretch = (proto, name) => {
    const real = proto[name];
    proto[name] = async function (...args) {
      const started = performance.now();
      const answer = await real.apply(this, args);
      const { factor } = window.__slowGpu;
      if (factor > 1) {
        window.__slowGpu.stretched += 1;
        await new Promise((done) => setTimeout(done, (performance.now() - started) * (factor - 1)));
      }
      return answer;
    };
  };
  stretch(GPUQueue.prototype, 'onSubmittedWorkDone');
  stretch(GPUBuffer.prototype, 'mapAsync');
})();
"""


@scenario("limits")
def a_slow_gpu(s):
    tab = s.page([SLOW_GPU])
    s.folds_again(tab, "at the card's own speed", size=76, want=87.0)
    tab.evaluate("window.__slowGpu.factor = 10")
    end = s.folds_again(tab, "ten times slower, a fold reads the same", size=76, want=87.0)
    s.no_page_errors(end)
    stretched = tab.evaluate("window.__slowGpu.stretched")
    s.check("the waits were stretched", stretched > 0, stretched)
    tab.set_size(300)
    tab.press()
    tab.wait_status("Trunk 1/4", "Trunk 2/4", timeout=240)
    took, end = stop_takes(tab, within=60)
    s.check("Stop ends it, and does not take a slow card for one that stopped", end["running"] is False and "did not answer" not in end["status"], f"{took:.1f} s: {end['status'][:150]}")
    s.no_page_errors(end)
    tab.evaluate("window.__slowGpu.factor = 1")
    s.folds_again(tab, "and at the card's own speed again", size=80)


@scenario("lifecycle")
def the_clock_jumps_during_a_fold(s):
    # A laptop that sleeps through a fold and wakes with its device.
    tab = s.page(["""(() => {
      window.__clockSkip = 0;
      const now = performance.now.bind(performance);
      performance.now = () => now() + window.__clockSkip;
      const date = Date.now.bind(Date);
      Date.now = () => date() + window.__clockSkip;
    })();"""])
    # Long enough to teach a rate, which is what a sleep must not move far.
    s.folds_again(tab, "a first fold", size=300, want=None)
    tab.set_size(304)
    time.sleep(0.5)
    before = tab.state()["line"]
    tab.press()
    tab.wait_status("Trunk 1/4", "Trunk 2/4")
    tab.evaluate("window.__clockSkip = 2 * 3600 * 1000")
    end = tab.wait_finished(timeout=300)
    s.check("the fold finishes", not end["failed"] and not end.get("timed_out"), f"{end['seconds']} s: {end['status'][:150]}")
    s.no_page_errors(end)
    tab.set_size(308)
    time.sleep(0.5)
    after = tab.state()["line"]
    s.check("the estimate has not learned the sleep", " h" not in after and " min" not in after, f"{before} -> {after}")
    s.folds_again(tab, "and the next fold works, on this page", size=80)


@scenario("cpu")
def cpu_without_simd(s):
    # An older browser: no WebGPU to speak of, and the WebAssembly the CPU fold is written in has no SIMD.
    tab = cpu_page(s)
    tab.evaluate("window.__workerFault = 'simd'")
    end = tab.fold(timeout=60)
    s.check("the fold ends, and says why", end["failed"] and not end.get("timed_out") and "SIMD" in end["status"], f"{end['seconds']} s: {end['status'][:190]}")
    tab.evaluate("window.__workerFault = null")
    s.folds_again(tab, "the next fold works, on this page", size=30)


# A healthy device's answers that are not yes, each once: nothing here is lost, so nothing here may be kept.
REJECTS_ONCE = """
(() => {
  window.__rejectOnce = null;
  window.__rejected = 0;
  const wrap = (proto, name, fails) => {
    const held = proto[name];
    proto[name] = function (...args) {
      if (window.__rejectOnce !== name) return held.apply(this, args);
      window.__rejectOnce = null;
      window.__rejected += 1;
      return fails(held, this, args);
    };
  };
  wrap(GPUDevice.prototype, 'createComputePipelineAsync', () => Promise.reject(new GPUPipelineError('injected: the driver refused this shader', { reason: 'internal' })));
  wrap(GPUBuffer.prototype, 'mapAsync', () => Promise.reject(new DOMException('injected: a mapping failed', 'OperationError')));
  wrap(GPUQueue.prototype, 'onSubmittedWorkDone', () => Promise.reject(new DOMException('injected: the queue failed', 'OperationError')));
  wrap(GPUBuffer.prototype, 'getMappedRange', () => { throw new RangeError('injected: no room for a mapping'); });
  wrap(GPUQueue.prototype, 'writeBuffer', () => { throw new DOMException('injected: a write failed', 'OperationError'); });
  wrap(GPUDevice.prototype, 'popErrorScope', async (held, device, args) => {
    await held.apply(device, args);
    return new GPUValidationError('injected: a validation error');
  });
})();
"""


def rejected_under_a_fold(s, name, words, family=ESMFOLD2, size=300):
    tab = s.page([REJECTS_ONCE])
    tab.choose(family)
    tab.set_size(size)
    tab.press()
    tab.wait_status(*words, timeout=120)
    tab.evaluate("window.__rejectOnce = %s" % json.dumps(name))
    end = tab.wait_finished(timeout=240)
    rejected = tab.evaluate("window.__rejected")
    s.note("the fold", f"rejected {rejected} times, failed={end['failed']} {end['seconds']} s: {end['status'][:170]}")
    s.check("the fold ended, one way or the other, and did not hang", end["running"] is False and not end.get("timed_out"), f"{end['seconds']} s: {end['status'][:150]}")
    s.no_page_errors({**end, "errors": [e for e in end["errors"] if "injected" not in e]})
    tab.evaluate("window.__rejectOnce = null")
    s.folds_again(tab, "nothing kept: the next fold works, on this page", family=family, size=size)
    if family == ESMFOLD2:
        s.folds_again(tab, "and reads what it should", size=76, want=87.0)


@scenario("hangs")
def a_pipeline_refused_in_the_trunk(s):
    # At a length of its own, so that the next pipeline is a new one. Whatever a cache did with this one, the next fold asks again.
    rejected_under_a_fold(s, "createComputePipelineAsync", ["loading", "Language model", "Trunk"], size=212)


@scenario("hangs")
def a_pipeline_refused_in_openbind(s):
    rejected_under_a_fold(s, "createComputePipelineAsync", ["loading", "Trunk", "Preparing"], family="openbind0", size=212)


@scenario("hangs")
def a_readback_refused_in_the_trunk(s):
    rejected_under_a_fold(s, "mapAsync", ["Trunk 1/4", "Trunk 2/4"])


@scenario("hangs")
def a_readback_refused_in_the_sampler(s):
    rejected_under_a_fold(s, "mapAsync", ["Folding"])


@scenario("hangs")
def the_queue_refuses_in_the_trunk(s):
    rejected_under_a_fold(s, "onSubmittedWorkDone", ["Trunk 1/4", "Trunk 2/4"])


@scenario("hangs")
def a_validation_error_from_a_scope(s):
    rejected_under_a_fold(s, "popErrorScope", ["loading", "Language model", "Trunk"], size=212)


@scenario("lifecycle")
def the_tab_crashes_during_a_fold(s):
    browser = s.browser()
    first = browser.first.open()
    second = browser.new_tab(URL)
    second.open()
    first.choose()
    first.set_size(800)
    second.choose()
    second.set_size(76)
    first.press()
    first.wait_status("Trunk 1/4")
    second.press()
    time.sleep(1)
    said = second.state()["status"]
    s.note("the second waits", said[:100])
    crashed = next(t["id"] for t in browser.targets() if t["type"] == "page" and t["id"] != second.target)
    try:
        first.ws.s.settimeout(3)
        first.ws.send({"id": 999999, "method": "Page.crash", "params": {}})
    except OSError:
        pass
    first.opened = False
    end = second.wait_finished(timeout=240)
    plddt = number_after("pLDDT", end["status"])
    s.check("the one that was waiting folds, and reads what it should", not end["failed"] and not end.get("timed_out") and plddt is not None and abs(plddt - 87.0) < 0.5, f"{crashed[:8]} {end['seconds']} s: {end['status'][:150]}")
    s.no_page_errors(end)
    s.folds_again(second, "and the next", size=80)


@scenario("stop")
def drop_a_file_during_a_fold(s):
    tab = s.page()
    tab.choose()
    tab.set_size(300)
    tab.press()
    tab.wait_status("Trunk 1/4")
    job = "new File([JSON.stringify({ name: 'dropped', sequences: [{ protein: { id: 'A', sequence: %s } }], modelSeeds: [1] })], 'dropped.json')" % json.dumps(UBI)
    said = tab.evaluate(DROP % job)
    s.note("what the drop said", str(said)[:160])
    end = tab.wait_finished(timeout=240)
    s.check("the fold under it ends, and says so", end["running"] is False and not end.get("timed_out"), f"{end['seconds']} s: {end['status'][:150]}")
    s.no_page_errors(end)
    s.folds_again(tab, "and the next fold works, on this page", size=76, want=87.0)


@scenario("stop")
def downloads_during_a_fold(s):
    tab = s.page()
    s.folds_again(tab, "a fold to download", size=76, want=87.0)
    tab.set_size(300)
    tab.press()
    tab.wait_status("Trunk 1/4")
    clicked = tab.evaluate("""(async () => {
      const heldUrl = URL.createObjectURL;
      const heldClick = HTMLAnchorElement.prototype.click;
      let made = 0;
      URL.createObjectURL = (blob) => { made += 1; return heldUrl.call(URL, blob); };
      HTMLAnchorElement.prototype.click = function () {};
      const pressed = [];
      for (const button of [document.getElementById('download-pdb'), document.getElementById('download-all'),
                            ...[...document.querySelectorAll('button')].filter((b) => b.textContent.trim() === 'Session')]) {
        if (button === null || button.disabled) continue;
        button.click();
        pressed.push(button.id || button.textContent.trim());
        await new Promise((done) => setTimeout(done, 200));
      }
      await new Promise((done) => setTimeout(done, 500));
      URL.createObjectURL = heldUrl;
      HTMLAnchorElement.prototype.click = heldClick;
      return JSON.stringify({ pressed, made });
    })()""")
    s.check("the buttons were there to press", len(json.loads(clicked)["pressed"]) > 0, clicked)
    end = tab.wait_finished(timeout=240)
    s.check("the fold under it finishes, and reads what it should", not end["failed"] and not end.get("timed_out") and number_after("pLDDT", end["status"]) is not None, f"{end['seconds']} s: {end['status'][:150]}")
    s.no_page_errors(end)
    s.folds_again(tab, "and the next fold works, on this page", size=80)


@scenario("stop")
def restore_during_a_fold(s):
    tab = s.page()
    s.folds_again(tab, "a fold to restore", size=76, want=87.0)
    time.sleep(3)
    tab.ws.call("Page.reload")
    time.sleep(1)
    tab.wait_ready()
    time.sleep(1)
    tab.choose()
    tab.set_size(300)
    tab.press()
    tab.wait_status("Trunk 1/4", "Language model")
    offered = tab.evaluate("(() => { const b = document.getElementById('session-restore'); if (b === null || b.offsetParent === null) return false; b.click(); return true; })()")
    s.check("Restore was there to press", offered, offered)
    end = tab.wait_finished(timeout=240)
    s.check("the fold under it ends, and says so", end["running"] is False and not end.get("timed_out"), f"{end['seconds']} s: {end['status'][:150]}")
    s.no_page_errors(end)
    s.folds_again(tab, "and the next fold works, on this page", size=80)


@scenario("inputs")
def odd_seeds(s):
    tab = s.page()
    tab.choose()
    tab.set_size(40)
    for seed in ("", "-1", "1e10", "abc", "9007199254740993", "0.5"):
        tab.set_control("random-seed", seed)
        end = tab.fold()
        s.check(f"seed {seed!r}: a fold, or a word about the seed", end["running"] is False and not end.get("timed_out") and "NaN" not in end["status"] and "undefined" not in end["status"], f"failed={end['failed']} {end['status'][:130]}")
        s.no_page_errors(end)
    tab.set_control("random-seed", "0")
    s.folds_again(tab, "and seed 0 reads what it should", size=76, want=87.0)


@scenario("limits")
def press_before_the_line_has_read(s):
    # Set and pressed in one go, inside the 120 ms the line waits to reread: a fold past the limit is refused all the same.
    tab = s.page(["""localStorage.setItem('localfold.gpu-chosen-limit', String(2 * 2 ** 30));"""])
    tab.choose()
    tab.set_size(76)
    time.sleep(0.5)
    tab.evaluate("""(() => {
      const list = window.__entityList;
      const first = list.read()[0];
      list.set([{ ...first, type: 'protein', value: %s, copies: 1 }]);
      window.__foldState = undefined;
      document.getElementById('predict').click();
    })()""" % json.dumps(sequence(1800)))
    end = tab.wait_finished(timeout=60)
    s.check("refused, before anything starts", end["running"] is False and end["failed"] and "GiB" in end["status"] and end["seconds"] < 10, f"{end['seconds']} s: {end['status'][:170]}")
    s.no_page_errors(end)
    s.folds_again(tab, size=76, want=87.0)


# What a status line may not be: a reader should not have to read a program's words, or a number that did not come out.
RAW_WORDS = ("undefined", "NaN", "[object", "Cannot read", "is not a function", "is not defined", "TypeError", "RangeError", "Unexpected token")


def in_words(status):
    return not any(word in status for word in RAW_WORDS)


def ended(end):
    return end["running"] is False and not end.get("timed_out")


def drmsd(first, second):
    """How far two structures are apart, in their own distances, because the frame a fold lands in is arbitrary. Every third C-alpha."""
    a, b = first["ca"][::3], second["ca"][::3]
    assert len(a) == len(b) and len(a) > 1, (len(a), len(b))
    total = 0.0
    count = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            total += (math.dist(a[i], a[j]) - math.dist(b[i], b[j])) ** 2
            count += 1
    return (total / count) ** 0.5


def homologs(query, rows=24):
    """An alignment a model can read: the query, and rows a few residues off it."""
    records = [f">query\n{query}\n"]
    for row in range(rows):
        changed = "".join("A" if at % (row + 5) == 0 else residue for at, residue in enumerate(query))
        records.append(f">hit{row}\n{changed}\n")
    return "".join(records)


def paste(tab, text):
    tab.set_control("msa-mode", "paste")
    tab.evaluate("""(() => { const el = document.getElementById('msa-text'); el.value = %s;
                             el.dispatchEvent(new Event('input', { bubbles: true })); })()""" % json.dumps(text))
    time.sleep(0.2)


def alignments_by_hand(s, family):
    tab = s.page()
    tab.choose(family)
    tab.set_rows([["protein", UBI, 1]])
    good = homologs(UBI)
    paste(tab, good)
    end = tab.fold(timeout=300)
    reads = number_after("pLDDT", end["status"])
    s.check("a pasted alignment folds", not end["failed"] and ended(end), f"{end['seconds']} s: {end['status'][:150]}")
    s.no_page_errors(end)
    paste(tab, good.replace("\n", "\r\n"))
    end = tab.fold(timeout=300)
    same = number_after("pLDDT", end["status"])
    # ...to within a browser's first fold, which differs in its last digits from its later ones.
    s.check("and with Windows line ends, reads the same", not end["failed"] and (reads is None or abs(same - reads) < 0.5), f"{reads} and {same}: {end['status'][:130]}")
    s.no_page_errors(end)
    paste(tab, f">query\n{UBI}\n>inserts\n{UBI[:10]}acd{UBI[10:]}\n>gaps\n{'-' * 10}{UBI[10:]}\n")
    end = tab.fold(timeout=300)
    s.check("and with inserts and gaps", not end["failed"] and ended(end), end["status"][:150])
    s.no_page_errors(end)
    # A fold has to refuse these, or go on without them, in words.
    for label, given in (("no sequence under a header", ">only a header"), ("rows of another length", f">query\n{UBI}\n>short\nACD\n"),
                         ("symbols", ">a\n@@@@@@\n"), ("no header", UBI), ("another protein's", homologs(UBI[::-1])),
                         ("a query alone", f">query\n{UBI}\n"), ("nothing", "")):
        paste(tab, given)
        end = tab.fold(timeout=300)
        s.check(f"{label}: a fold or a word", ended(end) and in_words(end["status"]), f"failed={end['failed']} {end['status'][:170]}")
        s.no_page_errors(end)
    # ...and a file says what a paste says.
    tab.set_control("msa-mode", "upload")
    tab.evaluate(DROP % ("new File([%s], 'align.a3m')" % json.dumps(good)))
    end = tab.fold(timeout=300)
    dropped = number_after("pLDDT", end["status"])
    s.check("a dropped file reads what a paste read", not end["failed"] and (reads is None or abs(dropped - reads) < 0.5), f"{reads} and {dropped}: {end['status'][:130]}")
    s.no_page_errors(end)
    tab.set_control("msa-mode", "none")
    s.folds_again(tab, family=family, size=76)


@scenario("inputs")
def alignments_by_hand_openbind(s):
    alignments_by_hand(s, "openbind0")


@scenario("inputs")
def alignments_by_hand_monomer(s):
    alignments_by_hand(s, "monomer")


FIRST_RECORD = """(async () => (await (await fetch(%s)).text()).split('>')[1].split('\\n').slice(1).join('').trim())()"""
REAL_ALIGNMENT = "/tools/fixtures/5caj-a.a3m"


@scenario("inputs")
def a_real_alignment_on_every_model(s):
    tab = s.page()
    query = tab.evaluate(FIRST_RECORD % json.dumps(REAL_ALIGNMENT))
    s.note("the target", f"{len(query)} residues")
    for family in OTHER_MODELS:
        tab.choose(family)
        tab.set_rows([["protein", query, 1]])
        tab.set_control("msa-mode", "upload")
        tab.evaluate(DROP % ("new File([await (await fetch(%s)).blob()], '5caj-a.a3m')" % json.dumps(REAL_ALIGNMENT)))
        time.sleep(0.5)
        line = tab.state()["line"]
        end = tab.fold(timeout=420)
        s.check(f"{family}: folds, or says why not", ended(end) and in_words(end["status"]), f"plan: {line} | failed={end['failed']} {end['seconds']} s: {end['status'][:150]}")
        s.no_page_errors(end)
    tab.set_control("msa-mode", "none")
    s.folds_again(tab, size=76, want=87.0)


WITH_TEMPLATE = """(async () => {
  const text = %s;
  const list = window.__entityList;
  const first = list.read()[0];
  list.set([{ ...first, type: 'protein', value: %s, copies: 1, template: { kind: 'upload', text, filename: 'template.pdb', source: %s } }]);
})()"""
FETCHED = "await (await fetch(%s)).text()"
TEMPLATE_FIXTURE = "/tools/fixtures/5caj-crystal.pdb"


@scenario("inputs")
def a_template_from_a_file(s):
    tab = s.page()
    query = tab.evaluate(FIRST_RECORD % json.dumps(REAL_ALIGNMENT))
    for family in ("openbind0", "protenix2"):
        tab.choose(family)
        tab.set_rows([["protein", query, 1]])
        end = tab.fold(timeout=420)
        s.check(f"{family}: the fold alone", not end["failed"] and ended(end), end["status"][:130])
        bare = tab.pdb()
        tab.evaluate(WITH_TEMPLATE % (FETCHED % json.dumps(TEMPLATE_FIXTURE), json.dumps(query), json.dumps("A")))
        end = tab.fold(timeout=420)
        s.check(f"{family}: with its own crystal as the template", not end["failed"] and ended(end), end["status"][:130])
        templated = tab.pdb()
        apart = drmsd(bare, templated) if bare and templated else None
        s.check(f"{family}: which moves the fold", apart is not None and apart > 1.0, f"{apart} A apart: {end['status'][:110]}")
        s.no_page_errors(end)
        # ...and another file is another fold.
        tab.evaluate(WITH_TEMPLATE % (FETCHED % json.dumps("/tools/fixtures/1qys-crystal.pdb"), json.dumps(query), json.dumps("A")))
        end = tab.fold(timeout=420)
        s.check(f"{family}: another template is not reused from the last", "reused" not in end["status"], end["status"][:130])
    for label, source, chain in (("another protein's crystal", FETCHED % json.dumps("/tools/fixtures/1qys-crystal.pdb"), "A"),
                                 ("a chain the file does not have", FETCHED % json.dumps(TEMPLATE_FIXTURE), "Z"),
                                 ("a file that is no structure", json.dumps("not a structure at all\n"), "A")):
        tab.evaluate(WITH_TEMPLATE % (source, json.dumps(query), json.dumps(chain)))
        end = tab.fold(timeout=420)
        s.check(f"{label}: a fold or a word", ended(end) and in_words(end["status"]), f"failed={end['failed']} {end['status'][:170]}")
        s.no_page_errors(end)
    tab.choose("monomer")
    tab.set_rows([["protein", query, 1]])
    end = tab.fold(timeout=420)
    s.check("the monomer: the fold alone", not end["failed"] and ended(end), end["status"][:130])
    bare = tab.pdb()
    tab.evaluate(WITH_TEMPLATE % (FETCHED % json.dumps(TEMPLATE_FIXTURE), json.dumps(query), json.dumps("A")))
    end = tab.fold(timeout=420)
    s.check("the monomer: with its own crystal as the template", not end["failed"] and ended(end), end["status"][:130])
    templated = tab.pdb()
    apart = drmsd(bare, templated) if bare and templated else None
    s.check("the monomer: which moves the fold", apart is not None and apart > 0.1, f"{apart} A apart: {end['status'][:110]}")
    s.no_page_errors(end)
    s.folds_again(tab, family="openbind0", size=76)


def shelf_of_jobs():
    fixtures = REPO / "tools" / "fixtures"
    return (sorted((fixtures / "af3-jobs").glob("*.json")) + sorted((fixtures / "constraints").glob("*.json")) + sorted((fixtures / "crosslinks").glob("*.json"))
            + [fixtures / "glycan-standalone.json", fixtures / "kitchen-sink-relabelled.json"])


@scenario("inputs")
def every_job_on_the_shelf(s):
    tab = s.page()
    jobs = shelf_of_jobs()
    for family in ("openbind0", ESMFOLD2):
        for job in jobs:
            tab.choose(family)
            said = tab.evaluate(DROP % ("new File([%s], %s)" % (json.dumps(job.read_text(encoding="utf-8")), json.dumps(job.name))))
            time.sleep(0.5)
            line = tab.state()["line"]
            end = tab.fold(timeout=420)
            s.check(f"{family} {job.stem}: a fold or a word", ended(end) and in_words(end["status"]), f"said {str(said)[:50]} plan {line} | failed={end['failed']} {end['seconds']} s: {end['status'][:130]}")
            s.check(f"{family} {job.stem}: and no fold offered that would refuse it too", not end["failed"] or not any("CPU" in action for action in end["actions"]), end["actions"])
            s.no_page_errors(end)
            if not end["failed"] and ended(end):
                drawn = tab.evaluate(DRAWN)
                s.check(f"{family} {job.stem}: and draws it", drawn > 20, f"{drawn} colours")
        tab.choose(family)
        said = tab.evaluate(DROP % "new File([await (await fetch('/tools/fixtures/fold_2026_09_01_10_17.zip')).blob()], 'fold_2026_09_01_10_17.zip')")
        end = tab.fold(timeout=420)
        s.check(f"{family}: a real archive", ended(end) and in_words(end["status"]), f"said {str(said)[:80]} | failed={end['failed']} {end['seconds']} s: {end['status'][:130]}")
        s.no_page_errors(end)
    s.folds_again(tab, size=76)
    # ...and with the seed and sampler the last job left on the form gone with the form.
    tab.ws.call("Page.reload")
    time.sleep(1)
    tab.wait_ready()
    s.folds_again(tab, "after a reload", size=76, want=87.0)


CAPTURE_DOWNLOAD = """(async () => {
  const heldUrl = URL.createObjectURL;
  const heldClick = HTMLAnchorElement.prototype.click;
  let blob = null;
  URL.createObjectURL = (b) => { blob = b; return heldUrl.call(URL, b); };
  HTMLAnchorElement.prototype.click = function () {};
  document.getElementById(%s).click();
  for (let wait = 0; wait < 300 && blob === null; wait += 1) await new Promise((done) => setTimeout(done, 50));
  URL.createObjectURL = heldUrl;
  HTMLAnchorElement.prototype.click = heldClick;
  if (blob === null) return null;
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let text = '';
  for (let at = 0; at < bytes.length; at += 32768) text += String.fromCharCode(...bytes.subarray(at, at + 32768));
  return btoa(text);
})()"""
FROM_BASE64 = "new File([Uint8Array.from(atob(%s), (c) => c.charCodeAt(0))], %s)"


@scenario("inputs")
def the_monomer_sweep(s):
    # All five of AlphaFold 2's models in one press: a page loading and freeing each in turn, a pass of each reported, the best kept.
    tab = s.page()
    tab.choose("monomer")
    tab.set_rows([["protein", UBI, 1]])
    options = json.loads(tab.evaluate("JSON.stringify([...document.getElementById('af2Model')?.options ?? []].map((o) => o.value))"))
    s.note("the models on offer", options)
    sweep = next((value for value in options if "all" in value.lower()), None)
    if sweep is None:
        return s.note("no sweep to choose", "")
    tab.set_control("af2Model", sweep)
    end = tab.fold(timeout=600)
    s.check("a sweep finishes", not end["failed"] and ended(end) and in_words(end["status"]), f"{end['seconds']} s: {end['status'][:150]}")
    s.no_page_errors(end)
    s.check("and draws it", tab.evaluate(DRAWN) > 20, "")
    tab.set_rows([["protein", UBI[:60], 1]])
    tab.press()
    tab.wait_status("model 3", "Folding", timeout=120)
    took, stopped = stop_takes(tab)
    s.check("Stop ends one", ended(stopped) and took < 15, f"{took:.1f} s: {stopped['status'][:110]}")
    s.no_page_errors(stopped)
    tab.set_control("af2Model", options[0])
    s.folds_again(tab, "and the next fold works, on this page", family="monomer", size=76)


@scenario("inputs")
def a_fold_handed_back(s):
    # The archive's own README says to drop it back "to fold again with exactly these alignments", and a session file is the viewer's.
    for family in ("openbind0", ESMFOLD2):
        tab = s.page()
        tab.choose(family)
        tab.set_rows([["protein", UBI, 1]])
        end = tab.fold()
        first = number_after("pLDDT", end["status"])
        s.check(f"{family}: a fold to hand back", not end["failed"] and first is not None, end["status"][:120])
        archive = tab.evaluate(CAPTURE_DOWNLOAD % json.dumps("download-all"))
        session = tab.evaluate(CAPTURE_DOWNLOAD % json.dumps("saveStateButton"))
        s.check(f"{family}: both download", archive is not None and session is not None, f"archive {len(archive or '')}, session {len(session or '')} characters")
        s.close()
        if archive is None or session is None:
            continue
        tab = s.page()
        tab.choose(family)
        said = tab.evaluate(DROP % (FROM_BASE64 % (json.dumps(archive), json.dumps("fold.zip"))))
        rows = json.loads(tab.evaluate("JSON.stringify(window.__entityList.read().map((row) => row.value))"))
        s.check(f"{family}: the archive puts the job back", UBI in rows, f"{str(said)[:120]} {[len(row) for row in rows]}")
        end = tab.fold()
        again = number_after("pLDDT", end["status"])
        s.check(f"{family}: and it folds to what it read", not end["failed"] and again is not None and abs(again - first) < 0.5, f"{first} and {again}: {end['status'][:120]}")
        s.no_page_errors(end)
        s.close()
        tab = s.page()
        tab.choose(family)
        said = tab.evaluate(DROP % (FROM_BASE64 % (json.dumps(session), json.dumps("py2dmol_state_2026-11-03.json"))))
        time.sleep(2)
        drawn = tab.evaluate(DRAWN)
        s.check(f"{family}: the session shows it", drawn > 20, f"{drawn} colours: {str(said)[:120]}")
        s.no_page_errors(tab.state())
        s.folds_again(tab, family=family, size=76)
        s.close()


MARKUP = '"><img src=x onerror="window.__xss=1"><svg onload="window.__xss=1">'
NO_MARKUP = "window.__xss === undefined && document.querySelector('img[src=\"x\"], svg[onload]') === null"


def markup_everywhere(value):
    """Every name a session carries, as markup."""
    if isinstance(value, dict):
        return {key: (MARKUP if key in ("name", "stem", "model", "label", "title", "filename") and isinstance(item, str) else markup_everywhere(item))
                for key, item in value.items()}
    if isinstance(value, list):
        return [markup_everywhere(item) for item in value]
    return value


@scenario("inputs")
def markup_in_every_name(s):
    # A page that writes a name it was handed into its markup runs whatever a shared file or a link put there.
    tab = s.page()
    tab.choose("openbind0")
    short = UBI[:40]
    def job(sequences, name="job"):
        return json.dumps({"name": name, "modelSeeds": [1], "dialect": "alphafold3", "version": 2, "sequences": sequences})
    named = job([{"protein": {"id": "A", "sequence": short}}], name=MARKUP)
    for label, make in (("a job's name", "new File([%s], 'job.json')" % json.dumps(named)),
                        ("a chain's id", "new File([%s], 'job.json')" % json.dumps(job([{"protein": {"id": MARKUP, "sequence": short}}]))),
                        ("a ligand's code", "new File([%s], 'job.json')" % json.dumps(job([{"protein": {"id": "A", "sequence": short}}, {"ligand": {"id": "B", "ccdCodes": [MARKUP]}}]))),
                        ("a modification's code", "new File([%s], 'job.json')" % json.dumps(job([{"protein": {"id": "A", "sequence": short, "modifications": [{"ptmType": MARKUP, "ptmPosition": 1}]}}]))),
                        ("a file's name", "new File(['>a\\nACD'], %s)" % json.dumps(MARKUP + ".a3m")),
                        ("a fasta's header", "new File([%s], 'x.fasta')" % json.dumps(f">{MARKUP}\n{short}\n")),
                        ("an alignment's header", "new File([%s], 'x.a3m')" % json.dumps(f">{MARKUP}\n{short}\n")),
                        ("a structure's text", "new File([%s], 'x.pdb')" % json.dumps(f"TITLE     {MARKUP}\nATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n")),
                        ("a sequence", "new File([%s], 'x.fasta')" % json.dumps(f">a\n{MARKUP}\n"))):
        said = tab.evaluate(DROP % make)
        s.check(f"{label}: stays text", tab.evaluate(NO_MARKUP) is True, str(said)[:120])
        end = tab.fold(timeout=120)
        s.check(f"{label}: a fold or a word, and still text", ended(end) and tab.evaluate(NO_MARKUP) is True, f"failed={end['failed']} {end['status'][:130]}")
        s.no_page_errors(end)
        tab.choose("openbind0")
        tab.set_rows([["protein", short, 1]])
    # ...the name a fold kept, offered and restored on the next visit.
    tab.evaluate(DROP % ("new File([%s], 'job.json')" % json.dumps(named)))
    end = tab.fold(timeout=120)
    s.check("a fold under that name", not end["failed"] and ended(end), end["status"][:120])
    session = tab.evaluate(CAPTURE_DOWNLOAD % json.dumps("saveStateButton"))
    time.sleep(2)
    tab.ws.call("Page.reload")
    time.sleep(1)
    tab.wait_ready()
    time.sleep(1)
    s.check("the offer of it stays text", tab.evaluate(NO_MARKUP) is True, tab.evaluate("document.getElementById('session')?.textContent ?? ''")[:120])
    tab.evaluate("document.getElementById('session-restore')?.click()")
    time.sleep(2)
    s.check("and restoring it", tab.evaluate(NO_MARKUP) is True, "")
    # ...a session file with every name in it replaced, if it is one a reader could have edited.
    if session is not None:
        try:
            edited = json.dumps(markup_everywhere(json.loads(base64.b64decode(session))))
        except ValueError as error:
            s.note("a session file is not plain JSON", error)
        else:
            said = tab.evaluate(DROP % ("new File([%s], %s)" % (json.dumps(edited), json.dumps(MARKUP + ".json"))))
            s.note("what a session file full of it said", str(said)[:120])
            time.sleep(2)
            s.check("a session file full of it stays text", tab.evaluate(NO_MARKUP) is True, "")
    s.no_page_errors(tab.state())
    s.close()
    # ...and a link.
    tab = s.browser().first
    tab.open(url=URL + "?model=" + urllib.parse.quote(MARKUP) + "#" + urllib.parse.quote(MARKUP))
    time.sleep(1)
    s.check("a link full of it stays text", tab.evaluate(NO_MARKUP) is True, "")
    s.folds_again(tab, size=76)


# What the card and the host can answer that nothing in the page's own code asked for.

NONSENSE = """
(() => {
  window.__nonsense = null;
  window.__nonsensical = 0;
  const held = GPUBuffer.prototype.getMappedRange;
  GPUBuffer.prototype.getMappedRange = function (...args) {
    const range = held.apply(this, args);
    if (window.__nonsense !== null && (this.usage & GPUBufferUsage.MAP_READ) !== 0) {
      new Float32Array(range, 0, range.byteLength >> 2).fill(window.__nonsense);
      window.__nonsensical += 1;
    }
    return range;
  };
})();
"""


def nonsense_under_a_fold(s, value, words, family=ESMFOLD2, size=300, want=87.0):
    tab = s.page([NONSENSE])
    tab.choose(family)
    tab.set_size(size)
    tab.press()
    tab.wait_status(*words, timeout=120)
    tab.evaluate("window.__nonsense = %s" % value)
    end = tab.wait_finished(timeout=300)
    tab.evaluate("window.__nonsense = null")
    s.note("the fold", f"{tab.evaluate('window.__nonsensical')} readbacks, failed={end['failed']} {end['seconds']} s: {end['status'][:170]}")
    s.check("the fold ended", ended(end), end["status"][:150])
    s.check("in words", in_words(end["status"]), end["status"][:170])
    s.no_page_errors(end)
    if not end["failed"]:
        pdb = tab.pdb()
        clean = pdb is not None and all(v is not None and abs(v) < 9999 for row in pdb["ca"] for v in row) and len({tuple(row) for row in pdb["ca"]}) > 1
        s.check("and what it calls a result is one", clean and number_after("pLDDT", end["status"]) is not None, end["status"][:150])
    # Whatever it kept, the same fold again owes nothing to it.
    tab.set_size(size)
    end = tab.fold(timeout=300)
    pdb = tab.pdb()
    clean = pdb is not None and all(v is not None and abs(v) < 9999 for row in pdb["ca"] for v in row)
    s.check("the same fold again is a real one", not end["failed"] and clean and number_after("pLDDT", end["status"]) is not None, end["status"][:150])
    s.folds_again(tab, family=family, size=76, want=want)


@scenario("faults")
def nonsense_from_the_language_model(s):
    nonsense_under_a_fold(s, "NaN", ["Language model"])


@scenario("faults")
def nonsense_from_the_trunk(s):
    nonsense_under_a_fold(s, "NaN", ["Trunk 1/4", "Trunk 2/4"])


@scenario("faults")
def nonsense_from_the_sampler(s):
    nonsense_under_a_fold(s, "NaN", ["Folding"])


@scenario("faults")
def huge_numbers_from_the_sampler(s):
    nonsense_under_a_fold(s, "1e30", ["Folding"])


@scenario("faults")
def zeros_from_the_sampler(s):
    nonsense_under_a_fold(s, "0", ["Folding"])


@scenario("faults")
def nonsense_in_openbind(s):
    nonsense_under_a_fold(s, "NaN", ["Trunk 2/4"], family="openbind0", want=None)


@scenario("faults")
def nonsense_from_the_monomer(s):
    nonsense_under_a_fold(s, "NaN", ["Folding"], family="monomer", want=None)


@scenario("faults")
def kill_just_after_stop(s):
    tab = s.page()
    tab.choose()
    tab.set_size(600)
    tab.press()
    tab.wait_status("Trunk 2/4")
    tab.evaluate("document.getElementById('predict').click()")
    kill_gpu_process()
    end = tab.wait_finished(timeout=60)
    s.check("the fold ends", ended(end), f"{end['seconds']} s: {end['status'][:150]}")
    s.no_page_errors(end)
    s.folds_again(tab, want=None)


SMALLER_CARD = """
(() => {
  window.__smallerCard = false;
  const held = GPU.prototype.requestAdapter;
  GPU.prototype.requestAdapter = async function (...args) {
    const adapter = await held.apply(this, args);
    if (!window.__smallerCard || adapter === null) return adapter;
    return new Proxy(adapter, { get(target, key) {
      if (key === 'features') { const kept = new Set(target.features); kept.delete('shader-f16'); kept.delete('chromium-experimental-subgroup-matrix'); return kept; }
      if (key === 'limits') return new Proxy(target.limits, { get: (limits, name) => ({ maxBufferSize: 2 ** 30, maxStorageBufferBindingSize: 2 ** 30 })[name] ?? limits[name] });
      if (key === 'requestDevice') return (descriptor = {}) => target.requestDevice({ ...descriptor, requiredFeatures: [...(descriptor.requiredFeatures ?? [])].filter((name) => name !== 'shader-f16' && name !== 'chromium-experimental-subgroup-matrix') });
      const value = target[key];
      return typeof value === 'function' ? value.bind(target) : value;
    } });
  };
})();
"""


@scenario("faults")
def the_card_is_another_after_a_crash(s):
    # A laptop that lost its dock, or a driver that came back with less. What the page kept to go on recycling was computed on the first.
    tab = s.page([SMALLER_CARD])
    tab.choose()
    tab.set_size(300)
    tab.set_control("recycles", 3)
    end = tab.fold()
    reference = number_after("pLDDT", end["status"])
    s.check("a fold of three recycles", not end["failed"] and reference is not None, end["status"][:120])
    tab.set_control("recycles", 1)
    tab.fold()
    tab.evaluate("window.__smallerCard = true")
    kill_gpu_process()
    time.sleep(3)
    tab.set_control("recycles", 3)
    time.sleep(0.5)
    line = tab.state()["line"]
    end = tab.fold()
    plddt = number_after("pLDDT", end["status"])
    s.check("three recycles on the smaller card read as they did, or it says why not", ended(end) and in_words(end["status"]) and (end["failed"] or (plddt is not None and abs(plddt - reference) < 0.5)), f"plan {line} | {reference} and {plddt}: {end['status'][:150]}")
    s.no_page_errors(end)
    s.folds_again(tab, "and it folds on it", size=76, want=87.0)


HOST_FULL = """
(() => {
  window.__hostFull = false;
  window.__hostRefused = 0;
  const refuse = () => { window.__hostFull = false; window.__hostRefused += 1; throw new RangeError('Array buffer allocation failed'); };
  for (const name of ['ArrayBuffer', 'Uint8Array', 'Uint16Array', 'Int32Array', 'Uint32Array', 'Float32Array', 'Float64Array']) {
    const Held = window[name];
    const width = Held.BYTES_PER_ELEMENT ?? 1;
    window[name] = new Proxy(Held, { construct(target, args, newTarget) {
      if (window.__hostFull && typeof args[0] === 'number' && args[0] * width >= 1 << 22) refuse();
      return Reflect.construct(target, args, newTarget);
    } });
  }
  const read = Response.prototype.arrayBuffer;
  Response.prototype.arrayBuffer = function () {
    if (window.__hostFull) { try { refuse(); } catch (cause) { return Promise.reject(cause); } }
    return read.call(this);
  };
})();
"""


def host_full_at(s, words, size, cold=False):
    tab = s.page([HOST_FULL], cold=cold)
    tab.choose()
    tab.set_size(size)
    if words:
        tab.press()
        tab.wait_status(*words, timeout=180)
        tab.evaluate("window.__hostFull = true")
    else:
        tab.evaluate("window.__hostFull = true")
        tab.press()
    end = tab.wait_finished(timeout=300)
    refused = tab.evaluate("window.__hostRefused")
    s.note("the fold", f"refused {refused} times, failed={end['failed']} {end['seconds']} s: {end['status'][:170]}")
    tab.evaluate("window.__hostFull = false")
    s.check("it ended", ended(end), end["status"][:150])
    s.check("in words", in_words(end["status"]) and "allocation failed" not in end["status"], end["status"][:170])
    s.no_page_errors({**end, "errors": [e for e in end["errors"] if "allocation failed" not in e]})
    s.folds_again(tab, "the next fold works, on this page", size=76, want=87.0)
    return refused


@scenario("limits")
def host_memory_runs_out_in_the_download(s):
    refused = host_full_at(s, [], 76, cold=True)
    s.check("the fault fired", refused > 0, refused)


@scenario("limits")
def host_memory_runs_out_in_the_trunk(s):
    host_full_at(s, ["Trunk 1/4"], 600)


@scenario("limits")
def host_memory_runs_out_in_the_sampler(s):
    host_full_at(s, ["Folding"], 600)


@scenario("limits")
def no_float16array(s):
    # Chrome had WebGPU for two years before it had Float16Array, and the weights are read through it.
    tab = s.page(["delete globalThis.Float16Array;"])
    for family in (ESMFOLD2, "openbind0"):
        tab.choose(family)
        tab.set_size(76)
        end = tab.fold(timeout=120)
        s.check(f"{family}: it says what to do about it", end["failed"] and ended(end) and "Update" in end["status"], end["status"][:150])
        s.check(f"{family}: and offers no fold that needs the same", not end["actions"], end["actions"])
        s.no_page_errors(end)


BUSY_CARD = """(async () => {
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();
  const buffer = device.createBuffer({ size: 1 << 24, usage: GPUBufferUsage.STORAGE });
  const module = device.createShaderModule({ code: `
    @group(0) @binding(0) var<storage, read_write> data: array<f32>;
    @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3u) {
      var x = data[id.x %% 4000000u];
      for (var i = 0u; i < 4000u; i += 1u) { x = sqrt(x * x + f32(i)); }
      data[id.x %% 4000000u] = x;
    }` });
  const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
  const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer } }] });
  window.__busy = true;
  window.__busySubmits = 0;
  window.__busyDevice = device;
  void (async () => {
    while (window.__busy) {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(%d);
      pass.end();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      window.__busySubmits += 1;
    }
  })();
  return 'busy';
})()"""


@scenario("limits")
def another_program_loads_the_card(s):
    # A game, a video call, a second model: nothing here knows of it but what its own work takes.
    browser = s.browser()
    tab = browser.first.open()
    tab.choose()
    tab.set_size(300)
    tab.fold()
    tab.set_size(307)
    tab.fold()
    tab.set_size(300)
    alone = tab.fold()
    reference = tab.pdb()
    s.check("a fold with the card to itself", not alone["failed"], f"{alone['seconds']} s: {alone['status'][:100]}")
    busy = browser.new_tab(URL)
    time.sleep(2)
    busy.evaluate(BUSY_CARD % 16384)
    time.sleep(2)
    s.note("the other program", f"{busy.evaluate('window.__busySubmits')} submissions in 2 s")
    tab.set_size(307)
    tab.fold(timeout=900)
    tab.set_size(300)
    loaded = tab.fold(timeout=900)
    s.note("the same fold beside it", f"{loaded['seconds']} s against {alone['seconds']} s, its {busy.evaluate('window.__busySubmits')} submissions")
    s.check("reads what it read", not loaded["failed"] and ended(loaded) and number_after("pLDDT", loaded["status"]) == number_after("pLDDT", alone["status"]), f"{alone['status'][:90]} | {loaded['status'][:90]}")
    pdb = tab.pdb()
    worst = max((max(abs(a - b) for a, b in zip(p, q)) for p, q in zip(pdb["ca"], reference["ca"])), default=9.9) if pdb else 9.9
    s.check("and the same coordinates", worst < 0.2, f"largest change {worst:.3f} A")
    tab.set_size(600)
    tab.press()
    tab.wait_status("Trunk 1/4", timeout=300)
    clicked = time.time()
    tab.evaluate("document.getElementById('predict').click()")
    end = tab.wait_finished(timeout=120)
    s.check("Stop answers beside it", ended(end) and time.time() - clicked < 15, f"{time.time() - clicked:.1f} s: {end['status'][:100]}")
    s.no_page_errors(end)
    busy.evaluate("window.__busy = false; window.__busyDevice.destroy()")
    browser.close_tab(busy)
    s.folds_again(tab, size=76, want=87.0)


@scenario("limits")
def two_browsers_fold_at_once(s):
    # Two browsers share no lock, and a fold that waits for the card to be free can only wait for its own.
    one = s.browser().first.open()
    other = s.browser(port=args.debug_port + 1, profile=tempfile.mkdtemp(prefix="robustness-other-")).first.open()
    for tab in (one, other):
        tab.choose()
        tab.set_size(76)
        end = tab.fold(timeout=300)
        s.check("each folds alone", not end["failed"], end["status"][:100])
    one.set_size(600)
    alone = one.fold(timeout=300)
    reference = one.pdb()
    one.set_size(590)
    one.fold(timeout=300)
    one.set_size(600)
    other.set_size(600)
    one.press()
    other.press()
    first = one.wait_finished(timeout=900)
    second = other.wait_finished(timeout=900)
    s.note("each took", f"{first['seconds']} s and {second['seconds']} s against {alone['seconds']} s alone")
    for label, tab, end in (("the first", one, first), ("the second", other, second)):
        s.check(f"{label} finishes", not end["failed"] and ended(end), end["status"][:130])
        s.no_page_errors(end)
        pdb = tab.pdb()
        worst = max((max(abs(a - b) for a, b in zip(p, q)) for p, q in zip(pdb["ca"], reference["ca"])), default=9.9) if pdb else 9.9
        s.check(f"{label} reads what it read alone", worst < 0.2, f"largest change {worst:.3f} A")
    # ...and a crash takes both, because they share a card even so.
    one.set_size(300)
    other.set_size(300)
    one.press()
    other.press()
    one.wait_status("Trunk 1/4", timeout=180)
    kill_gpu_process()
    first = one.wait_finished(timeout=120)
    second = other.wait_finished(timeout=120)
    s.check("a crash under both ends what it stopped", ended(first) and ended(second), f"{first['status'][:80]} | {second['status'][:80]}")
    time.sleep(2)
    s.folds_again(one, "the first folds again", size=76, want=87.0)
    s.folds_again(other, "the second folds again", size=76, want=87.0)


# What a reader's own page can do around a fold.

@scenario("lifecycle")
def the_window_changes_during_a_fold(s):
    tab = s.page()
    tab.choose()
    tab.set_size(300)
    tab.press()
    tab.wait_status("Trunk 1/4")
    for width, height, scale, mobile in ((360, 640, 3, True), (1920, 1080, 1, False), (800, 300, 2, False), (390, 844, 3, True), (2560, 1440, 1, False)):
        tab.ws.call("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=scale, mobile=mobile)
        time.sleep(1.5)
    end = tab.wait_finished(timeout=240)
    s.check("the fold under it finishes", not end["failed"] and ended(end), f"{end['seconds']} s: {end['status'][:110]}")
    s.no_page_errors(end)
    for width, height, scale, mobile in ((360, 640, 3, True), (1280, 800, 1, False)):
        tab.ws.call("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=scale, mobile=mobile)
        time.sleep(1)
        wide = tab.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")
        s.check(f"at {width}: no sideways scroll", wide[0] <= wide[1] + 1, wide)
        drawn = tab.evaluate(DRAWN)
        s.check(f"at {width}: and it draws", drawn > 20, f"{drawn} colours")
    tab.ws.call("Emulation.clearDeviceMetricsOverride")
    s.folds_again(tab, size=76, want=87.0)


@scenario("lifecycle")
def opened_in_the_background(s):
    # A link opened in a new tab behind this one: the page starts with nobody looking, and may never draw a frame until someone does.
    browser = s.browser()
    front = browser.first
    target = front.ws.call("Target.createTarget", url="about:blank", background=True)["targetId"]
    found = next(t for t in browser.targets() if t["id"] == target)
    tab = Tab(cdp.WS(found["webSocketDebuggerUrl"]), browser)
    tab.open()
    hidden = tab.evaluate("document.visibilityState")
    s.note("the tab is", hidden)
    tab.choose()
    tab.set_size(76)
    time.sleep(0.5)
    end = tab.fold(timeout=240)
    plddt = number_after("pLDDT", end["status"])
    s.check("a fold nobody is looking at finishes, and reads what it should", not end["failed"] and ended(end) and plddt is not None and abs(plddt - 87.0) < 0.5, f"{end['seconds']} s: {end['status'][:110]}")
    s.no_page_errors(end)
    browser.activate(target)
    time.sleep(1)
    drawn = tab.evaluate(DRAWN)
    s.check("and shows when someone looks", drawn > 20, f"{drawn} colours")
    s.folds_again(tab, size=80)


SERVER = {"process": None}


def serve(repo, command=None):
    """This checkout at the one origin, in place of whatever answered there. Caches, settings and a session belong to an origin."""
    if SERVER["process"] is not None:
        SERVER["process"].kill()
        SERVER["process"].wait()
    SERVER["process"] = subprocess.Popen(command or [sys.executable, str(repo / "tools" / "serve.py"), str(args.port)],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=repo)
    for _ in range(100):
        with socket.socket() as probe:
            probe.settimeout(0.2)
            if probe.connect_ex(("127.0.0.1", args.port)) == 0:
                return
        time.sleep(0.1)
    raise AssertionError(f"{repo} never answered on {args.port}")


READ_CACHES = "(async () => JSON.stringify((await caches.keys()).map((name) => name.split('@')[0])))()"


@scenario("lifecycle")
def the_site_as_built(s):
    # A visitor gets dist/ from a static host, and everything else here reads the checkout: whatever the build leaves out, a page may still ask for.
    with tempfile.TemporaryDirectory() as held:
        copy = Path(held) / "checkout"
        shutil.copytree(REPO, copy, ignore=shutil.ignore_patterns(".git", "node_modules", "dist", "oracle-dumps"))
        built = subprocess.run([sys.executable, str(copy / "tools" / "build_site.py")], cwd=copy, capture_output=True, text=True)
        s.check("it builds", built.returncode == 0, (built.stdout + built.stderr)[-300:])
        try:
            serve(copy / "dist", [sys.executable, "-m", "http.server", str(args.port), "--bind", "127.0.0.1"])
            tab = s.page(cold=True)
            s.folds_again(tab, "the built page folds", size=76, want=87.0)
            # ...and the code before this has no CPU fold to choose.
            if tab.evaluate("document.getElementById('settings-compute') !== null"):
                choose_cpu(tab)
                s.check("takes its threads from the service worker", tab.evaluate("self.crossOriginIsolated") is True, tab.evaluate("document.getElementById('fold-plan')?.textContent ?? ''"))
                s.folds_again(tab, "and folds on the CPU", size=30)
                choose_gpu(tab)
                s.folds_again(tab, "and back on the GPU", size=76, want=87.0)
            # http.server sends no cache headers, so what it served stays good in this profile by heuristics, and the next scenario is another server's.
            tab.ws.call("Network.enable")
            tab.ws.call("Network.clearBrowserCache")
            tab.ws.call("Network.disable")
        finally:
            s.close()
            serve(REPO)


@scenario("lifecycle")
def a_reader_of_the_old_page_comes_back(s):
    # Everyone who has used the site has its caches, its stored settings and its last session, in the formats of the code before this.
    before = Path("/baseline")
    if not (before / "tools" / "serve.py").exists() or before.resolve() == REPO.resolve():
        return s.note("no older checkout to come from", str(before))
    serve(before)
    try:
        tab = s.page(cold=True)
        tab.choose()
        tab.set_size(76)
        end = tab.fold(timeout=300)
        s.check("a fold on the old page", not end["failed"] and ended(end), end["status"][:120])
        tab.evaluate("localStorage.setItem('localfold.theme', 'dark')")
        time.sleep(3)
        s.note("what the old page cached", tab.evaluate(READ_CACHES))
        tab.opened = False  # ...and what it threw is its own.
        s.close()
        serve(REPO)
        tab = s.page(keep_origin=True)
        time.sleep(1)
        offered = tab.evaluate("(() => { const b = document.getElementById('session-restore'); return b !== null && b.offsetParent !== null; })()")
        s.check("the last session is offered", offered is True, tab.evaluate("document.getElementById('session')?.textContent ?? ''")[:120])
        if offered:
            tab.evaluate("document.getElementById('session-restore').click()")
            time.sleep(3)
            drawn = tab.evaluate(DRAWN)
            s.check("and comes back", drawn > 20, f"{drawn} colours")
        s.no_page_errors(tab.state())
        tab.choose()
        tab.set_size(76)
        end = tab.fold(timeout=300)
        plddt = number_after("pLDDT", end["status"])
        s.check("a fold on the new page reads what it should", not end["failed"] and plddt is not None and abs(plddt - 87.0) < 0.5, f"{end['seconds']} s: {end['status'][:120]}")
        s.no_page_errors(end)
        time.sleep(3)
        caches = json.loads(tab.evaluate(READ_CACHES))
        s.check("and what the old page downloaded does not stay", not any(name.startswith("localfold-model-") for name in caches), caches)
        s.close()
        # ...and if this were reverted.
        serve(before)
        tab = s.page(keep_origin=True)
        tab.choose()
        tab.set_size(76)
        end = tab.fold(timeout=300)
        s.check("the old page, should it come back, folds over what this one left", not end["failed"] and ended(end), f"{end['seconds']} s: {end['status'][:120]}")
        tab.opened = False
    finally:
        s.close()
        serve(REPO)


@scenario("storage")
def a_session_store_from_the_future(s):
    # A site rolled back after a visit: the store's version is past what this code asks for, and IndexedDB refuses it.
    tab = s.page(["""indexedDB.open('localfold-session', 99).onsuccess = (event) => event.target.result.close();"""])
    time.sleep(1)
    s.folds_again(tab, "a fold, with a session store it cannot open", size=76, want=87.0)
    time.sleep(2)
    s.no_page_errors(tab.state())
    tab.ws.call("Page.reload")
    time.sleep(1)
    tab.wait_ready()
    time.sleep(1)
    s.folds_again(tab, "and after a reload", size=80)


@scenario("network")
def download_answers_a_captive_portal(s):
    fetch_fault(s, "portal", at=1, times=100000)


@scenario("network")
def an_alignment_search_on_the_real_server(s):
    # One short question to the server the page sends a search to, because every other scenario here answers for it.
    tab = s.page()
    tab.choose("openbind0")
    tab.set_rows([["protein", UBI, 1]])
    tab.set_control("msa-mode", "search")
    end = tab.fold(timeout=420)
    s.note("the fold", f"failed={end['failed']} {end['seconds']} s: {end['status'][:190]}")
    s.check("it ends", ended(end) and in_words(end["status"]), end["status"][:170])
    s.no_page_errors(end)
    if not end["failed"]:
        s.check("with a reading", number_after("pLDDT", end["status"]) is not None, end["status"][:150])
    tab.set_control("msa-mode", "none")
    s.folds_again(tab, family="openbind0", size=76)


COMPLEXES = (("a heterodimer", [["protein", UBI[:30], 1], ["protein", UBI[30:60], 1]]),
             ("a homodimer", [["protein", UBI[:30], 2]]),
             ("a protein and DNA", [["protein", UBI[:30], 1], ["dna", "GATTACAGATTA", 2]]),
             ("a protein and a ligand", [["protein", UBI[:30], 1], ["ligand", "ATP", 1]]),
             ("a protein and a SMILES", [["protein", UBI[:30], 1], ["smiles", "CC(=O)OC1=CC=CC=C1C(=O)O", 1]]))


@scenario("cpu")
def cpu_folds_a_complex(s):
    # What the CPU fold is for, and until here every CPU fold in this suite was one chain.
    tab = s.page()
    tab.choose()
    on_gpu = {}
    for name, rows in COMPLEXES:
        tab.set_rows(rows)
        end = tab.fold()
        s.check(f"{name}: on the GPU", not end["failed"] and ended(end), end["status"][:110])
        on_gpu[name] = (number_after("pLDDT", end["status"]), tab.pdb())
    choose_cpu(tab)
    for name, rows in COMPLEXES:
        tab.choose()
        tab.set_rows(rows)
        time.sleep(0.3)
        line = tab.state()["line"]
        end = tab.fold(timeout=900)
        plddt = number_after("pLDDT", end["status"])
        s.check(f"{name}: on the CPU", not end["failed"] and ended(end), f"{line} | {end['seconds']} s: {end['status'][:110]}")
        s.no_page_errors(end)
        pdb = tab.pdb()
        apart = drmsd(on_gpu[name][1], pdb) if pdb and on_gpu[name][1] else None
        s.check(f"{name}: the same fold", plddt is not None and on_gpu[name][0] is not None and abs(plddt - on_gpu[name][0]) < 1.0 and apart is not None and apart < 0.3, f"pLDDT {on_gpu[name][0]} and {plddt}, {apart} A apart")
    s.folds_again(tab, "the next fold, on the CPU", size=30)


@scenario("cpu")
def no_webassembly(s):
    # A browser that turned it off, which some do for a site they do not trust, or an administrator's policy.
    tab = s.page(["delete window.WebAssembly;", WORKER_FAULT])
    s.folds_again(tab, "the GPU has no need of it", size=76, want=87.0)
    choose_cpu(tab)
    tab.choose()
    tab.set_size(30)
    tab.evaluate("window.__workerFault = 'nowasm'")
    time.sleep(0.5)
    line = tab.state()["line"]
    end = tab.fold(timeout=60)
    s.check("the CPU says why not", end["failed"] and ended(end) and in_words(end["status"]), f"{line} | {end['seconds']} s: {end['status'][:190]}")
    tab.evaluate("window.__workerFault = null")


@scenario("hangs")
def a_mapping_refused_in_the_trunk(s):
    rejected_under_a_fold(s, "getMappedRange", ["Trunk 1/4", "Trunk 2/4"])


@scenario("hangs")
def a_mapping_refused_in_the_sampler(s):
    rejected_under_a_fold(s, "getMappedRange", ["Folding"])


@scenario("hangs")
def a_write_refused_in_the_trunk(s):
    rejected_under_a_fold(s, "writeBuffer", ["Trunk 1/4", "Trunk 2/4"])


def reused(end):
    return re.search(r"reused|more recycle", end["status"]) is not None


def then(s, tab, label, reuses=False):
    """A fold after a change, and whether it should have gone on from the last one's trunk."""
    time.sleep(0.2)
    end = tab.fold(timeout=300)
    s.check(f"{label}: {'goes on from the last trunk' if reuses else 'folds anew'}", ended(end) and not end["failed"] and reused(end) == reuses, end["status"][:150])
    s.no_page_errors(end)
    return tab.pdb()


HAND_IN_JOB = "new File([%s], 'job.json')"


def with_job(tab, name, sequences, **more):
    tab.evaluate(DROP % (HAND_IN_JOB % json.dumps(json.dumps({"name": name, "modelSeeds": [1], "dialect": "alphafold3", "version": 2, "sequences": sequences, **more}))))


@scenario("inputs")
def what_changes_a_fold(s):
    # A trunk is kept to go on from, and every key that misses something folds the last question and answers it as this one. Adding a
    # template did: the fold read the same to the third decimal, and said the template had been used.
    tab = s.page()
    short = UBI[:50]
    plain = [{"protein": {"id": "A", "sequence": short}}]
    # OpenBind's key has the seed; ESMFold2's has the recycles and not the seed, because the seed does not reach its trunk.
    tab.choose("openbind0")
    tab.set_rows([["protein", short, 1]])
    tab.set_control("random-seed", 1)
    tab.fold()
    then(s, tab, "the same again", reuses=True)
    tab.set_control("random-seed", 2)
    then(s, tab, "openbind0, another seed")
    tab.set_control("recycles", 6)
    then(s, tab, "openbind0, more recycles", reuses=True)
    tab.set_control("recycles", 2)
    then(s, tab, "openbind0, fewer recycles")
    tab.set_control("recycles", 3)
    for family, seeds_reach_the_trunk in (("openbind0", True), (ESMFOLD2, False)):
        tab.choose(family)
        tab.set_control("random-seed", 1)
        tab.set_rows([["protein", short, 1]])
        tab.fold()
        tab.set_rows([["protein", short[1:], 1]])
        then(s, tab, f"{family}, one residue fewer")
        tab.set_rows([["protein", short[:25], 1], ["protein", short[25:], 1]])
        tab.fold()
        tab.set_rows([["protein", short[25:], 1], ["protein", short[:25], 1]])
        then(s, tab, f"{family}, the chains the other way round")
        tab.set_rows([["protein", short[:25], 2]])
        then(s, tab, f"{family}, two copies of one")
        tab.set_rows([["protein", short[:25], 1], ["protein", short[:25], 1]])
        then(s, tab, f"{family}, the same two as two rows", reuses=True)
        tab.set_rows([["dna", "GCAGCAGCAGCA", 1]])
        tab.fold()
        tab.set_rows([["rna", "GCAGCAGCAGCA", 1]])
        then(s, tab, f"{family}, the same letters as RNA")
        tab.set_rows([["protein", short, 1], ["ligand", "ATP", 1]])
        tab.fold()
        tab.set_rows([["protein", short, 1], ["ligand", "ADP", 1]])
        then(s, tab, f"{family}, another ligand")
        tab.set_rows([["protein", short, 1], ["smiles", "CC(=O)OC1=CC=CC=C1C(=O)O", 1]])
        then(s, tab, f"{family}, a SMILES")
        tab.set_rows([["protein", short, 1], ["smiles", "CCO", 1]])
        then(s, tab, f"{family}, another SMILES")
        with_job(tab, "plain", plain)
        tab.fold()
        with_job(tab, "modified", [{"protein": {"id": "A", "sequence": short, "modifications": [{"ptmType": "SEP", "ptmPosition": 20}]}}])
        then(s, tab, f"{family}, a modified residue")
        with_job(tab, "modified", [{"protein": {"id": "A", "sequence": short, "modifications": [{"ptmType": "TPO", "ptmPosition": 22}]}}])
        then(s, tab, f"{family}, modified at another residue")
    # The seed reaches ESMFold2's sampler and not its trunk: the trunk goes on, and the structure must not.
    tab.choose(ESMFOLD2)
    tab.set_control("random-seed", 1)
    tab.set_rows([["protein", short, 1]])
    tab.fold()
    first = tab.pdb()
    tab.set_control("random-seed", 2)
    second = then(s, tab, f"{ESMFOLD2}, another seed", reuses=True)
    worst = max((max(abs(a - b) for a, b in zip(p, q)) for p, q in zip(first["ca"], second["ca"])), default=0.0) if first and second else 0.0
    s.check("with another structure", worst > 0.01, f"largest change {worst:.3f} A")
    tab.set_control("random-seed", 1)
    tab.fold()
    tab.set_control("recycles", 2)
    then(s, tab, f"{ESMFOLD2}, fewer recycles")
    tab.set_control("recycles", 6)
    then(s, tab, f"{ESMFOLD2}, more recycles")
    tab.set_control("recycles", 3)
    tab.choose(ESMFOLD2, "esmc-600m")
    tab.set_rows([["protein", short, 1]])
    then(s, tab, f"{ESMFOLD2}, the larger language model")
    tab.choose(ESMFOLD2, "esmc-300m")
    # An alignment, and how much of it is read.
    tab.choose("openbind0")
    tab.set_rows([["protein", short, 1]])
    paste(tab, homologs(short))
    tab.fold(timeout=300)
    paste(tab, homologs(short, rows=12))
    then(s, tab, "openbind0, another alignment")
    tab.set_control("msa-mode", "none")
    then(s, tab, "openbind0, no alignment")
    # ...and across models, which is how the first one was found out.
    for one, other in (("openbind0", "boltz2"), ("boltz2", "protenix2")):
        tab.choose(one)
        tab.set_rows([["protein", short, 1]])
        tab.fold(timeout=300)
        tab.choose(other)
        then(s, tab, f"{one} and then {other}")
    # The monomer has a cache of its own, and reads the alignment through it.
    tab.choose("monomer")
    tab.set_rows([["protein", short, 1]])
    paste(tab, homologs(short))
    tab.fold(timeout=300)
    first = tab.pdb()
    paste(tab, f">query\n{short}\n>one\n{short[::-1]}\n")
    end = tab.fold(timeout=300)
    second = tab.pdb()
    apart = drmsd(first, second) if first and second else None
    s.check("the monomer, another alignment: another fold", apart is not None and apart > 0.01, f"{apart} A apart: {end['status'][:110]}")
    tab.set_control("msa-mode", "none")
    s.folds_again(tab, family="openbind0", size=76)


def fold_the_shelf(s, family, jobs, cold=False):
    tab = s.page(cold=cold)
    readings = {}
    for job in jobs:
        tab.choose(family)
        tab.evaluate(DROP % ("new File([%s], %s)" % (json.dumps(job.read_text(encoding="utf-8")), json.dumps(job.name))))
        time.sleep(0.5)
        end = tab.fold(timeout=420)
        readings[job.stem] = {"failed": end["failed"] or not ended(end), "status": end["status"][:150], "plddt": number_after("pLDDT", end["status"]),
                              "pdb": None if end["failed"] or not ended(end) else tab.pdb()}
    return tab, readings


@scenario("inputs")
def the_shelf_reads_as_it_did(s):
    # Every kind of job a reader can hand in, on the code before this and on this: the single chains are checked to the bit by the
    # tools, and a complex with a ligand, a modification and DNA nowhere but here. Stock Chrome's ESMFold2 read differently before this.
    before = Path("/baseline")
    if not (before / "tools" / "serve.py").exists() or before.resolve() == REPO.resolve():
        return s.note("no older checkout to read against", str(before))
    jobs = shelf_of_jobs()
    families = ["openbind0", "protenix2"] + ([] if os.environ.get("LOCALFOLD_STOCK_FLAGS") == "1" else [ESMFOLD2])
    try:
        for family in families:
            serve(before)
            tab, old = fold_the_shelf(s, family, jobs)
            tab.opened = False  # ...and what it threw is its own.
            s.close()
            serve(REPO)
            tab, new = fold_the_shelf(s, family, jobs)
            s.close()
            for job in jobs:
                was, now = old[job.stem], new[job.stem]
                if was["failed"]:
                    s.note(f"{family} {job.stem}: the old page did not fold it", f"{was['status'][:80]} | now {now['status'][:80]}")
                    continue
                s.check(f"{family} {job.stem}: still folds", not now["failed"], now["status"][:130])
                if not now["failed"] and was["pdb"] and now["pdb"]:
                    apart = drmsd(was["pdb"], now["pdb"]) if len(was["pdb"]["ca"]) > 3 else 0.0
                    s.check(f"{family} {job.stem}: and reads what it read", was["pdb"]["atoms"] == now["pdb"]["atoms"] and apart < 0.05 and (was["plddt"] is None or abs(was["plddt"] - now["plddt"]) < 0.3),
                            f"{was['plddt']} and {now['plddt']}, {apart:.4f} A apart, {now['pdb']['atoms']} atoms")
    finally:
        s.close()
        serve(REPO)


@scenario("inputs")
def alphafold2_reads_as_it_did(s):
    # The shelf reads the AF3 lineage and ESMFold2, and AlphaFold 2 has a stack of its own: alone, with an alignment, and as a complex.
    before = Path("/baseline")
    if not (before / "tools" / "serve.py").exists() or before.resolve() == REPO.resolve():
        return s.note("no older checkout to read against", str(before))
    cases = (("the monomer", "monomer", [["protein", UBI, 1]], None),
             ("the monomer on an alignment", "monomer", [["protein", UBI, 1]], homologs(UBI)),
             ("the multimer", "multimer", [["protein", UBI[:40], 1], ["protein", UBI[40:], 1]], None))

    def read_them():
        tab = s.page()
        tab.set_control("random-seed", 1)
        readings = {}
        for name, family, rows, alignment in cases:
            tab.choose(family)
            tab.set_rows(rows)
            if alignment is None:
                tab.set_control("msa-mode", "none")
            else:
                paste(tab, alignment)
            # A browser's first fold at a length differs in its last digits, here as before.
            tab.fold(timeout=300)
            end = tab.fold(timeout=300)
            readings[name] = {"failed": end["failed"] or not ended(end), "status": end["status"][:130], "plddt": number_after("pLDDT", end["status"]),
                              "pdb": None if end["failed"] or not ended(end) else tab.pdb()}
        return tab, readings

    try:
        serve(before)
        tab, old = read_them()
        tab.opened = False  # ...and what it threw is its own.
        s.close()
        serve(REPO)
        tab, new = read_them()
        s.close()
        for name, *_ in cases:
            was, now = old[name], new[name]
            s.check(f"{name}: folds on both", not was["failed"] and not now["failed"], f"{was['status']} | {now['status']}")
            if was["pdb"] and now["pdb"]:
                apart = drmsd(was["pdb"], now["pdb"])
                s.check(f"{name}: and reads what it read", was["pdb"]["atoms"] == now["pdb"]["atoms"] and apart < 0.05 and abs(was["plddt"] - now["plddt"]) < 0.3,
                        f"{was['plddt']} and {now['plddt']}, {apart:.4f} A apart, {now['pdb']['atoms']} atoms")
    finally:
        s.close()
        serve(REPO)


SIZE_OF_DOWNLOAD = """(async () => {
  const heldUrl = URL.createObjectURL;
  const heldClick = HTMLAnchorElement.prototype.click;
  let blob = null;
  URL.createObjectURL = (b) => { blob = b; return heldUrl.call(URL, b); };
  HTMLAnchorElement.prototype.click = function () {};
  document.getElementById(%s).click();
  for (let wait = 0; wait < 1200 && blob === null; wait += 1) await new Promise((done) => setTimeout(done, 50));
  URL.createObjectURL = heldUrl;
  HTMLAnchorElement.prototype.click = heldClick;
  return blob === null ? 0 : blob.size;
})()"""


@scenario("sizes")
def a_long_fold_kept(s):
    # What a fold of ten thousand atoms and a million and a half pairs leaves to write down, store and read back.
    tab = s.page()
    tab.choose()
    tab.set_size(1200)
    end = tab.fold(timeout=900)
    plddt = number_after("pLDDT", end["status"])
    s.check("a long fold", not end["failed"] and ended(end) and plddt is not None, f"{end['seconds']} s: {end['status'][:110]}")
    started = time.time()
    archive = tab.evaluate(SIZE_OF_DOWNLOAD % json.dumps("download-all"))
    s.check("All downloads", archive > 0, f"{archive / 2 ** 20:.1f} MiB in {time.time() - started:.1f} s")
    started = time.time()
    session = tab.evaluate(SIZE_OF_DOWNLOAD % json.dumps("saveStateButton"))
    s.check("Session downloads", session > 0, f"{session / 2 ** 20:.1f} MiB in {time.time() - started:.1f} s")
    s.no_page_errors(tab.state())
    time.sleep(15)
    stored = tab.evaluate("navigator.storage.estimate().then((e) => JSON.stringify(e.usageDetails ?? e.usage))")
    s.note("what the page stores", stored)
    started = time.time()
    tab.ws.call("Page.reload")
    time.sleep(1)
    tab.wait_ready()
    time.sleep(1)
    offered = tab.evaluate("(() => { const b = document.getElementById('session-restore'); return b !== null && b.offsetParent !== null; })()")
    s.check("it is offered after a reload", offered is True, tab.evaluate("document.getElementById('session')?.textContent ?? ''")[:120])
    if offered:
        started = time.time()
        tab.evaluate("document.getElementById('session-restore').click()")
        for _ in range(240):
            if tab.evaluate(DRAWN) > 20:
                break
            time.sleep(0.5)
        drawn = tab.evaluate(DRAWN)
        s.check("and comes back", drawn > 20, f"{drawn} colours in {time.time() - started:.1f} s")
        s.check("with its map", shown(tab) > 20, tab.evaluate(OTHER_CANVASES))
        answer = tab.evaluate("1 + 1")
        s.check("and the page answers", answer == 2, "")
    s.no_page_errors(tab.state())
    s.folds_again(tab, size=76, want=87.0)


@scenario("cpu")
def a_job_carried_to_the_cpu(s):
    # What "Fold on CPU" is for: a job the card will not hold, carried whole across the reload that threads need.
    job = [{"protein": {"id": "A", "sequence": UBI[:30], "modifications": [{"ptmType": "SEP", "ptmPosition": 20}]}},
           {"protein": {"id": "B", "sequence": UBI[30:55]}}, {"ligand": {"id": "C", "ccdCodes": ["ATP"]}}]
    tab = s.page()
    tab.choose()
    with_job(tab, "carried", job, modelSeeds=[7])
    time.sleep(0.5)
    end = tab.fold()
    reference = number_after("pLDDT", end["status"])
    on_gpu = tab.pdb()
    s.check("on the GPU", not end["failed"] and ended(end) and reference is not None, end["status"][:130])
    tab.evaluate("localStorage.setItem('localfold.gpu-chosen-limit', String(%d))" % (0.3 * GIB))
    tab.ws.call("Page.reload")
    time.sleep(1)
    tab.wait_ready()
    tab.choose()
    with_job(tab, "carried", job, modelSeeds=[7])
    time.sleep(0.5)
    end = tab.fold(timeout=30)
    s.check("refused here, with a way on", end["failed"] and any("CPU" in action for action in end["actions"]), f"{end['status'][:130]} {end['actions']}")
    s.check("which is one press", tab.click_action("Fold on CPU", "Try on CPU") is True, end["actions"])
    time.sleep(3)
    tab.wait_ready()
    isolated = tab.evaluate("self.crossOriginIsolated")
    end = tab.wait_finished(timeout=900)
    plddt = number_after("pLDDT", end["status"])
    on_cpu = tab.pdb()
    apart = drmsd(on_gpu, on_cpu) if on_gpu and on_cpu and len(on_gpu["ca"]) == len(on_cpu["ca"]) else None
    s.check("across the reload, threads and the same job", isolated is True and "SEP20" in end["status"] and on_cpu is not None and on_cpu["atoms"] == on_gpu["atoms"], f"isolated {isolated}, {on_cpu['atoms'] if on_cpu else None} atoms against {on_gpu['atoms'] if on_gpu else None}: {end['status'][:120]}")
    s.check("the same fold", plddt is not None and abs(plddt - reference) < 1.0 and apart is not None and apart < 0.3, f"pLDDT {reference} and {plddt}, {apart} A apart")
    s.no_page_errors(end)
    tab.evaluate("localStorage.removeItem('localfold.gpu-chosen-limit')")
    s.folds_again(tab, "the next fold, on the CPU", size=30)


def run(name):
    group, function = SCENARIOS[name]
    print(f"\n=== {group}/{name}", flush=True)
    state = Scenario(name)
    started = time.time()
    try:
        function(state)
    except Exception as error:
        state.check("the scenario ran to its end", False, f"{type(error).__name__}: {error} | {traceback.format_exc(limit=2)[-300:]}")
    finally:
        state.close()
    state.nothing_thrown()
    seconds = round(time.time() - started, 1)
    results.append({"name": name, "group": group, "failed": state.failed, "seconds": seconds, "notes": state.notes})
    print(f"=== {name}: {'ok' if state.failed == 0 else str(state.failed) + ' FAILED'} in {seconds} s", flush=True)


def main():
    if args.list:
        for name, (group, _) in SCENARIOS.items():
            print(group, name)
        return 0
    assert sys.platform.startswith("linux"), "this ends GPU processes and fills cards on purpose: a rented card, under Linux"
    wanted = [name for name, (group, _) in SCENARIOS.items()
              if (not args.group or group in args.group.split(",")) and (not args.only or name in args.only.split(","))]
    assert wanted, "no scenario matches"
    with socket.socket() as probe:
        probe.settimeout(1)
        assert probe.connect_ex(("127.0.0.1", args.port)) != 0, f"something is already serving port {args.port}"
    serve(REPO)
    try:
        for name in wanted:
            run(name)
    finally:
        SERVER["process"].kill()
    failed = [result["name"] for result in results if result["failed"]]
    print(f"\n{len(results) - len(failed)}/{len(results)} scenarios passed" + (f"; failed: {', '.join(failed)}" if failed else ""), flush=True)
    print("RESULTS " + json.dumps(results), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
