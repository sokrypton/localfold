"""A minimal CDP client: just enough WebSocket to drive headless Chrome.

A straight copy of py2Dmol's tests/cdp.py, because the fault it exists for is
the same one and this page IS that page's layout. Used by
tools/mobile-layout.py.

🔴 IT EXISTS BECAUSE --window-size CLAMPS AT 500px. Measured: --window-size=390
and =320 both report an innerWidth of 500, --headless=old clamps identically,
and --force-device-scale-factor does not help because --window-size is already
in CSS pixels. A responsive layout that is only ever measured at 500 is not
measured at all - the whole band a phone lives in is below it.

Emulation.setDeviceMetricsOverride gives a TRUE viewport at any width, and
Page.captureScreenshot gives an image, which is the other thing the flag-only
harness cannot do: --screenshot never returns on a page with a running rAF
loop, and --virtual-time-budget does not end one either.

No dependency: the WebSocket framing below is about sixty lines, against
adding websockets/playwright to a project that has none.
"""
import base64, json, os, socket, struct, subprocess, sys, time, urllib.request, shutil


class WS:
    def __init__(self, url):
        _, rest = url.split("://", 1)
        hostport, path = rest.split("/", 1)
        host, port = hostport.split(":")
        self.s = socket.create_connection((host, int(port)))
        self.s.settimeout(60)
        key = base64.b64encode(os.urandom(16)).decode()
        self.s.sendall(("GET /%s HTTP/1.1\r\nHost: %s\r\nUpgrade: websocket\r\n"
                        "Connection: Upgrade\r\nSec-WebSocket-Key: %s\r\n"
                        "Sec-WebSocket-Version: 13\r\n\r\n" % (path, hostport, key)).encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            buf += self.s.recv(4096)
        self.buf = buf.split(b"\r\n\r\n", 1)[1]
        self.id = 0

    def _recv(self, n):
        while len(self.buf) < n:
            d = self.s.recv(65536)
            if not d: raise IOError("closed")
            self.buf += d
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def send(self, obj):
        p = json.dumps(obj).encode()
        h = bytearray([0x81])
        n = len(p)
        if n < 126: h.append(0x80 | n)
        elif n < 65536: h.append(0x80 | 126); h += struct.pack(">H", n)
        else: h.append(0x80 | 127); h += struct.pack(">Q", n)
        m = os.urandom(4); h += m
        self.s.sendall(bytes(h) + bytes(b ^ m[i % 4] for i, b in enumerate(p)))

    def recv(self):
        payload = b""
        while True:
            b0, b1 = self._recv(2)
            fin, op = b0 & 0x80, b0 & 0x0F
            n = b1 & 0x7F
            if n == 126: n = struct.unpack(">H", self._recv(2))[0]
            elif n == 127: n = struct.unpack(">Q", self._recv(8))[0]
            payload += self._recv(n)
            if fin: break
        return json.loads(payload) if payload else {}

    def call(self, method, **params):
        self.id += 1
        mid = self.id
        self.send({"id": mid, "method": method, "params": params})
        while True:
            m = self.recv()
            if m.get("id") == mid:
                if "error" in m: raise RuntimeError("%s: %s" % (method, m["error"]))
                return m.get("result", {})


# 🔴 THE HARDCODED /Applications PATH MADE EVERY PAGE TOOL MAC-ONLY, and
# tools/gpu-chrome.mjs had already been through this and fixed it for the
# GPU tools. This file had not, so fold-in-page.py - the gate that exists
# BECAUSE a contact map failed to appear three times in a row - could not run
# on Linux at all, and the contact overlay broke again and nothing caught it.
# LOCALFOLD_CHROME overrides; otherwise the Mac bundle on darwin and the first
# Chrome on PATH elsewhere.
def chrome_binary():
    override = os.environ.get("LOCALFOLD_CHROME")
    if override:
        return override
    if sys.platform == "darwin":
        return "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        found = shutil.which(name)
        if found:
            return found
    raise RuntimeError("no Chrome found; set LOCALFOLD_CHROME to its path")


# 🔴 AND ON LINUX/NVIDIA `--headless=new` GETS YOU NO ADAPTER AT ALL. Headless
# Chrome wants VK_EXT_headless_surface, which the NVIDIA driver does not
# implement, so the page reports "No compatible WebGPU adapter was found" and
# the fold never starts. It has to run HEADFUL against an X server, which on a
# GPU box means Xvfb and DISPLAY=:99. These are the same flags
# tools/gpu-chrome.mjs arrived at; the note at its top has the whole discovery
# order. LOCALFOLD_HEADLESS=1 forces headless back on.
LINUX_FLAGS = ["--use-angle=vulkan", "--enable-features=Vulkan", "--use-vulkan=native",
               "--ignore-gpu-blocklist", "--no-sandbox",
               "--enable-dawn-features=vulkan_enable_f16_on_nvidia"]


def chrome_flags():
    # 🔴 macOS KEEPS EXACTLY THE FLAGS IT HAD. That path works and is the one
    # the project's own machine runs; only Linux, which could not launch at
    # all, gets anything new.
    if not sys.platform.startswith("linux"):
        return ["--headless=new"]
    headless = ["--headless=new"] if os.environ.get("LOCALFOLD_HEADLESS") == "1" else []
    return headless + LINUX_FLAGS + ["--enable-unsafe-webgpu", "--disable-gpu-sandbox"]


def launch(port, profile, keep=False):
    """Start Chrome on `profile`, wiping it first unless `keep`.

    🔴 WIPING IT IS WHY EVERY PAGE TIMING HERE IS A FIRST VISIT. A fresh
    user-data-dir has no HTTP cache and no shader cache, so a run pays the
    whole weight download AND compiles every pipeline - which is the right
    default for a checker (CLAUDE.md's note about a cached ES module looking
    exactly like a broken feature is about the other direction). It also means
    nothing here has ever measured what a RETURNING user pays, and Chrome
    caches compiled pipelines on disk.
    """
    if not keep:
        shutil.rmtree(profile, ignore_errors=True)
    p = subprocess.Popen([chrome_binary()] + chrome_flags() + [
        "--user-data-dir=" + profile, "--no-first-run",
        "--hide-scrollbars", "--remote-debugging-port=%d" % port, "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    end = time.time() + 25
    while time.time() < end:
        try:
            js = json.load(urllib.request.urlopen("http://127.0.0.1:%d/json/list" % port))
            for t in js:
                if t.get("type") == "page": return p, WS(t["webSocketDebuggerUrl"])
        except Exception: time.sleep(0.3)
    p.kill(); raise RuntimeError("no CDP target")


def evaluate(ws, expr, await_promise=True):
    r = ws.call("Runtime.evaluate", expression=expr, awaitPromise=await_promise,
                returnByValue=True)
    if "exceptionDetails" in r:
        raise RuntimeError(str(r["exceptionDetails"].get("exception", {}).get("description")))
    return r["result"].get("value")


def wait_for(ws, expr, timeout=45, what="", progress=None):
    """Poll until an expression is truthy.

    🔴 A FIXED SLEEP IS NOT A WAIT. `time.sleep(3.5)` after a navigate passed
    every time this ran alone and failed in the suite, where the ui lane starts
    a dozen browsers at once and the page had not finished its 34 scripts:
    `window.processFiles is not a function`. A probe that only passes when the
    machine is idle is worse than no probe.

    🔴 AND A TIMEOUT THAT NAMES NO STAGE COSTS THE WHOLE WAIT AGAIN. A 25-minute
    fold that ends in "timed out waiting for the fold to finish" says nothing
    about whether it was the alignment server, the template download or the
    sampler - so the only way to find out is to run it again with a print in a
    different place. `progress` is an expression polled alongside the condition
    (the page's own status line, normally) and printed WHENEVER IT CHANGES, so a
    long wait narrates itself and a timeout ends with the last stage it reached.
    """
    end = time.time() + timeout
    last = None
    seen = None
    next_progress = 0.0
    while time.time() < end:
        try:
            if evaluate(ws, expr, False):
                return True
        except Exception as e:
            last = e
        if progress is not None and time.time() >= next_progress:
            next_progress = time.time() + 2
            try:
                now = evaluate(ws, progress, False)
            except Exception:
                now = None
            if now and now != seen:
                seen = now
                print("  ...%s" % now, flush=True)
        time.sleep(0.25)
    raise RuntimeError("timed out waiting for %s%s%s"
                       % (what or expr, (" (last stage: %s)" % seen) if seen else "",
                          (" (last: %s)" % last) if last else ""))
