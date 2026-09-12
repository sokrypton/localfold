/**
 * Run a WebGPU module in headless Chrome and print what it returns.
 *
 * WHY THIS EXISTS. The Dawn lane (`npm run test:gpu`, package `webgpu`) cannot
 * load on macOS 13: every published build of dawn.node, back to 0.3.5, links
 * `_OBJC_CLASS_$_MTLLogStateDescriptor`, which is macOS 15. So on this machine
 * there is no native WebGPU for node, and without a GPU lane no AF3 kernel can
 * have the differential test AGENTS.md requires. Chrome has WebGPU on this OS
 * (Apple, metal-3, shader-f16), so the lane runs there instead.
 *
 * 🔴 PLAYWRIGHT CANNOT DRIVE THIS. Launched through Playwright - headless or
 * headed, with --enable-unsafe-webgpu, with default args dropped - Chrome
 * exposes no `navigator.gpu` at all. Chrome launched directly does. Do not
 * "simplify" this file back onto Playwright without re-checking that, because
 * the failure is silent: the page just sees an undefined `navigator.gpu` and
 * every GPU test skips rather than fails.
 *
 * The entry module must export `main(device, args)` and return something
 * JSON-serialisable. It is fetched over http rather than file://, so its
 * relative imports into src/ resolve exactly as they do on the served page.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/bench-triangle.js --lengths=300,600
 */
import { spawn, spawnSync } from "node:child_process";
import { createReadStream, rmSync } from "node:fs";
import { stat } from "node:fs/promises";
import { createServer } from "node:http";
import { extname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(fileURLToPath(new URL("..", import.meta.url)));
// 🔴 THE BROWSER IS NOT ALWAYS AT THE MAC PATH. This was a hard-coded
// /Applications path, which makes every GPU tool in this repository - the
// benches, the probes, the differential checkers - unrunnable anywhere but a
// Mac. LOCALFOLD_CHROME overrides it; otherwise it is the Mac bundle on darwin
// and `google-chrome` on the PATH elsewhere.
const CHROME = process.env.LOCALFOLD_CHROME
  ?? (process.platform === "darwin"
    ? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    : "google-chrome");

// 🔴 THE SAME HARNESS, ON A PHONE. `LOCALFOLD_GPU_ANDROID=1` runs the module in
// Chrome on a USB-attached Android device instead of locally, and it needs no
// remote-debugging protocol at all: everything here is already one HTTP origin
// - the runner page, every ES module under src/, the weight shards and the
// POST that carries the result back - so `adb reverse` pointing the phone's
// 127.0.0.1 at this server is the whole of the port.
//
// It matters because PRIORS has two entries. Every other GPU in the world takes
// DEFAULT_TUNING, which costs 1.5x on AF2 and ESMFold2 (see --no-prior), and
// the device class most users actually hold has never run a kernel from this
// repository.
const ANDROID = process.env.LOCALFOLD_GPU_ANDROID === "1";
const ADB = process.env.LOCALFOLD_ADB
  ?? `${process.env.HOME}/Library/Android/sdk/platform-tools/adb`;
// 🔴 A FIXED PORT, BECAUSE THE INTENT HAS TO NAME IT. The desktop path listens
// on 0 and lets the OS choose, which is right when the launcher is this same
// process; `adb reverse` has to be told a number before Chrome is started.
const ANDROID_PORT = Number(process.env.LOCALFOLD_GPU_PORT ?? 8974);
const SERIAL = process.env.LOCALFOLD_ANDROID_SERIAL;

/** adb, with the serial if one was named, failing loudly. */
function adb(...argv) {
  const result = spawnSync(ADB, [...(SERIAL ? ["-s", SERIAL] : []), ...argv],
                           { encoding: "utf8" });
  if (result.error) throw new Error(`${ADB}: ${result.error.message}`);
  if (result.status !== 0) {
    throw new Error(`adb ${argv.join(" ")} failed (${result.status}): `
      + `${(result.stderr || result.stdout || "").trim()}`);
  }
  return result.stdout;
}

// 🔴 AND ON LINUX/NVIDIA THE DEFAULTS GET YOU SwiftShader, SILENTLY. Chrome
// answers `requestAdapter` with its software fallback and the page runs - at
// about 1/1000 of the device - so a bench that has not checked
// `adapter.info.vendor` is measuring a CPU. Four things are needed together
// and the discovery order was: `--enable-features=Vulkan --use-vulkan=native`
// to bring up Dawn's Vulkan backend at all; a Vulkan LOADER new enough to know
// `VK_EXT_surface_maintenance1` (Ubuntu 22.04's 1.3.204 predates the extension
// and `vkCreateInstance` fails -7, VK_ERROR_EXTENSION_NOT_PRESENT); and then -
// the one that costs the most time to find - NO `--headless=new`, because
// headless Chrome demands `VK_EXT_headless_surface`, which the NVIDIA driver
// does not implement and Mesa's llvmpipe does. So it runs HEADFUL against an
// X server, which on a GPU box means Xvfb. `LOCALFOLD_HEADLESS=1` forces the
// headless flag back on for a machine whose driver does have that extension.
const LINUX_HEADLESS = process.env.LOCALFOLD_HEADLESS === "1";
//
// 🔴 AND `shader-f16` IS OFF ON EVERY NVIDIA GPU UNTIL A DAWN TOGGLE SAYS
// OTHERWISE. This is not the driver and not the hardware. Dawn's
// PhysicalDeviceVk.cpp passes this device on all four of its stated
// conditions - `VK_KHR_shader_float16_int8`, `shaderFloat16`, `shaderInt16`
// and `storageBuffer16BitAccess` are all true here, checked against the exact
// structs it reads - and then refuses anyway, a few hundred lines further
// down, in a second gate nothing in the first hints at:
//
//     // TODO(crbug.com/42251215): Investigate f16 CTS test failures ...
//     if (gpu_info::IsNvidia(mVendorId) &&
//         !toggles.IsEnabled(Toggle::VulkanEnableF16OnNvidia)) { ... }
//
// So it is a policy block on the whole vendor pending a CTS investigation,
// and `vulkan_enable_f16_on_nvidia` lifts it. With the toggle this adapter
// advertises 24 features instead of 23 and the whole f16 path in src/ - which
// is most of this repository's fast path - switches back on. It carries
// Dawn's own caveat: they turned it off because f16 CTS tests were failing.
// docs/A100.md records what this repository's differential checkers say
// about that, which is the only evidence here that bears on it.
//
// 🔴 AND TWO OF THESE ARE DEVELOPER FLAGS A USER'S CHROME DOES NOT HAVE, WHICH
// MAKES EVERY NUMBER TAKEN HERE A CONFIGURATION AND NOT A DELIVERY. Measured on
// this card, Chrome 152, by asking the adapter with each flag removed:
//
//   both flags                     shader-f16 YES  subgroup-matrix YES  25 features
//   without the dawn f16 toggle    shader-f16 NO   subgroup-matrix YES  24
//   without --enable-unsafe-webgpu shader-f16 YES  subgroup-matrix NO   20
//   without either                 shader-f16 NO   subgroup-matrix NO   19
//
// So on an NVIDIA GPU a stock Chrome gives this port neither its f16 storage
// nor its matrix units. `LOCALFOLD_STOCK_FLAGS=1` drops both, which is how to
// ask what a visitor actually gets rather than what this box can be made to do.
// tools/cdp.py carries the same two flags, so the PAGE numbers are no different.
const STOCK_FLAGS = process.env.LOCALFOLD_STOCK_FLAGS === "1";
const PLATFORM_FLAGS = process.platform === "linux"
  ? ["--use-angle=vulkan", "--enable-features=Vulkan", "--use-vulkan=native",
     "--ignore-gpu-blocklist", "--no-sandbox",
     ...(STOCK_FLAGS ? [] : ["--enable-dawn-features=vulkan_enable_f16_on_nvidia"])]
  : [];
const TYPES = {
  ".js": "text/javascript", ".mjs": "text/javascript", ".html": "text/html",
  ".json": "application/json", ".wgsl": "text/plain", ".bin": "application/octet-stream",
};

const [entry, ...args] = process.argv.slice(2);
if (entry === undefined) {
  console.error("usage: node tools/gpu-chrome.mjs <module.js> [args...]");
  process.exit(2);
}

/**
 * The page. It owns the device so the entry module does not have to, and it
 * reports failures as results rather than letting them vanish into a headless
 * console nobody reads.
 */
function runnerPage(modulePath, moduleArgs) {
  return `<!doctype html><meta charset="utf-8"><title>gpu-chrome</title><body>
<script type="module">
// fetch rather than sendBeacon: a beacon is capped near 64 kB and fails
// silently past it, which would turn a large result into a hang.
const post = (body) => fetch("/__result", { method: "POST", body: JSON.stringify(body) });
const logs = [];
for (const level of ["log", "warn", "error"]) {
  const original = console[level].bind(console);
  console[level] = (...parts) => {
    logs.push(parts.map((part) => typeof part === "string" ? part : JSON.stringify(part)).join(" "));
    original(...parts);
  };
}
try {
  if (!navigator.gpu) throw new Error("no navigator.gpu: Chrome was launched without WebGPU");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (adapter === null) throw new Error("no WebGPU adapter");
  // 🔴 THE SAME DEVICE THE PAGE ASKS FOR, FROM THE SAME PLACE. This used to
  // hand-roll its own requiredLimits, and had already drifted: it was missing
  // maxComputeWorkgroupsPerDimension, so a bench could pass on a device the
  // page does not get. A kernel that picks its shape from device.limits - the
  // diffusion transformer picks its token tile that way - would then be
  // measured in one configuration and shipped in another.
  const { requestAlphaFoldDevice } = await import("/src/runtime/device.js");
  const device = await requestAlphaFoldDevice(adapter);
  device.addEventListener("uncapturederror", (event) => {
    post({ ok: false, error: "uncaptured: " + event.error.message, logs });
  });
  // 🔴 ONE SWITCH, AND EVERY TOOL GETS IT. --f16=off forces the f32 path on a
  // device that HAS shader-f16, so the two arms can be compared without
  // relaunching the browser - which is what makes them comparable at all on a
  // machine that drifts. --f16=on is the default and is accepted so a script
  // can name both arms symmetrically. NOTE: no backticks in this comment; it
  // is inside the runner page's template literal and one would end it.
  // 🔴 --no-prior MAKES THIS DEVICE ANSWER AS AN UNRECOGNISED ONE. Two
  // architectures have priors and every other GPU takes DEFAULT_TUNING; this is
  // how a machine that HAS a prior measures what the machines that do not are
  // getting. See ignoreDevicePrior in src/runtime/device-profile.js.
  // --no-prior alone drops every prior knob; --no-prior=a,b keeps those two AT
  // THE PRIOR'S OWN VALUES and drops the rest, which is how a sweep asks what
  // one knob is worth without having to spell an object on a command line.
  // NOTE: no backticks anywhere in this comment - it is inside the runner
  // page's template literal and one would end it, which it just did.
  const priorArg = ${JSON.stringify(moduleArgs)}.find((a) => a === "--no-prior" || a.startsWith("--no-prior="));
  if (priorArg !== undefined) {
    const { ignoreDevicePrior, deviceTuning: deviceTuningOf }
      = await import("/src/runtime/device-profile.js");
    const keep = priorArg.includes("=")
      ? priorArg.slice("--no-prior=".length).split(",").filter(Boolean) : [];
    ignoreDevicePrior(device, keep);
    console.log("[gpu-chrome] device priors ignored"
      + (keep.length === 0 ? "" : ", keeping " + keep.join(" ")));
    // 🔴 AND AN UNRECOGNISED DEVICE MEASURES ITS OWN WIDTH, which is half of
    // what this arm is for. requestAlphaFoldDevice starts that only where the
    // tuning has no diffusion rule, and at the moment it ran this device still
    // had its prior - so without this the arm would report what an unrecognised
    // device gets MINUS the measurement it would have made. Awaited, because a
    // tool folds immediately where a page has a download to hide it behind.
    if (deviceTuningOf(device).diffusionSplitK === null
        && !${JSON.stringify(moduleArgs)}.some((a) => a.startsWith("--occupancy="))) {
      const { measureDeviceOccupancy, deviceOccupancyDetail }
        = await import("/src/runtime/occupancy.js");
      const width = await measureDeviceOccupancy(device);
      console.log("[gpu-chrome] measured saturation: " + width + " workgroups  "
        + JSON.stringify(deviceOccupancyDetail(device)));
    }
  }
  // --default-tuning strips the CAPABILITY layer as well as the prior, which
  // is the only way to measure what DEFAULT_TUNING alone is worth now that a
  // device with matrix units gets them without a prior. --no-prior is the
  // other question: an unrecognised device with whatever units it has.
  if (${JSON.stringify(moduleArgs)}.includes("--default-tuning")) {
    const { ignoreDevicePrior, ignoreDeviceCapabilities }
      = await import("/src/runtime/device-profile.js");
    ignoreDevicePrior(device, []);
    ignoreDeviceCapabilities(device);
    console.log("[gpu-chrome] prior AND capability tuning ignored");
  }
  // 🔴 --occupancy=<n> ANSWERS AS A DEVICE OF THAT WIDTH. The width-driven
  // derivations are the default for every GPU with no prior, and one machine
  // can only measure its own; this is how it asks what a narrower one gets.
  const occupancyArg = ${JSON.stringify(moduleArgs)}.find((a) => a.startsWith("--occupancy="));
  if (occupancyArg !== undefined) {
    const { setDeviceOccupancy } = await import("/src/runtime/occupancy.js");
    setDeviceOccupancy(device, Number(occupancyArg.slice("--occupancy=".length)));
    console.log("[gpu-chrome] answering as a device of "
      + occupancyArg.slice("--occupancy=".length) + " workgroups");
  }
  // 🔴 --tune-json IS HOW AN OBJECT-VALUED KNOB IS REACHED AT ALL. Each tool's
  // own --tune splits its argument on commas, so a knob whose value is an
  // object or a list - matrixLinear, diffusionSplitK, atomRowTile,
  // diffusionTokenTile, trianglePairProjectTile - could not be written on a
  // command line and was therefore never in any arm. matrixLinear being one of
  // them is exactly how its missing off position survived: the audit that would
  // have caught it could not express the knob. One JSON object, parsed whole.
  //     --tune-json={"matrixLinear":false}
  const tuneJson = ${JSON.stringify(moduleArgs)}.find((a) => a.startsWith("--tune-json="));
  if (tuneJson !== undefined) {
    const { setDeviceTuning } = await import("/src/runtime/device-profile.js");
    const patch = JSON.parse(tuneJson.slice("--tune-json=".length));
    setDeviceTuning(device, patch);
    console.log("[gpu-chrome] tuning patched with " + JSON.stringify(patch));
  }
  const f16Arg = ${JSON.stringify(moduleArgs)}.find((a) => a.startsWith("--f16="));
  if (f16Arg !== undefined) {
    const { setHalfPrecision } = await import("/src/runtime/device-profile.js");
    const wanted = f16Arg.slice("--f16=".length);
    if (wanted !== "on" && wanted !== "off") throw new Error("--f16 takes on or off, not " + wanted);
    setHalfPrecision(device, wanted === "on");
    console.log("[gpu-chrome] half precision forced " + wanted
      + " (device " + (device.features.has("shader-f16") ? "has" : "lacks") + " shader-f16)");
  }
  const module = await import(${JSON.stringify(modulePath)});
  if (typeof module.main !== "function") throw new Error("module exports no main(device, args)");
  const value = await module.main(device, ${JSON.stringify(moduleArgs)});
  post({ ok: true, value, logs, adapter: {
    vendor: adapter.info?.vendor, architecture: adapter.info?.architecture,
    description: adapter.info?.description, features: [...device.features],
  } });
} catch (error) {
  post({ ok: false, error: error && error.stack ? error.stack : String(error), logs });
}
</script>`;
}

async function main() {
  const modulePath = "/" + relative(ROOT, resolve(entry)).split("\\").join("/");
  const page = runnerPage(modulePath, args);

  let settle;
  const finished = new Promise((res) => { settle = res; });

  const server = createServer((request, response) => {
    const url = new URL(request.url, "http://127.0.0.1");
    if (url.pathname === "/__result" && request.method === "POST") {
      const chunks = [];
      request.on("data", (chunk) => chunks.push(chunk));
      request.on("end", () => {
        response.writeHead(204).end();
        try { settle(JSON.parse(Buffer.concat(chunks).toString("utf8"))); }
        catch (error) { settle({ ok: false, error: `unparseable result: ${error.message}` }); }
      });
      return;
    }
    if (url.pathname === "/__runner") {
      response.writeHead(200, { "content-type": "text/html" }).end(page);
      return;
    }
    // ...everything else is the repo, so the entry module's imports resolve.
    const target = join(ROOT, decodeURIComponent(url.pathname));
    if (!target.startsWith(ROOT)) { response.writeHead(403).end(); return; }
    stat(target).then((info) => {
      if (!info.isFile()) throw new Error("not a file");
      response.writeHead(200, {
        "content-type": TYPES[extname(target)] ?? "application/octet-stream",
        "content-length": info.size,
        // 🔴 NO-STORE, BECAUSE THE ANDROID PATH REUSES THE USER'S OWN CHROME
        // PROFILE. The desktop path gets a fresh user-data-dir every run and so
        // could never see this, but on the phone a second run would be served
        // yesterday's src/ out of the HTTP cache and the change would look like
        // it had not landed - which is the trap CLAUDE.md records costing three
        // sessions on the page. Costs the desktop nothing: its cache is empty.
        "cache-control": "no-store",
      });
      createReadStream(target).pipe(response);
    }).catch(() => response.writeHead(404).end("not found"));
  });

  await new Promise((res) => server.listen(ANDROID ? ANDROID_PORT : 0, "127.0.0.1", res));
  const port = server.address().port;

  // 🔴 A FRESH PROFILE EVERY RUN. Chrome refuses a second headless instance on
  // a profile already in use, so a shared one would make two concurrent tests
  // fail in a way that looks like a GPU error.
  //
  // The phone gets neither: its Chrome is the user's own, with the user's
  // profile, and nothing here may take it away from them. That is also why the
  // server sends no-store above.
  const profile = ANDROID ? null
    : join(process.env.TMPDIR ?? "/tmp", `gpu-chrome-${process.pid}-${Date.now()}`);
  const chrome = ANDROID ? null : spawn(CHROME, [
    ...(process.platform === "linux" && !LINUX_HEADLESS ? [] : ["--headless=new"]),
    ...(STOCK_FLAGS ? [] : ["--enable-unsafe-webgpu"]), "--disable-gpu-sandbox",
    ...PLATFORM_FLAGS,
    // performance.memory rounds to 100 KiB without this, which is too coarse to
    // see a tensor cache being dropped. It affects nothing else.
    "--enable-precise-memory-info",
    // ...and a collection has to be requestable, or a heap reading counts
    // whatever garbage has not been swept yet and cannot see a cache dropped.
    "--js-flags=--expose-gc",
    "--no-first-run", "--no-default-browser-check", `--user-data-dir=${profile}`,
    `http://127.0.0.1:${port}/__runner`,
  ], { stdio: ["ignore", "ignore", "pipe"] });

  const stderr = [];
  if (chrome !== null) {
    chrome.stderr.on("data", (chunk) => stderr.push(chunk.toString("utf8")));
    chrome.on("exit", (code) => {
      settle({ ok: false, error: `Chrome exited (${code}) before reporting\n${stderr.join("")}` });
    });
  } else {
    // 🔴 THE REVERSE TUNNEL BEFORE THE INTENT, or the page loads against
    // nothing and Chrome shows its own offline error - which reaches this
    // process as a timeout naming no cause at all.
    adb("reverse", `tcp:${port}`, `tcp:${port}`);
    // 🔴 AND THE SCREEN HAS TO STAY ON. Android throttles and then suspends a
    // backgrounded tab; a fold that takes minutes will simply stop, and the
    // result POST never comes. This is the phone's own setting and it is left
    // as it was found - see the teardown.
    adb("shell", "svc", "power", "stayon", "usb");
    adb("shell", "am", "start", "-a", "android.intent.action.VIEW",
        "-n", "com.android.chrome/com.google.android.apps.chrome.Main",
        "-d", `http://127.0.0.1:${port}/__runner`);
  }

  const timeoutMs = Number(process.env.LOCALFOLD_GPU_TIMEOUT_MS ?? 600_000);
  const timer = setTimeout(() => {
    settle({ ok: false, error: `timed out after ${timeoutMs} ms\n${stderr.join("")}` });
  }, timeoutMs);

  const result = await finished;
  clearTimeout(timer);
  if (chrome !== null) {
    chrome.removeAllListeners("exit");
    chrome.kill("SIGKILL");
  } else {
    // Best effort, all of it: a failed run must still print WHY it failed
    // rather than an adb error raised while tidying up after it.
    for (const argv of [["shell", "am", "force-stop", "com.android.chrome"],
                        ["shell", "svc", "power", "stayon", "false"],
                        ["reverse", "--remove", `tcp:${port}`]]) {
      try { adb(...argv); } catch { /* best effort */ }
    }
  }
  server.close();

  // 🔴 AND THEN DELETE THE PROFILE, WHICH THIS DID NOT DO FOR A YEAR. A fresh
  // user-data-dir per run is right; leaving it behind is not. Each is 3 MiB to
  // 2.5 GiB depending on how much the run cached, and they accumulate silently
  // in TMPDIR because macOS only sweeps files untouched for three days and a
  // busy week never lets them go cold. This machine reached 1265 of them and
  // 428 GB - a full disk, which surfaces as git failing to write its index and
  // every redirection producing an empty file, naming nothing.
  //
  // rmSync after the kill, not before: Chrome writes on the way out.
  if (profile !== null) {
    try { rmSync(profile, { recursive: true, force: true }); } catch { /* best effort */ }
  }

  for (const line of result.logs ?? []) console.log(line);
  if (!result.ok) {
    console.error(result.error);
    process.exit(1);
  }
  if (result.value !== undefined && result.value !== null) {
    console.log(typeof result.value === "string" ? result.value
      : JSON.stringify(result.value, null, 2));
  }
}

await main();
