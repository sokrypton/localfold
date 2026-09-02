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
import { spawn } from "node:child_process";
import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { createServer } from "node:http";
import { extname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(fileURLToPath(new URL("..", import.meta.url)));
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
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
      });
      createReadStream(target).pipe(response);
    }).catch(() => response.writeHead(404).end("not found"));
  });

  await new Promise((res) => server.listen(0, "127.0.0.1", res));
  const port = server.address().port;

  // 🔴 A FRESH PROFILE EVERY RUN. Chrome refuses a second headless instance on
  // a profile already in use, so a shared one would make two concurrent tests
  // fail in a way that looks like a GPU error.
  const profile = join(process.env.TMPDIR ?? "/tmp", `gpu-chrome-${process.pid}-${Date.now()}`);
  const chrome = spawn(CHROME, [
    "--headless=new", "--enable-unsafe-webgpu", "--disable-gpu-sandbox",
    "--no-first-run", "--no-default-browser-check", `--user-data-dir=${profile}`,
    `http://127.0.0.1:${port}/__runner`,
  ], { stdio: ["ignore", "ignore", "pipe"] });

  const stderr = [];
  chrome.stderr.on("data", (chunk) => stderr.push(chunk.toString("utf8")));
  chrome.on("exit", (code) => {
    settle({ ok: false, error: `Chrome exited (${code}) before reporting\n${stderr.join("")}` });
  });

  const timeoutMs = Number(process.env.LOCALFOLD_GPU_TIMEOUT_MS ?? 600_000);
  const timer = setTimeout(() => {
    settle({ ok: false, error: `timed out after ${timeoutMs} ms\n${stderr.join("")}` });
  }, timeoutMs);

  const result = await finished;
  clearTimeout(timer);
  chrome.removeAllListeners("exit");
  chrome.kill("SIGKILL");
  server.close();

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
