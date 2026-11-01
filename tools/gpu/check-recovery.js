/**
 * WebGPU device-loss recovery in the studio: destroy the device mid-generation and between runs; the failed run
 * must say the GPU was reset, and the next Generate must succeed on a new device (web/app.js `_watchDevice`).
 *
 *   node tools/gpu-chrome.mjs tools/gpu/check-recovery.js
 */
export async function main() {
  const iframe = document.createElement('iframe');
  iframe.style.width = '1440px';
  iframe.style.height = '960px';
  document.body.appendChild(iframe);
  await new Promise((resolve, reject) => { iframe.onload = resolve; iframe.onerror = reject; iframe.src = '/index.html'; });
  const win = iframe.contentWindow;
  const t0 = performance.now();
  while (!win.rfd4pApp || win.rfd4pApp.els.predictBtn.disabled) {
    if (performance.now() - t0 > 60000) throw new Error('app init timeout');
    await new Promise((r) => setTimeout(r, 100));
  }
  const app = win.rfd4pApp;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const run = async () => {
    await app.runDesign({ nsteps: 8, recycleSchedule: 'none' });
    return { status: app.els.statusMessage.textContent.slice(0, 160), ok: Boolean(app.lastResult?.cif) && !app.els.statusMessage.classList.contains('error') };
  };
  const out = {};
  out.first = await run();
  const dev1 = app.gpuDevice;
  // 1. Lose the device mid-generation.
  app.lastResult = null;
  const pending = run();
  await sleep(1500);
  dev1.destroy();
  out.midRun = await pending;
  out.lostInfo = app.gpuDeviceLost;
  // 2. The next Generate must recover on a new device.
  app.lastResult = null;
  out.afterMidRun = await run();
  out.newDevice = app.gpuDevice && app.gpuDevice !== dev1;
  // 3. Lose it between runs.
  const dev2 = app.gpuDevice;
  if (!dev2) return { ...out, allOk: false, note: 'no device after the mid-run recovery', lostAfter: app.gpuDeviceLost };
  dev2.destroy();
  await sleep(200);
  app.lastResult = null;
  out.afterIdleLoss = await run();
  out.thirdDevice = app.gpuDevice && app.gpuDevice !== dev2;
  out.allOk = out.first.ok && !out.midRun.ok && /GPU was reset/.test(out.midRun.status) && out.afterMidRun.ok &&
    out.newDevice && out.afterIdleLoss.ok && out.thirdDevice;
  return out;
}
