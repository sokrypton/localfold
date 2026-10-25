/**
 * Compare two `check-rfd4p-length.js --arm=` runs on the host.
 *
 *   node tools/gpu/compare-rfd4p-arms.mjs f16.json f32.json
 *
 * 🔴 IT IS A SEPARATE PROCESS BECAUSE THE RUNNER HAS A DEADLINE. `tools/gpu-chrome.mjs`
 * gives a module 600 seconds and both arms of a 60-step generation at n = 400 take about
 * fifteen minutes, so the one-process version returned a timeout and no measurement. Each
 * arm is its own run now and the arithmetic that compares them does not need a GPU.
 *
 * The REFERENCE is the f32 arm: it is the same code path the oracle holds at 100 tokens.
 * The bound is 0.5 A, which is what `tools/gpu/check-generate.js` already holds its own
 * half arm to - and it is not widened here for a longer chain.
 */
import { readFileSync } from 'node:fs';

const TOL_A = 0.5;

/** A runner's stdout may carry a banner line before the JSON. */
function readReport(path) {
  const text = readFileSync(path, 'utf8');
  const at = text.indexOf('{');
  if (at < 0) throw new Error(`${path}: no JSON in it`);
  return JSON.parse(text.slice(at));
}

function caDeviation(a, b) {
  const n = Math.min(a.length, b.length) / 3;
  let sum = 0;
  let worst = 0;
  let nonFinite = 0;
  for (let i = 0; i < n; i += 1) {
    if (!Number.isFinite(a[i * 3]) || !Number.isFinite(b[i * 3])) nonFinite += 1;
    const dx = a[i * 3] - b[i * 3];
    const dy = a[i * 3 + 1] - b[i * 3 + 1];
    const dz = a[i * 3 + 2] - b[i * 3 + 2];
    sum += Math.sqrt(dx * dx + dy * dy + dz * dz) * 10;
    worst = Math.max(worst, Math.sqrt(dx * dx + dy * dy + dz * dz) * 10);
  }
  return { meanA: sum / Math.max(n, 1), worstA: worst, nonFinite, residues: n };
}

const [left, right] = process.argv.slice(2);
if (!left || !right) {
  console.error('usage: compare-rfd4p-arms.mjs <one.json> <other.json>');
  process.exit(2);
}
const one = readReport(left);
const other = readReport(right);
const f16 = one.arm === 'f16' ? one : other;
const f32 = one.arm === 'f32' ? one : other;
if (f16.arm !== 'f16' || f32.arm !== 'f32') {
  throw new Error(`need one f16 arm and one f32 arm, got ${one.arm} and ${other.arm}`);
}
// 🔴 TWO RUNS OF DIFFERENT THINGS ARE NOT A COMPARISON.
for (const field of ['length', 'nsteps', 'seed']) {
  if (f16[field] !== f32[field]) {
    throw new Error(`${field} differs: ${f16[field]} vs ${f32[field]} - not the same design`);
  }
}

const deviation = caDeviation(f16.ca, f32.ca);
const ok = deviation.nonFinite === 0 && deviation.meanA <= TOL_A && f32.isProtein === true;
const report = {
  length: f16.length,
  nsteps: f16.nsteps,
  tolA: TOL_A,
  f16: { triF16: f16.triF16, chunks: f16.chunks, wallMs: f16.wallMs, spacing: f16.spacing, isProtein: f16.isProtein },
  f32: { triF16: f32.triF16, chunks: f32.chunks, wallMs: f32.wallMs, spacing: f32.spacing, isProtein: f32.isProtein },
  speedup: Number((f32.wallMs / f16.wallMs).toFixed(2)),
  deviation: {
    meanA: Number(deviation.meanA.toFixed(3)),
    worstA: Number(deviation.worstA.toFixed(3)),
    nonFinite: deviation.nonFinite,
    residues: deviation.residues,
  },
  referenceIsProtein: f32.isProtein === true,
  ok,
};
console.log(JSON.stringify(report, null, 2));
if (!ok) process.exit(1);
