import test from 'node:test';
import assert from 'node:assert/strict';

import { ConvergenceMonitor, caRmsdA } from '../shared/rfd4p/convergence.js';

const helixNm = (n, jitter = 0) => Float32Array.from({ length: n * 3 }, (_, k) => {
  const i = Math.floor(k / 3);
  const d = k % 3;
  const v = d === 0 ? 0.23 * Math.cos(i * 1.745) : d === 1 ? 0.23 * Math.sin(i * 1.745) : 0.15 * i;
  return v + jitter * Math.sin(17 * k + 3);
});

test('caRmsdA is zero under rigid motion and masks tokens', () => {
  const a = helixNm(30);
  const c = Math.cos(0.7), s = Math.sin(0.7);
  const b = new Float32Array(a.length);
  for (let i = 0; i < 30; i += 1) {
    const [x, y, z] = [a[i * 3], a[i * 3 + 1], a[i * 3 + 2]];
    b[i * 3] = c * x - s * y + 1.5;
    b[i * 3 + 1] = s * x + c * y - 0.4;
    b[i * 3 + 2] = z + 2;
  }
  assert.ok(caRmsdA(a, b) < 1e-3);
  const moved = Float32Array.from(a);
  moved[0] += 1; // 10 Å on token 0 only
  assert.ok(caRmsdA(a, moved) > 0.5);
  const mask = Uint8Array.from({ length: 30 }, (_, i) => (i === 0 ? 0 : 1));
  assert.ok(caRmsdA(a, moved, mask) < 1e-3);
});

test('ConvergenceMonitor stops after `patience` still steps, not before minFrac, never on the last step', () => {
  const n = 20;
  const seqA = new Int32Array(n).fill(5);
  const run = (nsteps, settleAt, cfg) => {
    const m = new ConvergenceMonitor(nsteps, cfg);
    for (let s = 0; s < nsteps; s += 1) {
      const still = s >= settleAt;
      const ca = helixNm(n, still ? 0 : 0.05 * ((s % 2) + 1));
      const seq = still ? seqA : Int32Array.from(seqA, (v, i) => (i === s % n ? v + 1 : v));
      if (m.observe(s, ca, seq).stop) return s;
    }
    return null;
  };
  // settles at step 10; the 5th unchanged comparison is step 15
  assert.equal(run(60, 10, { patience: 5, minFrac: 0 }), 15);
  // minFrac 0.5 of 60 steps holds it until step index 29
  assert.equal(run(60, 10, { patience: 5, minFrac: 0.5 }), 29);
  // still from the start but too short a run: the last step is never an early stop
  assert.equal(run(6, 0, { patience: 5, minFrac: 0 }), null);
  // a sequence change resets the streak
  const m = new ConvergenceMonitor(60, { patience: 2, minFrac: 0 });
  m.observe(0, helixNm(n), seqA);
  m.observe(1, helixNm(n), seqA);
  const changed = Int32Array.from(seqA);
  changed[3] = 9;
  assert.equal(m.observe(2, helixNm(n), changed).streak, 0);
  assert.equal(m.observe(3, helixNm(n), changed).streak, 1);
  assert.equal(m.observe(4, helixNm(n), changed).stop, true);
});
