// The diffusion's two discretisation options (cpu/af3/diffusion/diffusion-sampler.js; cuda/af3/src/sampler.cuh has
// the same): the exponential update and the geometric schedule, against their closed forms.
import test from "node:test";
import assert from "node:assert/strict";
import { noiseSchedule, noiseLevels, samplerStep } from "../cpu/af3/diffusion/diffusion-sampler.js";

test("the exponential update is r^S x + (1 - r^S) D, and stays between x and D where Euler overshoots", () => {
  const x = Float32Array.from([10, -4]), d = Float32Array.from([1, 2]);
  const tHat = 100, level = 20, S = 1.5, keep = Math.pow(level / tHat, S);
  const got = samplerStep(x, d, tHat, level, { stepScale: S, update: "exponential" });
  for (let i = 0; i < 2; ++i) assert.ok(Math.abs(got[i] - (keep * x[i] + (1 - keep) * d[i])) < 1e-5);
  // Euler at r = 0.2, S = 1.5: x + 1.5 (0.2 - 1) (x - D) = -0.2 x + 1.2 D - past D
  const euler = samplerStep(x, d, tHat, level, { stepScale: S });
  assert.ok(Math.abs(euler[0] - (-0.2 * 10 + 1.2 * 1)) < 1e-5);
  assert.ok(got[0] > d[0] && got[0] < x[0]);
});

test("the default update is Euler's, unchanged", () => {
  const x = Float32Array.from([3]), d = Float32Array.from([1]);
  assert.deepEqual(samplerStep(x, d, 10, 9, { stepScale: 1.5 }), samplerStep(x, d, 10, 9, { stepScale: 1.5, update: "euler" }));
});

test("the geometric schedule keeps Karras's endpoints and spaces the levels evenly in log sigma", () => {
  const karras = noiseLevels(8, {}), geo = noiseLevels(8, { schedule: "geometric" });
  assert.ok(Math.abs(geo[0] - karras[0]) < 1e-9 && Math.abs(geo[8] - karras[8]) < 1e-12);
  const ratios = Array.from({ length: 8 }, (_, i) => geo[i + 1] / geo[i]);
  for (const r of ratios) assert.ok(Math.abs(r - ratios[0]) < 1e-9);
  assert.equal(noiseSchedule(0.5, {}), noiseSchedule(0.5, { schedule: "karras" }));
});

test("an unknown schedule or update is refused by name", () => {
  assert.throws(() => noiseSchedule(0.5, { schedule: "fast" }), /karras or geometric/);
  assert.throws(() => samplerStep(new Float32Array(1), new Float32Array(1), 2, 1, { update: "heun" }), /euler or exponential/);
});
