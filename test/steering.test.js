import { describe, expect, it } from "./harness.js";
import { steerDenoised, steeringEnergy, steeringSchedule } from "../cpu/af3/diffusion/steering.js";

// 🔴 STEERING THE SAMPLER TOWARD AN EPITOPE (2026-10-28, rewritten 2026-10-29): the translation is
// written by hand, so it is held to a finite difference of the energy it claims to descend - and the
// step is held to never tearing the binder or overshooting, which is what the first version did.

/** Two atoms a token, the second slot of token 2 masked; tokens 0-1 the binder, 2-3 the pocket. */
const dense = 2;
const atomMask = [1, 1, 1, 1, 1, 0, 1, 1];
const positions = Float32Array.from([
  0, 0, 0, 1, 0, 0,
  2, 1, 0, 3, 1, 0,
  20, 0, 1, 99, 99, 99,
  24, 3, 0, 25, 2, 1,
]);
const steer = { binder: [0, 1], pocket: [2, 3], cutoff: 8 };
const translated = (shift) => {
  const out = Float64Array.from(positions);
  for (let atom = 0; atom < 4; atom += 1) for (let k = 0; k < 3; k += 1) out[atom * 3 + k] += shift[k];
  return out;
};

describe("steering toward an epitope", () => {
  it("has the gradient of its energy, for a translation of the binder", () => {
    const { energy, push } = steeringEnergy(positions, atomMask, dense, steer);
    expect(energy > 0).toBe(true);
    const h = 1e-3;
    for (let k = 0; k < 3; k += 1) {
      const up = [0, 0, 0]; up[k] = h;
      const down = [0, 0, 0]; down[k] = -h;
      const numeric = (steeringEnergy(translated(up), atomMask, dense, steer).energy
        - steeringEnergy(translated(down), atomMask, dense, steer).energy) / (2 * h);
      expect(Math.abs(numeric - push[k]) < 1e-4).toBe(true);
    }
  });

  it("moves the binder as one body, by at most the weight, and never the target", () => {
    const out = steerDenoised(positions, atomMask, dense, { ...steer, weight: 1.5 }, 0);
    const shift = [0, 1, 2].map((k) => out[k] - positions[k]);
    expect(Math.abs(Math.hypot(...shift) - 1.5) < 1e-5).toBe(true);
    for (let atom = 0; atom < 4; atom += 1) {
      for (let k = 0; k < 3; k += 1) expect(Math.abs(out[atom * 3 + k] - positions[atom * 3 + k] - shift[k]) < 1e-5).toBe(true);
    }
    expect([...out.slice(12)]).toEqual([...positions.slice(12)]);
    expect(steeringEnergy(out, atomMask, dense, steer).energy
      < steeringEnergy(positions, atomMask, dense, steer).energy).toBe(true);
  });

  it("does not overshoot however strong it is asked to be", () => {
    const { largest } = steeringEnergy(positions, atomMask, dense, steer);
    const out = steerDenoised(positions, atomMask, dense, { ...steer, weight: 1000 }, 0);
    expect(Math.abs(Math.hypot(out[0] - positions[0], out[1] - positions[1], out[2] - positions[2]) - largest) < 1e-4).toBe(true);
  });

  it("leaves a binder already there alone, and stops for the last quarter of the walk", () => {
    const close = Float32Array.from(translated([18, 0, 0]));
    expect(steeringEnergy(close, atomMask, dense, steer).energy).toBe(0);
    expect([...steerDenoised(close, atomMask, dense, { ...steer, weight: 1 }, 0)]).toEqual([...close]);
    expect([steeringSchedule(0), steeringSchedule(0.5), steeringSchedule(0.8)]).toEqual([1, 0.5, 0]);
    expect([...steerDenoised(positions, atomMask, dense, { ...steer, weight: 1 }, 0.8)]).toEqual([...positions]);
  });

  it("refuses a token with no atoms", () => {
    expect(() => steeringEnergy(positions, [0, 0, ...atomMask.slice(2)], dense, steer)).toThrow(/no atoms/);
  });
});
