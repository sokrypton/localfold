import { describe, expect, it } from "./harness.js";
import { steerDenoised, steeringEnergy } from "../cpu/af3/diffusion/steering.js";

// 🔴 STEERING THE SAMPLER TOWARD AN EPITOPE (2026-10-28): the gradient is written by hand, so it is
// held to a finite difference of the energy it claims to descend.

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

describe("steering toward an epitope", () => {
  it("has the gradient of its energy", () => {
    const { energy, gradient } = steeringEnergy(positions, atomMask, dense, steer);
    expect(energy > 0).toBe(true);
    const h = 1e-3;
    for (let index = 0; index < positions.length; index += 1) {
      const up = Float64Array.from(positions); up[index] += h;
      const down = Float64Array.from(positions); down[index] -= h;
      const numeric = (steeringEnergy(up, atomMask, dense, steer).energy
        - steeringEnergy(down, atomMask, dense, steer).energy) / (2 * h);
      expect(Math.abs(numeric - gradient[index]) < 1e-3 * Math.max(1, Math.abs(numeric))).toBe(true);
    }
    // a masked atom is not part of any centroid and is not moved
    expect([...gradient.slice(15, 18)]).toEqual([0, 0, 0]);
  });

  it("pulls the binder toward the pocket, and leaves a binder already there alone", () => {
    const before = steeringEnergy(positions, atomMask, dense, steer).energy;
    const after = steeringEnergy(steerDenoised(positions, atomMask, dense, { ...steer, weight: 0.05 }, 80, 160),
      atomMask, dense, steer).energy;
    expect(after < before).toBe(true);
    const close = Float32Array.from(positions);
    for (let index = 0; index < 12; index += 3) close[index] += 18;
    expect(steeringEnergy(close, atomMask, dense, steer).energy).toBe(0);
    // ...and nothing at the end of the walk
    expect([...steerDenoised(positions, atomMask, dense, { ...steer, weight: 1 }, 0, 160)]).toEqual([...positions]);
  });

  it("refuses a token with no atoms", () => {
    expect(() => steeringEnergy(positions, [0, 0, ...atomMask.slice(2)], dense, steer)).toThrow(/no atoms/);
  });
});
