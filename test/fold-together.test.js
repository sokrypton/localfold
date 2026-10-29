import { describe, expect, it } from "./harness.js";
import { meanOfDenoisers } from "../cpu/af3/diffusion/together.js";

// 🔴 FOLDING WITH SEVERAL MODELS AT ONCE (2026-10-29): their denoised answers are fitted onto the
// leader's and averaged slot by slot, so a partner in a rotated frame must not smear the mean.

const atoms = 4;
const own = Float32Array.from([0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3]);
const rotate = (p) => Float32Array.from({ length: p.length }, (_, i) => {
  const [x, y, z] = [p[i - (i % 3)], p[i - (i % 3) + 1], p[i - (i % 3) + 2]];
  return [-y + 5, x - 1, z + 2][i % 3];   // 90 degrees about z, then moved
});

describe("folding together", () => {
  it("fits a partner onto the leader before averaging", () => {
    const mean = meanOfDenoisers(own, [1, 1, 1, 1], atoms, [{ positions: rotate(own), atomMask: [1, 1, 1, 1] }]);
    for (let i = 0; i < own.length; i += 1) expect(Math.abs(mean[i] - own[i]) < 1e-4).toBe(true);
  });

  it("averages where both keep an atom, and keeps the leader's where only it does", () => {
    const other = Float32Array.from(own); other[9 + 2] = 5;
    const mean = meanOfDenoisers(own, [1, 1, 1, 1], atoms, [{ positions: other, atomMask: [1, 1, 1, 0] }]);
    expect([...mean.slice(0, 9)].every((v, i) => Math.abs(v - own[i]) < 1e-4)).toBe(true);
    expect(Math.abs(mean[11] - 3) < 1e-4).toBe(true);
  });

  it("refuses a partner laid out differently", () => {
    expect(() => meanOfDenoisers(own, [1, 1, 1, 1], atoms, [{ positions: new Float32Array(9), atomMask: [1, 1, 1] }]))
      .toThrow(/lay the input out alike/);
  });
});
