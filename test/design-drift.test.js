import { describe, expect, it } from "./harness.js";
import { designedDrift } from "../cpu/design/align.js";

/** CA-only residues of `chain` along a bent line, each point moved by `shift` and turned by `turn`. */
function chain(letter, n, shift = [0, 0, 0], turn = 0, offset = [0, 0, 0]) {
  return Array.from({ length: n }, (_, at) => {
    let p = [at * 3.8 + offset[0], (at % 3) * 1.7 + offset[1], (at % 2) * 2.1 + offset[2]];
    p = [p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]];
    const [c, s] = [Math.cos(turn), Math.sin(turn)];
    p = [c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]];
    return `ATOM  ${String(at + 1).padStart(5)}  CA  ALA ${letter}${String(at + 1).padStart(4)}    `
      + p.map((v) => v.toFixed(3).padStart(8)).join("") + "  1.00  0.00           C";
  });
}
const pdb = (...lines) => `${lines.flat().join("\n")}\nEND\n`;

describe("how far a designed chain moved between cycles", () => {
  // 🔴 THE CASE THE FIRST MEASURE COULD NOT SEE: the binder keeps its shape and slides 5 A along its
  // target - superposed on itself that is 0 A, and it is exactly what the viewer shows moving
  it("counts a binder sliding along its target, whatever frame the fold came out in", () => {
    const before = pdb(chain("A", 12, [0, 0, 0], 0, [0, 12, 0]), chain("B", 20));
    const slid = pdb(chain("A", 12, [5, 0, 0], 0.7, [0, 12, 0]), chain("B", 20, [0, 0, 0], 0.7));
    expect(Math.abs(designedDrift(before, slid, "A") - 5) < 1e-2).toBe(true);
    expect(designedDrift(before, before, "A") < 1e-3).toBe(true);
  });

  it("measures a monomer on itself, as the viewer fits it", () => {
    const before = pdb(chain("A", 12));
    expect(designedDrift(before, pdb(chain("A", 12, [9, 4, 0], 1.1)), "A") < 1e-2).toBe(true);
  });

  it("refuses two cycles of different lengths", () => {
    expect(() => designedDrift(pdb(chain("A", 12), chain("B", 20)), pdb(chain("A", 11), chain("B", 20)), "A"))
      .toThrow(/12 residues in the cycle before and 11 in this one/);
  });
});
