import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { superposePdbOnto } from "../web/pdb-superpose.js";

// 🔴 A NATIVE FOLD'S RESULT IS DRAWN MOVED ONTO ITS LAST STREAMED FRAME (web/app.js followRemoteFold), so this is what
// keeps the picture from jumping when the result lands: a structure moved by a known rotation and translation must
// come back onto its original, every atom, and the text outside the coordinates must not change.
const crystal = readFileSync(new URL("../tools/fixtures/6mrr-crystal.pdb", import.meta.url), "utf8");
const atoms = (text) => text.split("\n").filter((l) => l.startsWith("ATOM") || l.startsWith("HETATM"));
const xyz = (line) => [30, 38, 46].map((at) => Number.parseFloat(line.slice(at, at + 8)));

function moved(text, angle, axis, shift) {
  const [ux, uy, uz] = axis.map((v) => v / Math.hypot(...axis));
  const c = Math.cos(angle), s = Math.sin(angle), t = 1 - c;
  const r = [[t * ux * ux + c, t * ux * uy - s * uz, t * ux * uz + s * uy],
             [t * ux * uy + s * uz, t * uy * uy + c, t * uy * uz - s * ux],
             [t * ux * uz - s * uy, t * uy * uz + s * ux, t * uz * uz + c]];
  return text.split("\n").map((line) => {
    if (!line.startsWith("ATOM") && !line.startsWith("HETATM")) return line;
    const p = xyz(line);
    const q = r.map((row, k) => row[0] * p[0] + row[1] * p[1] + row[2] * p[2] + shift[k]);
    return line.slice(0, 30) + q.map((v) => v.toFixed(3).padStart(8)).join("") + line.slice(54);
  }).join("\n");
}

describe("a PDB superposed onto another of the same atoms", () => {
  for (const [angle, axis] of [[2.4, [1, 2, 3]], [Math.PI, [0, 0, 1]], [0.01, [1, 0, 0]]]) {
    it(`comes back onto its original (a ${angle.toFixed(2)} rad turn)`, () => {
      const back = superposePdbOnto(moved(crystal, angle, axis, [37, -12, 60]), crystal);
      const a = atoms(back), b = atoms(crystal);
      assert.equal(a.length, b.length);
      let worst = 0;
      for (let i = 0; i < a.length; i += 1) {
        worst = Math.max(worst, Math.hypot(...xyz(a[i]).map((v, k) => v - xyz(b[i])[k])));
        assert.equal(a[i].slice(0, 30) + a[i].slice(54), b[i].slice(0, 30) + b[i].slice(54));
      }
      assert.ok(worst < 0.01, `an atom ${worst.toFixed(4)} A from its place`);
    });
  }
  it("leaves a text with other atoms unmoved", () => {
    const shorter = crystal.split("\n").filter((l, i) => !(l.startsWith("ATOM") && i % 7 === 0)).join("\n");
    const far = moved(shorter, 1.0, [0, 1, 0], [50, 0, 0]);
    assert.equal(superposePdbOnto(far, crystal), far);
  });
});
