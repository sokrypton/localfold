import { test } from "node:test";
import assert from "node:assert/strict";
import { inpaintDenoised, motifSlots } from "../cpu/af3/diffusion/inpaint.js";

// two tokens of chain 1 (dense 3: N, CA, C) and a three-atom ligand chain 2, one token an atom
const names = [["N", "CA", "C"], ["N", "CA", "C"], ["C1", "", ""], ["O1", "", ""], ["C2", "", ""]];
const batch = {
  asymId: Int32Array.from([1, 1, 2, 2, 2]),
  predDenseAtomMask: Float32Array.from(names.flatMap((row) => row.map((n) => (n ? 1 : 0)))),
  refAtomNameChars: Int32Array.from(names.flatMap((row) => row.flatMap((n) => [0, 1, 2, 3].map((k) => (n[k] ? n.charCodeAt(k) - 32 : 0))))),
};

test("a motif's token-and-name entries resolve to dense slots, a negative token a whole chain", () => {
  const { slots } = motifSlots(batch, [
    { token: 1, atom: "CA", xyz: [0, 0, 0] }, { token: -2, atom: "O1", xyz: [1, 0, 0] }, { token: -2, atom: "C2", xyz: [0, 1, 0] },
  ]);
  assert.deepEqual([...slots], [4, 9, 12]);
});

test("a name that is not exactly one atom is refused, never guessed past", () => {
  assert.throws(() => motifSlots(batch, [{ token: -1, atom: "CA", xyz: [0, 0, 0] }, { token: 0, atom: "N", xyz: [0, 0, 0] },
    { token: 0, atom: "C", xyz: [0, 0, 0] }]), /2 atoms/);
  assert.throws(() => motifSlots(batch, [{ token: 0, atom: "CB", xyz: [0, 0, 0] }, { token: 0, atom: "N", xyz: [0, 0, 0] },
    { token: 0, atom: "C", xyz: [0, 0, 0] }]), /0 atoms/);
  assert.throws(() => motifSlots(batch, [{ token: 0, atom: "N", xyz: [0, 0, 0] }]), /three/);
});

test("an optional name the input lacks is left out and counted; an optional duplicate is still refused", () => {
  const { slots, dropped } = motifSlots(batch, [{ token: 0, atom: "OXT", xyz: [0, 0, 0], optional: true },
    { token: 0, atom: "N", xyz: [0, 0, 0] }, { token: 0, atom: "CA", xyz: [1, 0, 0] }, { token: 0, atom: "C", xyz: [0, 1, 0] }]);
  assert.deepEqual([...slots], [0, 1, 2]);
  assert.equal(dropped, 1);
  assert.throws(() => motifSlots(batch, [{ token: -1, atom: "CA", xyz: [0, 0, 0], optional: true },
    { token: 0, atom: "N", xyz: [0, 0, 0] }, { token: 0, atom: "C", xyz: [0, 0, 0] }]), /2 atoms/);
});

const target = Float32Array.from([0, 0, 0, 1.5, 0, 0, 0, 1.4, 0, 0.3, 0.2, 1.1]);
const motif = { slots: Int32Array.from([1, 3, 6, 8]), xyz: target };
/** `target` under a rotation about z and a translation, written into a 15-slot answer at the motif's slots. */
function answerFrom(points, theta, shift) {
  const out = new Float32Array(15 * 3).map((_, i) => 0.37 * i);
  const c = Math.cos(theta), s = Math.sin(theta);
  motif.slots.forEach((slot, k) => {
    const [x, y, z] = points.subarray(k * 3, k * 3 + 3);
    out.set([c * x - s * y + shift[0], s * x + c * y + shift[1], z + shift[2]], slot * 3);
  });
  return out;
}

test("an answer that already holds the motif comes back unchanged", () => {
  const answer = answerFrom(target, 0.7, [3, -2, 5]);
  const out = inpaintDenoised(answer, motif);
  for (let i = 0; i < out.length; i += 1) assert.ok(Math.abs(out[i] - answer[i]) < 1e-4, `coordinate ${i}`);
});

test("a distorted motif is replaced by the target's exact shape, and nothing else moves", () => {
  const bent = Float32Array.from(target).map((v, i) => v + (i % 5 === 0 ? 0.6 : -0.2));
  const answer = answerFrom(bent, -1.1, [10, 4, -7]);
  const out = inpaintDenoised(answer, motif);
  const d = (a, i, j) => Math.hypot(a[i * 3] - a[j * 3], a[i * 3 + 1] - a[j * 3 + 1], a[i * 3 + 2] - a[j * 3 + 2]);
  for (let i = 0; i < 4; i += 1) for (let j = i + 1; j < 4; j += 1) {
    assert.ok(Math.abs(d(out.subarray(0), motif.slots[i], motif.slots[j]) - d(target, i, j)) < 1e-4, `pair ${i}-${j}`);
  }
  for (let slot = 0; slot < 15; slot += 1) {
    if (motif.slots.includes(slot)) continue;
    for (let a = 0; a < 3; a += 1) assert.equal(out[slot * 3 + a], answer[slot * 3 + a]);
  }
  assert.notEqual(out, answer);
});
