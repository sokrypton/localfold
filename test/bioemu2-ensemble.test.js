/**
 * The BioEmu2 ensemble's PCA, contact frequency and play-bar order, and how an ensemble splits
 * into denoiser batches.
 *
 * The PCA is held to a case with a known answer: two clusters along one axis must separate on
 * PC1, and a pure stretch must put the whole variance there. A PCA that returned noise would
 * still plot points, which is why the separation is asserted rather than looked at.
 */
import assert from "node:assert/strict";
import { describe, it } from "./harness.js";
import { distanceFeatures, ensembleContacts, ensemblePca, orderAlongPc1, pca2, representativeAtoms }
  from "../web/bioemu2-ensemble.js";
import { bioemu2Batches } from "../webgpu/bioemu2/fold.js";

// a straight chain of `n` atoms along x, with atom i at i * spacing; `bend` lifts the last atom
const chain = (n, spacing, bend = 0) => {
  const x = new Float32Array(n * 3);
  for (let i = 0; i < n; i += 1) x[i * 3] = i * spacing;
  x[(n - 1) * 3 + 1] = bend;
  return x;
};

describe("the ensemble PCA", () => {
  it("separates two shapes on PC1, and orders the play bar across them", () => {
    // five open and five bent chains, drawn interleaved
    const samples = [];
    for (let s = 0; s < 10; s += 1) samples.push(chain(6, 3.8 + 0.01 * s, s % 2 === 0 ? 0 : 12));
    const reps = Int32Array.from([0, 1, 2, 3, 4, 5]);
    const pca = ensemblePca(samples, reps);
    const bent = (s) => s % 2 === 1;
    const pc1 = pca.scores.map((score) => score[0]);
    const lowBent = Math.max(...pc1.filter((_, s) => bent(s)));
    const highOpen = Math.min(...pc1.filter((_, s) => !bent(s)));
    assert.ok(lowBent < highOpen || Math.min(...pc1.filter((_, s) => bent(s))) > Math.max(...pc1.filter((_, s) => !bent(s))),
      `the two shapes overlap on PC1: ${pc1.map((v) => v.toFixed(2)).join(" ")}`);
    assert.ok(pca.explained[0] > 0.9, `PC1 explains ${pca.explained[0]}`);
    const order = orderAlongPc1(pca, samples.length);
    const groups = order.map(bent);
    // sorted along PC1, the shapes come out as two runs, not interleaved
    assert.equal(groups.slice(1).filter((g, i) => g !== groups[i]).length, 1, `order ${order}`);
  });

  it("puts a one-parameter family's whole variance on PC1", () => {
    const rows = [1, 2, 3, 4, 5].map((t) => Float64Array.from([t, 2 * t, -t]));
    const pca = pca2(rows);
    assert.ok(Math.abs(pca.explained[0] - 1) < 1e-9);
    assert.ok(Math.abs(pca.explained[1]) < 1e-9);
    // the most extreme sample scores positive, so the sign does not flip as samples arrive
    const pc1 = pca.scores.map((score) => score[0]);
    const extreme = pc1.reduce((best, v) => (Math.abs(v) > Math.abs(best) ? v : best), 0);
    assert.ok(extreme > 0);
  });

  it("is invariant to rotating a sample, because it reads distances", () => {
    const x = chain(5, 3.8, 6);
    const r = new Float32Array(x.length);
    for (let i = 0; i < 5; i += 1) { r[i * 3] = -x[i * 3 + 1]; r[i * 3 + 1] = x[i * 3]; r[i * 3 + 2] = x[i * 3 + 2]; }
    const atoms = [0, 1, 2, 3, 4];
    const a = distanceFeatures(x, atoms);
    const b = distanceFeatures(r, atoms);
    assert.ok(a.every((v, k) => Math.abs(v - b[k]) < 1e-4));
  });

  it("returns nothing below three samples", () => {
    assert.equal(ensemblePca([chain(4, 3.8), chain(4, 3.9)], Int32Array.from([0, 1, 2, 3])), null);
    assert.deepEqual(orderAlongPc1(null, 2), [0, 1]);
  });
});

describe("the ensemble's contact frequency", () => {
  it("is the fraction of samples with the pair inside 8 A", () => {
    const near = chain(3, 4);   // 0-2 at 8 A: not inside
    const close = chain(3, 3);  // 0-2 at 6 A: inside
    const freq = ensembleContacts([near, close, close, close], Int32Array.from([0, 1, 2]));
    assert.equal(freq[0 * 3 + 2], 0.75);
    assert.equal(freq[2 * 3 + 0], 0.75);
    assert.equal(freq[0 * 3 + 1], 1);
  });

  it("picks a token's CA or C1', else its first atom", () => {
    const topology = { names: ["N", "CA", "C", "P", "C1'", "C1"], tokenIdxAtAtom: [0, 0, 0, 1, 1, 2] };
    assert.deepEqual(Array.from(representativeAtoms(topology, 3)), [1, 4, 5]);
  });
});

describe("BioEmu2's denoiser batches", () => {
  it("are as few as fit and all one size", () => {
    assert.deepEqual(bioemu2Batches(1), { batches: 1, size: 1 });
    assert.deepEqual(bioemu2Batches(5), { batches: 1, size: 5 });
    assert.deepEqual(bioemu2Batches(12), { batches: 3, size: 4 });
    assert.deepEqual(bioemu2Batches(100), { batches: 20, size: 5 });
    assert.deepEqual(bioemu2Batches(8, 8), { batches: 1, size: 8 });
  });
});
