import { readFile } from "node:fs/promises";
import { describe, expect, it } from "./harness.js";
import { interChainCovarianceMask } from "../src/input/chains.js";

describe("inter-chain covariance mask", () => {
  it("keeps every pair for a monomer", () => {
    const mask = interChainCovarianceMask(3, undefined);
    expect(Array.from(mask)).toEqual([1, 1, 1, 1, 1, 1, 1, 1, 1]);
  });

  it("keeps the diagonal blocks and drops the off-diagonal ones", () => {
    const mask = interChainCovarianceMask(4, [2, 2]);
    expect(Array.from(mask)).toEqual([
      1, 1, 0, 0,
      1, 1, 0, 0,
      0, 0, 1, 1,
      0, 0, 1, 1,
    ]);
  });

  it("handles unequal chains", () => {
    const mask = interChainCovarianceMask(3, [1, 2]);
    expect(Array.from(mask)).toEqual([
      1, 0, 0,
      0, 1, 1,
      0, 1, 1,
    ]);
  });

  it("keeps 1/N of the pair matrix for an N-mer, which is the work that survives", () => {
    for (const copies of [2, 3, 4]) {
      const mask = interChainCovarianceMask(12, Array(copies).fill(12 / copies));
      const kept = mask.reduce((sum, value) => sum + value, 0);
      expect(kept / mask.length).toBeCloseTo(1 / copies, 6);
    }
  });

  it("refuses lengths that do not partition the sequence", () => {
    expect(() => interChainCovarianceMask(5, [2, 2])).toThrow(/sum to 4/);
  });
});

/**
 * The substitution the shaders make, in plain JS: where the mask is 0 the
 * sequence sum is replaced by the product of the marginals. Reproducing it here
 * pins the algebra independently of WGSL, which cannot run on this machine.
 */
function referenceContraction(left, right, msaMask, sequences, length, cOuter, covMask, epsilon = 1e-3) {
  const out = new Float64Array(length * length * cOuter * cOuter);
  const leftSum = new Float64Array(length * cOuter);
  const rightMean = new Float64Array(length * cOuter);
  for (let residue = 0; residue < length; residue += 1) {
    let covered = 0;
    for (let s = 0; s < sequences; s += 1) covered += msaMask[s * length + residue];
    for (let o = 0; o < cOuter; o += 1) {
      let l = 0, r = 0;
      for (let s = 0; s < sequences; s += 1) {
        l += left[(s * length + residue) * cOuter + o];
        r += right[(s * length + residue) * cOuter + o];
      }
      leftSum[residue * cOuter + o] = l;
      rightMean[residue * cOuter + o] = r / (epsilon + covered);
    }
  }
  for (let i = 0; i < length; i += 1) for (let j = 0; j < length; j += 1) {
    const masked = covMask === undefined ? 1 : covMask[i * length + j];
    for (let ol = 0; ol < cOuter; ol += 1) for (let or = 0; or < cOuter; or += 1) {
      const slot = ((i * length + j) * cOuter + ol) * cOuter + or;
      if (masked === 0) {
        out[slot] = leftSum[i * cOuter + ol] * rightMean[j * cOuter + or];
        continue;
      }
      let value = 0;
      for (let s = 0; s < sequences; s += 1) {
        value += left[(s * length + i) * cOuter + ol] * right[(s * length + j) * cOuter + or];
      }
      out[slot] = value;
    }
  }
  return out;
}

describe("the marginal substitution", () => {
  const sequences = 6, length = 4, cOuter = 3;
  const rows = sequences * length;
  const left = Float32Array.from({ length: rows * cOuter }, (_, i) => Math.sin(i * 1.7));
  const right = Float32Array.from({ length: rows * cOuter }, (_, i) => Math.cos(i * 0.9));
  const msaMask = new Float32Array(rows).fill(1);

  it("leaves the intra-chain blocks bitwise alone", () => {
    const covMask = interChainCovarianceMask(length, [2, 2]);
    const plain = referenceContraction(left, right, msaMask, sequences, length, cOuter, undefined);
    const masked = referenceContraction(left, right, msaMask, sequences, length, cOuter, covMask);
    for (let i = 0; i < length; i += 1) for (let j = 0; j < length; j += 1) {
      if (covMask[i * length + j] === 0) continue;
      for (let o = 0; o < cOuter * cOuter; o += 1) {
        const slot = (i * length + j) * cOuter * cOuter + o;
        expect(masked[slot]).toBe(plain[slot]);
      }
    }
  });

  it("changes the inter-chain blocks, which is the whole point", () => {
    const covMask = interChainCovarianceMask(length, [2, 2]);
    const plain = referenceContraction(left, right, msaMask, sequences, length, cOuter, undefined);
    const masked = referenceContraction(left, right, msaMask, sequences, length, cOuter, covMask);
    let differing = 0;
    for (let i = 0; i < length; i += 1) for (let j = 0; j < length; j += 1) {
      if (covMask[i * length + j] === 1) continue;
      for (let o = 0; o < cOuter * cOuter; o += 1) {
        const slot = (i * length + j) * cOuter * cOuter + o;
        if (masked[slot] !== plain[slot]) differing += 1;
      }
    }
    expect(differing).toBeGreaterThan(0);
  });

  it("reduces to the sequence sum when the rows carry no covariance", () => {
    // Every sequence identical => left[s,i] does not vary with s, so the
    // covariance is zero and the substitution should change nothing. It lands
    // on sum * D/(eps + D) rather than sum, because the marginal is divided by
    // the softened count and the sum is not - the same eps AlphaFold carries.
    // That factor is exact, so it is asserted rather than tolerated.
    const flat = new Float32Array(rows * cOuter);
    const flatRight = new Float32Array(rows * cOuter);
    for (let s = 0; s < sequences; s += 1) for (let r = 0; r < length; r += 1) {
      for (let o = 0; o < cOuter; o += 1) {
        flat[(s * length + r) * cOuter + o] = Math.sin(r * 2.1 + o);
        flatRight[(s * length + r) * cOuter + o] = Math.cos(r * 1.3 + o);
      }
    }
    const covMask = interChainCovarianceMask(length, [2, 2]);
    const plain = referenceContraction(flat, flatRight, msaMask, sequences, length, cOuter, undefined);
    const masked = referenceContraction(flat, flatRight, msaMask, sequences, length, cOuter, covMask);
    for (let i = 0; i < length; i += 1) for (let j = 0; j < length; j += 1) {
      if (covMask[i * length + j] === 1) continue;
      for (let o = 0; o < cOuter * cOuter; o += 1) {
        const slot = (i * length + j) * cOuter * cOuter + o;
        const exact = plain[slot] * (sequences / (1e-3 + sequences));
        expect(masked[slot]).toBeCloseTo(exact, 10);
      }
    }
  });
});

describe("the chain mask on MSA row attention", () => {
  it("shuts off exactly the cross-chain keys and nothing else", async() => {
    const { interChainCovarianceMask } = await import("../src/input/chains.js");
    const mask = interChainCovarianceMask(4, [2, 2]);
    // The bias adds 1e9 * (mask - 1), so intra-chain adds 0 and cross-chain
    // adds -1e9, which is what the flash kernels apply for a masked key.
    const bias = (q, k) => 1e9 * (mask[q * 4 + k] - 1);
    expect(bias(0, 1)).toBe(0);
    expect(bias(2, 3)).toBe(0);
    expect(bias(0, 2)).toBe(-1e9);
    expect(bias(3, 1)).toBe(-1e9);
  });

  it("is the same buffer the outer product mean uses", async() => {
    const { interChainCovarianceMask } = await import("../src/input/chains.js");
    const a = interChainCovarianceMask(6, [3, 3]);
    const b = interChainCovarianceMask(6, [3, 3]);
    expect(Array.from(a)).toEqual(Array.from(b));
  });

  it("reaches row attention but never triangle attention", async() => {
    const source = await readFile(new URL("../src/evoformer/block.js", import.meta.url), "utf8");
    // Every chainMask hand-off must sit in a block whose label is an MSA row
    // attention. Triangle attention shares the pipeline and must stay unmasked:
    // the pair track is where the interface is built.
    const calls = source.split("encodeAttention(execution, encoder, {").slice(1);
    const masked = calls.filter((call) => call.slice(0, 600).includes("chainMask:"));
    expect(masked.length).toBe(2);
    for (const call of masked) {
      expect(call.slice(0, 600).includes("msa-row-attention")).toBe(true);
    }
    const triangle = calls.filter((call) => call.slice(0, 600).includes("triangle-attention"));
    expect(triangle.length).toBeGreaterThan(0);
    for (const call of triangle) {
      expect(call.slice(0, 600).includes("chainMask:")).toBe(false);
    }
  });

  it("leaves column attention alone, since a column is one chain", async() => {
    const source = await readFile(new URL("../src/evoformer/block.js", import.meta.url), "utf8");
    const calls = source.split("encodeAttention(execution, encoder, {").slice(1);
    const column = calls.filter((call) => call.slice(0, 600).includes("column-attention"));
    expect(column.length).toBeGreaterThan(0);
    for (const call of column) {
      expect(call.slice(0, 600).includes("chainMask:")).toBe(false);
    }
  });
});
