// The fused template embedder's features cross sparse (webgpu/af3/trunk/template.js, sparseTemplateFeatures):
// each row's nonzero (column, value) pairs in column order. The WebGPU shader and the native port both rebuild
// the dense rows from them, so the sparse form must expand to EXACTLY the dense one - an empty slot's included,
// which is written from its columns and never built dense.
import { test } from "node:test";
import assert from "node:assert/strict";
import { fusedTemplateFeatures, fusedTemplateFeaturesSparse, sparseTemplateFeatures, SPARSE_PAD }
  from "../shared/af3/featurise/template-fused-features.js";
import { PROTENIX2, BOLTZ2, ROSETTAFOLD3 } from "../shared/af3/dialect.js";

function expand(packed, rows, width) {
  const K = packed[0], dense = new Float32Array(rows * width);
  const bits = new Uint32Array(1), value = new Float32Array(bits.buffer);
  let lastColumn = -1;
  for (let r = 0; r < rows; r += 1) {
    lastColumn = -1;
    for (let k = 0; k < K; k += 1) {
      const c = packed[1 + (r * K + k) * 2];
      if (c === SPARSE_PAD) break;
      assert.ok(c > lastColumn, `row ${r}: columns in order`);
      lastColumn = c;
      bits[0] = packed[2 + (r * K + k) * 2];
      dense[r * width + c] = value[0];
    }
  }
  return dense;
}

test("an arbitrary matrix round-trips through the sparse form", () => {
  const rows = 37, width = 23, dense = new Float32Array(rows * width);
  let seed = 7;
  for (let i = 0; i < dense.length; i += 1) {
    seed = (seed * 1103515245 + 12345) >>> 0;
    if (seed % 5 === 0) dense[i] = (seed % 1000) / 37 - 13;
  }
  dense.fill(0, 3 * width, 4 * width);                  // (an all-zero row: padding only)
  assert.deepEqual(expand(sparseTemplateFeatures(dense, width), rows, width), dense);
});

test("an empty slot's sparse rows are its dense rows, under each fused dialect", () => {
  const tokens = 9;
  for (const [dialect, width] of [[PROTENIX2, 108], [BOLTZ2, 109], [ROSETTAFOLD3, 66]]) {
    for (const useGap of [true, false]) {
      let dense;
      try { dense = fusedTemplateFeatures(undefined, tokens, width, dialect, undefined, useGap); }
      catch { assert.throws(() => fusedTemplateFeaturesSparse(undefined, tokens, width, dialect, undefined, useGap)); continue; }
      const packed = fusedTemplateFeaturesSparse(undefined, tokens, width, dialect, undefined, useGap);
      assert.deepEqual(expand(packed, tokens * tokens, width), dense, `${width} wide, useGap ${useGap}`);
    }
  }
});
