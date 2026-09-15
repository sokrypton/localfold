/**
 * Every confidence head runs its pairformer in f32 with the matrix kernels off.
 *
 * 🔴 OPENDDE'S HEAD SHIPPED WITHOUT THE PINS AlphaFold 3's CARRIES. pLDDT and PAE
 * are softmaxes over 50 and 64 bins, the most amplifying thing these models
 * emit, and `Af3ConfidenceHeadGpu` measured all four of its outputs failing with
 * f16 staging or the matrix pair kernels on. `openddeConfidence` built its
 * stack with the bundle's weight precision alone, so the page computed
 * OpenDDE's pLDDT and PAE at the trunk's precision: PAE 1.04e-2 against
 * af3-any-model on an M2, 9.11e-7 once pinned.
 *
 * The oracle that measured it needs dumps this repository does not ship, so it
 * cannot run here. This reads the source instead: each head's options must name
 * all four pins, and dropping any one fails it.
 */
import { strict as assert } from "node:assert";
import { af3Source } from "./helpers/af3-source.js";
import { describe, it } from "node:test";

const PINS = [
  /stagedPrecision:\s*"f32"/,
  /weightPrecision:\s*"f32"/,
  /accumulatePrecision:\s*"f32"/,
  /pairMatrixKernels:\s*false/,
];

const HEADS = {
  "AlphaFold 3 (and its lineage)": "confidence-webgpu.js",
  OpenDDE: "opendde-confidence.js",
};

describe("the confidence heads' pairformer precision", () => {
  for (const [name, path] of Object.entries(HEADS)) {
    it(`pins all four axes for ${name}`, () => {
      const source = af3Source(path);
      for (const pin of PINS) assert.match(source, pin, `${name} is missing ${pin}`);
    });
  }
});
