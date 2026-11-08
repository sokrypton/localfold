/**
 * The atom block's row tile against the device's workgroup storage.
 *
 * 🔴 AT 590 TOKENS EVERY AlphaFold 3 FOLD FAILED TO CREATE ITS PIPELINE ON A MINIMUM DEVICE: 4800 query rows take a
 * tile of 8, whose output kernel stages 20,480 bytes where WebGPU guarantees 16,384. `test:spec-floor` found it once it
 * folded something that large. See createAtomBlockShaders.
 */
import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import { createAtomBlockShaders, workgroupBytes } from "../webgpu/af3/diffusion/atom-encoder.js";

// AlphaFold 3's atom widths at 590 tokens: 150 subsets of 32 queries.
const SHAPE = { subsets: 150, queries: 32, keys: 128, channels: 128, heads: 4, dimension: 32 };
const SPEC_FLOOR = 16384;

describe("the atom block's row tile", () => {
  it("is 8 at 4800 rows with no limit, and takes more than the standard guarantees", () => {
    const sources = createAtomBlockShaders("", SHAPE);
    assert.equal(sources.outputRowTile, 8);
    assert.equal(workgroupBytes(sources.output), 20480);
  });

  it("steps down to what fits a minimum device", () => {
    const sources = createAtomBlockShaders("", { ...SHAPE, workgroupStorageBytes: SPEC_FLOOR });
    assert.equal(sources.outputRowTile, 4);
    assert.ok(workgroupBytes(sources.output) <= SPEC_FLOOR);
  });

  it("is untouched where it fits", () => {
    assert.equal(createAtomBlockShaders("", { ...SHAPE, workgroupStorageBytes: 49152 }).outputRowTile, 8);
  });

  it("refuses a tile the caller names, rather than changing it", () => {
    assert.throws(() => createAtomBlockShaders("", { ...SHAPE, outputRowTile: 8, workgroupStorageBytes: SPEC_FLOOR }),
      /row tile of 8 takes 20480 bytes of workgroup storage, past this device's 16384/);
  });

  it("counts every array type the kernels declare", () => {
    assert.equal(workgroupBytes("var<workgroup> a: array<vec4<f32>, 3>;\nvar<workgroup> b : array<f32, 64>;"), 304);
    assert.throws(() => workgroupBytes("var<workgroup> a: array<f16, 3>;"), /no size for f16/);
  });
});
