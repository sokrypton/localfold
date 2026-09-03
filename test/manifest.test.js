import { describe, expect, it } from "./harness.js";
import { DEFAULT_MANIFEST } from "../src/reference/manifest.js";

describe("DEFAULT_MANIFEST", () => {
  it("defines the expected AlphaFold model_1_ptm metadata", () => {
    expect(DEFAULT_MANIFEST.formatVersion).toBe(1);
    expect(DEFAULT_MANIFEST.model.name).toBe("model_1");
    expect(DEFAULT_MANIFEST.bundle.model).toBe("model_1_ptm");
    // 🔴 STILL EIGHT SHARDS. The distogram head is embedded in the manifest
    // as base64 rather than sharded, so nothing published changes and no
    // remote has to catch up before the page can use it.
    expect(DEFAULT_MANIFEST.bundle.shards).toBe(8);
    expect(DEFAULT_MANIFEST.bundle.tensors).toBe(335);
  });

  it("carries the distogram head, embedded rather than sharded", () => {
    // 🔴 THE HEAD ALPHAFOLD ALWAYS HAD AND THIS BUNDLE NEVER SHIPPED. Without
    // it there is no contact map for AF2 at all - the confidence heads are
    // pLDDT and PAE only.
    const head = DEFAULT_MANIFEST.distogramHead;
    expect(typeof head).toBe("object");
    expect(head.bins).toBe(64);
    expect(head.firstBreak).toBe(2);
    expect(head.lastBreak).toBe(22);
    expect(head.weightsShape).toEqual([128, 64]);
    expect(head.biasShape).toEqual([64]);
    expect(head.encoding).toBe("base64-float32-le");
    // 🔴 CARRIED BY THE MANIFEST, NOT BY A SHARD. A shard would have to reach
    // the pinned remote before the page could load at all - which is exactly
    // how the first version of this broke every AF2 fold.
    expect(typeof head.weights).toBe("string");
    expect(head.weights.length > 40000).toBe(true);
  });

  it("contains all 335 tensor entries with valid shapes and dtypes", () => {
    const tensorKeys = Object.keys(DEFAULT_MANIFEST.tensors);
    expect(tensorKeys.length).toBe(335);

    const validDtypes = new Set(["int8", "float32", "float16"]);
    for (const [name, tensor] of Object.entries(DEFAULT_MANIFEST.tensors)) {
      expect(typeof tensor.file).toBe("string");
      expect(tensor.file.startsWith("weights-")).toBe(true);
      expect(Array.isArray(tensor.shape)).toBe(true);
      expect(tensor.shape.length).toBeGreaterThan(0);
      expect(typeof tensor.byteOffset).toBe("number");
      expect(validDtypes.has(tensor.dtype)).toBe(true);
      if (tensor.dtype === "int8") {
        expect(typeof tensor.scaleOffset).toBe("number");
        expect(tensor.block).toBe(64);
      }
    }
  });

  it("has all required module parameter mappings", () => {
    expect(DEFAULT_MANIFEST.evoformerStack.blocks).toBe(48);
    expect(DEFAULT_MANIFEST.extraMsaStack.blocks).toBe(4);
    expect(DEFAULT_MANIFEST.structureModule.iterations).toBe(8);
    expect(Object.keys(DEFAULT_MANIFEST.embedding.parameters).length).toBeGreaterThan(0);
    expect(Object.keys(DEFAULT_MANIFEST.templateEmbedding.parameters).length).toBeGreaterThan(0);
    expect(Object.keys(DEFAULT_MANIFEST.confidenceHeads.parameters).length).toBeGreaterThan(0);
    expect(DEFAULT_MANIFEST.residueGeometry.tensors.length).toBe(6);
  });
});
