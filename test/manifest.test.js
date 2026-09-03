import { describe, expect, it } from "./harness.js";
import { DEFAULT_MANIFEST } from "../src/reference/manifest.js";

describe("DEFAULT_MANIFEST", () => {
  it("defines the expected AlphaFold model_1_ptm metadata", () => {
    expect(DEFAULT_MANIFEST.formatVersion).toBe(1);
    expect(DEFAULT_MANIFEST.model.name).toBe("model_1");
    expect(DEFAULT_MANIFEST.bundle.model).toBe("model_1_ptm");
    // 🔴 NINE SHARDS AND 337 TENSORS SINCE THE DISTOGRAM HEAD WAS APPENDED.
    // tools/add_distogram_head.py adds AlphaFold 2's half_logits weights and
    // bias in a shard of their own rather than re-sharding the bundle, so the
    // first eight shards and their digests are byte-identical to what is
    // published and only the new 33 KB file has to be uploaded.
    expect(DEFAULT_MANIFEST.bundle.shards).toBe(9);
    expect(DEFAULT_MANIFEST.bundle.tensors).toBe(337);
  });

  it("carries the distogram head, in its own shard", () => {
    // 🔴 THE HEAD ALPHAFOLD ALWAYS HAD AND THIS BUNDLE NEVER SHIPPED. Without
    // it there is no contact map for AF2 at all - the confidence heads are
    // pLDDT and PAE only.
    const head = DEFAULT_MANIFEST.distogramHead;
    expect(typeof head).toBe("object");
    expect(head.bins).toBe(64);
    expect(head.firstBreak).toBe(2);
    expect(head.lastBreak).toBe(22);
    const weights = DEFAULT_MANIFEST.tensors[head.parameters.halfLogitsWeights];
    const bias = DEFAULT_MANIFEST.tensors[head.parameters.halfLogitsBias];
    expect(weights.shape).toEqual([128, 64]);
    expect(bias.shape).toEqual([64]);
    expect(weights.dtype).toBe("float32");
    expect(bias.dtype).toBe("float32");
    // Its own shard, so nothing already published moved.
    expect(weights.file).toBe(bias.file);
    expect(weights.file === "weights-00.bin").toBe(false);
  });

  it("contains all 337 tensor entries with valid shapes and dtypes", () => {
    const tensorKeys = Object.keys(DEFAULT_MANIFEST.tensors);
    expect(tensorKeys.length).toBe(337);

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
