import { describe, expect, it } from "./harness.js";
import { DEFAULT_MANIFEST } from "../src/reference/manifest.js";

describe("DEFAULT_MANIFEST", () => {
  it("defines the expected AlphaFold model_1_ptm metadata", () => {
    expect(DEFAULT_MANIFEST.formatVersion).toBe(1);
    expect(DEFAULT_MANIFEST.model.name).toBe("model_1");
    expect(DEFAULT_MANIFEST.bundle.model).toBe("model_1_ptm");
    // 🔴 STILL EIGHT SHARDS, AND TWO MORE TENSORS THAN THERE WERE. The
    // distogram head is appended to the LAST shard rather than given one of
    // its own, so the 227 MB before it are untouched and an upload transfers
    // one shard.
    expect(DEFAULT_MANIFEST.bundle.shards).toBe(8);
    expect(DEFAULT_MANIFEST.bundle.tensors).toBe(337);
  });

  it("carries the distogram head as tensors in the shards", () => {
    // 🔴 THE HEAD ALPHAFOLD ALWAYS HAD AND THIS BUNDLE NEVER SHIPPED. Without
    // it there is no contact map for AF2 at all - the confidence heads are
    // pLDDT and PAE only.
    const head = DEFAULT_MANIFEST.distogramHead;
    expect(typeof head).toBe("object");
    expect(head.bins).toBe(64);
    expect(head.firstBreak).toBe(2);
    expect(head.lastBreak).toBe(22);
    // 🔴 IT NAMES TENSORS, IT DOES NOT CARRY BYTES. The head was 44 KB of
    // base64 in this manifest for a while, which existed only to avoid
    // rewriting published shards - and the cost was a bundle that was not the
    // whole model, readable only through a special case in the loader.
    expect(typeof head.weights).toBe("string");
    expect(typeof head.bias).toBe("string");
    expect(head.encoding).toBe(undefined);
    const weights = DEFAULT_MANIFEST.tensors[head.weights];
    const bias = DEFAULT_MANIFEST.tensors[head.bias];
    expect(weights.shape).toEqual([128, 64]);
    expect(bias.shape).toEqual([64]);
    // ...float32, because 33 KB is not worth a codec and the store already
    // reads float32 from these same shards for the PAE bin edges.
    expect(weights.dtype).toBe("float32");
    expect(bias.dtype).toBe("float32");
    // ...and both in the last shard, which is what keeps the earlier ones byte
    // for byte what they were.
    expect(weights.file).toBe(bias.file);
    expect(bias.byteOffset).toBe(weights.byteOffset + 128 * 64 * 4);
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
