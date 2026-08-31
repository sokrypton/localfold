import { describe, expect, it } from "./harness.js";
import { DEFAULT_MANIFEST } from "../src/reference/manifest.js";

describe("DEFAULT_MANIFEST", () => {
  it("defines the expected AlphaFold model_1_ptm metadata", () => {
    expect(DEFAULT_MANIFEST.formatVersion).toBe(1);
    expect(DEFAULT_MANIFEST.model.name).toBe("model_1");
    expect(DEFAULT_MANIFEST.bundle.model).toBe("model_1_ptm");
    expect(DEFAULT_MANIFEST.bundle.shards).toBe(8);
    expect(DEFAULT_MANIFEST.bundle.tensors).toBe(335);
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
