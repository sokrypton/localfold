import { describe, expect, it } from "./harness.js";
import { DEFAULT_MANIFEST } from "../shared/bundles/manifest.js";

describe("DEFAULT_MANIFEST", () => {
  it("defines the expected AlphaFold model_1_ptm metadata", () => {
    expect(DEFAULT_MANIFEST.formatVersion).toBe(1);
    expect(DEFAULT_MANIFEST.model.name).toBe("model_1");
    expect(DEFAULT_MANIFEST.bundle.model).toBe("model_1_ptm");
    // 🔴 NINE SHARDS: the distogram head was appended to the LAST shard, and
    // the template single features (tools/append_template_single.py) came in a
    // ninth of their own, so the eight before it are byte-identical and an
    // upload transfers one small file.
    expect(DEFAULT_MANIFEST.bundle.shards).toBe(9);
    expect(DEFAULT_MANIFEST.bundle.tensors).toBe(341);
  });

  it("carries the template single features, float32, in a section of their own", () => {
    // AF2 monomer turns a template's torsion angles into MSA rows through
    // these two layers; without them 5CAJ with its own crystal folds to 2.53 A
    // against 0.22 (native/af2). A section of their own, so nothing that reads
    // templateEmbedding meets them.
    const params = DEFAULT_MANIFEST.templateSingle.parameters;
    const shapes = {
      template_single_embedding: { weights: [57, 256], bias: [256] },
      template_projection: { weights: [256, 256], bias: [256] },
    };
    for (const [module, leaves] of Object.entries(shapes)) {
      for (const [leaf, shape] of Object.entries(leaves)) {
        const record = DEFAULT_MANIFEST.tensors[params[module][leaf]];
        expect(record.shape).toEqual(shape);
        expect(record.dtype).toBe("float32");
      }
    }
  });

  it("carries the distogram head as tensors in the shards", () => {
    // 🔴 THE HEAD ALPHAFOLD ALWAYS HAD AND THIS BUNDLE NEVER SHIPPED. Without
    // it there is no contact map for AF2 at all - the confidence heads are
    // pLDDT and PAE only.
    const head = DEFAULT_MANIFEST.distogramHead;
    expect(typeof head).toBe("object");
    expect(head.bins).toBe(64);
    // 🔴 AND ITS BREAKS ARE ALPHAFOLD'S OWN, NOT THE ROUND NUMBERS. `config.py`
    // says `first_break: 2.3125, last_break: 21.6875`, which is an exact
    // 0.3125 A grid; the manifest said 2 and 22 for a long time, which is the
    // same grid's CENTRE form misread as its break form and shifts every edge
    // by up to a bin. See shared/heads/distogram.js.
    expect(head.firstBreak).toBe(2.3125);
    expect(head.lastBreak).toBe(21.6875);
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
    // 🔴 THEY USED TO BE PINNED TO ONE SHARD, ADJACENT, AND THAT RULE HAS
    // EXPIRED. `add_distogram_head.py` APPENDED the head to shards that were
    // already published, so putting both in the last one and the bias directly
    // after the weights is what kept the other 227 MB byte for byte and made
    // the upload one file. The bundle is exported whole now - int5 asymmetric,
    // every shard rewritten - so there is nothing to preserve and the packer
    // lays them wherever the sizes fall. What still matters is that they are
    // float32, the right shape, and IN the table, which is asserted above:
    // they are 33 KB and they are the contact map.
    expect(typeof weights.file).toBe("string");
    expect(typeof bias.byteOffset).toBe("number");
  });

  it("contains all 341 tensor entries with valid shapes and dtypes", () => {
    const tensorKeys = Object.keys(DEFAULT_MANIFEST.tensors);
    expect(tensorKeys.length).toBe(341);

    // 🔴 int5 IS IN THE LIST BECAUSE THE BUNDLE IS int5 NOW - asymmetric, group
    // 32, 73 MiB against the int8 export's 98 and measured free on a fold. A
    // packed dtype carries a zero point as well as a scale, which is what
    // "asymmetric" means and what int8 symmetric did not have.
    const validDtypes = new Set(["int8", "int5", "float32", "float16"]);
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
      if (tensor.dtype === "int5") {
        expect(typeof tensor.scaleOffset).toBe("number");
        expect(typeof tensor.zeroOffset).toBe("number");
        expect(tensor.block).toBe(32);
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

describe("the monomer deltas and the template single features", async () => {
  // A delta inherits every base tensor its header does not name, so each
  // delta must decide these four: model_2_ptm carries its own WHOLE, and
  // model_3/4/5_ptm, which have no template embedder, mark them absent - and
  // the store then drops the section, as it drops templateEmbedding.
  const { DeltaTensorStore } = await import("../shared/bundles/delta-tensor-store.js");
  const names = Object.values(DEFAULT_MANIFEST.templateSingle.parameters).flatMap((l) => Object.values(l));
  it("model_2 carries them whole", async () => {
    const { MANIFEST } = await import("../shared/bundles/manifests/monomer-2.js");
    for (const name of names) {
      expect(MANIFEST.delta.whole.includes(name)).toBe(true);
      expect(MANIFEST.tensors[name].dtype).toBe("float32");
    }
  });
  for (const k of [3, 4, 5]) {
    it(`model_${k} has none, and the store drops the section`, async () => {
      const { MANIFEST } = await import(`../shared/bundles/manifests/monomer-${k}.js`);
      for (const name of names) expect(MANIFEST.delta.absent.includes(name)).toBe(true);
      const store = new DeltaTensorStore({ manifest: DEFAULT_MANIFEST }, { manifest: MANIFEST });
      expect(store.manifest.templateSingle).toBe(undefined);
      expect(store.manifest.templateEmbedding).toBe(undefined);
      for (const name of names) expect(store.manifest.tensors[name]).toBe(undefined);
    });
  }
});
