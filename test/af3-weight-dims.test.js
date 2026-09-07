/**
 * Every width AF3's loader used to declare, derived from the tensor that states it.
 *
 * 🔴 A DECLARED WIDTH IS RIGHT FOR EXACTLY ONE CHECKPOINT AND SILENT FOR EVERY
 * OTHER. `pairChannels: 128` typed beside the tensor it describes loads a
 * 384-channel pair representation without complaint, and what comes out is a
 * kernel dispatched over a third of its own tensor - no error, no shape
 * mismatch, a fold. OpenDDE is the checkpoint that makes that concrete: its
 * pair track is 384 wide, its MSA 128, its grid attention 12 heads in the trunk
 * and 2 in the template stack, and its distogram 96 bins.
 *
 * So the loader reads them off the weights. This test is the other half of that
 * change: it pins every derived width to the constant it replaced, on the two
 * shipping bundles, because a shape lookup that silently returns the WRONG
 * number is exactly as quiet as the literal it replaced. The numbers below are
 * AlphaFold 3's, copied from the source before the derivation went in.
 */
import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import {
  confidenceWeights, distogramWeights, embedderWeights,
  msaBlockWeights, pairformerBlockWeights, templateWeights,
} from "../src/af3/weights.js";
import { MANIFEST as AF3 } from "../src/reference/manifests/af3.js";
import { MANIFEST as OPENBIND0 } from "../src/reference/manifests/openbind0.js";
import { MANIFEST as OPENDDE } from "../src/reference/manifests/opendde.js";

/**
 * A store that answers shapes truthfully and tensors with zeros.
 *
 * The widths are the whole subject here, so the CONTENT is irrelevant and a
 * real bundle is 265 MiB over the network. Shapes come from the compiled
 * manifest, which is the same table the page reads.
 */
function shapeOnlyStore(manifest) {
  return {
    manifest,
    shape(name) { return manifest.tensors[name]?.shape; },
    async tensor(name) {
      const shape = manifest.tensors[name]?.shape;
      if (shape === undefined) throw new Error(`no tensor ${name}`);
      return new Float32Array(shape.reduce((a, b) => a * b, 1));
    },
  };
}

/** AlphaFold 3's own widths, as they were written in weights.js. */
const AF3_WIDTHS = {
  pairChannels: 128, singleChannels: 384, msaChannels: 64,
  targetFeatWidth: 447, relativeWidth: 139,
  gridHeads: 4, gridDimension: 32,
  templateGridHeads: 4, templateGridDimension: 16, queryChannels: 128,
  msaHeads: 8, msaDimension: 8, outerChannels: 32,
  singleHeads: 16, singleDimension: 24,
  bins: 64,
};

for (const [family, manifest] of [["af3", AF3], ["openbind0", OPENBIND0]]) {
  describe(`AF3 weight widths, derived (${family})`, () => {
    const store = shapeOnlyStore(manifest);
    const W = AF3_WIDTHS;

    it("takes the embedder's five widths from three tensors", async () => {
      const w = await embedderWeights(store);
      for (const key of ["pairChannels", "singleChannels", "msaChannels",
                         "targetFeatWidth", "relativeWidth"]) {
        assert.equal(w[key], W[key], key);
      }
    });

    it("takes the MSA stack's widths and both head counts from its own tensors",
      async () => {
        const w = await msaBlockWeights(store, 0);
        assert.equal(w.pairChannels, W.pairChannels);
        assert.equal(w.msaChannels, W.msaChannels);
        assert.equal(w.outerProductMean.outerChannels, W.outerChannels);
        assert.equal(w.msaAttention1.heads, W.msaHeads);
        // 🔴 THE VALUE DIMENSION IS NOT msaChannels / heads. It is 8 here and
        // 64/8 is also 8, so AF3 cannot tell the two rules apart - but
        // OpenDDE's MSA track is 128 wide with 8 heads and a value dim of 8,
        // which upstream calls "decoupled per-head width". Dividing would give
        // it 16. Only `v_projection`'s own shape says so.
        assert.equal(w.msaAttention1.dimension, W.msaDimension);
        assert.equal(w.pairAttention1.heads, W.gridHeads);
        assert.equal(w.pairAttention1.dimension, W.gridDimension);
      });

    it("takes the trunk pairformer's single attention from its q projection",
      async () => {
        const w = await pairformerBlockWeights(store, 0);
        assert.equal(w.pairChannels, W.pairChannels);
        assert.equal(w.singleChannels, W.singleChannels);
        assert.equal(w.singleAttention.heads, W.singleHeads);
        assert.equal(w.singleAttention.dimension, W.singleDimension);
        assert.equal(w.pairAttention1.heads, W.gridHeads);
        assert.equal(w.pairAttention1.dimension, W.gridDimension);
      });

    it("gives the template stack its OWN grid shape, not the trunk's", async () => {
      const w = await templateWeights(store);
      assert.equal(w.queryChannels, W.queryChannels);
      assert.equal(w.blocks.length, 2);
      for (const block of w.blocks) {
        assert.equal(block.pairAttention1.heads, W.templateGridHeads);
        assert.equal(block.pairAttention1.dimension, W.templateGridDimension);
      }
      // The trunk's is 4x32 and this is 4x16: a stack that inherited the
      // trunk's numbers would read twice its own tensor.
      assert.notEqual(W.templateGridDimension, W.gridDimension);
    });

    it("takes the confidence head's widths from the head and from its stack",
      async () => {
        const w = await confidenceWeights(store);
        assert.equal(w.pairChannels, W.pairChannels);
        assert.equal(w.singleChannels, W.singleChannels);
        assert.equal(w.targetFeatWidth, W.targetFeatWidth);
        assert.equal(w.blocks.length, 4);
        assert.equal(w.blocks[0].singleAttention.heads, W.singleHeads);
        assert.equal(w.blocks[0].singleAttention.dimension, W.singleDimension);
      });

    it("takes the distogram's bin count from its logits", async () => {
      const w = await distogramWeights(store);
      assert.equal(w.pairChannels, W.pairChannels);
      // 🔴 OpenDDE's is 96. Nothing but this tensor says how many bins a
      // distogram head has, and a head read at the wrong count produces a
      // contact map rather than an error.
      assert.equal(w.bins, W.bins);
    });

    it("refuses a width it cannot derive rather than assuming one", async () => {
      const blind = { ...shapeOnlyStore(manifest), shape() { return undefined; } };
      await assert.rejects(() => embedderWeights(blind), /a width cannot be derived/);
    });
  });
}


/**
 * OpenDDE, which is the reason any of the above is derived rather than declared.
 *
 * 🔴 THESE NUMBERS ARE UPSTREAM'S, NOT OURS. Every one is stated independently
 * in `model_registry.OPENDDE_SETTINGS` - pair 384, MSA 128, trunk and MSA and
 * confidence triangle attention 12 heads, template 2, head dim 32, distogram 96
 * bins - and every one is READ HERE off the tensor that carries it. Two
 * independent statements of one profile agreeing is what says the derivation is
 * a measurement rather than a restatement.
 */
describe("AF3 weight widths, derived (opendde)", () => {
  const store = shapeOnlyStore(OPENDDE);

  it("reads OpenDDE's own widths, not AlphaFold 3's", async () => {
    const w = await embedderWeights(store);
    assert.equal(w.pairChannels, 384);
    assert.equal(w.msaChannels, 128);
    assert.equal(w.singleChannels, 384);
    // Unchanged from AF3: the feature widths are the FEATURISER's, and OpenDDE
    // rides AlphaFold 3's featurisation here.
    assert.equal(w.targetFeatWidth, 447);
    assert.equal(w.relativeWidth, 139);
  });

  it("reads 12 triangle-attention heads in the trunk and 2 in the template stack",
    async () => {
      const trunk = await pairformerBlockWeights(store, 0);
      assert.equal(trunk.pairAttention1.heads, 12);
      assert.equal(trunk.pairAttention1.dimension, 32);
      const template = await templateWeights(store);
      // 🔴 2 x 32, WHERE AlphaFold 3's IS 4 x 16. Both come to 64 channels, and
      // neither number can be derived from the other or from the trunk's - which
      // is why the stack reads its own projection rather than inheriting.
      assert.equal(template.blocks[0].pairAttention1.heads, 2);
      assert.equal(template.blocks[0].pairAttention1.dimension, 32);
      // ...and the template embedder's INPUT is the trunk pair, so it moves too.
      assert.equal(template.queryChannels, 384);
    });

  it("reads an MSA value dim that is NOT msaChannels / heads", async () => {
    // 🔴 THE CASE ALPHAFOLD 3 COULD NEVER HAVE SHOWN. AF3 is 64 channels and 8
    // heads with a value dim of 8, so 64/8 = 8 and the two rules agree.
    // OpenDDE is 128 channels and 8 heads with a value dim of STILL 8 -
    // upstream calls it a decoupled per-head width - so dividing gives 16 and
    // reads twice its own tensor.
    const w = await msaBlockWeights(store, 0);
    assert.equal(w.msaChannels, 128);
    assert.equal(w.msaAttention1.heads, 8);
    assert.equal(w.msaAttention1.dimension, 8);
    assert.notEqual(w.msaAttention1.dimension, w.msaChannels / w.msaAttention1.heads);
    assert.equal(w.pairAttention1.heads, 12);
  });

  it("reads 96 distogram bins and the trained bias AlphaFold 3 has not", async () => {
    const w = await distogramWeights(store);
    assert.equal(w.pairChannels, 384);
    assert.equal(w.bins, 96);
    assert.equal(w.halfLogitsBias.length, 96);
  });

  it("refuses a bundle whose bias and dialect disagree", async () => {
    // The bias is applied TWICE by the symmetrisation, so losing it silently
    // spreads the softmax across every bin. Presence and dialect must agree.
    await assert.rejects(
      () => distogramWeights(store, { distogramBias: false }),
      /carries .*half_logits\/bias/);
    await assert.rejects(
      () => distogramWeights(shapeOnlyStore(AF3), { distogramBias: true }),
      /does not carry .*half_logits\/bias/);
  });
});
