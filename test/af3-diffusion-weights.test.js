/**
 * Which tensors the diffusion head's loader asks the checkpoint for.
 *
 * 🔴 THE POINT IS THE _1 SUFFIX. Four of the atom encoder's pair tensors exist
 * under two names with IDENTICAL shapes: the unsuffixed pair belongs to the
 * conditioning computed over a token's own 24 dense atom slots, and the _1 pair
 * to the queries-keys layout the atom transformer actually works in. Loading
 * the wrong four does not throw, does not change a shape, and does not stop the
 * model folding a protein - it just folds a different one, with side chains
 * about 8% compressed and rings irregular. Only the ORACLE caught it, and only
 * after the checkers that would have caught it were found to build their
 * weights by hand rather than through this loader.
 */
import { describe, expect, it } from "./harness.js";
import { diffusionWeights, targetFeatureWeights } from "../src/af3/weights/diffusion-weights.js";

/** A store that answers every request with zeros and remembers what was asked. */
function recordingStore(asked) {
  const TX = "diffuser/~/diffusion_head/transformer/__layer_stack_with_per_layer";
  const shapes = new Map([
    [`${TX}/pair_logits_projection/weights`, [6, 128, 4, 16]],
    // 🔴 THE TOKEN TRANSFORMER'S SHAPE COMES OFF THESE TWO NOW, so a stub that
    // answers `[24]` for everything makes `txShape` raise - which is the
    // derivation working, and is why the stub has to state them. AlphaFold 3's
    // real values: q_projection is
    // [superBlocks, blocksPerSuperBlock, channels, heads, dimension] and the
    // SwiGLU transition's last axis is `channels * factor * 2`.
    [`${TX}/${"__layer_stack_with_per_layer"}/transformerq_projection/weights`,
      [6, 4, 768, 16, 48]],
    [`${TX}/${"__layer_stack_with_per_layer"}/transformerffw_transition1/weights`,
      [6, 4, 768, 3072]],
  ]);
  return {
    // A bundle names the graph it was converted for, and the loaders read it -
    // see af3Dialect. Stock AF3, which is what these tensor names are.
    manifest: { model: { name: "alphafold3" } },
    async tensor(name) {
      asked.add(name);
      const shape = shapes.get(name);
      const length = shape === undefined
        ? 64 : shape.reduce((a, b) => a * b, 1);
      return new Float32Array(length);
    },
    shape(name) {
      return shapes.get(name) ?? [24];
    },
  };
}

/**
 * 🔴 THIS SUITE ASSERTED THE `_1` FORM AND THE `_1` FORM IS THE WRONG ONE.
 *
 * Four of these tensors exist twice, unsuffixed and `_1`: haiku numbers a
 * module the second time its constructor runs, and both instantiations are
 * created during `init`. Only the FIRST is CALLED at inference - tracing
 * af3-any-model's whole fold with `hk.intercept_methods` shows
 * `diffusion_single_to_pair_cond_row` and its `evoformer_conditioning` twin
 * firing twice each and neither `_1` firing at all.
 *
 * For AlphaFold 3 the two are different trained tensors (rms 0.088 against
 * 0.406 for the row projection), so this was not a naming preference. Its
 * denoise step read relRMS 4.19e-1 against af3-any-model and 1.55e-5 after -
 * and no gate here could see it, because every one of them feeds ONE bundle to
 * both sides and the per-module checkers build their weight dict the same wrong
 * way. `tools/gpu/check-af3-denoise.js` is the gate that can.
 */
describe("the diffusion head's weight names", () => {
  it("takes the UNSUFFIXED queries-keys pair tensors, not the _1 form", async () => {
    const asked = new Set();
    await diffusionWeights(recordingStore(asked), 1);
    const head = "diffuser/~/diffusion_head";
    for (const leaf of ["diffusion_single_to_pair_cond_row",
                        "diffusion_single_to_pair_cond_col",
                        "diffusion_embed_pair_offsets",
                        "diffusion_embed_pair_distances"]) {
      expect(asked.has(`${head}/${leaf}/weights`)).toBe(true);
      // ...and NOT the `_1` one, which is a module the graph never calls.
      expect(asked.has(`${head}/${leaf}_1/weights`)).toBe(false);
    }
    // The one with no _1 form, so the set above cannot be "fixed" wholesale.
    expect(asked.has(`${head}/diffusion_embed_pair_offsets_valid/weights`)).toBe(true);
  });

  it("takes the same four, unsuffixed, for target_feat's own atom encoder", async () => {
    const asked = new Set();
    await targetFeatureWeights(recordingStore(asked));
    const root = "diffuser/evoformer_conditioning";
    for (const leaf of ["single_to_pair_cond_row", "single_to_pair_cond_col",
                        "embed_pair_offsets", "embed_pair_distances"]) {
      expect(asked.has(`${root}_${leaf}/weights`)).toBe(true);
      expect(asked.has(`${root}_${leaf}_1/weights`)).toBe(false);
    }
  });
});
