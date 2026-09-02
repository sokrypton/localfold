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
import { diffusionWeights, targetFeatureWeights } from "../src/af3/diffusion-weights.js";

/** A store that answers every request with zeros and remembers what was asked. */
function recordingStore(asked) {
  const shapes = new Map([
    ["diffuser/~/diffusion_head/transformer/__layer_stack_with_per_layer"
      + "/pair_logits_projection/weights", [6, 128, 4, 16]],
  ]);
  return {
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

describe("the diffusion head's weight names", () => {
  it("takes the queries-keys pair tensors, which carry the _1 suffix", async () => {
    const asked = new Set();
    await diffusionWeights(recordingStore(asked), 1);
    const head = "diffuser/~/diffusion_head";
    for (const leaf of ["diffusion_single_to_pair_cond_row",
                        "diffusion_single_to_pair_cond_col",
                        "diffusion_embed_pair_offsets",
                        "diffusion_embed_pair_distances"]) {
      expect(asked.has(`${head}/${leaf}_1/weights`)).toBe(true);
      // ...and NOT the unsuffixed one, which has the same shape and is a
      // different module's.
      expect(asked.has(`${head}/${leaf}/weights`)).toBe(false);
    }
    // The one with no _1 form, so the set above cannot be "fixed" wholesale.
    expect(asked.has(`${head}/diffusion_embed_pair_offsets_valid/weights`)).toBe(true);
  });

  it("takes the same four, suffixed, for target_feat's own atom encoder", async () => {
    const asked = new Set();
    await targetFeatureWeights(recordingStore(asked));
    const root = "diffuser/evoformer_conditioning";
    for (const leaf of ["single_to_pair_cond_row", "single_to_pair_cond_col",
                        "embed_pair_offsets", "embed_pair_distances"]) {
      expect(asked.has(`${root}_${leaf}_1/weights`)).toBe(true);
      expect(asked.has(`${root}_${leaf}/weights`)).toBe(false);
    }
  });
});
