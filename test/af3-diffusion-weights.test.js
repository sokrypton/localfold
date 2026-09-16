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
 * 🔴 THIS SUITE ASSERTED THE UNSUFFIXED FORM, AND THE UNSUFFIXED FORM IS THE
 * WRONG ONE. It has now been wrong in both directions, so read the evidence
 * rather than the last edit.
 *
 * Four of these tensors exist twice in AlphaFold 3's own checkpoint,
 * unsuffixed and `_1`, at identical shapes. That is not a typo: haiku numbers a
 * module the second time its constructor runs, and `atom_cross_attention.py`
 * builds a Linear called `<root>_single_to_pair_cond_row` at TWO call sites -
 * inside `_per_atom_conditioning`, over a token's own 24 dense slots, and again
 * in the encoder, in the queries-keys layout. The unsuffixed set belongs to the
 * first; `_1` belongs to the encoder, which is the one this port runs.
 *
 * 🔴 THE TRACE THAT ARGUED FOR UNSUFFIXED WAS TAKEN ON A REGRESSED REFERENCE.
 * `hk.intercept_methods` showed `_1` never firing - true, and only because
 * af3-any-model's 041ab187 ("stop computing three things the models then throw
 * away") had deleted the first call, on the grounds that its result is assigned
 * to `_`. It is discarded; deleting it nevertheless RENAMES the second call,
 * which then silently reads the first one's weights. Bisected over 395 commits
 * on an A10: `run_alphafold.py` on Google's own af3.bin.zst folds 6MRR with mean
 * CA-CB 1.5315 at 041ab187^ and 1.2610 at 041ab187, and reverting that one file
 * alone restores 1.5315.
 *
 * For AlphaFold 3 the two are different trained tensors - rms 0.088 against
 * 0.406 for the row projection, 0.576 against 0.014 for the offsets - so this
 * was never a naming preference. Measured on 6MRR, side-chain bond rms:
 * unsuffixed 0.339, `_1` 0.059, genuine AF3 0.051, the deposited crystal 0.049.
 * Every PORTED bundle writes one tensor into both names, so protenix2, boltz2,
 * intellifold2 and rosettafold3 are unaffected either way - which is why
 * AlphaFold 3 was the lineage's only outlier.
 *
 * 🔴 AND NO FOLD GATE COULD SEE IT: CA-RMSD to the crystal moves 0.660 -> 0.583
 * and pLDDT 83.11 -> 85.70. It is a BOND-LENGTH defect, and
 * `tools/gpu/bond-geometry.js` is the only instrument here that measures one.
 */
describe("the diffusion head's weight names", () => {
  it("takes the _1 queries-keys pair tensors, not the unsuffixed form", async () => {
    const asked = new Set();
    await diffusionWeights(recordingStore(asked), 1);
    const head = "diffuser/~/diffusion_head";
    for (const leaf of ["diffusion_single_to_pair_cond_row",
                        "diffusion_single_to_pair_cond_col",
                        "diffusion_embed_pair_offsets",
                        "diffusion_embed_pair_distances"]) {
      expect(asked.has(`${head}/${leaf}_1/weights`)).toBe(true);
      // ...and NOT the unsuffixed one, which belongs to the per-token branch.
      expect(asked.has(`${head}/${leaf}/weights`)).toBe(false);
    }
    // The one with no _1 form, so the set above cannot be "fixed" wholesale.
    expect(asked.has(`${head}/diffusion_embed_pair_offsets_valid/weights`)).toBe(true);
  });

  it("takes the same four, _1, for target_feat's own atom encoder", async () => {
    const asked = new Set();
    await targetFeatureWeights(recordingStore(asked));
    const root = "diffuser/evoformer_conditioning";
    for (const leaf of ["single_to_pair_cond_row", "single_to_pair_cond_col",
                        "embed_pair_offsets", "embed_pair_distances"]) {
      expect(asked.has(`${root}_${leaf}_1/weights`)).toBe(true);
      expect(asked.has(`${root}_${leaf}/weights`)).toBe(false);
    }
    expect(asked.has(`${root}_embed_pair_offsets_valid/weights`)).toBe(true);
  });
});
