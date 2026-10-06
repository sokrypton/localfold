/**
 * Do the PINNED registry manifests still describe weights this loader can read?
 *
 * 🔴 PUSHING TO `main` IS THE DEPLOY, AND THE WEIGHTS DO NOT TRAVEL WITH IT.
 * `src/bundles/manifests/*.js` is a committed copy of a manifest pinned to a
 * Hugging Face commit; the shards live there and are updated by hand. So a
 * change to what the LOADER reads can be correct, gated, and reviewed - and
 * still take the live site down, because the bundle it now needs is not the one
 * that is published.
 *
 * That is not hypothetical. OpenDDE joined `PADDED_SINGLE_COND` upstream, so
 * its diffusion single conditioning is 833 channels; the published bundle
 * carries an 831-wide `single_cond_initial_norm/scale`, and the loader RAISES
 * on it rather than folding at the wrong width - which is the right behaviour
 * and means the published bundle no longer loads at all.
 *
 * This is the cheapest thing that catches it: no GPU, no network, just the
 * committed manifest against the dialect the same code derives.
 */
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";

const DIR = new URL("../src/bundles/manifests/", import.meta.url);
const HEAD = "diffuser/~/diffusion_head";

/** Every registry module that carries an AF3-lineage tensor table. */
function manifests() {
  const out = [];
  for (const name of readdirSync(DIR)) {
    if (!name.endsWith(".js") || name === "index.js") continue;
    const text = readFileSync(new URL(name, DIR), "utf8");
    const at = text.indexOf("export const MANIFEST = ");
    if (at < 0) continue;
    const from = text.indexOf("{", at);
    // The literal runs to the end of its statement; JSON.parse finds the end.
    const end = text.indexOf("};", from);
    let manifest;
    try {
      manifest = JSON.parse(text.slice(from, end < 0 ? undefined : end + 1));
    } catch { continue; }
    if (manifest?.tensors?.[`${HEAD}/single_cond_initial_norm/scale`] === undefined) continue;
    out.push([name, manifest]);
  }
  return out;
}

describe("the pinned registry manifests", () => {
  const found = manifests();

  it("finds the AF3-lineage tables at all", () => {
    // 🔴 A RULE THAT STOPS MATCHING PASSES BY FINDING NOTHING. Two of these
    // exist today; a rename that hides them must fail here, not pass quietly.
    assert.ok(found.length >= 2, `only ${found.length} AF3-lineage manifests parsed`);
  });

  for (const [name, manifest] of found) {
    it(`${name} carries a single conditioning this loader can read`, async () => {
      const { dialectFor } = await import("../src/af3/dialect.js");
      const model = manifest.model?.name;
      assert.ok(typeof model === "string", `${name} names no model`);
      const dialect = dialectFor(model);
      const tensors = manifest.tensors;
      const scale = tensors[`${HEAD}/single_cond_initial_norm/scale`].shape[0];
      // The same three terms `diffusionConditioning` adds up, from the same
      // tensors: the trunk single, target_feat, and the padding the dialect
      // asks for. `single_activations` is [target_feat, trunk single].
      const [targetFeat, trunkSingle] =
        tensors["diffuser/evoformer/single_activations/weights"].shape;
      const pad = dialect.padSingleCondUnknownDna ? 2 : 0;
      assert.equal(scale, trunkSingle + targetFeat + pad,
        `${name}: the published bundle's single_cond_initial_norm is ${scale} wide `
        + `and this loader builds ${trunkSingle} + ${targetFeat} + ${pad} = `
        + `${trunkSingle + targetFeat + pad}. The bundle must be re-exported and `
        + "re-published before this reaches main - see docs/HOSTING.md.");
    });

    it(`${name} carries the encoder pair tensors the loader reads`, () => {
      // The `_1` form is a module the graph never calls; see
      // test/af3-diffusion-weights.test.js. A bundle exported before that was
      // understood may carry ZEROS under the name the loader now reads, which
      // no shape check can see - but its ABSENCE can be.
      for (const leaf of ["diffusion_single_to_pair_cond_row",
                          "diffusion_single_to_pair_cond_col",
                          "diffusion_embed_pair_offsets",
                          "diffusion_embed_pair_distances"]) {
        assert.ok(manifest.tensors[`${HEAD}/${leaf}/weights`] !== undefined,
          `${name} has no ${leaf}; the loader stopped reading the _1 form`);
      }
    });
  }
});
