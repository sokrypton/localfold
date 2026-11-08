/**
 * WEBGL2_FAMILIES against the bundles themselves.
 *
 * 🔴 A HAND-WRITTEN CAPABILITY LIST IS A CLAIM, AND THIS IS WHAT MAKES IT A MEASUREMENT. The page
 * refuses a family on the WebGL2 fallback by name, BEFORE the weights download, which is the only
 * place a refusal is worth anything to a visitor - but it can only do that from a list, and a list
 * drifts the moment a bundle is re-exported. So the list is held to the thing it is about: whether
 * that bundle carries AlphaFold 3's UNFUSED template embedder, which webgl/af3/trunk/template.js is
 * the port of. `template_pair_embedding_8` is the tensor that names it.
 *
 * 🔴 AND IN BOTH DIRECTIONS, because the dangerous failure is the quiet one. A family wrongly ON the
 * list downloads half a gigabyte and dies inside the template stack; a family wrongly OFF it is
 * refused on a browser that could have folded it. Neither announces itself.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { AF3_FAMILIES, WEBGL2_FAMILIES } from "../shared/bundles/manifests/index.js";

/** The one tensor that says a bundle uses the unfused template embedder. */
const UNFUSED = "template_pair_embedding_8";

/** OpenDDE has the unfused embedder and still cannot fold here: it re-tokenises after the trunk. */
const STRUCTURAL_TOKENS = ["opendde"];

async function tensorNames(family) {
  const module = await import(`../shared/bundles/manifests/${family}.js`);
  const manifest = module.default ?? Object.values(module)[0];
  return Object.keys(manifest?.tensors ?? {});
}

test("WEBGL2_FAMILIES is a subset of the AF3 lineage", () => {
  for (const family of WEBGL2_FAMILIES) {
    assert.ok(AF3_FAMILIES.includes(family), `${family} is not an AF3-lineage family`);
  }
});

test("every family on the list has the unfused template embedder the WebGL2 port implements",
  async () => {
    for (const family of WEBGL2_FAMILIES) {
      const names = await tensorNames(family);
      assert.ok(names.some((name) => name.includes(UNFUSED)),
        `${family} is on WEBGL2_FAMILIES but its manifest has no ${UNFUSED}:`
        + " the fold would die inside webgl/af3/trunk/template.js after the download");
      assert.ok(!STRUCTURAL_TOKENS.includes(family),
        `${family} re-tokenises after the trunk, which the WebGL2 fold has not ported`);
    }
  });

test("every lineage family left off the list is off it for a stated reason", async () => {
  for (const family of AF3_FAMILIES) {
    if (WEBGL2_FAMILIES.includes(family)) continue;
    const names = await tensorNames(family);
    const fused = !names.some((name) => name.includes(UNFUSED));
    assert.ok(fused || STRUCTURAL_TOKENS.includes(family),
      `${family} is NOT on WEBGL2_FAMILIES, yet it carries ${UNFUSED} and does not re-tokenise`
      + " - so a browser without WebGPU is being refused a fold it could have had");
  }
});
