import test from "node:test";
import assert from "node:assert/strict";
import { af3Sources } from "./helpers/af3-source.js";
import { DIALECTS } from "../src/af3/dialect.js";

/**
 * A dialect flag reaches code by three routes, and this pins which.
 *
 * 🔴 BECAUSE ONE CONVENTION TAKES ALL THREE AT ONCE AND NOTHING CHECKED THEY
 * AGREE. `diffusionNoResidual` is read four times from three different sources:
 * `input.dialect` in the atom encoder's shape, `weights.blocks[0]` in the atom
 * decoder, `weights` (one block) in the CPU reference, and `dialect` directly
 * for the token transformer. The loader copies the dialect's value onto every
 * block, so today they agree by construction - but the decoder reads block
 * ZERO where the reference reads the block it was handed, so a copy that failed
 * for a later block leaves two of the four right and says nothing.
 *
 * docs/ARCHITECTURE.md calls collapsing the routes a design decision. This is
 * the differential that has to come first either way: the grid's weight-order
 * collapse was only safe because its differential landed before it.
 *
 * 🔴 IT IS STRUCTURAL, OVER THE SOURCE, because the three sources are not all
 * reachable from one process without a bundle - and the failure it guards is a
 * new read site added against a new source, which is a property of the text.
 */
const SOURCE = af3Sources();

/**
 * The flags the weight loader copies onto every atom block, read out of the
 * loader itself rather than typed here - a list typed twice is the thing this
 * file exists to stop.
 */
const copied = [...SOURCE.get("diffusion-weights.js")
  .matchAll(/^\s*block\.([A-Za-z0-9_]+) = dialect\.\1;/gm)].map((m) => m[1]);

/** Which expression a flag is read off, per file, ignoring comments. */
function routesOf(flag) {
  const found = [];
  for (const [name, text] of SOURCE) {
    for (const line of text.split("\n")) {
      const code = line.replace(/^\s*(\/\/|\*).*$/, "");
      if (!code.includes(flag)) continue;
      if (new RegExp(`block\\.${flag} = dialect\\.${flag};`).test(code)) continue;
      if (new RegExp(`(input\\.dialect\\??\\.|dialect\\??\\.)${flag}`).test(code)) {
        found.push(`${name}:dialect`);
      } else if (new RegExp(`(weights|block)[A-Za-z0-9_.[\\]?]*\\.${flag}`).test(code)) {
        found.push(`${name}:block`);
      }
    }
  }
  return [...new Set(found)].sort();
}

test("how a dialect flag reaches code", async (t) => {
  await t.test("the loader copies exactly the flags this file knows about", () => {
    // 🔴 A RULE THAT STOPS MATCHING PASSES BY FINDING NOTHING.
    assert.ok(copied.length >= 3, `parsed ${copied.length} copied flags from the loader`);
    assert.deepEqual(copied.slice().sort(),
      ["chainedAtomLayerNorm", "diffusionNoResidual", "keyMaskedAtomAttention",
       "maskAtomActPerBlock"].slice().sort());
  });

  await t.test("every copied flag is a real convention every dialect states", () => {
    for (const flag of copied) {
      for (const [name, dialect] of Object.entries(DIALECTS)) {
        assert.ok(flag in dialect, `${name} does not state ${flag}`);
        assert.notEqual(dialect[flag], undefined, `${name}.${flag} is undefined`);
      }
    }
  });

  // 🔴 THE PIN. Each entry is a read site's SOURCE, and a new one - a fourth
  // route, or an existing flag suddenly read off the dialect where it was read
  // off the block - turns this red. That is the point: it is not that the
  // routing is wrong, it is that a change to it must be deliberate.
  await t.test("the routes are the ones docs/ARCHITECTURE.md records", () => {
    // 🔴 THREE OF THE FOUR COPIED FLAGS TAKE EXACTLY ONE ROUTE - the block -
    // and are consistent everywhere.
    for (const flag of ["maskAtomActPerBlock", "chainedAtomLayerNorm",
                        "keyMaskedAtomAttention"]) {
      assert.deepEqual(routesOf(flag), [
        "atom-decoder-webgpu.js:block",
        "atom-encoder-reference.js:block",
        "atom-encoder-webgpu.js:block",
        "diffusion-weights.js:block",
      ], `${flag} no longer takes one route`);
    }
    // 🔴 AND `diffusionNoResidual` USED TO TAKE TWO, WHICH WAS THE FINDING THAT
    // PRODUCED THIS FILE. The atom encoder read it off `input.dialect` while
    // the decoder and the CPU reference read the copy on the block - one
    // convention, two sources, agreeing only because the loader writes the
    // dialect's value onto every block. It reads off the block now, so every
    // copied flag takes exactly one route. The loader keeps its own
    // `dialect.` read, which is the assignment and its undefined guard.
    assert.deepEqual(routesOf("diffusionNoResidual"), [
      "atom-decoder-webgpu.js:block",
      "atom-encoder-reference.js:block",
      "atom-encoder-webgpu.js:block",
      "diffusion-weights.js:block",
      "diffusion-weights.js:dialect",
    ]);
    // 🔴 THE RULE, RATHER THAN THE LIST: outside the loader, a copied flag is
    // read off the BLOCK and never off a dialect. That is what makes a fifth
    // read site against a new source fail, whatever it is called.
    for (const flag of copied) {
      const outside = routesOf(flag).filter((r) => !r.startsWith("diffusion-weights.js"));
      assert.deepEqual([...new Set(outside.map((r) => r.split(":")[1]))], ["block"],
                       `${flag} is read off something other than the block: ${outside}`);
    }
  });

  // 🔴 AND THE ONE THAT WOULD ACTUALLY BITE: the decoder reads block ZERO, the
  // reference reads the block it was handed. They agree only while the loader
  // copies the same value onto every block - which it does, and which nothing
  // said out loud until this.
  await t.test("the decoder's block-zero read is equivalent to a per-block one", () => {
    const decoder = SOURCE.get("atom-decoder-webgpu.js");
    assert.ok(/weights\.blocks\[0\]\?\.diffusionNoResidual/.test(decoder),
              "the decoder no longer reads block zero - re-check this rule");
    const loader = SOURCE.get("diffusion-weights.js");
    // The copy is inside the per-block builder, so every block gets it.
    assert.ok(/async function atomBlockWith\([^)]*dialect\)/.test(loader),
              "the per-block builder no longer takes the dialect");
  });
});
