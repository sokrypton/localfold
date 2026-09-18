import { readFileSync } from "node:fs";
import { describe, expect, it } from "./harness.js";
import { MODEL_BUNDLES, graphFamily } from "../src/bundles/manifests/index.js";
import { sourceFiles } from "./helpers/source-files.js";

const ROOT = new URL("..", import.meta.url).pathname;

/**
 * 🔴 A DELTA FAMILY IS ITS BASE'S GRAPH, AND ASKING ITS NAME IS THE BUG.
 *
 * AlphaFold 2 is five models per family and 2 to 5 ship as a difference on
 * model_1, so the registry carries `multimer-2` ... `monomer-5` beside the two
 * whole bundles. Everything decided by the GRAPH - which driver folds it,
 * whether its alignment is paired, how its passes are ranked - must resolve
 * through `graphFamily`; everything decided by the WEIGHTS - the shard cache
 * key, the download stem, the label - keeps the resolved name.
 *
 * Four call sites got that wrong at once, and the loudest reached a visitor:
 * `const multimer = family === "multimer"` sent `multimer-2` through the
 * MONOMER driver, which has no multimer template embedder and died in
 * QueryOnlyTemplateGpu with `Cannot read properties of undefined (reading
 * 'embeddingBias')` - after downloading 116 MiB. The quiet one was the
 * alignment: `monomer-3` would have been handed PAIRED rows, which is a
 * silently worse fold rather than an error.
 *
 * The behaviour half of this rule is in test/mmseqs2-api.test.js. This is the
 * structural half, because the next delta family will be added by somebody who
 * has not read either.
 */
describe("a delta family runs its base's graph", () => {
  it("resolves a delta to its base and leaves every other family alone", () => {
    expect(graphFamily("multimer-2")).toBe("multimer");
    expect(graphFamily("monomer-5")).toBe("monomer");
    expect(graphFamily("multimer")).toBe("multimer");
    expect(graphFamily("af3")).toBe("af3");
    // An unknown name is returned unchanged rather than resolved to nothing:
    // every caller here already refuses one, with a message that names it.
    expect(graphFamily("nonesuch")).toBe("nonesuch");
  });

  it("every delta in the registry has a base that is itself a whole bundle", () => {
    for (const [family, bundle] of Object.entries(MODEL_BUNDLES)) {
      if (bundle.delta === undefined) continue;
      const base = MODEL_BUNDLES[bundle.delta.base];
      expect(`${family}: ${base === undefined ? "missing" : base.delta === undefined}`)
        .toBe(`${family}: true`);
    }
  });

  // 🔴 THE PATTERN, NOT THE FOUR SITES. `x === "multimer"` on a variable named
  // for a family is right only where the value came from the model ROW (which
  // `chosenFamily` has not yet folded the number into) - and those are named
  // `chosen` and `row` for exactly that reason.
  it("no page or source file tests a resolved family name for a graph", () => {
    const offenders = [];
    const files = [...sourceFiles(`${ROOT}src`), ...sourceFiles(`${ROOT}web`)];
    const pattern = /\b(family|model|choice)\s*[!=]==\s*"(monomer|multimer)"/g;
    // ...over the CODE, not the comments. This rule is worth stating in prose
    // where it is broken most easily, and a gate that forbids describing it is
    // a gate people write around.
    const withoutComments = (text) => text
      .replace(/\/\*[\s\S]*?\*\//g, " ").replace(/^\s*\/\/.*$/gm, " ");
    for (const file of files) {
      const text = withoutComments(readFileSync(file, "utf8"));
      for (const match of text.matchAll(pattern)) {
        offenders.push(`${file.slice(ROOT.length)}: ${match[0]}`);
      }
    }
    expect(offenders).toEqual([]);
    // ...and it must be able to find one, or stripping the comments has
    // quietly turned it into a rule over an empty string.
    expect([...withoutComments('if (family === "multimer") {').matchAll(pattern)].length)
      .toBe(1);
  });
});
