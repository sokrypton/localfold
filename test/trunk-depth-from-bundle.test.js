/**
 * Does anything still TYPE IN how deep a trunk is?
 *
 * 🔴 THIS IS THE GATE FOR A BUG THAT PRODUCED A PLAUSIBLE STRUCTURE AND NO
 * ERROR. `web/af3-model.js` loaded `trunkWeights(store, 48, 4)` with both
 * counts written out - right for four of the five AF3-lineage families and
 * wrong for boltz2, whose trunk pairformer is SIXTY-FOUR blocks. A stack is one
 * stacked tensor, so asking for 48 of 64 reads the first 48 slices, runs them,
 * and returns a trunk that never finished. The page folded 6MRR's 128-row
 * alignment to pLDDT 72.4 against 72.1 with no alignment at all, which reads as
 * "boltz2 is not using the MSA" and is nothing of the kind - the CLI, which
 * read the depth from the bundle, gave 96.1 on the identical batch.
 *
 * Two rules, because either alone passes while the other is broken:
 *   1. every published family's depths come from its OWN manifest, and boltz2's
 *      is 64 - so a bundle re-export that changed a depth fails here.
 *   2. no fold path passes a literal count. A bench may (`allowPrefix`), and
 *      says so at the call site.
 */
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";

const MANIFESTS = new URL("../src/reference/manifests/", import.meta.url);
const PAIRFORMER = "diffuser/evoformer/__layer_stack_no_per_layer_1"
  + "/trunk_pairformer/single_attention_q_projection/bias";
const MSA = "diffuser/evoformer/__layer_stack_no_per_layer/msa_stack"
  + "/outer_product_mean/output_b";

function manifests() {
  const out = [];
  for (const name of readdirSync(MANIFESTS)) {
    if (!name.endsWith(".js") || name === "index.js") continue;
    const text = readFileSync(new URL(name, MANIFESTS), "utf8");
    const at = text.indexOf("export const MANIFEST = ");
    if (at < 0) continue;
    const from = text.indexOf("{", at);
    const end = text.indexOf("};", from);
    let manifest;
    try { manifest = JSON.parse(text.slice(from, end < 0 ? undefined : end + 1)); }
    catch { continue; }
    const tensors = manifest?.tensors ?? {};
    const pairformer = Object.keys(tensors).find((k) => k.endsWith(PAIRFORMER.slice(PAIRFORMER.indexOf("__layer_stack"))));
    const msa = Object.keys(tensors).find((k) => k.endsWith(MSA.slice(MSA.indexOf("__layer_stack"))));
    if (pairformer === undefined || msa === undefined) continue;
    out.push([manifest.model?.name ?? name,
      tensors[pairformer].shape[0], tensors[msa].shape[0]]);
  }
  return out;
}

describe("a trunk's depth is the bundle's", () => {
  const found = manifests();

  it("finds the AF3-lineage tables at all", () => {
    // A rule that stops matching passes by finding nothing.
    assert.ok(found.length >= 3, `only ${found.length} AF3-lineage manifests parsed`);
  });

  it("does not agree on one depth, which is the whole point", () => {
    const depths = new Set(found.map(([, pairformer]) => pairformer));
    assert.ok(depths.size >= 2,
      `every family reports ${[...depths]} pairformer blocks - if that is really`
      + " true this gate is measuring nothing, and if it is not, the reader is broken");
  });

  it("still has boltz2 at 64 pairformer blocks", () => {
    const boltz2 = found.find(([name]) => name === "boltz2");
    assert.ok(boltz2 !== undefined, "no boltz2 manifest in the registry");
    assert.equal(boltz2[1], 64);
    assert.equal(boltz2[2], 4);
  });

  it("has no fold path with a literal block count", () => {
    // 🔴 THE CALL SITES, NOT THE DEPTHS. `trunkWeights` raises on a mismatch
    // now, but a bench legitimately walks a prefix and passes `allowPrefix` -
    // so the rule is that a LITERAL never reaches it, from anywhere. A count
    // read from the store or from a flag is fine; `48` is not.
    const roots = ["../src/", "../web/", "../tools/gpu/"];
    const offenders = [];
    let scanned = 0;
    const walk = (dir) => {
      for (const entry of readdirSync(dir, { withFileTypes: true })) {
        const at = new URL(`${entry.name}${entry.isDirectory() ? "/" : ""}`, dir);
        if (entry.isDirectory()) { if (entry.name !== "node_modules") walk(at); continue; }
        if (!entry.name.endsWith(".js")) continue;
        scanned += 1;
        const text = readFileSync(at, "utf8");
        for (const line of text.split("\n")) {
          if (line.trimStart().startsWith("//") || line.trimStart().startsWith("*")) continue;
          const call = /trunkWeights\(\s*[A-Za-z0-9_.]+\s*,\s*([^),]+)/.exec(line);
          if (call === null) continue;
          if (/^\s*\d+\s*$/.test(call[1])) {
            offenders.push(`${entry.name}: ${line.trim()}`);
          }
        }
      }
    };
    for (const root of roots) walk(new URL(root, import.meta.url));
    assert.ok(scanned > 100, `only ${scanned} modules scanned - the sweep is broken`);
    assert.deepEqual(offenders, []);
  });
});
