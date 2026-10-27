import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "./harness.js";
import { HUNTER_CANNOT, HUNTER_WOULD_DIFFER, presetSuits } from "../cpu/design/preset-methods.js";
import { chainsOf, contigOf, drawLengths } from "../web/contig-ui.js";
import { exampleSpans } from "../web/example-spans.js";
import { parseCifToAtomArray } from "../shared/rfd4p/pipeline/cif.js";
import { decompress } from "../shared/rfd4p/vendor/fzstd.js";
import { RFD4P_PRESETS } from "../shared/rfd4p/presets.js";

/** The structure an example ships with, or null when it names none. */
const structureOf = (preset) => {
  if (preset.cifUrl === null || preset.cifUrl === undefined) return null;
  const path = fileURLToPath(new URL(`../${preset.cifUrl.replace(/^\.\//, "")}`,
    import.meta.url));
  let bytes;
  try {
    bytes = new Uint8Array(readFileSync(path));
  } catch {
    return null;
  }
  return parseCifToAtomArray(new TextDecoder().decode(
    path.endsWith(".zst") ? decompress(bytes) : bytes));
};

/**
 * 🔴 THE SEQUENCE SOURCE THAT CAN SAY NO. An earlier version of this check
 * handed `chainsOf` a resolver that returned a sequence for ANY range, which
 * made every preset look runnable and hid two real mismatches. It answers only
 * for residues the structure actually contains.
 */
const sequenceFrom = (atoms) => (chain, first, last) => {
  if (atoms === null) return null;
  const present = new Set();
  for (let at = 0; at < atoms.nAtoms; at += 1) {
    if (String(atoms.chain_id[at]) === chain) present.add(Number(atoms.res_id[at]));
  }
  for (let residue = first; residue <= last; residue += 1) {
    if (!present.has(residue)) return null;
  }
  return "A".repeat(last - first + 1);
};

/** What Protein Hunter says about an example, the way the page builds it. */
const hunterProblems = (preset) => {
  // 🔴 THE PAGE'S OWN RULE (web/example-spans.js), so what this measures is what the page loads:
  // an annotation-driven example as its file describes it, any other as its contig moved onto it
  const atoms = structureOf(preset);
  const spans = exampleSpans(preset, atoms);
  if (spans === null) return ["the editor cannot parse this contig"];
  return [...new Set([
    ...contigOf(spans, "hunter").problems,
    ...chainsOf(drawLengths(spans, { run: 0, seed: 0 }),
      { sequenceOf: sequenceFrom(atoms) }).problems,
  ])];
};

describe("which examples each method can run", () => {
  // 🔴 THE DECLARED LIST AGAINST THE MEASUREMENT, BOTH WAYS. A preset the page
  // offers under Hunter that Hunter then refuses is an invitation to a dead
  // end; a preset the page HIDES that would have worked is a feature quietly
  // withheld. Neither is visible without loading the structures, which is why
  // the list cannot be derived from the presets' own fields.
  it("matches what Protein Hunter can actually build", () => {
    for (const preset of RFD4P_PRESETS) {
      const problems = hunterProblems(preset);
      const declared = presetSuits(preset.id, "hunter");
      // ...except the two that build and are withheld because they would MEAN something else there,
      // which must still build (or the reason given is the wrong one)
      const withheld = HUNTER_WOULD_DIFFER.has(preset.id);
      expect(`${preset.id}: ${declared ? "offered" : "hidden"}`)
        .toBe(`${preset.id}: ${problems.length === 0 && !withheld ? "offered" : "hidden"}`);
      if (withheld) expect(`${preset.id}: ${problems.join("; ")}`).toBe(`${preset.id}: `);
    }
  });

  // RFD4 wrote these examples, so it takes every one of them.
  it("offers every example under RFD4-Proteina", () => {
    for (const preset of RFD4P_PRESETS) expect(presetSuits(preset.id, "rfd4p")).toBe(true);
  });

  // 🔴 AND EVERY EXCLUSION NAMES A PRESET THAT EXISTS. An id that no longer
  // ships would hide nothing and read as a limitation that is still in force.
  it("excludes only presets that exist", () => {
    const ids = new Set(RFD4P_PRESETS.map((preset) => preset.id));
    for (const id of Object.keys(HUNTER_CANNOT)) expect(ids.has(id)).toBe(true);
  });

  // The count is asserted so that a change to it is a decision rather than a
  // drift - six of ten: the binder and the DNA binder joined when the examples loaded as their files
  // describe them, and partial diffusion and inverse folding left as meaning something else there.
  it("leaves Protein Hunter six of the ten", () => {
    const offered = RFD4P_PRESETS.filter((preset) => presetSuits(preset.id, "hunter"));
    expect(offered.length).toBe(6);
  });
});
