import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "./harness.js";
import { parseCifToAtomArray } from "../shared/rfd4p/pipeline/cif.js";
import { decompress } from "../shared/rfd4p/vendor/fzstd.js";
import { ROLE_LABELS, designSpec } from "../shared/rfd4p/design-spec.js";

// 🔴 NO StudioDoc. Upstream's version of this test builds its atom arrays
// through the AtomWorks Studio document model - one class out of 126 KB of
// editor API that this repository deliberately did not take - when the examples
// already carry their condition annotations in the mmCIF and
// `parseCifToAtomArray` preserves them. So the fixtures are read straight off
// disk and `designSpec` is held to its output.
const example = (name) => {
  const path = fileURLToPath(new URL(`../web/examples/${name}.cif.zst`, import.meta.url));
  const text = new TextDecoder().decode(decompress(new Uint8Array(readFileSync(path))));
  return designSpec(parseCifToAtomArray(text));
};

describe("what a design job asks for, read off a structure", () => {
  // 🔴 THIS IS THE CONTIG MODEL, AND THESE SEVEN CASES ARE WHY IT IS THE RIGHT
  // ABSTRACTION: every modality falls out of two per-span axes (is the geometry
  // given, is the sequence given) plus a length range, with no per-modality
  // branch anywhere. The contig strings below are RFdiffusion's own notation.
  const CASES = {
    // nothing given: one generated span
    uncond_short: { key: "unconditional", contig: "100", roles: [["A", "new", 100, 100]] },
    // a generated binder of a RANGE of lengths, against a chain that is given whole
    ppi_pdl1: { key: "binder", contig: "50-120/0 B1-115", roles: [["A", "new", 50, 120], ["B", "fixed", 115, 115]] },
    // 🔴 THE PAIR THAT PROVES THE MODEL CARRIES "indexed" vs "unindexed": the
    // same protein, the same motif, differing ONLY in whether the motif's
    // position is pinned. Indexed spells the generated spans out around the two
    // given segments; unindexed puts the motif in brackets and lets the method
    // place it. Nothing but the annotations differs.
    motif_idx_1prw: {
      key: "motif", contig: "5-20/A1-20/10-25/A22-41/5-20", roles: [["A", "motif", 60, 105]],
    },
    motif_unidx_1prw: { key: "motif", contig: "60-105/[A1-40]", roles: [["A", "motif", 60, 105]] },
    // a ligand is a span with no residues of its own
    sm_sam: { key: "sm_binder", contig: "120-165 + S;M", roles: [["B", "ligand", 0, 0], ["A", "new", 120, 165]] },
    // two given nucleic chains and a generated protein
    dna_7m5w: {
      key: "binder", contig: "A1-9/0 B1-10/0 120-150",
      roles: [["A", "fixed", 9, 9], ["B", "fixed", 10, 10], ["C", "new", 120, 150]],
    },
    // geometry given, sequence free - which is inverse folding, and needs no
    // generated span at all
    "5o45_cropped": { key: "redesign_all", contig: "115", roles: [["A", "free", 115, 115]] },
  };

  for (const [name, wanted] of Object.entries(CASES)) {
    it(`reads ${name} as ${wanted.key}`, () => {
      const spec = example(name);
      expect(spec.task.key).toBe(wanted.key);
      expect(spec.contig).toBe(wanted.contig);
      expect(spec.chains.map((chain) => [chain.id, chain.role, chain.lengthMin, chain.lengthMax]))
        .toEqual(wanted.roles);
    });
  }

  // 🔴 A ROLE WITH NO LABEL IS A ROLE THE PAGE CANNOT NAME, and the page's whole
  // job is saying what is being decided. Every role these fixtures produce has
  // to have a word for it.
  it("has a label for every role the fixtures produce", () => {
    const seen = new Set();
    for (const name of Object.keys(CASES)) {
      for (const chain of example(name).chains) seen.add(chain.role);
    }
    expect(seen.size).toBeGreaterThan(3);
    for (const role of seen) expect(typeof ROLE_LABELS[role]).toBe("string");
  });

  // Pure: the same atoms give the same answer, and nothing is cached across calls.
  it("is pure - the same structure reads the same twice", () => {
    expect(example("ppi_pdl1").contig).toBe(example("ppi_pdl1").contig);
  });

  // An empty input is a legal question with an empty answer, not a throw: the
  // page calls this on every edit, including the edit that removes everything.
  it("answers for nothing at all", () => {
    const spec = designSpec(undefined);
    expect(Array.isArray(spec.chains)).toBe(true);
    expect(spec.chains.length).toBe(0);
  });
});
