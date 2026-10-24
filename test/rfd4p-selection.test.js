import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "./harness.js";
import { parseCifToAtomArray } from "../shared/rfd4p/pipeline/cif.js";
import { decompress } from "../shared/rfd4p/vendor/fzstd.js";
import { evalPathSelection } from "../shared/rfd4p/pipeline/spec.js";
import { RFD4P_PRESETS } from "../shared/rfd4p/presets.js";

const example = (name) => {
  const path = fileURLToPath(new URL(`../web/examples/${name}.cif.zst`, import.meta.url));
  return parseCifToAtomArray(
    new TextDecoder().decode(decompress(new Uint8Array(readFileSync(path)))));
};

describe("RFD4-Proteina's selection DSL, and how to write a residue list in it", () => {
  const atoms = example("ppi_pdl1");
  const count = (selection) => evalPathSelection(atoms, selection).filter(Boolean).length;

  // 🔴 A LIST OF RESIDUES MUST BE BRACKETED, AND THE BARE FORM FAILS SILENTLY.
  // `evalPathSelection` splits the WHOLE selection on commas before it splits a
  // path on "/", so `B/*/54,56,66/*` is four tokens - `B/*/54`, `56`, `66`,
  // `*` - of which only the first is a path. The other three match no chain and
  // contribute nothing. `expandPathField` exists for exactly this and handles
  // `[54,56,66]` and `[54-56]`.
  //
  // This is pinned because anything generating a selection has to know it, and
  // because getting it wrong produces a selection that works, returns atoms, and
  // conditions on a fraction of what was asked.
  it("takes a bracketed residue list and quietly drops a bare one", () => {
    expect(count("B/*/*/*")).toBe(924);
    expect(count("B/*/54/*")).toBe(4);
    expect(count("B/*/[54,56,66]/*")).toBe(24);
    // the bare form: residue 54's four atoms, and nothing for 56 or 66
    expect(count("B/*/54,56,66/*")).toBe(4);
    expect(count("B/*/[54-56]/*")).toBe(22);
  });

  // 🔴 AND THE SHIPPED `ppi_pdl1` PRESET WAS WRITTEN THE BARE WAY. Its own
  // description says it conditions on target hotspots, and its selection names
  // four residues; three of them were being dropped. Fixed to the bracketed
  // form, which this test now holds it to - a preset is the thing people copy.
  it("has every preset's hotspot selection bracketed", () => {
    for (const preset of RFD4P_PRESETS) {
      for (const [condition, arms] of Object.entries(preset.conditions ?? {})) {
        for (const arm of Object.values(arms)) {
          for (const entry of arm) {
            const selection = String(entry.select ?? "");
            const residues = selection.replace(/^sel\(['"]|['"]\)$/g, "").split("/")[2] ?? "";
            if (!residues.includes(",")) continue;
            expect({ preset: preset.id, condition, selection, bracketed: residues.startsWith("[") })
              .toEqual({ preset: preset.id, condition, selection, bracketed: true });
          }
        }
      }
    }
  });
});
