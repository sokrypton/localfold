// Which partners a token's certainty is averaged over - the rule a ligand broke.
//
// 🔴 A LIGAND IS ONE TOKEN PER HEAVY ATOM, so the old rule - exclude a partner
// when the TOKEN INDICES are within `separation` - meant nothing there. Two
// consequences, both measured on ubiquitin plus ATP (76 residues, 31 atoms) and
// both reproduced by folding rather than argued:
//
// * The ligand's whole self-block was averaged in. Its atoms are a few
//   angstroms apart and the model is handed their geometry as the reference
//   conformer, so it predicts them sharply - and the ligand read certainty
//   0.8173 off its own conformer while the distogram predicted NO
//   protein-ligand contact at all (0 predicted against 64 the structure makes).
//   Under this rule it reads 0.2193, which is the same model being honest.
// * 62% of the fold's reported contacts (195 of 314) were the ligand's own
//   pairs, so the precision and recall a checker prints were mostly a statement
//   about ATP's internal geometry.
//
// 🔴 AND IT IS THE OLD RULE EXACTLY FOR ONE UNMODIFIED PROTEIN CHAIN, which is
// what every measurement behind CERTAINTY's three constants was made on. That
// is the gate this file exists for: a fold of ubiquitin alone comes back with
// the certainty vector IDENTICAL to every digit, before and after.
import { describe, expect, it } from "./harness.js";
import {
  CERTAINTY, createCertaintyShader, partnerKeys,
} from "../src/esmfold2/distogram-webgpu.js";

describe("the certainty's partner rule", () => {
  it("carries the asym id and the residue number, one pair per token", () => {
    // A two-residue chain, then a three-atom ligand: AF3's featuriser gives
    // every atom of a component the same residue number.
    const keys = partnerKeys({ asymId: [0, 0, 1, 1, 1],
                               residueIndex: [0, 1, 0, 0, 0] }, 5);
    expect(Array.from(keys)).toEqual([0, 0, 0, 1, 1, 0, 1, 0, 1, 0]);
  });

  it("drops a ligand's whole self-block, whatever the separation", () => {
    // The shader's own test, applied on the host: same asym, and the residue
    // numbers within `separation`.
    const keys = partnerKeys({ asymId: [0, 0, 1, 1, 1],
                               residueIndex: [0, 1, 0, 0, 0] }, 5);
    const excluded = (i, j) => keys[i * 2] === keys[j * 2]
      && Math.abs(keys[i * 2 + 1] - keys[j * 2 + 1]) <= CERTAINTY.separation;
    for (const [i, j] of [[2, 3], [2, 4], [3, 4]]) expect(excluded(i, j)).toBe(true);
    // ...and keeps every protein-ligand pair, which is the only thing the model
    // is actually predicting about where the ligand goes.
    for (const [i, j] of [[0, 2], [0, 4], [1, 3]]) expect(excluded(i, j)).toBe(false);
  });

  it("is the token-index rule on one unmodified protein chain", () => {
    const tokens = 40;
    const keys = partnerKeys({
      asymId: new Int32Array(tokens),
      residueIndex: Int32Array.from({ length: tokens }, (_, t) => t),
    }, tokens);
    for (let i = 0; i < tokens; i += 1) {
      for (let j = 0; j < tokens; j += 1) {
        const byResidue = keys[i * 2] === keys[j * 2]
          && Math.abs(keys[i * 2 + 1] - keys[j * 2 + 1]) <= CERTAINTY.separation;
        expect(byResidue).toBe(Math.abs(i - j) <= CERTAINTY.separation);
      }
    }
  });

  it("reads the rule from a buffer, so the shader cannot regrow the old one", () => {
    // 🔴 ASSERT ON THE GENERATED WGSL. A partner rule that never reaches the
    // kernel reports agreement with itself; the shape of the mistake this file
    // is about is a token index arriving where a residue number was meant.
    const wgsl = createCertaintyShader({ tokens: 8, separation: 3, modeCutoffBin: 25 });
    expect(wgsl).toContain("partner: array<vec2<i32>>");
    expect(wgsl).toContain("here.x == there.x");
    expect(wgsl).toContain("abs(here.y - there.y) <= 3");
    expect(wgsl.includes("other > token")).toBe(false);
  });
});
