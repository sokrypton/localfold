import assert from "node:assert/strict";
import { describe, expect, it } from "./harness.js";
import { ESMFOLD2_ATOMISED_STEPS, ESMFOLD2_COUNTS, ESMFOLD2_LONG_RESIDUES, esmfold2StepsFor } from "../web/esmfold2-model.js";
import { SAMPLER_PRESETS } from "../webgpu/esmfold2/fold.js";

/**
 * ESMFold2's step floor for per-atom tokens.
 *
 * 🔴 AT ITS ELEVEN DEFAULT STEPS ESMFold2 TEARS A LIGAND - biotin's atoms 0.2-0.7 A apart, on both ports and
 * on the vendor's own model - and 45 hold it (web/esmfold2-model.js, ESMFOLD2_ATOMISED_STEPS). The page applies
 * the floor in samplerPreset and python/localfold/worker.py applies it to a job no page sent, from constants of its own:
 * two copies of one number, which is the drift this file exists to stop. The worker's fold is gated for real
 * by `npm run test:native` (its biotin cases); this is the CPU half.
 */
describe("ESMFold2's step floor for a ligand or a modified residue", () => {
  it("raises the dial to the floor only for per-atom tokens", () => {
    expect(esmfold2StepsFor(15, { atomised: true })).toBe(ESMFOLD2_ATOMISED_STEPS);
    expect(esmfold2StepsFor(15, { atomised: false })).toBe(15);
    expect(esmfold2StepsFor(15)).toBe(15);
    expect(esmfold2StepsFor(200, { atomised: true })).toBe(200);
  });

  it("is a preset the sampler has and a count the dial offers", () => {
    assert.ok(SAMPLER_PRESETS[`diffusion-${ESMFOLD2_ATOMISED_STEPS}`] !== undefined);
    expect(ESMFOLD2_COUNTS.diffusion.values).toContain(ESMFOLD2_ATOMISED_STEPS);
  });

});

/**
 * ...and the same floor for a long input: past ESMFOLD2_LONG_RESIDUES the checkpoint's eleven steps leave the backbone
 * SHORT (copies of 6MRR: 15.9% of CA-CA out of band at 3,060 residues, 91.5% at 4,080; 0.4% and 0.1% with the floor).
 * Three copies of the rule - the page, the worker, the native binary's standalone - held to one here.
 */
describe("ESMFold2's step floor for a long input", () => {
  it("raises the dial past the length, and only past it", () => {
    expect(esmfold2StepsFor(15, { residues: ESMFOLD2_LONG_RESIDUES })).toBe(15);
    expect(esmfold2StepsFor(15, { residues: ESMFOLD2_LONG_RESIDUES + 1 })).toBe(ESMFOLD2_ATOMISED_STEPS);
    expect(esmfold2StepsFor(200, { residues: 4080 })).toBe(200);
  });

});
