import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "./harness.js";
import { ESMFOLD2_ATOMISED_STEPS, ESMFOLD2_COUNTS, esmfold2StepsFor } from "../web/esmfold2-model.js";
import { SAMPLER_PRESETS } from "../webgpu/esmfold2/fold.js";

/**
 * ESMFold2's step floor for per-atom tokens, and the CUDA worker's copy of it.
 *
 * 🔴 AT ITS ELEVEN DEFAULT STEPS ESMFold2 TEARS A LIGAND - biotin's atoms 0.2-0.7 A apart, on both ports and
 * on the vendor's own model - and 45 hold it (web/esmfold2-model.js, ESMFOLD2_ATOMISED_STEPS). The page applies
 * the floor in samplerPreset and cuda/worker.py applies it to a job no page sent, from constants of its own:
 * two copies of one number, which is the drift this file exists to stop. The worker's fold is gated for real
 * by `npm run test:cuda` (its biotin cases); this is the CPU half.
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

  it("is the number cuda/worker.py applies, and its default is the dial's", () => {
    const worker = readFileSync(new URL("../cuda/worker.py", import.meta.url), "utf8");
    const line = worker.match(/^ESMFOLD2_DEFAULT_STEPS, ESMFOLD2_ATOMISED_STEPS = (\d+), (\d+)$/m);
    assert.ok(line !== null, "cuda/worker.py no longer states ESMFOLD2_DEFAULT_STEPS, ESMFOLD2_ATOMISED_STEPS");
    expect(Number(line[1])).toBe(ESMFOLD2_COUNTS.diffusion.preferred);
    expect(Number(line[2])).toBe(ESMFOLD2_ATOMISED_STEPS);
  });
});
