// `lm_mask_pct`: the training-time input corruption, and why it is off here.
//
// 🔴 THE CONFIG CLASS'S DOCSTRING WOULD TALK YOU INTO TURNING IT ON.
// `EsmFold2Config` documents `lm_mask_pct` as "Single-sequence checkpoints set
// this to 0.1" - and base600M-step1500k IS a single-sequence checkpoint
// (`disable_msa_features: true`) that does not set it, so it takes the 0.0
// default and upstream's own `if lm_mask_pct:` never fires. A port that masked
// a tenth of every sequence would produce a plausible structure and be a
// different model, with nothing in the shapes to say so.
import { describe, expect, it } from "./harness.js";
import {
  ESM_BOS, ESM_EOS, ESM_MASK, ESM_PAD, maskLanguageModelInput,
} from "../src/esmfold2/featurise.js";
import { uniforms } from "../src/esmfold2/sampler-reference.js";

/** The packed run the tower is handed: [BOS] A [EOS BOS] B [EOS]. */
const packed = () => Int32Array.from([
  ESM_BOS, 4, 5, 6, 7, 8, ESM_EOS, ESM_BOS, 9, 10, 11, 12, ESM_EOS]);

describe("the language model's input masking", () => {
  it("does nothing at all at zero, which is what this checkpoint asks for", () => {
    const ids = packed();
    const before = Array.from(ids);
    // 🔴 AND IT DRAWS NOTHING EITHER. A generator handed in but consulted
    // anyway would shift every later value, so a fold at fraction zero would
    // stop matching a fold from before this existed.
    let draws = 0;
    const counted = () => { draws += 1; return 0; };
    expect(maskLanguageModelInput(ids, 0, counted)).toBe(0);
    expect(Array.from(ids)).toEqual(before);
    expect(draws).toBe(0);
  });

  it("never masks a separator, which would merge two chains", () => {
    // 🔴 THE IDS ARE ONE PACKED RUN, so a mask landing on the [EOS BOS] in the
    // middle would hand the tower a single chain of the wrong length. Upstream
    // exempts bos, eos and pad by value; so does this.
    const ids = packed();
    maskLanguageModelInput(ids, 1, () => 0);
    expect(ids[0]).toBe(ESM_BOS);
    expect(ids[6]).toBe(ESM_EOS);
    expect(ids[7]).toBe(ESM_BOS);
    expect(ids[12]).toBe(ESM_EOS);
    // ...and everything that is a residue IS masked at fraction 1.
    for (const at of [1, 2, 3, 4, 5, 8, 9, 10, 11]) expect(ids[at]).toBe(ESM_MASK);
  });

  it("masks about the fraction asked for, and only residues", () => {
    const ids = new Int32Array(10000).fill(7);
    ids[0] = ESM_BOS; ids[1] = ESM_EOS; ids[2] = ESM_PAD;
    const masked = maskLanguageModelInput(ids, 0.1, uniforms(11));
    // 9997 residues at a tenth; a binomial's spread here is about 30.
    expect(masked).toBeGreaterThan(850);
    expect(masked).toBeLessThan(1150);
    expect(ids[0]).toBe(ESM_BOS);
    expect(ids[1]).toBe(ESM_EOS);
    expect(ids[2]).toBe(ESM_PAD);
  });

  it("is reproducible from the seed, and different across seeds", () => {
    const run = (seed) => {
      const ids = new Int32Array(2000).fill(7);
      maskLanguageModelInput(ids, 0.1, uniforms(seed));
      return Array.from(ids).join("");
    };
    expect(run(3)).toBe(run(3));
    expect(run(3) === run(4)).toBe(false);
    // ...and seed 0 is its own stream, which xorshift's fixed point once ate.
    expect(run(0) === run(1)).toBe(false);
  });
});
