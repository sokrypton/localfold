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
  ESM_BOS, ESM_EOS, ESM_MASK, ESM_PAD, featuriseForEsmfold2, maskLanguageModelInput,
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

/**
 * 🔴 AN ATOMISED RESIDUE IS UNKNOWN TO ESMFold2, NOT ITS PARENT - AND THE PORT
 * TOLD IT OTHERWISE. AF3 gives every atom token of a modified residue the
 * PARENT restype, so a phosphoserine's ten tokens all said SER: the model was
 * handed ten single atoms labelled as ten serines. Folded, the SEP came out at
 * a mean bond ratio of **2.349** against a 0.999 control in the same structure,
 * with `OG-P` at 2.06-4.25 A against a 1.610 ideal; the vendor `esm` package
 * reads 0.997 on the identical job, and this port now reads **0.994**.
 *
 * 🔴 THE TWO NUMBERS ARE THE VENDOR'S OWN, read off its atomised branch rather
 * than guessed: `TokenInfo(res_type=PROTEIN_UNK_RES_TYPE, input_id=DNA_RNA_LIGAND_INPUT_ID)`
 * - 22 and 24.
 *
 * 🔴 AND 24 IS NOT WHAT THE RESTYPE TABLE WOULD GIVE, which is the subtle half.
 * `AATYPE_TO_ESM_ID` maps the unknown restype to `<unk>` (3) deliberately: an
 * `X` in a sequence is a residue nobody identified, and the tower was trained
 * to see `<unk>` there. An atomised residue is a row of single ATOMS and the
 * tower sees a ligand's token. Two unknowns, two ids - so the fix cannot be a
 * single lookup and the note on that table stays true of its own case.
 *
 * Found by comparing against the vendor after a user reported the fold; the
 * same cause and the same fix as sokrypton/alphafold3's `96d1958` on its side.
 * See docs/ESMFOLD2_PTM.md.
 */
describe("an atomised residue's restype and language-model id", () => {
  const PROTEIN_UNK_RES_TYPE = 22;
  const DNA_RNA_LIGAND_INPUT_ID = 24;

  // A three-atom stand-in, so this needs no network and no dictionary: what is
  // under test is which LABEL the atom tokens carry, not the chemistry.
  const modification = {
    code: "XYZ", chain: 0, position: 2,
    atoms: [0, 1, 2].map((slot) => ({
      name: ["N", "CA", "C"][slot], element: 6, charge: 0,
      x: slot, y: 0, z: 0, componentSlot: slot,
    })),
    bonds: [{ from: 0, to: 1, order: 1 }, { from: 1, to: 2, order: 1 }],
  };
  const features = () => featuriseForEsmfold2({
    sequence: "GWSTE", chainKinds: ["protein"], ligands: [],
    modifications: [modification],
  });

  it("labels every atom token UNKNOWN, not the parent residue", () => {
    const f = features();
    // residue 2 of GWSTE is W, atomised into three tokens at 1..3.
    for (const token of [1, 2, 3]) {
      expect(f.residueType[token]).toBe(PROTEIN_UNK_RES_TYPE);
    }
  });

  it("gives them the LIGAND language-model id, not `<unk>`", () => {
    const f = features();
    for (const token of [1, 2, 3]) {
      expect(f.inputIds[token]).toBe(DNA_RNA_LIGAND_INPUT_ID);
    }
  });

  /**
   * 🔴 AND THE PEPTIDE BONDS AT THE JUNCTION, which AF3 drops. It extracts
   * inter-residue bonds only where one side is a LIGAND chain, so a residue
   * atomised inside a polymer keeps its own CCD bonds and loses the backbone
   * bond to each neighbour. Counted against `esm` 3.4.1's own featuriser on a
   * SEP + glycerol job: inside the SEP block 18 against 18, inside the
   * glycerol 10 against 10 - this port already symmetrises - but
   * SEP-to-neighbours **0 against 2**. It matters more for this model than for
   * the AF3 lineage because ESMFold2's atom attention has no pair bias, so
   * this matrix is its only statement that two atom tokens are bonded.
   *
   * The fold barely moves on it (SEP 0.994 -> 0.989 at 138 steps, one sample
   * each, which is noise) - it is here because the feature now MATCHES, not
   * because the number did.
   */
  it("bonds the atomised residue to its chain neighbours", () => {
    const f = features();
    const tokens = f.tokens;
    let across = 0;
    for (let token = 1; token <= 3; token += 1) {
      for (let other = 0; other < tokens; other += 1) {
        if (other >= 1 && other <= 3) continue;
        if (f.tokenBonds[token * tokens + other]
            || f.tokenBonds[other * tokens + token]) across += 1;
      }
    }
    expect(across).toBe(2);
  });

  it("leaves every unmodified residue alone", () => {
    const f = features();
    // G before it and S, T, E after: real restypes, real ESM ids. Asserted as
    // the whole row rather than token by token, because "not 22" would pass on
    // a row that had gone wrong some other way.
    expect([...f.residueType]).toEqual([9, 22, 22, 22, 17, 18, 8]);
    expect([...f.inputIds]).toEqual([6, 24, 24, 24, 8, 11, 9]);
  });
});
