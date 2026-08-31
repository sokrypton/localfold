import { describe, expect, it } from "./harness.js";
import { makeQueryOnlyFeatures } from "../src/input/query-only-features.js";
import { AlphaFoldFixture } from "../src/reference/alphafold-fixture.js";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { errorMetrics } from "../src/triangle/types.js";

const MANIFEST = "test/fixtures/evoformer/model1-query-59-stack/manifest.json";
const SEQUENCE = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK";

describe("query-only feature construction", () => {
  it("applies chain breaks without changing sequence feature shapes", async() => {
    const fixture = AlphaFoldFixture.fromStore(await FileTensorStore.open(MANIFEST));
    const generated = makeQueryOnlyFeatures("ACDEF", await fixture.queryOnlyFeatureTables(), {
      recycles: 0, chainLengths: [2, 3], maskedMsaCodes: [Float32Array.of(0, 4, 3, 6, 13)],
    })[0];
    expect(Array.from(generated.residueIndex)).toEqual([0, 1, 202, 203, 204]);
    expect(generated.targetFeatures.length).toBe(5 * 22);
    expect(generated.msaFeatures.length).toBe(5 * 49);
  });

  it("reconstructs official processed tensors from sequence and seeded mask codes", async() => {
    const fixture = AlphaFoldFixture.fromStore(await FileTensorStore.open(MANIFEST));
    const codes = [];
    for (let recycle = 0; recycle < 4; recycle += 1) {
      const msa = await fixture.tensor(`feature_msa_feat_recycle${recycle}`);
      const row = new Float32Array(SEQUENCE.length);
      for (let residue = 0; residue < SEQUENCE.length; residue += 1) {
        let code = 0;
        for (let channel = 1; channel < 23; channel += 1) {
          if (msa[residue * 49 + channel] > msa[residue * 49 + code]) code = channel;
        }
        row[residue] = code;
      }
      codes.push(row);
    }
    const generated = makeQueryOnlyFeatures(SEQUENCE, await fixture.queryOnlyFeatureTables(), {
      recycles: 3, maskedMsaCodes: codes,
    });
    for (let recycle = 0; recycle < 4; recycle += 1) {
      const actual = generated[recycle];
      expect(errorMetrics(actual.targetFeatures, await fixture.tensor(`feature_target_feat_recycle${recycle}`)).maxAbsoluteError).toBe(0);
      expect(errorMetrics(actual.msaFeatures, await fixture.tensor(`feature_msa_feat_recycle${recycle}`)).maxAbsoluteError).toBeLessThan(1e-7);
      expect(errorMetrics(actual.aatype, await fixture.tensor(`feature_aatype_recycle${recycle}`)).maxAbsoluteError).toBe(0);
      expect(errorMetrics(actual.atom37ToAtom14, await fixture.tensor(`feature_residx_atom37_to_atom14_recycle${recycle}`)).maxAbsoluteError).toBe(0);
      expect(errorMetrics(actual.atom37Mask, await fixture.tensor(`feature_atom37_atom_exists_recycle${recycle}`)).maxAbsoluteError).toBe(0);
    }
  });

  it("leaves residues unmasked when randomMasking is disabled", async() => {
    const fixture = AlphaFoldFixture.fromStore(await FileTensorStore.open(MANIFEST));
    const generated = makeQueryOnlyFeatures("ACDEF", await fixture.queryOnlyFeatureTables(), {
      recycles: 2, randomMasking: false,
    });
    expect(generated.length).toBe(3);
    for (const pass of generated) {
      // Residues A, C, D, E, F correspond to aatypes 0, 4, 3, 6, 13
      const expectedAatype = [0, 4, 3, 6, 13];
      for (let r = 0; r < 5; r += 1) {
        const code = expectedAatype[r];
        expect(pass.msaFeatures[r * 49 + code]).toBe(1);
        expect(pass.msaFeatures[r * 49 + 22]).toBe(0); // mask token 22 is never set
      }
    }
  });
});
