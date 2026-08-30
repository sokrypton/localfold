import { afterAll, beforeAll, describe, expect, it } from "./harness.js";
import { create, globals } from "webgpu";
import { AlphaFoldMonomerGpu } from "../src/model/monomer.js";
import { AlphaFoldFixture } from "../src/reference/alphafold-fixture.js";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { requestAlphaFoldDevice } from "../src/runtime/device.js";

const enabled = process.env.LOCALFOLD_GPU_TESTS === "1";
const A3M_MANIFEST = "test/fixtures/evoformer/model1-a3m-59-stack/manifest.json";
const WEIGHT_MANIFEST = "test/fixtures/evoformer/model1-query-59-stack/manifest.json";

describe.skipIf(!enabled)("end-to-end uploaded A3M AlphaFold WebGPU", () => {
  let gpu; let device;
  beforeAll(async() => { Object.assign(globalThis, globals); gpu = create([]); const adapter = await gpu.requestAdapter();
    if (adapter === null) throw new Error("no WebGPU adapter"); device = await requestAlphaFoldDevice(adapter); });
  afterAll(() => device?.destroy());
  it("runs 508 clustered and 1,024 extra rows through four recycling passes", async() => {
    const input = AlphaFoldFixture.fromStore(await FileTensorStore.open(A3M_MANIFEST));
    const model = AlphaFoldFixture.fromStore(await FileTensorStore.open(WEIGHT_MANIFEST));
    const features = [];
    for (let recycle = 0; recycle < 4; recycle += 1) {
      const msa = `feature_msa_feat_recycle${recycle}`; const extra = `feature_extra_msa_recycle${recycle}`;
      features.push({
        targetFeatures: await input.tensor(`feature_target_feat_recycle${recycle}`), msaFeatures: await input.tensor(msa),
        msaMask: await input.tensor(`feature_msa_mask_recycle${recycle}`), extraMsa: await input.tensor(extra),
        extraHasDeletion: await input.tensor(`feature_extra_has_deletion_recycle${recycle}`),
        extraDeletionValue: await input.tensor(`feature_extra_deletion_value_recycle${recycle}`),
        extraMsaMask: await input.tensor(`feature_extra_msa_mask_recycle${recycle}`),
        residueIndex: await input.tensor(`feature_residue_index_recycle${recycle}`),
        aatype: await input.tensor(`feature_aatype_recycle${recycle}`), seqMask: await input.tensor(`feature_seq_mask_recycle${recycle}`),
        atom37ToAtom14: await input.tensor(`feature_residx_atom37_to_atom14_recycle${recycle}`),
        atom37Mask: await input.tensor(`feature_atom37_atom_exists_recycle${recycle}`),
        msaSequences: input.shape(msa)[0], extraSequences: input.shape(extra)[0],
        targetChannels: 22, msaFeatureChannels: 49,
      });
    }
    const [embedding, template, extraStack, mainStack, structure, confidence, geometry] = await Promise.all([
      model.embeddingWeights(), model.templateWeights(), model.extraStackWeights(), model.mainStackWeights(),
      model.structureWeights(), model.confidenceWeights(), model.geometryTables(),
    ]);
    const prediction = await new AlphaFoldMonomerGpu(device).predict(features, {
      embedding, template, extraStack, mainStack, structure, lddt: confidence.lddt, pae: confidence.pae, geometry,
    }, await model.tensor("confidencePaeBreaks"));
    const reference = (input.store.manifest).referencePrediction.recycleMetrics;
    for (let recycle = 0; recycle < 4; recycle += 1) {
      expect(Math.abs(prediction.recycles[recycle] .confidence.meanPlddt - reference[recycle] .meanPlddt)).toBeLessThan(0.05);
      expect(Math.abs(prediction.recycles[recycle] .confidence.ptm - reference[recycle] .ptm)).toBeLessThan(0.001);
    }
    expect(prediction.final.confidence.meanPlddt).toBeGreaterThan(95);
  }, 240_000);
});
