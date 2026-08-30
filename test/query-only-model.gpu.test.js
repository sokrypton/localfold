import { afterAll, beforeAll, describe, expect, it } from "./harness.js";
import { create, globals } from "webgpu";
import { AlphaFoldQueryOnlyGpu } from "../src/model/query-only.js";
import { AlphaFoldFixture } from "../src/reference/alphafold-fixture.js";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { errorMetrics } from "../src/triangle/types.js";

const enabled = process.env.LOCALFOLD_GPU_TESTS === "1";
const MANIFEST = "test/fixtures/evoformer/model1-query-59-stack/manifest.json";
const SEQUENCE = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK";

describe.skipIf(!enabled)("end-to-end query-only AlphaFold model WebGPU", () => {
  let gpu; let device;
  beforeAll(async() => {
    Object.assign(globalThis, globals); gpu = create([]); const adapter = await gpu.requestAdapter();
    if (adapter === null) throw new Error("no WebGPU adapter"); device = await adapter.requestDevice();
  });
  afterAll(() => device?.destroy());
  it("runs four recycling passes and matches official model-1 confidence", async() => {
    const fixture = AlphaFoldFixture.fromStore(await FileTensorStore.open(MANIFEST));
    const maskedMsaCodes = [];
    for (let recycle = 0; recycle < 4; recycle += 1) {
      const msaName = `feature_msa_feat_recycle${recycle}`;
      const msa = await fixture.tensor(msaName);
      const codes = new Float32Array(SEQUENCE.length);
      for (let residue = 0; residue < SEQUENCE.length; residue += 1) {
        let code = 0;
        for (let channel = 1; channel < 23; channel += 1) {
          if (msa[residue * 49 + channel] > msa[residue * 49 + code]) code = channel;
        }
        codes[residue] = code;
      }
      maskedMsaCodes.push(codes);
    }
    const [embedding, template, extraStack, mainStack, structure, confidence, geometry] = await Promise.all([
      fixture.embeddingWeights(), fixture.templateWeights(), fixture.extraPairStackWeights(), fixture.mainStackWeights(),
      fixture.structureWeights(), fixture.confidenceWeights(), fixture.geometryTables(),
    ]);
    const prediction = await new AlphaFoldQueryOnlyGpu(device).predictSequence(SEQUENCE, {
      embedding, template, extraStack, mainStack, structure, lddt: confidence.lddt, pae: confidence.pae, geometry,
    }, await fixture.queryOnlyFeatureTables(), { recycles: 3, maskedMsaCodes }, await fixture.tensor("confidencePaeBreaks"));
    const references = (fixture.store.manifest).confidenceHeads.recycleReferences;
    for (let recycle = 0; recycle < 4; recycle += 1) {
      const actual = prediction.recycles[recycle];
      const reference = references[recycle];
      expect(Math.abs(actual.confidence.meanPlddt - reference.meanPlddt)).toBeLessThan(0.02);
      expect(Math.abs(actual.confidence.ptm - reference.ptm)).toBeLessThan(0.001);
      const pair = errorMetrics(actual.pair, await fixture.tensor(`stackRecycle${recycle}ExpectedPair`));
      expect(pair.meanAbsoluteError).toBeLessThan(0.08);
    }
    const atoms = errorMetrics(prediction.final.structure.atom37, await fixture.tensor("structureFinalAtomPositions"));
    expect(atoms.meanAbsoluteError).toBeLessThan(0.25);
    expect(prediction.final.confidence.meanPlddt).toBeGreaterThan(60);
  }, 120_000);
});
