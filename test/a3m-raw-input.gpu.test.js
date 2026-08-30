import { readFile } from "node:fs/promises";
import { afterAll, beforeAll, describe, expect, it } from "./harness.js";
import { create, globals } from "webgpu";
import { AlphaFoldMonomerGpu } from "../src/model/monomer.js";
import { AlphaFoldFixture } from "../src/reference/alphafold-fixture.js";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { requestAlphaFoldDevice } from "../src/runtime/device.js";

const enabled = process.env.LOCALFOLD_GPU_TESTS === "1";
const MANIFEST = "test/fixtures/evoformer/model1-query-59-stack/manifest.json";
describe.skipIf(!enabled)("raw A3M input to WebGPU prediction", () => {
  let gpu; let device;
  beforeAll(async() => { Object.assign(globalThis, globals); gpu = create([]); const adapter = await gpu.requestAdapter();
    if (adapter === null) throw new Error("no WebGPU adapter"); device = await requestAlphaFoldDevice(adapter); });
  afterAll(() => device?.destroy());
  it("predicts the literal uploaded test.a3m without Python preprocessing", async() => {
    const fixture = AlphaFoldFixture.fromStore(await FileTensorStore.open(MANIFEST));
    const [embedding, template, extraStack, mainStack, structure, confidence, geometry, featureTables] = await Promise.all([
      fixture.embeddingWeights(), fixture.templateWeights(), fixture.extraStackWeights(), fixture.mainStackWeights(),
      fixture.structureWeights(), fixture.confidenceWeights(), fixture.geometryTables(), fixture.queryOnlyFeatureTables(),
    ]);
    const prediction = await new AlphaFoldMonomerGpu(device).predictA3m(await readFile("test.a3m", "utf8"), {
      embedding, template, extraStack, mainStack, structure, lddt: confidence.lddt, pae: confidence.pae, geometry,
    }, featureTables, { recycles: 0, randomSeed: 0 }, await fixture.tensor("confidencePaeBreaks"));
    expect(prediction.final.confidence.meanPlddt).toBeGreaterThan(90);
    expect(prediction.final.confidence.ptm).toBeGreaterThan(0.65);
  }, 120_000);
});
