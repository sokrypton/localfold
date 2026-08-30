import { afterAll, beforeAll, describe, expect, it } from "./harness.js";
import { create, globals } from "webgpu";
import { EvoformerStackGpu } from "../src/evoformer/stack.js";
import { AlphaFoldFixture } from "../src/reference/alphafold-fixture.js";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { errorMetrics } from "../src/triangle/types.js";
import { requestAlphaFoldDevice } from "../src/runtime/device.js";

const enabled = process.env.LOCALFOLD_GPU_TESTS === "1";
const MANIFEST = "test/fixtures/evoformer/model1-a3m-59-stack/manifest.json";
describe.skipIf(!enabled)("512-row main Evoformer stack WebGPU", () => {
  let gpu; let device;
  beforeAll(async() => { Object.assign(globalThis, globals); gpu = create([]); const adapter = await gpu.requestAdapter();
    if (adapter === null) throw new Error("no WebGPU adapter"); device = await requestAlphaFoldDevice(adapter); });
  afterAll(() => device?.destroy());
  it("matches all 48 official blocks on the uploaded A3M", async() => {
    const fixture = AlphaFoldFixture.fromStore(await FileTensorStore.open(MANIFEST)); const weights = await fixture.mainStackWeights();
    const sequences = fixture.shape("feature_msa_feat_recycle0")[0]; const length = 59; const cM = 256;
    const inputMsa = await fixture.tensor("stackRecycle0InputMsa");
    const result = await new EvoformerStackGpu(device).run({
      msa: inputMsa.subarray(0, sequences * length * cM), pair: await fixture.tensor("stackRecycle0InputPair"),
      msaMask: await fixture.tensor("feature_msa_mask_recycle0"), pairMask: await fixture.tensor("stackPairMask"),
      sequences, length, cM, cZ: 128, cOuter: 32, triangleHidden: 128, blockWeights: weights,
    });
    const expectedMsa = (await fixture.tensor("stackRecycle0ExpectedMsa")).subarray(0, sequences * length * cM);
    const msa = errorMetrics(result.msa, expectedMsa);
    const pair = errorMetrics(result.pair, await fixture.tensor("stackRecycle0ExpectedPair"));
    expect(msa.meanAbsoluteError).toBeLessThan(0.02);
    expect(msa.maxAbsoluteError).toBeLessThan(0.8);
    expect(pair.meanAbsoluteError).toBeLessThan(0.03);
    expect(pair.maxAbsoluteError).toBeLessThan(0.8);
  }, 120_000);
});
