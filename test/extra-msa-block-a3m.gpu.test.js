import { afterAll, beforeAll, describe, expect, it } from "./harness.js";
import { create, globals } from "webgpu";
import { ExtraMsaStackGpu } from "../src/evoformer/stack.js";
import { AlphaFoldFixture } from "../src/reference/alphafold-fixture.js";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { errorMetrics } from "../src/triangle/types.js";

const enabled = process.env.LOCALFOLD_GPU_TESTS === "1";
const MANIFEST = "test/fixtures/evoformer/model1-a3m-59-stack/manifest.json";
describe.skipIf(!enabled)("one extra-MSA block WebGPU", () => {
  let gpu; let device;
  beforeAll(async() => { Object.assign(globalThis, globals); gpu = create([]); const adapter = await gpu.requestAdapter();
    if (adapter === null) throw new Error("no WebGPU adapter"); device = await adapter.requestDevice(); });
  afterAll(() => device?.destroy());
  it("matches the official 1,024-row MSA and pair block", async() => {
    const fixture = AlphaFoldFixture.fromStore(await FileTensorStore.open(MANIFEST)); const weights = await fixture.extraStackWeights();
    const result = await new ExtraMsaStackGpu(device).run({
      msa: await fixture.tensor("extraStackRecycle0InputMsa"), pair: await fixture.tensor("extraStackRecycle0InputPair"),
      msaMask: await fixture.tensor("feature_extra_msa_mask_recycle0"), pairMask: await fixture.tensor("extraStackPairMask"),
      sequences: 1024, length: 59, cM: 64, cZ: 128, cOuter: 32, triangleHidden: 128,
      blockWeights: [weights[0]],
    });
    const transitionInput = await fixture.tensor("stage_005_input");
    const transitionUpdate = await fixture.tensor("stage_006_output");
    const expectedMsa = transitionInput.map((value, index) => value + transitionUpdate[index]);
    const msa = errorMetrics(result.msa, expectedMsa);
    const pair = errorMetrics(result.pair, await fixture.tensor("extraReplayPair"));
    expect(msa.meanAbsoluteError).toBeLessThan(2e-3);
    expect(msa.maxAbsoluteError).toBeLessThan(0.08);
    expect(pair.meanAbsoluteError).toBeLessThan(3e-3);
    expect(pair.maxAbsoluteError).toBeLessThan(0.08);
  }, 120_000);
});
