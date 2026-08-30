import { afterAll, beforeAll, describe, expect, it } from "./harness.js";
import { create, globals } from "webgpu";
import { ExtraMsaStackGpu } from "../src/evoformer/stack.js";
import { AlphaFoldFixture } from "../src/reference/alphafold-fixture.js";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { errorMetrics } from "../src/triangle/types.js";

const enabled = process.env.LOCALFOLD_GPU_TESTS === "1";
const MANIFEST = "test/fixtures/evoformer/model1-a3m-59-stack/manifest.json";

describe.skipIf(!enabled)("full extra-MSA stack WebGPU", () => {
  let gpu; let device;
  beforeAll(async() => {
    Object.assign(globalThis, globals); gpu = create([]); const adapter = await gpu.requestAdapter();
    if (adapter === null) throw new Error("no WebGPU adapter"); device = await adapter.requestDevice();
  });
  afterAll(() => device?.destroy());
  it("matches all four official blocks on the 1,024-row uploaded A3M activation", async() => {
    const fixture = AlphaFoldFixture.fromStore(await FileTensorStore.open(MANIFEST));
    const weights = await fixture.extraStackWeights();
    const result = await new ExtraMsaStackGpu(device).run({
      msa: await fixture.tensor("extraStackRecycle0InputMsa"),
      pair: await fixture.tensor("extraStackRecycle0InputPair"),
      msaMask: await fixture.tensor("feature_extra_msa_mask_recycle0"),
      pairMask: await fixture.tensor("extraStackPairMask"),
      sequences: 1024, length: 59, cM: 64, cZ: 128,
      cOuter: weights[0] .outerProductMean.leftBias.length,
      triangleHidden: weights[0] .triangleMultiplicationOutgoing.linearAPBias.length,
      blockWeights: weights,
    });
    const msa = errorMetrics(result.msa, await fixture.tensor("extraStackRecycle0ExpectedMsa"));
    const pair = errorMetrics(result.pair, await fixture.tensor("extraStackRecycle0ExpectedPair"));
    expect(msa.meanAbsoluteError).toBeLessThan(8e-3);
    expect(msa.maxAbsoluteError).toBeLessThan(0.5);
    expect(pair.meanAbsoluteError).toBeLessThan(8e-3);
    expect(pair.maxAbsoluteError).toBeLessThan(0.3);
  }, 120_000);
});
