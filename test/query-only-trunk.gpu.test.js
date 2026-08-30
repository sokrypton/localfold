import { afterAll, beforeAll, describe, expect, it } from "./harness.js";
import { create, globals } from "webgpu";
import { InputEmbedderGpu } from "../src/evoformer/input-embedder.js";
import { EvoformerStackGpu, ExtraMsaPairStackGpu } from "../src/evoformer/stack.js";
import { QueryOnlyTemplateGpu } from "../src/evoformer/template.js";
import { AlphaFoldFixture } from "../src/reference/alphafold-fixture.js";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { ElementwiseAddGpu } from "../src/runtime/elementwise.js";
import { errorMetrics } from "../src/triangle/types.js";

const enabled = process.env.LOCALFOLD_GPU_TESTS === "1";
const MANIFEST = "test/fixtures/evoformer/model1-query-59-stack/manifest.json";

describe.skipIf(!enabled)("query-only AlphaFold trunk WebGPU", () => {
  let gpu;
  let device;
  beforeAll(async() => {
    Object.assign(globalThis, globals);
    gpu = create([]);
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === null) throw new Error("no WebGPU adapter");
    device = await adapter.requestDevice();
  });
  afterAll(() => device?.destroy());

  it("runs recycle 0 from processed sequence features through all 54 pair blocks", async() => {
    const fixture = AlphaFoldFixture.fromStore(await FileTensorStore.open(MANIFEST));
    const target = await fixture.tensor("feature_target_feat_recycle0");
    const msaFeatures = await fixture.tensor("feature_msa_feat_recycle0");
    const targetShape = fixture.shape("feature_target_feat_recycle0");
    const msaFeatureShape = fixture.shape("feature_msa_feat_recycle0");
    const extraShape = fixture.shape("feature_extra_msa_recycle0");
    const length = targetShape[0];
    const [embeddingWeights, templateWeights, extraWeights, mainWeights] = await Promise.all([
      fixture.embeddingWeights(), fixture.templateWeights(), fixture.extraPairStackWeights(), fixture.mainStackWeights(),
    ]);
    const embedding = await new InputEmbedderGpu(device).run({
      targetFeatures: target,
      msaFeatures,
      extraMsa: await fixture.tensor("feature_extra_msa_recycle0"),
      extraHasDeletion: await fixture.tensor("feature_extra_has_deletion_recycle0"),
      extraDeletionValue: await fixture.tensor("feature_extra_deletion_value_recycle0"),
      residueIndex: await fixture.tensor("feature_residue_index_recycle0"),
      aatype: await fixture.tensor("feature_aatype_recycle0"),
      previousMsaFirstRow: new Float32Array(length * 256),
      previousPair: new Float32Array(length * length * 128),
      previousPositions: new Float32Array(length * 37 * 3),
      length,
      msaSequences: msaFeatureShape[0],
      extraSequences: extraShape[0],
      targetChannels: targetShape[1],
      msaFeatureChannels: msaFeatureShape[2],
      msaChannels: 256,
      pairChannels: 128,
      extraMsaChannels: 64,
      weights: embeddingWeights,
    });
    const template = await new QueryOnlyTemplateGpu(device).run({
      length,
      templateChannels: 64,
      pairChannels: 128,
      pairMask: await fixture.tensor("extraStackPairMask"),
      weights: templateWeights,
    });
    const pairWithTemplate = await new ElementwiseAddGpu(device).run(
      embedding.pairWithoutTemplates, template.pairUpdate,
    );
    const extra = await new ExtraMsaPairStackGpu(device).run({
      msa: embedding.extraMsa,
      pair: pairWithTemplate,
      msaMask: await fixture.tensor("feature_extra_msa_mask_recycle0"),
      pairMask: await fixture.tensor("extraStackPairMask"),
      sequences: extraShape[0],
      length,
      cM: 64,
      cZ: 128,
      cOuter: extraWeights[0] .outerProductMean.leftBias.length,
      triangleHidden: extraWeights[0] .triangleMultiplicationOutgoing.linearAPBias.length,
      blockWeights: extraWeights,
    });
    const trunk = await new EvoformerStackGpu(device).run({
      msa: embedding.msa,
      pair: extra.pair,
      msaMask: await fixture.tensor("feature_msa_mask_recycle0"),
      pairMask: await fixture.tensor("extraStackPairMask"),
      sequences: 1,
      length,
      cM: 256,
      cZ: 128,
      cOuter: mainWeights[0] .outerProductMean.leftBias.length,
      triangleHidden: mainWeights[0] .triangleMultiplicationOutgoing.linearAPBias.length,
      blockWeights: mainWeights,
    });
    const expectedMsa = (await fixture.tensor("stackRecycle0ExpectedMsa")).subarray(0, length * 256);
    const msaMetrics = errorMetrics(trunk.msa, expectedMsa);
    const pairMetrics = errorMetrics(trunk.pair, await fixture.tensor("stackRecycle0ExpectedPair"));
    expect(msaMetrics.meanAbsoluteError).toBeLessThan(5e-3);
    expect(msaMetrics.maxAbsoluteError).toBeLessThan(1e-1);
    expect(pairMetrics.meanAbsoluteError).toBeLessThan(8e-3);
    expect(pairMetrics.maxAbsoluteError).toBeLessThan(3e-1);
  }, 60_000);
});
