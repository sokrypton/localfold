import { afterAll, beforeAll, describe, expect, it } from "./harness.js";
import { create, globals } from "webgpu";
import { InputEmbedderGpu } from "../src/evoformer/input-embedder.js";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { errorMetrics } from "../src/triangle/types.js";

const enabled = process.env.LOCALFOLD_GPU_TESTS === "1";
const MANIFEST = "test/fixtures/evoformer/model1-query-59-stack/manifest.json";

describe.skipIf(!enabled)("AlphaFold input/recycling embedder WebGPU", () => {
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

  it("matches recycle-0 embeddings before the template branch", async() => {
    const store = await FileTensorStore.open(MANIFEST);
    const parameters = (store.manifest).embedding.parameters;
    const parameter = async(module, name) => {
      const tensor = parameters[module]?.[name];
      if (tensor === undefined) throw new Error(`missing ${module}/${name}`);
      return store.tensor(tensor);
    };
    const weights = {
      preprocess1dWeight: await parameter("preprocess_1d", "weights"),
      preprocess1dBias: await parameter("preprocess_1d", "bias"),
      preprocessMsaWeight: await parameter("preprocess_msa", "weights"),
      preprocessMsaBias: await parameter("preprocess_msa", "bias"),
      leftSingleWeight: await parameter("left_single", "weights"),
      leftSingleBias: await parameter("left_single", "bias"),
      rightSingleWeight: await parameter("right_single", "weights"),
      rightSingleBias: await parameter("right_single", "bias"),
      previousPositionWeight: await parameter("prev_pos_linear", "weights"),
      previousPositionBias: await parameter("prev_pos_linear", "bias"),
      previousMsaNormScale: await parameter("prev_msa_first_row_norm", "scale"),
      previousMsaNormOffset: await parameter("prev_msa_first_row_norm", "offset"),
      previousPairNormScale: await parameter("prev_pair_norm", "scale"),
      previousPairNormOffset: await parameter("prev_pair_norm", "offset"),
      relativePositionWeight: await parameter("pair_activiations", "weights"),
      relativePositionBias: await parameter("pair_activiations", "bias"),
      extraMsaWeight: await parameter("extra_msa_activations", "weights"),
      extraMsaBias: await parameter("extra_msa_activations", "bias"),
    };
    const target = await store.tensor("feature_target_feat_recycle0");
    const msaFeatures = await store.tensor("feature_msa_feat_recycle0");
    const targetShape = store.shape("feature_target_feat_recycle0");
    const msaShape = store.shape("feature_msa_feat_recycle0");
    const extraShape = store.shape("feature_extra_msa_recycle0");
    const length = targetShape[0];
    const result = await new InputEmbedderGpu(device).run({
      targetFeatures: target,
      msaFeatures,
      extraMsa: await store.tensor("feature_extra_msa_recycle0"),
      extraHasDeletion: await store.tensor("feature_extra_has_deletion_recycle0"),
      extraDeletionValue: await store.tensor("feature_extra_deletion_value_recycle0"),
      residueIndex: await store.tensor("feature_residue_index_recycle0"),
      aatype: await store.tensor("feature_aatype_recycle0"),
      previousMsaFirstRow: new Float32Array(length * 256),
      previousPair: new Float32Array(length * length * 128),
      previousPositions: new Float32Array(length * 37 * 3),
      length,
      msaSequences: msaShape[0],
      extraSequences: extraShape[0],
      targetChannels: targetShape[1],
      msaFeatureChannels: msaShape[2],
      msaChannels: 256,
      pairChannels: 128,
      extraMsaChannels: 64,
      weights,
    });
    const expectedMsaAll = await store.tensor("stackRecycle0InputMsa");
    const expectedMsa = expectedMsaAll.subarray(0, length * 256);
    const msaMetrics = errorMetrics(result.msa, expectedMsa);
    const pairWithTemplates = await store.tensor("extraStackRecycle0InputPair");
    const templateUpdate = await store.tensor("templatePairUpdateRecycle0");
    const expectedPair = new Float32Array(pairWithTemplates.length);
    for (let index = 0; index < expectedPair.length; index += 1) {
      expectedPair[index] = pairWithTemplates[index] - templateUpdate[index];
    }
    const pairMetrics = errorMetrics(result.pairWithoutTemplates, expectedPair);
    const extraMetrics = errorMetrics(result.extraMsa, await store.tensor("extraStackRecycle0InputMsa"));
    expect(msaMetrics.meanAbsoluteError).toBeLessThan(1e-4);
    expect(msaMetrics.maxAbsoluteError).toBeLessThan(1e-3);
    expect(pairMetrics.meanAbsoluteError).toBeLessThan(1e-4);
    expect(pairMetrics.maxAbsoluteError).toBeLessThan(1e-3);
    expect(extraMetrics.meanAbsoluteError).toBeLessThan(1e-4);
    expect(extraMetrics.maxAbsoluteError).toBeLessThan(5e-4);
  });
});
