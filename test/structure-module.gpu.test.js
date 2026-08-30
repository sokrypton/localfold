import { afterAll, beforeAll, describe, expect, it } from "./harness.js";
import { create, globals } from "webgpu";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { StructureModuleGpu } from "../src/structure/module.js";
import { errorMetrics } from "../src/triangle/types.js";

const enabled = process.env.LOCALFOLD_GPU_TESTS === "1";
const MANIFEST = "test/fixtures/evoformer/model1-query-59-stack/manifest.json";

describe.skipIf(!enabled)("complete AlphaFold structure module WebGPU", () => {
  let gpu; let device;
  beforeAll(async() => {
    Object.assign(globalThis, globals); gpu = create([]); const adapter = await gpu.requestAdapter();
    if (adapter === null) throw new Error("no WebGPU adapter"); device = await adapter.requestDevice();
  });
  afterAll(() => device?.destroy());
  it("matches official final representation and atom positions", async() => {
    const store = await FileTensorStore.open(MANIFEST);
    const manifest = store.manifest;
    const p = async(map, module, name) => {
      const tensor = map[module]?.[name]; if (tensor === undefined) throw new Error(`missing ${module}/${name}`);
      return store.tensor(tensor);
    };
    const sp = manifest.structureModule.parameters;
    const ep = manifest.embedding.parameters;
    const root = "fold_iteration";
    const ipa = `${root}/invariant_point_attention`;
    const sc = `${root}/rigid_sidechain`;
    const weights = {
      initialize: {
        singleProjectionWeight: await p(ep, "single_activations", "weights"),
        singleProjectionBias: await p(ep, "single_activations", "bias"),
        singleNormScale: await p(sp, "single_layer_norm", "scale"),
        singleNormOffset: await p(sp, "single_layer_norm", "offset"),
        initialProjectionWeight: await p(sp, "initial_projection", "weights"),
        initialProjectionBias: await p(sp, "initial_projection", "bias"),
      },
      ipa: {
        pairNormScale: await p(sp, "pair_layer_norm", "scale"), pairNormOffset: await p(sp, "pair_layer_norm", "offset"),
        queryScalarWeight: await p(sp, `${ipa}/q_scalar`, "weights"), queryScalarBias: await p(sp, `${ipa}/q_scalar`, "bias"),
        keyValueScalarWeight: await p(sp, `${ipa}/kv_scalar`, "weights"), keyValueScalarBias: await p(sp, `${ipa}/kv_scalar`, "bias"),
        queryPointWeight: await p(sp, `${ipa}/q_point_local`, "weights"), queryPointBias: await p(sp, `${ipa}/q_point_local`, "bias"),
        keyValuePointWeight: await p(sp, `${ipa}/kv_point_local`, "weights"), keyValuePointBias: await p(sp, `${ipa}/kv_point_local`, "bias"),
        trainablePointWeights: await p(sp, ipa, "trainable_point_weights"),
        attention2dWeight: await p(sp, `${ipa}/attention_2d`, "weights"), attention2dBias: await p(sp, `${ipa}/attention_2d`, "bias"),
        outputWeight: await p(sp, `${ipa}/output_projection`, "weights"), outputBias: await p(sp, `${ipa}/output_projection`, "bias"),
      },
      postAttention: {
        attentionNormScale: await p(sp, `${root}/attention_layer_norm`, "scale"),
        attentionNormOffset: await p(sp, `${root}/attention_layer_norm`, "offset"),
        transitionWeights: [await p(sp, `${root}/transition`, "weights"), await p(sp, `${root}/transition_1`, "weights"), await p(sp, `${root}/transition_2`, "weights")],
        transitionBiases: [await p(sp, `${root}/transition`, "bias"), await p(sp, `${root}/transition_1`, "bias"), await p(sp, `${root}/transition_2`, "bias")],
        transitionNormScale: await p(sp, `${root}/transition_layer_norm`, "scale"),
        transitionNormOffset: await p(sp, `${root}/transition_layer_norm`, "offset"),
        affineWeight: await p(sp, `${root}/affine_update`, "weights"), affineBias: await p(sp, `${root}/affine_update`, "bias"),
      },
      sidechain: {
        inputWeight: await p(sp, `${sc}/input_projection`, "weights"), inputBias: await p(sp, `${sc}/input_projection`, "bias"),
        initialInputWeight: await p(sp, `${sc}/input_projection_1`, "weights"), initialInputBias: await p(sp, `${sc}/input_projection_1`, "bias"),
        residual1Weights: [await p(sp, `${sc}/resblock1`, "weights"), await p(sp, `${sc}/resblock2`, "weights")],
        residual1Biases: [await p(sp, `${sc}/resblock1`, "bias"), await p(sp, `${sc}/resblock2`, "bias")],
        residual2Weights: [await p(sp, `${sc}/resblock1_1`, "weights"), await p(sp, `${sc}/resblock2_1`, "weights")],
        residual2Biases: [await p(sp, `${sc}/resblock1_1`, "bias"), await p(sp, `${sc}/resblock2_1`, "bias")],
        angleWeight: await p(sp, `${sc}/unnormalized_angles`, "weights"), angleBias: await p(sp, `${sc}/unnormalized_angles`, "bias"),
      },
    };
    const length = store.shape("feature_aatype_recycle3")[0];
    const msa = await store.tensor("stackRecycle3ExpectedMsa");
    const result = await new StructureModuleGpu(device).run({
      msaFirstRow: msa.subarray(0, length * 256), pair: await store.tensor("structureInputPair"),
      mask: await store.tensor("feature_seq_mask_recycle3"), aatype: await store.tensor("feature_aatype_recycle3"),
      atom37ToAtom14: await store.tensor("feature_residx_atom37_to_atom14_recycle3"),
      atom37Mask: await store.tensor("feature_atom37_atom_exists_recycle3"), length, weights,
      geometry: {
        defaultFrames: await store.tensor("geometryDefaultFrames"), atom14ToGroup: await store.tensor("geometryAtom14ToGroup"),
        atom14Positions: await store.tensor("geometryAtom14Positions"), atom14Mask: await store.tensor("geometryAtom14Mask"),
      },
    });
    const representation = errorMetrics(result.finalRepresentation, await store.tensor("structureFinalRepresentation"));
    const atoms = errorMetrics(result.atom37, await store.tensor("structureFinalAtomPositions"));
    const affineStages = await store.tensor("structureStage_affine_output");
    const angleStages = await store.tensor("structureStage_angles");
    const affine = errorMetrics(result.affine, affineStages.subarray(7 * length * 7));
    const angles = errorMetrics(result.angles, angleStages.subarray(7 * length * 14));
    expect(representation.meanAbsoluteError).toBeLessThan(2e-4);
    expect(representation.maxAbsoluteError).toBeLessThan(3e-3);
    expect(affine.maxAbsoluteError).toBeLessThan(3e-3);
    expect(angles.maxAbsoluteError).toBeLessThan(3e-3);
    expect(atoms.meanAbsoluteError).toBeLessThan(2e-4);
    expect(atoms.maxAbsoluteError).toBeLessThan(3e-3);
  }, 30_000);
});
