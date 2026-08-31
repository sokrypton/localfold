/**
 * Check the diffusion head's conditioning against AF3.
 *
 *     python3 tools/oracle/dump_af3_trunk.py --blocks 48 --diffusion 1 --float32 \
 *       --capture 'diffusion_head/[a-z_0-9]+/__call__$|evoformer/__call__$' \
 *       --out af3-oracle-diffusion-f32.json
 *     node tools/oracle/check_af3_diffusion_conditioning.js
 *
 * The conditioning is everything the denoiser knows besides the atoms, and it
 * is the half of the diffusion head that can be checked WITHOUT reproducing the
 * sampler's random state: it depends only on the trunk's outputs and the noise
 * level. The denoiser itself needs `positions_noisy`, which the sampler draws
 * from a PRNG this cannot reach.
 *
 * 🔴 --diffusion 1 IS WHAT MAKES THIS AFFORDABLE, and it also fixes the noise
 * level. AF3's schedule is
 *     sigma(t) = SIGMA_DATA * (smax^(1/p) + t*(smin^(1/p) - smax^(1/p)))^p
 * with smin 0.0004, smax 160, p 7 - so the first step runs at t = 0, which is
 * SIGMA_DATA * smax = 2560. That the Fourier embedding then matches to 4e-7 is
 * the check that the level was inferred right, not an assumption fed in.
 */
import { join } from "node:path";

import { layerNormSlow } from "../../src/af3/atom-encoder-reference.js";
import { SIGMA_DATA, diffusionConditioning } from "../../src/af3/diffusion-reference.js";
import { noiseEmbedding } from "../../src/af3/noise-fourier.js";
import { linear } from "../../src/af3/pairformer-reference.js";
import { ROOT, captures, loadDump, loadTensors, report } from "./af3-bundle.js";

const HEAD = "diffuser/~/diffusion_head";
const EVO = "diffuser/evoformer";
const NOISE_CHANNELS = 256;

/** A conditioning transition: plain LayerNorm, SwiGLU, project. No gate. */
const transitionWeights = (T, name) => ({
  FfwLayerNormScale: T(`${name}ffw_layer_norm/scale`),
  FfwLayerNormOffset: T(`${name}ffw_layer_norm/offset`),
  FfwTransition1: T(`${name}ffw_transition1/weights`),
  FfwTransition2: T(`${name}ffw_transition2/weights`),
});

async function main() {
  const model = process.argv.includes("--model")
    ? process.argv[process.argv.indexOf("--model") + 1] : "model-af3-full-f32";
  const dump = await loadDump("af3-oracle-diffusion-f32.json");
  const { tensors } = await loadTensors(join(ROOT, model));
  const at = captures(dump, "dump_af3_trunk.py --blocks 48 --diffusion 1 --float32"
    + " (see the header of this file)");
  const T = (name) => {
    const tensor = tensors.get(`${HEAD}/${name}`);
    if (tensor === undefined) {
      throw new Error(`no tensor ${HEAD}/${name}; export with --include diffuser`);
    }
    return tensor.data;
  };

  const tokens = dump.tokens;
  const input = (name) => dump.inputs[name].data;
  const noiseLevel = SIGMA_DATA * 160;   // the schedule at t = 0

  const ours = diffusionConditioning({
    tokens,
    trunkSingle: at(`${EVO}/__call__:single`),
    trunkPair: at(`${EVO}/__call__:pair`),
    targetFeat: at(`${EVO}/__call__:target_feat`),
    noiseLevel,
    features: {
      residueIndex: input("residue_index"), tokenIndex: input("token_index"),
      asymId: input("asym_id"), entityId: input("entity_id"), symId: input("sym_id"),
    },
  }, {
    pairChannels: 128, seqChannels: 384, targetFeatWidth: 447, relativeWidth: 139,
    pairCondInitialNormScale: T("pair_cond_initial_norm/scale"),
    pairCondInitialProjection: T("pair_cond_initial_projection/weights"),
    pairTransitions: [transitionWeights(T, "pair_transition_0"),
                      transitionWeights(T, "pair_transition_1")],
    singleCondInitialNormScale: T("single_cond_initial_norm/scale"),
    singleCondInitialProjection: T("single_cond_initial_projection/weights"),
    noiseEmbeddingInitialNormScale: T("noise_embedding_initial_norm/scale"),
    noiseEmbeddingInitialProjection: T("noise_embedding_initial_projection/weights"),
    singleTransitions: [transitionWeights(T, "single_transition_0"),
                        transitionWeights(T, "single_transition_1")],
  });

  // ...AF3's own accumulation, rebuilt from its captured stages. Each term is
  // separate in the dump, so a disagreement says WHICH one rather than that the
  // total is off.
  const sum = (...arrays) => {
    const output = Float32Array.from(arrays[0]);
    for (let index = 1; index < arrays.length; index += 1) {
      for (let i = 0; i < output.length; i += 1) output[i] += arrays[index][i];
    }
    return output;
  };
  const broadcast = (row, count, width) => {
    const output = new Float32Array(count * width);
    for (let index = 0; index < count; index += 1) output.set(row, index * width);
    return output;
  };

  console.log(`${dump.model}, ${tokens} tokens, noise level ${noiseLevel}`
    + ` (schedule t=0), weights from ${model}/`);

  const embedded = noiseEmbedding(noiseLevel / SIGMA_DATA);
  report("noise", at(`${HEAD}/noise_embedding_initial_projection/__call__`),
         linear(layerNormSlow(embedded, 1, NOISE_CHANNELS,
                              T("noise_embedding_initial_norm/scale"), null),
                1, NOISE_CHANNELS, 384,
                T("noise_embedding_initial_projection/weights")));
  report("pair", sum(at(`${HEAD}/pair_cond_initial_projection/__call__`),
                     at(`${HEAD}/pair_transition_0ffw_transition2/__call__`),
                     at(`${HEAD}/pair_transition_1ffw_transition2/__call__`)),
         ours.pair);
  report("single", sum(at(`${HEAD}/single_cond_initial_projection/__call__`),
                       broadcast(at(`${HEAD}/noise_embedding_initial_projection/__call__`),
                                 tokens, 384),
                       at(`${HEAD}/single_transition_0ffw_transition2/__call__`),
                       at(`${HEAD}/single_transition_1ffw_transition2/__call__`)),
         ours.single);
}

await main();
