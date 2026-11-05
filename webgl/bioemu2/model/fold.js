/**
 * A whole BioEmu2 fold in WebGL2: features in, samples out.
 *
 *     s_inputs = input_embedder(features)                  (T, 455)
 *     z_init, s_init, m0 = sequence_embedder(s_inputs)
 *     four trunk passes, each 4 MSA layers and 52 pair blocks
 *     distogram(z)
 *     the EDM sampler, one denoiser call a step
 *
 * 🔴 THE SAMPLER IS `cpu/bioemu2/sampler.js`'s, NOT A SECOND COPY. The schedule,
 * the random rigid augmentation, the churn, the realignment and the Euler step
 * are the WGSL fold's own lines, moved so that both backends read them: a walk
 * re-derived for the second backend is how two backends come to anneal
 * differently, and nothing errors on a wrong constant - the walk simply returns
 * a plausible structure.
 *
 * 🔴 AND THE DENOISER WEIGHTS ARE 1.6 GiB OF f32 AND GO UP AFTER THE TRUNK'S
 * COME DOWN. `structure_module` is 424.6 M of the checkpoint's 511.7 M
 * parameters - 16 layers of two streams and a coupling, ~101 MiB a layer - where
 * the trunk's 52 pair blocks are 288 MiB and the rest is rounding. There is no
 * f16 in WebGL2 without an extension nobody can rely on, so this backend pays
 * f32 for all of it, and holding both stacks at once asks the device for 2.3
 * GiB it may not have. So the order here is deliberate: the trunk's weights are
 * released before the denoiser's are read, which costs nothing because no
 * sampler step touches the trunk.
 *
 * 🔴 AND THE DEFAULT STEP COUNT IS NOT THE MODEL'S. BioEmu2 samples at 200
 * steps; one WebGL2 denoiser call at 68 tokens is seconds, so a 200-step fold is
 * an hour. The caller passes `steps`, this defaults to 20, and a fold at 20 is
 * NOT a converged sample - it is a plumbing answer with a geometry floor. The
 * WGSL fold is the one to ask for a structure.
 */
import { BIOEMU2, BIOEMU2_SAMPLER } from "../../../shared/bioemu2/weights.js";
import { trunkBlockWeights } from "../../../shared/esmfold2/weights.js";
import { normalFrom } from "../../../shared/af3/random.js";
import { bioemu2Schedule, sampleMany, centre } from "../../../cpu/bioemu2/sampler.js";
import { uploadTriangleWeights, releaseTriangleWeights }
  from "../../af3/trunk/triangle.js";
import { uploadTransitionWeights, releaseTransitionWeights }
  from "../../af3/trunk/transition.js";
import { distogramLogits } from "../../esmfold2/distogram.js";
import { inputEmbedder } from "../embedder.js";
import { sequenceEmbedder } from "../sequence.js";
import { trunkPass } from "../trunk.js";
import { staticContext } from "../context.js";
import { bioemu2Denoise } from "../denoiser.js";

/** @typedef {import("../../runtime/storage.js").Tensor} Tensor */

const ATOM_HOST_NAMES = ["proj_atom_continuous/weight", "embed_element/weight",
  "embed_atom_name_chars_0/weight", "embed_atom_name_chars_1/weight",
  "embed_atom_name_chars_2/weight", "embed_atom_name_chars_3/weight",
  "embed_atompair_ref_pos/weight", "embed_atompair_ref_dist/weight",
  "embed_atompair_mask/weight", "c_to_p_trans_q_1/weight", "c_to_p_trans_k_1/weight",
  "p_mlp_1/weight", "p_mlp_3/weight", "p_mlp_5/weight"];

/** The six names an attention block reads, under whichever prefix carries it. */
const attentionNames = (prefix) => ({
  projQkv: `${prefix}/proj_qkv/weight`,
  projQBias: `${prefix}/proj_q_bias`,
  projG: `${prefix}/proj_g/weight`,
  normZ: `${prefix}/norm_z/weight`,
  projZ: `${prefix}/proj_z/weight`,
  projO: `${prefix}/proj_o/weight`,
});

/**
 * An uploader that remembers what it uploaded, so a stage's weights go back in
 * one call.
 *
 * 🔴 EVERY GROUP GETS ITS OWN, because the whole point of the order in
 * `foldBioemu2OnWebGl2` is that a stage's weights are released before the next
 * stage's are read. One shared list would hold all of them to the end.
 *
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {(name: string) => Promise<ArrayLike<number>>} read
 */
function group(execution, read) {
  const held = [];
  return {
    /** One tensor, by its full name. */
    up: async (name, label) => {
      // `uploadWeight`, not `upload`: with `--half-weights` a weight of 65,536
      // floats or more is held in float16, which halves what a big projection
      // streams. Off by default; see `Storage.halfWeights`.
      const tensor = execution.uploadWeight(
        Float32Array.from(await read(name)), label ?? name);
      held.push(tensor);
      return tensor;
    },
    /** A map of {key: name} uploaded under its keys. */
    tree: async function tree(names, label) {
      const out = {};
      for (const [key, name] of Object.entries(names)) {
        out[key] = await this.up(name, `${label}.${key}`);
      }
      return out;
    },
    release: () => {
      for (const tensor of held) execution.release(tensor);
      held.length = 0;
    },
    get count() { return held.length; },
  };
}

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {(name: string) => Promise<ArrayLike<number>>} read a bundle reader
 * @param {any} features `shared/bioemu2/featurise.js`'s
 * @param {{steps?: number, samples?: number, seed?: number, passes?: number,
 *   mode?: "structure" | "equilibrium", stepScale?: number, epsilon?: number,
 *   mixerBlocks?: number, onStage?: (name: string, detail?: any) => void,
 *   onStep?: (step: number, steps: number) => void}} [options]
 * @returns {Promise<{samples: Float32Array[], contact: Float32Array,
 *   distogram: Float32Array, tokens: number, atoms: number, calls: number,
 *   stages: [string, number][], mode: string, stepScale: number,
 *   peakMiB: number}>}
 */
export async function foldBioemu2OnWebGl2(execution, read, features, options = {}) {
  const f = features;
  const tokens = f.res_type.length;
  const atoms = f.atom_pad_mask.length;
  const epsilon = options.epsilon ?? 1e-5;
  const pairs = tokens * tokens;
  const sequences = f.msa.length / tokens;
  const mixerBlockCount = options.mixerBlocks ?? BIOEMU2.pairmixerBlocks;
  // 🔴 A STAGE'S TIME IS WHEN THE HOST GOT BACK TO HERE, CUMULATIVE FROM THE
  // START - and in this backend that means a stage ending in a readback absorbs
  // everything queued before it, which is the trap CLAUDE.md records for AF3's
  // own stage timers. Measured on the first 68-token fold: `trunk` read 293 ms
  // and `distogram` 8397, because the trunk's draws were still queued and the
  // read of `trunk.single` between them is what drained the queue. Read the
  // DIFFERENCES as a rough split and nothing finer.
  const stages = [];
  const clock = async (name, fn) => {
    const value = await fn();
    stages.push([name, Math.round(performance.now() - started)]);
    options.onStage?.(name);
    return value;
  };
  const started = performance.now();

  // --- the input embedder: s_inputs ---------------------------------------
  const embedder = group(execution, read);
  const sInputsHost = await clock("embed", async () => {
    const prefix = "input_embedder/atom_attention_encoder";
    const hostWeights = {};
    for (const name of [...ATOM_HOST_NAMES, "atom_to_token_trans_0/weight"]) {
      hostWeights[name] = Float32Array.from(await read(`${prefix}/${name}`));
    }
    const layers = [];
    for (let layer = 0; layer < BIOEMU2.inputLayers; layer += 1) {
      const base = `${prefix}/atom_encoder_layers_${layer}/transformer_layer`;
      const attn = `${base}/attention/attention_layer/pair_bias_attn`;
      const names = attentionNames(attn);
      layers.push({
        // 🔴 THE BIAS'S TWO WEIGHTS STAY ON THE HOST, because `windowBias` is
        // host arithmetic: a window of 128 keys a query over 16 channels.
        normZ: Float32Array.from(await read(names.normZ)),
        projZ: Float32Array.from(await read(names.projZ)),
        weights: {
          cond0Scale: await embedder.up(`${base}/cond_0/weight`, `ie.${layer}.cond0`),
          cond0Offset: await embedder.up(`${base}/cond_0/bias`, `ie.${layer}.cond0b`),
          cond2Weight: await embedder.up(`${base}/cond_2/weight`, `ie.${layer}.cond2`),
          cond2Bias: await embedder.up(`${base}/cond_2/bias`, `ie.${layer}.cond2b`),
          attention: {
            projQkv: await embedder.up(names.projQkv, `ie.${layer}.qkv`),
            projQBias: await embedder.up(names.projQBias, `ie.${layer}.qbias`),
            projG: await embedder.up(names.projG, `ie.${layer}.g`),
            projO: await embedder.up(names.projO, `ie.${layer}.o`),
          },
          transition: {
            fc1: await embedder.up(`${base}/transition/fc1/weight`, `ie.${layer}.fc1`),
            fc2: await embedder.up(`${base}/transition/fc2/weight`, `ie.${layer}.fc2`),
            fc3: await embedder.up(`${base}/transition/fc3/weight`, `ie.${layer}.fc3`),
          },
        },
      });
    }
    const out = inputEmbedder(execution, {
      features: f, hostWeights, layers, tokens, epsilon,
    });
    embedder.release();
    return out;
  });

  // --- the sequence embedder: z_init, s_init, m0 ---------------------------
  const embedded = await clock("sequence", async () => {
    const P = "sequence_embedder";
    const hostWeights = {};
    for (const name of ["s_init/weight", "z_init_1/weight", "z_init_2/weight",
      "method_conditioning/embedding/weight", "method_conditioning/pair_embedding/weight",
      "token_bonds/weight", "rel_pos/embed_rel_pos/weight", "rel_pos/embed_rel_token/weight",
      "rel_pos/embed_same_entity/weight", "rel_pos/embed_rel_chain/weight",
      "msa_module/msa_proj/weight", "msa_module/s_proj/weight"]) {
      hostWeights[name] = Float32Array.from(await read(`${P}/${name}`));
    }
    const contact = {};
    for (const name of ["encoder/weight", "encoder/bias", "encoding_unspecified",
      "encoding_unselected", "fourier_embedding/proj/weight",
      "fourier_embedding/proj/bias"]) {
      contact[name] = Float32Array.from(
        await read(`${P}/contact_conditioning/${name}`));
    }
    return sequenceEmbedder(execution, {
      features: f, sInputs: sInputsHost, tokens, sequences, hostWeights, contact,
    });
  });

  // --- the trunk: four passes ---------------------------------------------
  const trunkWeights = group(execution, read);
  const trunk = await clock("trunk", async () => {
    const P = "sequence_embedder";
    const msa = [];
    for (let layer = 0; layer < BIOEMU2.msaBlocks; layer += 1) {
      const L = `${P}/msa_module/layers/${layer}`;
      msa.push({
        pwa: await trunkWeights.tree({
          normMScale: `${L}/pair_weighted_averaging/norm_m/weight`,
          normMOffset: `${L}/pair_weighted_averaging/norm_m/bias`,
          projM: `${L}/pair_weighted_averaging/proj_m/weight`,
          projG: `${L}/pair_weighted_averaging/proj_g/weight`,
          normZScale: `${L}/pair_weighted_averaging/norm_z/weight`,
          normZOffset: `${L}/pair_weighted_averaging/norm_z/bias`,
          projZ: `${L}/pair_weighted_averaging/proj_z/weight`,
          projO: `${L}/pair_weighted_averaging/proj_o/weight`,
        }, `trunk.${layer}.pwa`),
        transition: await trunkWeights.tree({
          normScale: `${L}/msa_transition/norm/weight`,
          normOffset: `${L}/msa_transition/norm/bias`,
          fc1: `${L}/msa_transition/fc1/weight`,
          fc2: `${L}/msa_transition/fc2/weight`,
          fc3: `${L}/msa_transition/fc3/weight`,
        }, `trunk.${layer}.transition`),
        opm: await trunkWeights.tree({
          normScale: `${L}/outer_product_mean/norm/weight`,
          normOffset: `${L}/outer_product_mean/norm/bias`,
          projAb: `${L}/outer_product_mean/proj_ab/weight`,
          projO: `${L}/outer_product_mean/proj_o/weight`,
          projOBias: `${L}/outer_product_mean/proj_o/bias`,
        }, `trunk.${layer}.opm`),
      });
    }
    const weights = {
      ...(await trunkWeights.tree({
        sNormScale: `${P}/s_norm/weight`,
        sNormOffset: `${P}/s_norm/bias`,
        sRecycle: `${P}/s_recycle/weight`,
        zNormScale: `${P}/z_norm/weight`,
        zNormOffset: `${P}/z_norm/bias`,
        zRecycle: `${P}/z_recycle/weight`,
      }, "trunk.recycle")),
      msa,
    };
    const uploadBlock = async (layer, prefix) => {
      const block = await trunkBlockWeights(read, layer, prefix);
      return {
        triangleMultiplicationOutgoing: uploadTriangleWeights(
          execution, block.triangleMultiplicationOutgoing),
        triangleMultiplicationIncoming: uploadTriangleWeights(
          execution, block.triangleMultiplicationIncoming),
        pairTransition: uploadTransitionWeights(execution, block.pairTransition),
      };
    };
    const msaBlocks = [];
    for (let layer = 0; layer < BIOEMU2.msaBlocks; layer += 1) {
      msaBlocks.push(await uploadBlock(layer, "msa"));
    }
    const mixerBlocks = [];
    for (let layer = 0; layer < mixerBlockCount; layer += 1) {
      mixerBlocks.push(await uploadBlock(layer, "pairmixer"));
    }

    const zInit = embedded.zInit;
    const sInit = execution.upload(embedded.sInit, "bioemu2.s-init");
    const pairMask = execution.upload(
      new Float32Array(pairs).fill(1), "bioemu2.pair-mask");
    // 🔴 ZEROS, NOT AN ALLOCATION. The first trunk pass has no previous
    // activations and `trunkPass` normalises both of these, so a fresh
    // texture's contents ARE the answer here - and `allocate` does not promise
    // any (`texImage2D` with a null pointer leaves them undefined). Found by
    // `--nan-watch`, which named `layer-norm:68x384` inside the first pass; the
    // same fault in `webgl/af2/model/fold.js` moved that fold's pLDDT by 3.
    let previous = {
      pair: execution.zeros(pairs * BIOEMU2.pair, "bioemu2.z-prev"),
      single: execution.zeros(tokens * BIOEMU2.single, "bioemu2.s-prev"),
    };
    const passes = options.passes ?? BIOEMU2.trunkPasses;
    for (let pass = 0; pass < passes; pass += 1) {
      const out = trunkPass(execution, {
        zInit, sInit, m0: embedded.m0, zPrev: previous.pair, sPrev: previous.single,
        pairMask, tokens, sequences, valid: embedded.valid, weights,
        msaBlocks, mixerBlocks, epsilon,
      });
      execution.release(previous.pair);
      execution.release(previous.single);
      previous = out;
      options.onStage?.("trunk-pass", pass + 1);
    }
    for (const stack of [msaBlocks, mixerBlocks]) {
      for (const block of stack) {
        releaseTriangleWeights(execution, block.triangleMultiplicationOutgoing);
        releaseTriangleWeights(execution, block.triangleMultiplicationIncoming);
        releaseTransitionWeights(execution, block.pairTransition);
      }
    }
    trunkWeights.release();
    execution.release(pairMask);
    execution.release(sInit);
    execution.release(embedded.zInit);
    execution.release(embedded.m0);
    return previous;
  });
  const sTrunkHost = Float32Array.from(execution.read(trunk.single));

  // --- the distogram ------------------------------------------------------
  const head = group(execution, read);
  const { logits, contact } = await clock("distogram", async () => {
    const weights = await head.up("distogram_module/distogram/weight", "dg.weight");
    const bias = await head.up("distogram_module/distogram/bias", "dg.bias");
    const out = distogramLogits(execution, {
      pair: trunk.pair, tokens, channels: BIOEMU2.pair,
      bins: BIOEMU2.distogramBins, weights, bias, weightLayout: "out-major",
    });
    const values = Float32Array.from(execution.read(out));
    execution.release(out);
    head.release();
    // 🔴 BOLTZ'S BINS, 64 OVER 2-22 A, AND P(d < 8) IS THE FIRST 19. The edges
    // are not in this checkpoint's config; they are the reference's own, and the
    // WGSL head reads the same 19.
    const near = new Float32Array(pairs);
    for (let pair = 0; pair < pairs; pair += 1) {
      let largest = -Infinity;
      for (let bin = 0; bin < BIOEMU2.distogramBins; bin += 1) {
        largest = Math.max(largest, values[pair * BIOEMU2.distogramBins + bin]);
      }
      let total = 0;
      let close = 0;
      for (let bin = 0; bin < BIOEMU2.distogramBins; bin += 1) {
        const term = Math.exp(values[pair * BIOEMU2.distogramBins + bin] - largest);
        total += term;
        if (bin < 19) close += term;
      }
      near[pair] = close / total;
    }
    return { logits: values, contact: near };
  });

  // --- the denoiser's static context --------------------------------------
  const contextWeights = group(execution, read);
  const context = await clock("context", async () => {
    const P = "structure_module";
    const pc = `${P}/pairwise_conditioner`;
    const pa = `${P}/atom_feature_embedder`;
    const hostWeights = {};
    for (const name of [...ATOM_HOST_NAMES, "s_to_c_trans_0/weight",
      "s_to_c_trans_0/bias", "s_to_c_trans_1/weight"]) {
      hostWeights[name] = Float32Array.from(await read(`${pa}/${name}`));
    }
    // (cast because the uploader's `tree` answers a map and the context's
    // weights are a shape - the names are checked by the reads, not by TS)
    const weights = /** @type {any} */ ({
      relative: await contextWeights.tree({
        relPos: "rel_pos/embed_rel_pos/weight",
        relToken: "rel_pos/embed_rel_token/weight",
        sameEntity: "rel_pos/embed_same_entity/weight",
        relChain: "rel_pos/embed_rel_chain/weight",
      }, "ctx.rel"),
      ...(await contextWeights.tree({
        zToP0Scale: `${pa}/z_to_p_trans_0/weight`,
        zToP0Offset: `${pa}/z_to_p_trans_0/bias`,
        zToP1: `${pa}/z_to_p_trans_1/weight`,
        pairNormScale: `${pc}/dim_pairwise_init_proj_0/weight`,
        pairNormOffset: `${pc}/dim_pairwise_init_proj_0/bias`,
        pairProject: `${pc}/dim_pairwise_init_proj_1/weight`,
      }, "ctx")),
      transitions: [],
    });
    for (let index = 0; index < 2; index += 1) {
      weights.transitions.push(await contextWeights.tree({
        normScale: `${pc}/transitions_${index}/norm/weight`,
        normOffset: `${pc}/transitions_${index}/norm/bias`,
        fc1: `${pc}/transitions_${index}/fc1/weight`,
        fc2: `${pc}/transitions_${index}/fc2/weight`,
        fc3: `${pc}/transitions_${index}/fc3/weight`,
      }, `ctx.transition${index}`));
    }
    const out = staticContext(execution, {
      features: f, tokens, atoms, zTrunk: trunk.pair, sTrunk: sTrunkHost,
      hostWeights, weights, epsilon,
    });
    contextWeights.release();
    execution.release(trunk.pair);
    return out;
  });

  // --- the denoiser's weights ---------------------------------------------
  const denoiser = group(execution, read);
  const layers = BIOEMU2.denoiserLayers;
  const weights = await clock("denoiser-weights", async () => {
    const P = "structure_module";
    const sc = `${P}/single_conditioner`;
    const streamTree = async (base, attn, layer) => ({
      cond0Scale: await denoiser.up(`${base}/cond_0/weight`, `dn.${layer}.cond0`),
      cond0Offset: await denoiser.up(`${base}/cond_0/bias`, `dn.${layer}.cond0b`),
      cond2Weight: await denoiser.up(`${base}/cond_2/weight`, `dn.${layer}.cond2`),
      cond2Bias: await denoiser.up(`${base}/cond_2/bias`, `dn.${layer}.cond2b`),
      attention: await denoiser.tree(attentionNames(attn), `dn.${layer}.attn`),
      transition: await denoiser.tree({
        fc1: `${base}/transition/fc1/weight`,
        fc2: `${base}/transition/fc2/weight`,
        fc3: `${base}/transition/fc3/weight`,
      }, `dn.${layer}.transition`),
    });
    const singleConditioner = {
      ...(await denoiser.tree({
        normScale: `${sc}/norm_single/weight`,
        normOffset: `${sc}/norm_single/bias`,
        singleEmbedWeight: `${sc}/single_embed/weight`,
        singleEmbedBias: `${sc}/single_embed/bias`,
        fourierNormScale: `${sc}/norm_fourier/weight`,
        fourierNormOffset: `${sc}/norm_fourier/bias`,
        fourierToSingle: `${sc}/fourier_to_single/weight`,
      }, "dn.single")),
      transitions: [],
    };
    for (let index = 0; index < 2; index += 1) {
      singleConditioner.transitions.push(await denoiser.tree({
        normScale: `${sc}/transitions_${index}/norm/weight`,
        normOffset: `${sc}/transitions_${index}/norm/bias`,
        fc1: `${sc}/transitions_${index}/fc1/weight`,
        fc2: `${sc}/transitions_${index}/fc2/weight`,
        fc3: `${sc}/transitions_${index}/fc3/weight`,
      }, `dn.single.transition${index}`));
    }
    const atomBlocks = [];
    const tokenBlocks = [];
    const coupling = [];
    for (let layer = 0; layer < layers; layer += 1) {
      const L = `${P}/layers/${layer}`;
      const atomBase = `${L}/atom_block/transformer_layer`;
      atomBlocks.push(await streamTree(atomBase,
        `${atomBase}/attention/attention_layer/pair_bias_attn`, `atom${layer}`));
      tokenBlocks.push(await streamTree(`${L}/token_block`,
        `${L}/token_block/attention/pair_bias_attn`, `token${layer}`));
      coupling.push({
        qCoupling: await denoiser.tree({
          cond0Scale: `${L}/coupling/q_coupling_cond_0/weight`,
          cond0Offset: `${L}/coupling/q_coupling_cond_0/bias`,
          cond2Weight: `${L}/coupling/q_coupling_cond_2/weight`,
          cond2Bias: `${L}/coupling/q_coupling_cond_2/bias`,
        }, `dn.${layer}.qcond`),
        sCoupling: await denoiser.tree({
          cond0Scale: `${L}/coupling/s_coupling_cond_0/weight`,
          cond0Offset: `${L}/coupling/s_coupling_cond_0/bias`,
          cond2Weight: `${L}/coupling/s_coupling_cond_2/weight`,
          cond2Bias: `${L}/coupling/s_coupling_cond_2/bias`,
        }, `dn.${layer}.scond`),
        sToQ: await denoiser.tree({
          fc1: `${L}/coupling/s_to_q_mixer/fc1/weight`,
          fc2: `${L}/coupling/s_to_q_mixer/fc2/weight`,
          fc3: `${L}/coupling/s_to_q_mixer/fc3/weight`,
        }, `dn.${layer}.stoq`),
        qToS: await denoiser.tree({
          fc1: `${L}/coupling/q_to_s_mixer/fc1/weight`,
          fc2: `${L}/coupling/q_to_s_mixer/fc2/weight`,
          fc3: `${L}/coupling/q_to_s_mixer/fc3/weight`,
        }, `dn.${layer}.qtos`),
      });
    }
    return {
      singleConditioner,
      ...(await denoiser.tree({
        rToQ: `${P}/atom_feature_embedder/r_to_q_trans/weight`,
        sTrunkToA0Scale: `${P}/s_trunk_to_a_0/weight`,
        sTrunkToA0Offset: `${P}/s_trunk_to_a_0/bias`,
        sTrunkToA1: `${P}/s_trunk_to_a_1/weight`,
        rTokenToA: `${P}/r_token_to_a/weight`,
      }, "dn")),
      atomBlocks,
      tokenBlocks,
      coupling,
      readout: await denoiser.tree({
        qToHidden: `${P}/readout/q_to_hidden/weight`,
        sToHidden: `${P}/readout/s_to_hidden/weight`,
        outNormScale: `${P}/readout/to_out_0/weight`,
        toOut: `${P}/readout/to_out_1/weight`,
      }, "dn.readout"),
    };
  });
  const hostFourier = {
    weight: Float32Array.from(
      await read("structure_module/single_conditioner/fourier_embed/proj/weight")),
    bias: Float32Array.from(
      await read("structure_module/single_conditioner/fourier_embed/proj/bias")),
  };

  // --- the sampler --------------------------------------------------------
  const centreAtom = f.token_center_atom_index;
  const deviceFeatures = {
    atomMask: execution.upload(Float32Array.from(f.atom_mask), "bioemu2.atom-mask"),
    atomPad: execution.upload(Float32Array.from(f.atom_pad_mask), "bioemu2.atom-pad"),
    tokenPad: execution.upload(Float32Array.from(f.token_pad_mask), "bioemu2.token-pad"),
    centreAtom: execution.upload(Float32Array.from(centreAtom), "bioemu2.centre"),
    // 🔴 A PADDED TOKEN'S CENTRE AND A PADDED ATOM'S TOKEN ARE -1, so the gather
    // gives zeros - which is what masking them afterwards would give.
    centreAtomOrPad: execution.upload(
      Float32Array.from(centreAtom, (value, token) => (f.token_pad_mask[token] ? value : -1)),
      "bioemu2.centre-or-pad"),
    atomToTokenOrPad: execution.upload(
      Float32Array.from(f.token_idx_at_atom,
        (value, atom) => (f.atom_pad_mask[atom] ? value : -1)),
      "bioemu2.atom-token-or-pad"),
  };
  const sInputs = execution.upload(sInputsHost, "bioemu2.s-inputs");
  const sTrunk = execution.upload(sTrunkHost, "bioemu2.s-trunk");
  const dims = {
    atoms, tokens, layers,
    atomChannels: BIOEMU2.atom, tokenChannels: BIOEMU2.token,
    atomHeads: BIOEMU2.atomHeads, tokenHeads: BIOEMU2.tokenHeads,
    sigmaData: BIOEMU2.sigmaData,
  };
  let calls = 0;
  const denoise = async (stacked, sigma) => {
    calls += 1;
    const x = execution.upload(Float32Array.from(stacked), "bioemu2.x");
    const out = bioemu2Denoise(execution, {
      x, sigma, features: deviceFeatures, context, sInputs, sTrunk,
      weights, dims, hostFourier, epsilon,
    });
    const values = Float32Array.from(execution.read(out));
    execution.release(out);
    execution.release(x);
    return values;
  };

  const mode = options.mode ?? (f.method_feature[0] === 4 ? "equilibrium" : "structure");
  const stepScale = options.stepScale ?? BIOEMU2_SAMPLER.stepScale[mode];
  const schedule = bioemu2Schedule(options.steps ?? 20);
  const normal = normalFrom(options.seed ?? 1);
  const mask = Int32Array.from(f.atom_mask);
  const samples = [];
  await clock("sample", async () => {
    for (let index = 0; index < (options.samples ?? 1); index += 1) {
      // 🔴 ONE SAMPLE A CALL. The WGSL denoiser compiles for a batch of S and
      // runs them sample-major in one call; this one takes a single frame, so an
      // ensemble is S walks of the same compiled programs - the same arithmetic,
      // S times the draws, in the same order, because `sampleMany` consumes the
      // generator step by step either way.
      const walked = await sampleMany({
        denoise, setCompanion: () => {}, mask, N: atoms, S: 1, schedule,
        stepScale, normal, companions: null,
        onStep: (step, total) => options.onStep?.(step, total),
      });
      samples.push(centre(walked[0], mask, atoms));
      options.onStage?.("sample", index + 1);
    }
  });

  denoiser.release();
  for (const tensor of Object.values(deviceFeatures)) execution.release(tensor);
  execution.release(sInputs);
  execution.release(sTrunk);
  execution.release(trunk.single);
  for (const tensor of Object.values(context)) execution.release(tensor);

  return {
    samples, contact, distogram: logits, tokens, atoms, calls, stages, mode,
    stepScale, peakMiB: Math.round(execution.stats().peakBytes / 1048576),
  };
}
