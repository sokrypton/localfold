/**
 * BioEmu2's denoiser in WebGL2 against `cpu/bioemu2/denoiser.js`.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-bioemu2-denoiser.js [--falsify]
 *
 * 🔴 EVERY WEIGHT IN THIS MODEL IS OUT-MAJOR, which is the opposite of every
 * other model in this tree: the reference's `linear` reads `weight[o * inF + k]`
 * - torch's own `[out][in]`, untransposed at export - where AF3, AF2 and
 * ESMFold2 all arrive `[in][out]`. Nothing in a tensor's shape says which, so
 * this gate is the only thing standing between the port and the transpose of
 * every matrix in it.
 *
 * 🔴 AND sigma REACHES FOUR PLACES, so the gate runs THREE noise levels: `c_in`
 * on the input, the Fourier embedding in the conditioning, and `c_skip` and
 * `c_out` on the output. A port that scaled only the input returns a finite
 * structure at every level and the right one at none - and at sigma = sigma_data
 * the two output coefficients are equal, which is exactly where that mistake
 * hides.
 *
 * 🔴 AND THE TWO STREAMS COUPLE THROUGH THE CENTRE ATOM IN BOTH DIRECTIONS,
 * each direction a THREE-way slice of its own conditioning projection - alpha,
 * beta and a gate - so the layout gives the tokens DIFFERENT atom counts and a
 * centre atom that is not the token's first.
 *
 * 🔴 AND THE ATOM WINDOW IS THE VENDOR'S: query block `win` attends to atoms
 * `win * 32 - 48 ... + 128`. At 64 atoms that is two windows whose key lists
 * overlap and run off both ends, which is the case a sliding window gets wrong.
 */
import { bioemu2Denoise as cpuDenoise } from "../../cpu/bioemu2/denoiser.js";
import {
  bioemu2Denoise, WINDOW_KEYS, WINDOW_QUERIES,
} from "../../webgl/bioemu2/denoiser.js";
import { Execution } from "../../webgl/runtime/execution.js";
import { createWebGL2Context } from "../../webgl/index.js";
import { generator } from "./af2-synthetic-block.js";
import { relativeRms } from "./relative-rms.js";

const flag = (args, name) => (args ?? []).includes(`--${name}`);
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  const falsify = flag(args, "falsify");
  const bar = Number(option(args, "bar", "5e-6"));
  const windows = Number(option(args, "windows", "2"));
  const atoms = windows * WINDOW_QUERIES;
  const tokens = Number(option(args, "tokens", "8"));
  // 🔴 THE ATOM WIDTH AND BOTH HEAD COUNTS ARE THE REFERENCE'S OWN CONSTANTS,
  // not this gate's to choose. `denoiseOnTape` writes `tape.leaf(ctx.q0, [N,
  // 128])` and passes `heads: 4` for the atom stream and `heads: 16` for the
  // token one - so a narrower atom stream makes the reference claim 128 channels
  // for data that has fewer, and the first run of this gate died inside its
  // `concat` with "offset is out of bounds". The TOKEN width is free, because
  // every layer reads `x.shape[1]`; it is 64 here so that 16 heads leave a head
  // dimension that is a multiple of four.
  const atomChannels = Number(option(args, "atom-channels", "128"));
  const tokenChannels = Number(option(args, "token-channels", "64"));
  const atomHeads = Number(option(args, "atom-heads", "4"));
  const tokenHeads = Number(option(args, "token-heads", "16"));
  const atomPair = Number(option(args, "atom-pair", "8"));
  const tokenPair = Number(option(args, "token-pair", "8"));
  const layers = Number(option(args, "layers", "2"));
  const hidden = Number(option(args, "hidden", "32"));
  const fourierChannels = 16;
  const trunkChannels = Number(option(args, "trunk-channels", "12"));
  const inputChannels = Number(option(args, "input-channels", "20"));

  // 🔴 AND THE TWO HEAD COUNTS ARE HARDCODED IN THE REFERENCE, so they are not
  // the gate's to sweep: `denoiseOnTape` passes `heads: 4` and `heads: 16`
  // whatever a caller says. Asked for anything else, the two sides compute
  // different arithmetic and every arm reads NaN - which looks like a kernel
  // fault and is a gate asking two questions at once.
  if (atomHeads !== 4 || tokenHeads !== 16) {
    throw new RangeError(
      `bioemu2 denoiser gate: the reference hardcodes 4 atom heads and 16 token`
      + ` heads, so ${atomHeads} and ${tokenHeads} would compare two different`
      + " models; sweep the CHANNELS instead");
  }
  if (tokenChannels % (tokenHeads * 4) !== 0) {
    throw new RangeError(
      `bioemu2 denoiser gate: ${tokenChannels} channels over ${tokenHeads} heads`
      + " leaves a head dimension that is not a multiple of four, which this"
      + " port's attention kernel owns four channels of");
  }

  const random = generator(1618033);
  const noise = (n, spread, centre = 0) => {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i += 1) out[i] = centre + (random() * 2 - 1) * spread;
    return out;
  };

  // 🔴 TOKENS WITH DIFFERENT ATOM COUNTS AND A CENTRE THAT IS NOT THE FIRST.
  const atomToToken = new Int32Array(atoms);
  const centreAtom = new Int32Array(tokens);
  {
    let atom = 0;
    for (let token = 0; token < tokens; token += 1) {
      const count = Math.max(1, Math.floor(atoms / tokens) + (token % 3 === 0 ? 1 : -1));
      const from = atom;
      for (let i = 0; i < count && atom < atoms; i += 1, atom += 1) {
        atomToToken[atom] = token;
      }
      centreAtom[token] = Math.min(from + 1, atoms - 1);
    }
    // ...any tail belongs to the last token.
    for (; atom < atoms; atom += 1) atomToToken[atom] = tokens - 1;
  }
  const atomPad = new Float32Array(atoms).fill(1);
  for (let atom = 0; atom < atoms; atom += 1) if (atom % 11 === 9) atomPad[atom] = 0;
  const atomMask = Float32Array.from(atomPad);
  const tokenPad = new Float32Array(tokens).fill(1);
  if (tokens > 2) tokenPad[tokens - 1] = 0;

  const x = noise(atoms * 3, 10);
  const sInputs = noise(tokens * inputChannels, 1);
  const sTrunk = noise(tokens * trunkChannels, 1);
  const contextZ = noise(tokens * tokens * tokenPair, 1);
  const contextQ0 = noise(atoms * atomChannels, 1);
  const contextP = noise(windows * WINDOW_QUERIES * WINDOW_KEYS * atomPair, 1);

  // The weights, in BOTH shapes: the reference's flat `{data, shape}` map and
  // this port's tree. One source of numbers, two spellings.
  const flat = {};
  const put = (name, values, shape) => {
    flat[name] = { data: values, shape };
    return values;
  };
  const stackOf = (count, size) => {
    const values = noise(count * size, 1 / Math.sqrt(size / count));
    return { values, slice: (layer) => values.subarray(layer * size, (layer + 1) * size) };
  };
  const attentionNames = (prefix) => ({
    projQkv: `${prefix}/proj_qkv/weight`,
    projQBias: `${prefix}/proj_q_bias`,
    projG: `${prefix}/proj_g/weight`,
    normZ: `${prefix}/norm_z/weight`,
    projZ: `${prefix}/proj_z/weight`,
    projO: `${prefix}/proj_o/weight`,
  });
  const stacks = {};
  const stacked = (name, size) => {
    const stack = stackOf(layers, size);
    put(name, stack.values, [layers, size]);
    stacks[name] = stack;
    return stack;
  };

  // The single conditioner.
  put("single_conditioner/norm_single/weight",
    noise(trunkChannels + inputChannels, 0.2, 1), [trunkChannels + inputChannels]);
  put("single_conditioner/norm_single/bias",
    noise(trunkChannels + inputChannels, 0.1), [trunkChannels + inputChannels]);
  put("single_conditioner/single_embed/weight",
    noise((trunkChannels + inputChannels) * tokenChannels,
      1 / Math.sqrt(trunkChannels + inputChannels)),
    [tokenChannels, trunkChannels + inputChannels]);
  put("single_conditioner/single_embed/bias", noise(tokenChannels, 0.1), [tokenChannels]);
  put("single_conditioner/fourier_embed/proj/weight", noise(fourierChannels, 1),
    [fourierChannels, 1]);
  put("single_conditioner/fourier_embed/proj/bias", noise(fourierChannels, 0.5),
    [fourierChannels]);
  put("single_conditioner/norm_fourier/weight", noise(fourierChannels, 0.2, 1),
    [fourierChannels]);
  put("single_conditioner/norm_fourier/bias", noise(fourierChannels, 0.1),
    [fourierChannels]);
  put("single_conditioner/fourier_to_single/weight",
    noise(fourierChannels * tokenChannels, 1 / Math.sqrt(fourierChannels)),
    [tokenChannels, fourierChannels]);
  const conditionerTransitions = [];
  for (const index of [0, 1]) {
    const prefix = `single_conditioner/transitions_${index}`;
    put(`${prefix}/norm/weight`, noise(tokenChannels, 0.2, 1), [tokenChannels]);
    put(`${prefix}/norm/bias`, noise(tokenChannels, 0.1), [tokenChannels]);
    put(`${prefix}/fc1/weight`, noise(tokenChannels * hidden, 1 / Math.sqrt(tokenChannels)),
      [hidden, tokenChannels]);
    put(`${prefix}/fc2/weight`, noise(tokenChannels * hidden, 1 / Math.sqrt(tokenChannels)),
      [hidden, tokenChannels]);
    put(`${prefix}/fc3/weight`, noise(hidden * tokenChannels, 1 / Math.sqrt(hidden)),
      [tokenChannels, hidden]);
    conditionerTransitions.push(prefix);
  }

  put("atom_feature_embedder/r_to_q_trans/weight",
    noise(3 * atomChannels, 0.3), [atomChannels, 3]);
  put("s_trunk_to_a_0/weight", noise(trunkChannels, 0.2, 1), [trunkChannels]);
  put("s_trunk_to_a_0/bias", noise(trunkChannels, 0.1), [trunkChannels]);
  put("s_trunk_to_a_1/weight",
    noise(trunkChannels * tokenChannels, 1 / Math.sqrt(trunkChannels)),
    [tokenChannels, trunkChannels]);
  put("r_token_to_a/weight", noise(3 * tokenChannels, 0.3), [tokenChannels, 3]);

  // The two streams' layer stacks.
  const streamStacks = (prefix, channels, heads, pairChannels) => {
    stacked(`${prefix}/cond_0/weight`, tokenChannels);
    stacked(`${prefix}/cond_0/bias`, tokenChannels);
    stacked(`${prefix}/cond_2/weight`, tokenChannels * 6 * channels);
    stacked(`${prefix}/cond_2/bias`, 6 * channels);
    const attn = prefix.includes("atom")
      ? `${prefix}/attention/attention_layer/pair_bias_attn`
      : `${prefix}/attention/pair_bias_attn`;
    stacked(`${attn}/proj_qkv/weight`, channels * 3 * channels);
    stacked(`${attn}/proj_q_bias`, channels);
    stacked(`${attn}/proj_g/weight`, channels * channels);
    stacked(`${attn}/norm_z/weight`, pairChannels);
    stacked(`${attn}/proj_z/weight`, pairChannels * heads);
    stacked(`${attn}/proj_o/weight`, channels * channels);
    stacked(`${prefix}/transition/fc1/weight`, channels * hidden);
    stacked(`${prefix}/transition/fc2/weight`, channels * hidden);
    stacked(`${prefix}/transition/fc3/weight`, hidden * channels);
    return attn;
  };
  const atomAttn = streamStacks("layers/atom_block/transformer_layer",
    atomChannels, atomHeads, atomPair);
  const tokenAttn = streamStacks("layers/token_block", tokenChannels, tokenHeads, tokenPair);

  // The coupling.
  const width = tokenChannels + atomChannels;
  stacked("layers/coupling/q_coupling_cond_0/weight", tokenChannels);
  stacked("layers/coupling/q_coupling_cond_0/bias", tokenChannels);
  stacked("layers/coupling/q_coupling_cond_2/weight",
    tokenChannels * (2 * width + atomChannels));
  stacked("layers/coupling/q_coupling_cond_2/bias", 2 * width + atomChannels);
  stacked("layers/coupling/s_coupling_cond_0/weight", tokenChannels);
  stacked("layers/coupling/s_coupling_cond_0/bias", tokenChannels);
  stacked("layers/coupling/s_coupling_cond_2/weight",
    tokenChannels * (2 * width + tokenChannels));
  stacked("layers/coupling/s_coupling_cond_2/bias", 2 * width + tokenChannels);
  // 🔴 THE MIXERS ARE NOT SQUARE: each reads `[a | q]` - both widths added - and
  // writes the width of the stream it is mixing INTO. `fc3` is what says so.
  stacked("layers/coupling/s_to_q_mixer/fc1/weight", width * hidden);
  stacked("layers/coupling/s_to_q_mixer/fc2/weight", width * hidden);
  stacked("layers/coupling/s_to_q_mixer/fc3/weight", hidden * atomChannels);
  stacked("layers/coupling/q_to_s_mixer/fc1/weight", width * hidden);
  stacked("layers/coupling/q_to_s_mixer/fc2/weight", width * hidden);
  stacked("layers/coupling/q_to_s_mixer/fc3/weight", hidden * tokenChannels);

  put("readout/q_to_hidden/weight",
    noise(atomChannels * hidden, 1 / Math.sqrt(atomChannels)), [hidden, atomChannels]);
  put("readout/s_to_hidden/weight",
    noise(tokenChannels * hidden, 1 / Math.sqrt(tokenChannels)), [hidden, tokenChannels]);
  put("readout/to_out_0/weight", noise(hidden, 0.2, 1), [hidden]);
  put("readout/to_out_1/weight", noise(hidden * 3, 0.3 / Math.sqrt(hidden)), [3, hidden]);

  const gl = createWebGL2Context();
  const execution = new Execution(gl);
  const held = [];
  const upload = (values, label) => {
    const tensor = execution.upload(Float32Array.from(values), label);
    held.push(tensor);
    return tensor;
  };
  const layerTensor = (name, layer) => upload(stacks[name].slice(layer), `w.${name}.${layer}`);
  const attentionTree = (prefix, layer) => {
    const names = attentionNames(prefix);
    return Object.fromEntries(Object.entries(names)
      .map(([key, name]) => [key, layerTensor(name, layer)]));
  };
  const streamTree = (prefix, attn, layer) => ({
    cond0Scale: layerTensor(`${prefix}/cond_0/weight`, layer),
    cond0Offset: layerTensor(`${prefix}/cond_0/bias`, layer),
    cond2Weight: layerTensor(`${prefix}/cond_2/weight`, layer),
    cond2Bias: layerTensor(`${prefix}/cond_2/bias`, layer),
    attention: attentionTree(attn, layer),
    transition: {
      fc1: layerTensor(`${prefix}/transition/fc1/weight`, layer),
      fc2: layerTensor(`${prefix}/transition/fc2/weight`, layer),
      fc3: layerTensor(`${prefix}/transition/fc3/weight`, layer),
    },
  });
  const weights = {
    singleConditioner: {
      normScale: upload(flat["single_conditioner/norm_single/weight"].data, "w.norm-single"),
      normOffset: upload(flat["single_conditioner/norm_single/bias"].data, "w.norm-single-b"),
      singleEmbedWeight: upload(
        flat["single_conditioner/single_embed/weight"].data, "w.single-embed"),
      singleEmbedBias: upload(
        flat["single_conditioner/single_embed/bias"].data, "w.single-embed-b"),
      fourierNormScale: upload(
        flat["single_conditioner/norm_fourier/weight"].data, "w.norm-fourier"),
      fourierNormOffset: upload(
        flat["single_conditioner/norm_fourier/bias"].data, "w.norm-fourier-b"),
      fourierToSingle: upload(
        flat["single_conditioner/fourier_to_single/weight"].data, "w.fourier-to-single"),
      transitions: conditionerTransitions.map((prefix) => ({
        normScale: upload(flat[`${prefix}/norm/weight`].data, `w.${prefix}.norm`),
        normOffset: upload(flat[`${prefix}/norm/bias`].data, `w.${prefix}.norm-b`),
        fc1: upload(flat[`${prefix}/fc1/weight`].data, `w.${prefix}.fc1`),
        fc2: upload(flat[`${prefix}/fc2/weight`].data, `w.${prefix}.fc2`),
        fc3: upload(flat[`${prefix}/fc3/weight`].data, `w.${prefix}.fc3`),
      })),
    },
    rToQ: upload(flat["atom_feature_embedder/r_to_q_trans/weight"].data, "w.r-to-q"),
    sTrunkToA0Scale: upload(flat["s_trunk_to_a_0/weight"].data, "w.trunk-a0"),
    sTrunkToA0Offset: upload(flat["s_trunk_to_a_0/bias"].data, "w.trunk-a0-b"),
    sTrunkToA1: upload(flat["s_trunk_to_a_1/weight"].data, "w.trunk-a1"),
    rTokenToA: upload(flat["r_token_to_a/weight"].data, "w.r-token-a"),
    atomBlocks: Array.from({ length: layers },
      (_, layer) => streamTree("layers/atom_block/transformer_layer", atomAttn, layer)),
    tokenBlocks: Array.from({ length: layers },
      (_, layer) => streamTree("layers/token_block", tokenAttn, layer)),
    coupling: Array.from({ length: layers }, (_, layer) => ({
      qCoupling: {
        cond0Scale: layerTensor("layers/coupling/q_coupling_cond_0/weight", layer),
        cond0Offset: layerTensor("layers/coupling/q_coupling_cond_0/bias", layer),
        cond2Weight: layerTensor("layers/coupling/q_coupling_cond_2/weight", layer),
        cond2Bias: layerTensor("layers/coupling/q_coupling_cond_2/bias", layer),
      },
      sCoupling: {
        cond0Scale: layerTensor("layers/coupling/s_coupling_cond_0/weight", layer),
        cond0Offset: layerTensor("layers/coupling/s_coupling_cond_0/bias", layer),
        cond2Weight: layerTensor("layers/coupling/s_coupling_cond_2/weight", layer),
        cond2Bias: layerTensor("layers/coupling/s_coupling_cond_2/bias", layer),
      },
      sToQ: {
        fc1: layerTensor("layers/coupling/s_to_q_mixer/fc1/weight", layer),
        fc2: layerTensor("layers/coupling/s_to_q_mixer/fc2/weight", layer),
        fc3: layerTensor("layers/coupling/s_to_q_mixer/fc3/weight", layer),
      },
      qToS: {
        fc1: layerTensor("layers/coupling/q_to_s_mixer/fc1/weight", layer),
        fc2: layerTensor("layers/coupling/q_to_s_mixer/fc2/weight", layer),
        fc3: layerTensor("layers/coupling/q_to_s_mixer/fc3/weight", layer),
      },
    })),
    readout: {
      qToHidden: upload(flat["readout/q_to_hidden/weight"].data, "w.q-to-hidden"),
      sToHidden: upload(flat["readout/s_to_hidden/weight"].data, "w.s-to-hidden"),
      outNormScale: upload(flat["readout/to_out_0/weight"].data, "w.to-out-0"),
      toOut: upload(flat["readout/to_out_1/weight"].data, "w.to-out-1"),
    },
  };

  // 🔴 THE SKEW GOES TO THE DEVICE ALONE, and the coordinates are where it has
  // to land: they reach `c_in`, both streams' initial states and the blend.
  const skew = (values, period) => Float32Array.from(values,
    (v, i) => (i % period < period / 2 ? v * 1.01 + 0.01 : v));
  const features = {
    atomMask: upload(atomMask, "bioemu2.atom-mask"),
    atomPad: upload(atomPad, "bioemu2.atom-pad"),
    tokenPad: upload(tokenPad, "bioemu2.token-pad"),
    centreAtom: upload(Float32Array.from(centreAtom), "bioemu2.centre"),
    // A padded token's centre, and a padded atom's token, are -1 so the gather
    // gives zeros - which is what masking them after the fact would give.
    centreAtomOrPad: upload(
      Float32Array.from(centreAtom, (value, token) => (tokenPad[token] ? value : -1)),
      "bioemu2.centre-or-pad"),
    atomToTokenOrPad: upload(
      Float32Array.from(atomToToken, (value, atom) => (atomPad[atom] ? value : -1)),
      "bioemu2.atom-token-or-pad"),
  };
  const context = {
    z: upload(contextZ, "bioemu2.ctx-z"),
    q0: upload(contextQ0, "bioemu2.ctx-q0"),
    p: upload(contextP, "bioemu2.ctx-p"),
  };
  const inputs = {
    x: upload(falsify ? skew(x, 3) : x, "bioemu2.x"),
    sInputs: upload(sInputs, "bioemu2.s-inputs"),
    sTrunk: upload(sTrunk, "bioemu2.s-trunk"),
  };
  const dims = {
    atoms, tokens, layers, atomChannels, tokenChannels,
    atomHeads, tokenHeads, sigmaData: 16,
  };
  const hostFourier = {
    weight: flat["single_conditioner/fourier_embed/proj/weight"].data,
    bias: flat["single_conditioner/fourier_embed/proj/bias"].data,
  };

  const arms = [];
  const failures = [];
  const separations = {};
  const hostFeatures = {
    atom_mask: atomMask, atom_pad_mask: atomPad, token_pad_mask: tokenPad,
    token_idx_at_atom: atomToToken, token_center_atom_index: centreAtom,
    is_dynamics_conditioned: false,
  };
  const hostContext = { z: contextZ, q0: contextQ0, p: contextP };
  const answers = {};
  for (const sigma of [160, 16, 0.05]) {
    const out = bioemu2Denoise(execution, {
      ...inputs, sigma, features, context, weights, dims, hostFourier,
    });
    const mine = Float32Array.from(execution.read(out));
    execution.release(out);
    answers[sigma] = mine;
    // 🔴 THE REFERENCE'S dims ARE `N` AND `T`, NOT `atoms` AND `tokens`. Handing
    // it this port's spelling makes both undefined, every derived count NaN, and
    // the failure is "offset is out of bounds" inside its own `concat` - which
    // reads exactly like a wrong weight shape and is not.
    const theirs = cpuDenoise(flat, hostFeatures, hostContext, sInputs, sTrunk,
      x, sigma, { N: atoms, T: tokens });
    const relative = relativeRms(mine, theirs, `sigma ${sigma}`);
    const ok = relative <= bar;
    arms.push({
      arm: `D(x, ${sigma})`, relativeRms: Number(relative.toPrecision(4)), bar, ok,
      cSkip: Number((256 / (256 + sigma * sigma)).toPrecision(4)),
      cOut: Number(((16 * sigma) / Math.sqrt(256 + sigma * sigma)).toPrecision(4)),
    });
    if (!ok) {
      failures.push(`sigma ${sigma}: relRMS ${relative.toExponential(3)} over ${bar}`);
    }
  }
  // 🔴 AND THE ANSWER MUST MOVE WITH THE NOISE LEVEL, or sigma reached only one
  // of its four places.
  separations["sigma 160 against 16"] = Number(
    relativeRms(answers[160], answers[16], "levels").toPrecision(3));
  separations["sigma 0.05 against 16"] = Number(
    relativeRms(answers[0.05], answers[16], "levels").toPrecision(3));
  for (const [name, value] of Object.entries(separations)) {
    if (!(value > 1e-3)) failures.push(`${name} is INERT: ${value}`);
  }

  for (const tensor of held) execution.release(tensor);
  if (execution.stats().live !== 0) {
    failures.push(`${execution.stats().live} tensors left live`);
  }
  const red = arms.filter((row) => !row.ok).length;
  return {
    shape: {
      atoms, tokens, windows, atomChannels, tokenChannels, atomHeads, tokenHeads,
      atomPair, tokenPair, layers, hidden, trunkChannels, inputChannels,
      windowQueries: WINDOW_QUERIES, windowKeys: WINDOW_KEYS,
      paddedAtoms: atoms - atomMask.reduce((a, b) => a + b, 0),
      paddedTokens: tokens - tokenPad.reduce((a, b) => a + b, 0),
    },
    weightLayout: "out-major, which is the opposite of every other model here",
    bar,
    falsify,
    arms,
    separations,
    memory: execution.stats(),
    verdict: falsify
      ? (red === arms.length
        ? `ok: all ${arms.length} arms go red on a 1% perturbation`
        : `VACUOUS: only ${red} of ${arms.length} arms noticed it`)
      : (failures.length === 0
        ? `ok: ${arms.length} noise levels against the f64 reference`
        : `FAILED: ${failures.join("; ")}`),
  };
}
