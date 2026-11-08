/**
 * BioEmu2's input embedder in WebGL2 against the WGSL one, in the same page.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-bioemu2-embedder.js [--falsify]
 *
 * 🔴 THE REFERENCE IS THE OTHER BACKEND, BECAUSE THIS STAGE HAS NO f64 ONE.
 * `cpu/bioemu2/` holds the denoiser and its tape and nothing else; what checks
 * the input embedder anywhere is `check-bioemu2-network.js` against
 * `dump_bioemu2.py`'s dump, and that dump is not on this box. So the arm is the
 * WGSL implementation - itself gated against that dump elsewhere - driven on the
 * SAME weights and the SAME features, with its f16 weights turned off so that
 * what is compared is two implementations rather than two precisions.
 *
 * 🔴 AND THE HOST HALF IS SHARED, WHICH IS WHAT MAKES THE COMPARISON SHARP. The
 * atom features, the windowed pair geometry and its MLP, the window bias and the
 * 455-column assembly now live in `cpu/bioemu2/host.js` and both backends call
 * them - so a difference here is the three atom LAYERS and nothing else.
 *
 * 🔴 AND THE FEATURES ARE SYNTHETIC, which this stage can afford: every one of
 * them is an array the featuriser hands over, and the embedder reads them
 * elementwise. What a synthetic set cannot claim is anything about the
 * FEATURISER - and `shared/bioemu2/featurise.js` has its own gates for that.
 */
import { Bioemu2Gpu } from "../../webgpu/bioemu2/network.js";
import { BIOEMU2, bundleReader } from "../../shared/bioemu2/weights.js";
import { inputEmbedder } from "../../webgl/bioemu2/embedder.js";
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
  const bar = Number(option(args, "bar", "2e-4"));
  const bundle = option(args, "model", "/model-bioemu2-int5").replace(/\/$/, "");
  const windows = Number(option(args, "windows", "2"));
  const atoms = windows * BIOEMU2.windowQueries;
  const tokens = Number(option(args, "tokens", "8"));

  const random = generator(2718281);
  const noise = (n, spread, centre = 0) => {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i += 1) out[i] = centre + (random() * 2 - 1) * spread;
    return out;
  };

  // The features, with padding throughout and two reference spaces - the window
  // pair is masked by `ref_space_uid`, so one space everywhere would leave that
  // test unexercised.
  const atomPad = new Int32Array(atoms).fill(1);
  for (let atom = 0; atom < atoms; atom += 1) if (atom % 13 === 11) atomPad[atom] = 0;
  const tokenOfAtom = new Int32Array(atoms);
  {
    let atom = 0;
    for (let token = 0; token < tokens; token += 1) {
      const count = Math.max(1, Math.floor(atoms / tokens) + (token % 3 === 0 ? 1 : -1));
      for (let i = 0; i < count && atom < atoms; i += 1, atom += 1) tokenOfAtom[atom] = token;
    }
    for (; atom < atoms; atom += 1) tokenOfAtom[atom] = tokens - 1;
  }
  const features = {
    atom_pad_mask: atomPad,
    atom_mask: Float32Array.from(atomPad),
    template_coords: noise(atoms * 3, 6),
    charge: noise(atoms, 1),
    element: Int32Array.from({ length: atoms }, (_, i) => (i * 7 + 3) % 128),
    name_encoding: Int32Array.from({ length: atoms * 4 }, (_, i) => (i * 11 + 5) % 64),
    ref_space_uid: Int32Array.from({ length: atoms }, (_, i) => (i < atoms / 2 ? 0 : 1)),
    token_idx_at_atom: tokenOfAtom,
    res_type: Int32Array.from({ length: tokens }, (_, i) => (i * 5 + 1) % 32),
    profile: noise(tokens * 33, 0.3, 0.3),
    deletion_mean: noise(tokens, 0.2, 0.2),
    pocket_feature: new Int32Array(tokens),
  };
  if (falsify) {
    // 🔴 THE SKEW GOES TO THIS PORT'S FEATURES ALONE, and it lands on the
    // coordinates: they reach the atom features, the windowed pair geometry and
    // so every layer's bias.
    features.ours = Float32Array.from(features.template_coords,
      (v, i) => (i % 3 === 0 ? v * 1.01 + 0.01 : v));
  }

  const reader = bundleReader(bundle);
  // 🔴 f16 WEIGHTS OFF, so the two sides differ by implementation and not by
  // precision - the shipped WGSL arm halves its weights and this port has no f16.
  const net = new Bioemu2Gpu(device, reader, { half: false, trunkPrecision: "f32" });
  const theirs = await net.inputEmbedder(net.upload(features));

  // The same host group this port needs, read once through the same reader.
  const prefix = "input_embedder/atom_attention_encoder";
  const hostNames = ["proj_atom_continuous/weight", "embed_element/weight",
    ...[0, 1, 2, 3].map((i) => `embed_atom_name_chars_${i}/weight`),
    "embed_atompair_ref_pos/weight", "embed_atompair_ref_dist/weight",
    "embed_atompair_mask/weight", "c_to_p_trans_q_1/weight", "c_to_p_trans_k_1/weight",
    "p_mlp_1/weight", "p_mlp_3/weight", "p_mlp_5/weight", "atom_to_token_trans_0/weight"];
  const hostWeights = {};
  // 🔴 `bundleReader` RETURNS THE READ FUNCTION, not an object with a `read`.
  // Its extras hang off that function (`read.has`, `read.prefetch`), which is
  // what `reader.read is not a function` was telling me.
  for (const name of hostNames) hostWeights[name] = await reader(`${prefix}/${name}`);

  const gl = createWebGL2Context();
  const execution = new Execution(gl);
  const held = [];
  const upload = (values, label) => {
    const tensor = execution.upload(Float32Array.from(values), label);
    held.push(tensor);
    return tensor;
  };
  const layers = [];
  for (let layer = 0; layer < BIOEMU2.inputLayers; layer += 1) {
    const base = `${prefix}/atom_encoder_layers_${layer}/transformer_layer`;
    const attn = `${base}/attention/attention_layer/pair_bias_attn`;
    const read = (name) => reader(name);
    layers.push({
      normZ: await read(`${attn}/norm_z/weight`),
      projZ: await read(`${attn}/proj_z/weight`),
      weights: {
        cond0Scale: upload(await read(`${base}/cond_0/weight`), `w.${layer}.cond0`),
        cond0Offset: upload(await read(`${base}/cond_0/bias`), `w.${layer}.cond0b`),
        cond2Weight: upload(await read(`${base}/cond_2/weight`), `w.${layer}.cond2`),
        cond2Bias: upload(await read(`${base}/cond_2/bias`), `w.${layer}.cond2b`),
        attention: {
          projQkv: upload(await read(`${attn}/proj_qkv/weight`), `w.${layer}.qkv`),
          projQBias: upload(await read(`${attn}/proj_q_bias`), `w.${layer}.qbias`),
          projG: upload(await read(`${attn}/proj_g/weight`), `w.${layer}.g`),
          projO: upload(await read(`${attn}/proj_o/weight`), `w.${layer}.o`),
        },
        transition: {
          fc1: upload(await read(`${base}/transition/fc1/weight`), `w.${layer}.fc1`),
          fc2: upload(await read(`${base}/transition/fc2/weight`), `w.${layer}.fc2`),
          fc3: upload(await read(`${base}/transition/fc3/weight`), `w.${layer}.fc3`),
        },
      },
    });
  }

  const ourFeatures = falsify
    ? { ...features, template_coords: features.ours } : features;
  const mine = inputEmbedder(execution, {
    features: ourFeatures, hostWeights, layers, tokens,
  });

  const arms = [];
  const failures = [];
  const relative = relativeRms(mine, theirs, "s_inputs");
  const ok = relative <= bar;
  arms.push({
    arm: "s_inputs, against the WGSL embedder",
    relativeRms: Number(relative.toPrecision(4)), bar, ok,
  });
  if (!ok) failures.push(`s_inputs: relRMS ${relative.toExponential(3)} over ${bar}`);

  // 🔴 AND THE 455 COLUMNS ARE THREE DIFFERENT THINGS, so the arm is split: the
  // 384 pooled from the atoms are what the layers decide, and the 71 after them
  // are the featuriser's own one-hots, which must be EXACT on both sides.
  const pooled = (values) => {
    const out = new Float32Array(tokens * BIOEMU2.single);
    for (let token = 0; token < tokens; token += 1) {
      out.set(values.subarray(token * BIOEMU2.sInputs,
        token * BIOEMU2.sInputs + BIOEMU2.single), token * BIOEMU2.single);
    }
    return out;
  };
  const tail = (values) => {
    const width = BIOEMU2.sInputs - BIOEMU2.single;
    const out = new Float32Array(tokens * width);
    for (let token = 0; token < tokens; token += 1) {
      out.set(values.subarray(token * BIOEMU2.sInputs + BIOEMU2.single,
        (token + 1) * BIOEMU2.sInputs), token * width);
    }
    return out;
  };
  const pooledRms = relativeRms(pooled(mine), pooled(theirs), "pooled");
  arms.push({
    arm: "the 384 pooled from the atom layers",
    relativeRms: Number(pooledRms.toPrecision(4)), bar, ok: pooledRms <= bar,
  });
  if (!(pooledRms <= bar)) failures.push(`the pooled half: relRMS ${pooledRms}`);
  const tailRms = relativeRms(tail(mine), tail(theirs), "tail");
  arms.push({
    arm: "the 71 featuriser columns, exactly",
    relativeRms: tailRms, bar: 0, ok: tailRms === 0,
    falsifiable: false,
    reason: "the same host function on the same features, so a skew of the"
      + " coordinates cannot move it - what it asserts is that both sides"
      + " assemble the same columns",
  });
  if (tailRms !== 0) failures.push(`the featuriser columns differ by ${tailRms}`);

  for (const tensor of held) execution.release(tensor);
  if (execution.stats().live !== 0) {
    failures.push(`${execution.stats().live} tensors left live`);
  }
  const falsifiable = arms.filter((row) => row.falsifiable !== false);
  const red = falsifiable.filter((row) => !row.ok).length;
  return {
    bundle,
    shape: {
      atoms, tokens, windows, atom: BIOEMU2.atom, atomPair: BIOEMU2.atomPair,
      atomHeads: BIOEMU2.atomHeads, layers: BIOEMU2.inputLayers,
      sInputs: BIOEMU2.sInputs,
      paddedAtoms: atoms - atomPad.reduce((a, b) => a + b, 0),
      referenceSpaces: 2,
    },
    reference: "the WGSL input embedder, f16 weights off, same bundle",
    bar,
    falsify,
    arms,
    memory: execution.stats(),
    verdict: falsify
      ? (red === falsifiable.length
        ? `ok: all ${falsifiable.length} falsifiable arms go red on a 1% perturbation`
        : `VACUOUS: only ${red} of ${falsifiable.length} falsifiable arms noticed it`)
      : (failures.length === 0
        ? `ok: ${arms.length} arms against the WGSL embedder on a real bundle`
        : `FAILED: ${failures.join("; ")}`),
  };
}
