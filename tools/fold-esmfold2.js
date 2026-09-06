// A whole ESMFold2 fold on the CPU: features in, a structure out.
//
//     node tools/fold-esmfold2.js
//
// 🔴 THIS IS THE PORT FOLDING, NOT A DIFFERENTIAL. Every stage is LocalFold's
// own - the featuriser, the atom encoder, twenty-four trunk blocks four times
// over, the conditioning, the token transformer, the atom decoder and eleven
// sampler steps - and the only things taken from the dump are the input
// FEATURES and the language model's pair term, which has its own oracle in
// tools/gpu/check-esmc-tower.js. Each stage has already been checked against
// the native model separately; what this answers is whether they compose into
// a structure.
//
// 🔴 AND IT CANNOT PRODUCE THE MODEL'S COORDINATES, BY CONSTRUCTION. The
// sampler draws a random rotation, a random translation and a noise vector at
// every step from torch's RNG. This uses its own, so the fold is a DIFFERENT
// SAMPLE from the same distribution - and the honest comparison is RMSD after
// superposition, not a relRMS.
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import process from "node:process";

import { readTensor } from "../src/reference/dtype.js";
import { transition, triangleMultiplication } from "../src/af3/pairformer-reference.js";
import {
  recycleProjection, relativePositionEncoding, tokenBondEncoding, zInitFromInputs,
} from "../src/esmfold2/featuriser-reference.js";
import { atomDecoder, inputsEmbedder } from "../src/esmfold2/atom-encoder-reference.js";
import { denoiseStep } from "../src/esmfold2/diffusion-reference.js";
import { centreRandomAugmentation, churnFactors, gaussians, noiseLevels, noiseSchedule,
         samplerStep, weightedRigidAlign } from "../src/esmfold2/sampler-reference.js";

const ROOT = new URL("..", import.meta.url).pathname;
const flag = (name, fallback) => {
  const at = process.argv.find((a) => a.startsWith(`--${name}=`));
  return at === undefined ? fallback : at.slice(name.length + 3);
};
const bundleDirectory = flag("bundle", join(ROOT, "model-esmfold2-trunk-f32"));
const dumpPath = flag("dump", join(ROOT, "oracle-dumps", "esmfold2-trunk-40-lm.json"));
const seed = Number(flag("seed", "0"));
const out = flag("out", join(ROOT, "esmfold2-fold.pdb"));

function loadBundle(directory) {
  const manifest = JSON.parse(readFileSync(join(directory, "manifest.json"), "utf8"));
  const shards = new Map();
  const tensors = {};
  for (const [name, record] of Object.entries(manifest.tensors)) {
    if (!shards.has(record.file)) {
      const raw = readFileSync(join(directory, record.file));
      shards.set(record.file,
        raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
    }
    tensors[name] = readTensor(record, shards.get(record.file), record.byteOffset ?? 0, true);
  }
  return { manifest, tensors };
}

const dump = JSON.parse(readFileSync(dumpPath, "utf8"));
const { manifest, tensors } = loadBundle(bundleDirectory);
const M = manifest.trunk;
const n = dump.shapes.pair[1];
const pairs = n * n;
const channels = M.pairChannels;
const feature = (name) => Float32Array.from(dump.features[name].values);
const integers = (name) => Int32Array.from(dump.features[name].values);
const atoms = dump.features.ref_pos.shape[1];
const started = Date.now();
const say = (message) => console.log(
  `  [${((Date.now() - started) / 1000).toFixed(0).padStart(4)}s] ${message}`);

console.log(`ESMFold2 on the CPU: ${n} residues, ${atoms} atoms, seed ${seed}\n`);

// ---- the featuriser.
const atomShape = {
  atoms, tokens: n, channels: M.atomChannels, heads: M.atomHeads,
  blocks: M.atomBlocks, tokenChannels: 384, hidden: M.atomChannels * 2,
  windowSize: M.atomWindow,
};
const swaBlocks = (prefix, count) => {
  const list = [];
  for (let layer = 0; layer < count; layer += 1) {
    const at = (leaf) => tensors[`${prefix}/blocks/${layer}/${leaf}`];
    list.push({ adaln: at("adaln"), qkv: at("qkv"), attnGate: at("attnGate"),
                attnOut: at("attnOut"), ffnUp: at("ffnUp"), ffnDown: at("ffnDown") });
  }
  return list;
};
const baseFeatures = {
  refPos: feature("ref_pos"), refCharge: feature("ref_charge"),
  mask: feature("atom_attention_mask"), refElement: integers("ref_element"),
  refAtomNameChars: integers("ref_atom_name_chars"),
  refSpaceUid: feature("ref_space_uid"), atomToToken: integers("atom_to_token"),
};
const { tokenAct } = inputsEmbedder(baseFeatures, atomShape, {
  atomLinear: tensors["atom/linear"], atomNormScale: tensors["atom/norm/scale"],
  atomNormOffset: tensors["atom/norm/offset"], atomToToken: tensors["atom/toToken"],
  blocks: swaBlocks("atom", M.atomBlocks),
});
say("inputs embedder");

const singleInputs = M.singleInputs;
const embedderArgs = dump.block0.inputs_embedder.arguments;
const classes = embedderArgs.aatype.shape[2];
const aatype = Float32Array.from(embedderArgs.aatype.values);
const profile = Float32Array.from(embedderArgs.profile.values);
const deletion = Float32Array.from(embedderArgs.deletion_mean.values);
const sInputs = new Float32Array(n * singleInputs);
for (let token = 0; token < n; token += 1) {
  const to = token * singleInputs;
  for (let c = 0; c < 384; c += 1) sInputs[to + c] = tokenAct[token * 384 + c];
  for (let c = 0; c < classes; c += 1) {
    sInputs[to + 384 + c] = aatype[token * classes + c];
    sInputs[to + 384 + classes + c] = profile[token * classes + c];
  }
  sInputs[to + singleInputs - 1] = deletion[token];
}

const relPos = relativePositionEncoding({
  residueIndex: integers("residue_index"), asymId: integers("asym_id"),
  symId: integers("sym_id"), entityId: integers("entity_id"),
  tokenIndex: integers("token_index"),
}, n, channels, tensors["featuriser/relPos"]);
const zInit = zInitFromInputs(sInputs, n, singleInputs, channels,
  tensors["featuriser/zInit1"], tensors["featuriser/zInit2"]);
const bonds = tokenBondEncoding(
  Float32Array.from(dump.features.token_bonds.values), n, channels,
  tensors["featuriser/tokenBonds"]);
// 🔴 THE LANGUAGE MODEL'S PAIR COMES FROM THE DUMP. ESM-C's 36 blocks at 1152
// channels are a GPU job; src/esmc/ has the WebGPU port and its own oracle. A
// CPU fold that also ran the tower would take longer than this whole file and
// would say nothing the tower's own checker does not.
const lmPair = Float32Array.from(dump.block0.language_model.output);
for (let i = 0; i < zInit.length; i += 1) zInit[i] += relPos[i] + bonds[i] + lmPair[i];
say("z_init, all five terms");

// ---- the trunk.
const TRIANGLE = ["leftNormInputScale", "leftNormInputOffset", "centerNormScale",
  "centerNormOffset", "outputProjection", "gatingLinear", "projection", "gate"];
const TRANSITION = ["inputLayerNormScale", "inputLayerNormOffset",
  "transition1", "transition2"];
const group = (layer, name, leaves) => {
  const g = {};
  for (const leaf of leaves) g[leaf] = tensors[`blocks/${layer}/${name}/${leaf}`];
  return g;
};
const recycle = {
  scale: tensors["recycle/norm/scale"], offset: tensors["recycle/norm/offset"],
  projection: tensors["recycle/projection"],
};
const mask2d = new Float32Array(pairs).fill(1);
let pair = new Float32Array(pairs * channels);
for (let loop = 0; loop < dump.loops; loop += 1) {
  const projected = recycleProjection(pair, pairs, channels, recycle);
  pair = zInit.map((value, i) => value + projected[i]);
  const add = (delta) => { for (let i = 0; i < pair.length; i += 1) pair[i] += delta[i]; };
  for (let layer = 0; layer < M.blocks; layer += 1) {
    add(triangleMultiplication(pair, mask2d, n, channels, "outgoing",
      group(layer, "triangleMultiplicationOutgoing", TRIANGLE)));
    add(triangleMultiplication(pair, mask2d, n, channels, "incoming",
      group(layer, "triangleMultiplicationIncoming", TRIANGLE)));
    add(transition(pair, pairs, channels, group(layer, "pairTransition", TRANSITION)));
  }
  say(`trunk loop ${loop}`);
}

// ---- the sampler.
const s = dump.sampler;
const schedule = noiseSchedule({ steps: s.steps, sMax: s.sMax, sMin: s.sMin, p: s.p,
                                 sigmaData: s.sigmaData, maxSigma: 256 });
const gammas = churnFactors(schedule, s.gammaMin, s.gamma0);
const levels = noiseLevels(schedule, gammas);
const draw = gaussians(seed);
const mask = baseFeatures.mask;

const denoiseShape = {
  tokens: n, pairChannels: channels, singleInputs, tokenChannels: M.tokenChannels2,
  multiplier: M.transitionMultiplier, tokenHeads: M.tokenHeads, sigmaData: M.sigmaData,
};
const diffusionShape = { ...atomShape, tokenChannels: M.tokenChannels2 };
const encoder = {
  embed: (f) => inputsEmbedder({ ...baseFeatures, ...f }, diffusionShape, {
    atomLinear: tensors["diffusionAtomEncoder/linear"],
    atomNormScale: tensors["diffusionAtomEncoder/norm/scale"],
    atomNormOffset: tensors["diffusionAtomEncoder/norm/offset"],
    atomToToken: tensors["diffusionAtomEncoder/toToken"],
    coordsLinear: tensors["diffusionAtomEncoder/coordsLinear"],
    blocks: swaBlocks("diffusionAtomEncoder", M.atomBlocks),
  }),
  decode: (act, embedded) => atomDecoder(act, embedded.atomState, embedded.conditioning,
    embedded.rope, baseFeatures, diffusionShape, {
      tokenToAtom: tensors["diffusionAtomDecoder/tokenToAtom"],
      normScale: tensors["diffusionAtomDecoder/norm/scale"],
      normOffset: tensors["diffusionAtomDecoder/norm/offset"],
      outputLinear: tensors["diffusionAtomDecoder/outputLinear"],
      blocks: swaBlocks("diffusionAtomDecoder", M.atomBlocks),
    }),
};
const conditioningWeights = {
  zInputNormScale: tensors["diffusion/zInputNorm/scale"],
  zInputNormOffset: tensors["diffusion/zInputNorm/offset"],
  zProjection: tensors["diffusion/zProjection"],
  sInputNormScale: tensors["diffusion/sInputNorm/scale"],
  sInputNormOffset: tensors["diffusion/sInputNorm/offset"],
  sProjection: tensors["diffusion/sProjection"],
  fourierWeights: tensors["diffusion/fourier/weights"],
  fourierOffsets: tensors["diffusion/fourier/offsets"],
  noiseNormScale: tensors["diffusion/noiseNorm/scale"],
  noiseNormOffset: tensors["diffusion/noiseNorm/offset"],
  noiseProjection: tensors["diffusion/noiseProjection"],
  zTransitions: [0, 1].map((l) => ({
    normScale: tensors[`diffusion/zTransitions/${l}/norm/scale`],
    normOffset: tensors[`diffusion/zTransitions/${l}/norm/offset`],
    aProjection: tensors[`diffusion/zTransitions/${l}/aProjection`],
    bProjection: tensors[`diffusion/zTransitions/${l}/bProjection`],
    outProjection: tensors[`diffusion/zTransitions/${l}/outProjection`] })),
  sTransitions: [0, 1].map((l) => ({
    normScale: tensors[`diffusion/sTransitions/${l}/norm/scale`],
    normOffset: tensors[`diffusion/sTransitions/${l}/norm/offset`],
    aProjection: tensors[`diffusion/sTransitions/${l}/aProjection`],
    bProjection: tensors[`diffusion/sTransitions/${l}/bProjection`],
    outProjection: tensors[`diffusion/sTransitions/${l}/outProjection`] })),
};
const tokenBlocks = [];
for (let layer = 0; layer < M.tokenBlocks; layer += 1) {
  const at = (kind, leaf) => tensors[`diffusion/tokenBlocks/${layer}/${kind}/${leaf}`];
  const adaln = (kind) => ({
    singleScale: at(kind, "adaln/singleScale"), gateWeights: at(kind, "adaln/gateWeights"),
    gateBias: at(kind, "adaln/gateBias"), shiftWeights: at(kind, "adaln/shiftWeights") });
  tokenBlocks.push({
    attention: { adaln: adaln("attention"),
      queryWeights: at("attention", "queryWeights"), queryBias: at("attention", "queryBias"),
      kvWeights: at("attention", "kvWeights"), gateWeights: at("attention", "gateWeights"),
      outWeights: at("attention", "outWeights"),
      outGateWeights: at("attention", "outGateWeights"),
      outGateBias: at("attention", "outGateBias"),
      pairNormScale: at("attention", "pairNormScale"),
      pairNormOffset: at("attention", "pairNormOffset"),
      pairBiasWeights: at("attention", "pairBiasWeights") },
    transition: { adaln: adaln("transition"),
      swishWeights: at("transition", "swishWeights"),
      outWeights: at("transition", "outWeights"),
      outGateWeights: at("transition", "outGateWeights"),
      outGateBias: at("transition", "outGateBias") } });
}
const denoiserWeights = {
  conditioning: conditioningWeights, tokenBlocks,
  stepNormScale: tensors["diffusion/stepNorm/scale"],
  stepNormOffset: tensors["diffusion/stepNorm/offset"],
  singleToToken: tensors["diffusion/singleToToken"],
  tokenNormScale: tensors["diffusion/tokenNorm/scale"],
  tokenNormOffset: tensors["diffusion/tokenNorm/offset"],
};

let x = new Float32Array(atoms * 3);
for (let i = 0; i < x.length; i += 1) x[i] = schedule[0] * draw();
let cached;
for (let step = 0; step < levels.length; step += 1) {
  x = centreRandomAugmentation(x, mask, atoms, draw);
  const tHat = levels[step];
  const sigmaTm = schedule[step];
  const epsilon = s.noiseScale * Math.sqrt(Math.max(tHat * tHat - sigmaTm * sigmaTm, 0));
  const noisy = new Float32Array(x.length);
  for (let i = 0; i < noisy.length; i += 1) noisy[i] = x[i] + epsilon * draw();
  const answer = denoiseStep(noisy, tHat, baseFeatures, sInputs, pair, relPos,
                             denoiseShape, denoiserWeights, encoder, cached);
  // 🔴 THE PAIR CONDITIONING IS CACHED ACROSS THE STEPS AND THE SINGLE IS NOT,
  // as upstream does: only `s` depends on the noise level. Caching both would
  // freeze `t_hat` at step zero, which still converges to a structure.
  cached = answer.conditioning.pair;
  x = samplerStep(noisy, answer.coordinates, mask, atoms, tHat, schedule[step + 1],
                  s.stepScale);
  say(`sampler step ${step} (t_hat ${tHat.toExponential(2)})`);
}

// ---- what came out.
const native = Float32Array.from(dump.coordinates);
const aligned = weightedRigidAlign(x, native, mask, atoms);
let squared = 0, live = 0;
for (let atom = 0; atom < atoms; atom += 1) {
  if (mask[atom] === 0) continue;
  live += 1;
  for (let axis = 0; axis < 3; axis += 1) {
    const d = aligned[atom * 3 + axis] - native[atom * 3 + axis];
    squared += d * d;
  }
}
const rmsd = Math.sqrt(squared / live);

// Alpha carbons, by name: 'CA' is [35, 33, 0, 0] under chr(code + 32).
const names = baseFeatures.refAtomNameChars;
const alphas = [];
for (let atom = 0; atom < atoms; atom += 1) {
  if (mask[atom] !== 0 && names[atom * 4] === 35 && names[atom * 4 + 1] === 33
      && names[atom * 4 + 2] === 0) alphas.push(atom);
}
const spacing = [];
for (let i = 1; i < alphas.length; i += 1) {
  const a = alphas[i - 1] * 3, b = alphas[i] * 3;
  spacing.push(Math.hypot(x[b] - x[a], x[b + 1] - x[a + 1], x[b + 2] - x[a + 2]));
}
const mean = spacing.reduce((t, v) => t + v, 0) / spacing.length;

const lines = [];
for (let index = 0; index < alphas.length; index += 1) {
  const atom = alphas[index];
  lines.push(`ATOM  ${String(index + 1).padStart(5)}  CA  GLY A`
    + `${String(index + 1).padStart(4)}    `
    + [0, 1, 2].map((axis) => x[atom * 3 + axis].toFixed(3).padStart(8)).join("")
    + "  1.00  0.00           C");
}
writeFileSync(out, `${lines.join("\n")}\nEND\n`);

console.log(`\n  ${alphas.length} alpha carbons, ${live} live atoms`);
console.log(`  CA-CA spacing            ${mean.toFixed(3)} A  `
  + `(min ${Math.min(...spacing).toFixed(3)}, max ${Math.max(...spacing).toFixed(3)})`);
console.log(`  RMSD to the native fold  ${rmsd.toFixed(3)} A  after superposition`);
console.log(`  wrote ${out}`);
console.log("\n  🔴 A DIFFERENT SAMPLE, NOT A DIFFERENT ANSWER. The sampler draws a\n"
  + "     rotation, a translation and a noise vector per step; this used its own\n"
  + "     RNG, so an RMSD of a few angstroms is what agreement LOOKS like here.");
