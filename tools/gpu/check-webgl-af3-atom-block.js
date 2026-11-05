/**
 * AF3's atom cross-attention block in WebGL2 against
 * `cpu/af3/diffusion/atom-encoder.js`.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-af3-atom-block.js [--falsify]
 *
 * 🔴 198 M OF THE DIFFUSION'S 203 M PARAMETERS ARE THIS BLOCK AT ANOTHER WIDTH,
 * so it is the one module worth the most gating in the whole head: "the 24-block
 * transformer... is the AdaLN-conditioned attention already written for the atom
 * stack at different widths. Its atom encoder and decoder are 0.9 M each and are
 * literally atom-encoder.js with other weights."
 *
 * 🔴 FOUR DIALECT CHOICES LIVE IN THIS ONE BLOCK AND NONE OF THEM CHANGES A
 * SHAPE, so each is an arm:
 *
 *   - the mask bias is a PRODUCT under AF3 and a SUM under OpenDDE, Protenix and
 *     RoseTTAFold3. The product penalises a pair only when query AND key are
 *     padded; the sum is an OR. "The difference is large in a mostly-empty
 *     window... and it is nothing at all when every key is real", which is why
 *     this gate's layout leaves two thirds of its key slots empty.
 *   - the two adaptive LayerNorms are PARALLEL under AF3 and CHAINED under
 *     OpenDDE, whose second norm "re-centres and re-scales a tensor whose
 *     statistics the first already fixed - while seeing the first one's learned
 *     scale".
 *   - the transition reads the POST-attention activation, except in
 *     rosettafold3 where it reads the PRE-attention one and both land through
 *     ONE residual - "worth the atom encoder's whole skip connection".
 *   - and boltz2 has an extra up-gate reading the same adaptive-LayerNorm output
 *     the SwiGLU reads.
 *
 * 🔴 AND THE CONDITIONING IS SQUARE IN EVERY ARM, WHICH IS THE REFERENCE'S LIMIT
 * AND NOT THIS PORT'S. "In the atom stack both are 128 and the distinction is
 * invisible; in the diffusion transformer the conditioning is 384 and the
 * activation 768." This reference is the ATOM stack's block and calls its
 * adaptive pieces without their width arguments, so it cannot be asked about the
 * wide case - an arm that tried read 6e-3 against a reference that had assumed
 * square. The wide path is the diffusion TRANSFORMER's block, a different
 * function with its own reference, and this port's block takes both widths so
 * that one module serves both.
 *
 * 🔴 AND THE LAYOUT HERE IS SYNTHETIC, WHICH IS A STATED LIMIT RATHER THAN A
 * CHOICE. `check-af3-atom-encoder.js` uses a real batch's five gathers and says
 * why: "a hand-built layout would be regular in ways a real one is not, and the
 * padding is where the interesting disagreements live". That dump
 * (`oracle-dumps/af3-oracle-atom-f32.json`) is not on this box. What this gate
 * does instead is make the layout deliberately irregular - a window whose keys
 * come from scattered query rows, two thirds of them padded, and a query mask
 * that disagrees with the key mask - so the padding paths run; what it cannot
 * claim is that a real batch's gathers hold no case this one misses.
 */
import {
  crossAttentionBlock as cpuBlock, layerNormSlow,
} from "../../cpu/af3/diffusion/atom-encoder.js";
import {
  crossAttentionBlock, gather,
} from "../../webgl/af3/diffusion/atom-attention.js";
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
  const bar = Number(option(args, "bar", "3e-6"));
  const subsets = Number(option(args, "subsets", "2"));
  const queries = Number(option(args, "queries", "8"));
  const keys = Number(option(args, "keys", "12"));
  const channels = Number(option(args, "channels", "16"));
  // 🔴 THE CONDITIONING IS SQUARE HERE, AND THAT IS THE REFERENCE'S LIMIT RATHER
  // THAN THIS PORT'S. `crossAttentionBlock` in cpu/af3/diffusion/atom-encoder.js
  // is the ATOM stack's block: it calls `adaptiveLayerNorm` and
  // `adaptiveZeroInit` without their width arguments, so both default to
  // square - and its `adaptiveZeroInit` reads the gathered attention at
  // `channels` rather than at `heads * dimension`. The DIFFUSION TRANSFORMER's
  // block is a different function (cpu/af3/diffusion/diffusion.js) which passes
  // the widths explicitly, and that is where "the conditioning is 384 and the
  // activation 768" lives. This port's block takes both widths, so it serves
  // either; a `--cond-channels=32` arm here read 6e-3 against a reference that
  // had assumed square, which is the reference being asked the wrong question.
  const condChannels = channels;
  const heads = Number(option(args, "heads", "2"));
  const dimension = Number(option(args, "dimension", "8"));
  const width = heads * dimension;
  if (width !== channels) {
    throw new RangeError(
      `af3 atom block gate: heads * dimension is ${width} and channels ${channels}.`
      + " The f64 reference's `adaptiveZeroInit` reads the gathered attention at"
      + " `channels`, so the two must match for it to be answering the same"
      + " question - the diffusion transformer's block is where they differ.");
  }
  const queryRows = subsets * queries;
  const keyRows = subsets * keys;

  const random = generator(2718281);
  const noise = (n, spread, centre = 0) => {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i += 1) out[i] = centre + (random() * 2 - 1) * spread;
    return out;
  };

  const adaptiveGroup = (prefix) => ({
    [`${prefix}SingleCondLayerNormScale`]: noise(condChannels, 0.2, 1),
    [`${prefix}SingleCondScaleWeights`]: noise(
      condChannels * channels, 1 / Math.sqrt(condChannels)),
    [`${prefix}SingleCondScaleBias`]: noise(channels, 0.1),
    [`${prefix}SingleCondBias`]: noise(condChannels * channels, 0.2 / Math.sqrt(condChannels)),
  });
  const intermediate = channels * 2;
  const baseWeights = {
    ...adaptiveGroup("q"),
    ...adaptiveGroup("k"),
    ...adaptiveGroup("ffw"),
    qProjection: noise(channels * width, 1 / Math.sqrt(channels)),
    qBias: noise(width, 0.1),
    kProjection: noise(channels * width, 1 / Math.sqrt(channels)),
    vProjection: noise(channels * width, 1 / Math.sqrt(channels)),
    gatingQuery: noise(channels * width, 1 / Math.sqrt(channels)),
    Transition2: noise(width * channels, 0.3 / Math.sqrt(width)),
    AdaptiveZeroCondWeights: noise(condChannels * channels, 1 / Math.sqrt(condChannels)),
    AdaptiveZeroCondBias: noise(channels, 0.1),
    ffwTransition1: noise(channels * intermediate * 2, 1 / Math.sqrt(channels)),
    ffwTransition2: noise(intermediate * channels, 0.3 / Math.sqrt(intermediate)),
    ffwAdaptiveZeroCondWeights: noise(
      condChannels * channels, 1 / Math.sqrt(condChannels)),
    ffwAdaptiveZeroCondBias: noise(channels, 0.1),
  };

  // 🔴 AN IRREGULAR LAYOUT WITH REAL PADDING: each window's keys come from
  // scattered query rows, two thirds of the key slots are padded, and the query
  // mask disagrees with the key mask - which is what makes the two mask rules
  // tell apart.
  const queriesToKeys = new Int32Array(keyRows);
  const keysMask = new Float32Array(keyRows);
  for (let subset = 0; subset < subsets; subset += 1) {
    for (let key = 0; key < keys; key += 1) {
      const index = subset * keys + key;
      // A scattered source row inside this window's reach.
      queriesToKeys[index] = (subset * queries + (key * 5 + 3) % queries) % queryRows;
      keysMask[index] = key % 3 === 0 ? 1 : 0;
    }
  }
  const queriesMask = new Float32Array(queryRows);
  for (let index = 0; index < queryRows; index += 1) {
    queriesMask[index] = index % 4 === 3 ? 0 : 1;
  }

  const queriesAct = noise(queryRows * channels, 1);
  const queriesCond = noise(queryRows * condChannels, 1);
  const keysCond = noise(keyRows * condChannels, 1);
  const pairLogits = noise(subsets * heads * queries * keys, 0.5);

  const gl = createWebGL2Context();
  const execution = new Execution(gl);
  const held = [];
  const upload = (values, label) => {
    const tensor = execution.upload(Float32Array.from(values), label);
    held.push(tensor);
    return tensor;
  };
  // The control skews half of each query row's channels: the activation's first
  // stop is an unscaled LayerNorm, and a whole-row affine change is invisible to
  // one.
  const actTensor = upload(falsify
    ? Float32Array.from(queriesAct,
      (v, i) => (i % channels < channels / 2 ? v * 1.01 + 0.01 : v))
    : queriesAct, "af3.atom.act");
  const shared = {
    queriesToKeys: upload(queriesToKeys, "af3.atom.q2k"),
    queriesMask: upload(queriesMask, "af3.atom.qmask"),
    keysMask: upload(keysMask, "af3.atom.kmask"),
    queriesCond: upload(queriesCond, "af3.atom.qcond"),
    keysCond: upload(keysCond, "af3.atom.kcond"),
    pairLogits: upload(pairLogits, "af3.atom.logits"),
  };

  const arms = [];
  const failures = [];
  /** @type {{name: string, flags: object, upGate: boolean}[]} */
  const CASES = [
    { name: "AF3 (product mask, parallel norms, two residuals)", upGate: false, flags: {
      keyMaskedAtomAttention: false, chainedAtomLayerNorm: false, diffusionNoResidual: false,
    } },
    { name: "OpenDDE (sum mask, chained norms)", upGate: false, flags: {
      keyMaskedAtomAttention: true, chainedAtomLayerNorm: true, diffusionNoResidual: false,
    } },
    { name: "rosettafold3 (one residual)", upGate: false, flags: {
      keyMaskedAtomAttention: true, chainedAtomLayerNorm: false, diffusionNoResidual: true,
    } },
    { name: "boltz2 (the extra up-gate)", upGate: true, flags: {
      keyMaskedAtomAttention: true, chainedAtomLayerNorm: false, diffusionNoResidual: false,
    } },
  ];
  const outputs = {};
  for (const { name, flags, upGate } of CASES) {
    const weights = upGate
      ? { ...baseWeights, ffwAToB: noise(channels * intermediate, 1 / Math.sqrt(channels)) }
      : baseWeights;
    const uploaded = Object.fromEntries(Object.entries(weights).map(
      ([key, values]) => [key, upload(values, `w.af3.atom.${key}`)]));
    const out = crossAttentionBlock(execution, {
      queriesAct: actTensor, ...shared,
      subsets, queries, keys, channels, heads, dimension, condChannels,
      weights: uploaded, ...flags,
    });
    const mine = Float32Array.from(execution.read(out));
    execution.release(out);
    outputs[name] = mine;

    const expected = cpuBlock(queriesAct, {
      queriesToKeys: { indices: queriesToKeys, mask: keysMask, count: keyRows },
      queriesMask, keysMask, queriesCond, keysCond, pairLogits,
    }, { subsets, queries, keys, channels, heads, dimension }, { ...weights, ...flags });
    const relative = relativeRms(mine, expected, name);
    const ok = relative <= bar;
    arms.push({ arm: `${name} vs the f64 reference`,
      relativeRms: Number(relative.toPrecision(4)), bar, ok });
    if (!ok) failures.push(`${name}: relRMS ${relative.toExponential(3)} over ${bar}`);
  }

  // 🔴 AND EVERY DIALECT MUST MOVE THE ANSWER, or the four arms above are one arm
  // four times. Each is compared against AF3's, which is the configuration they
  // each differ from by exactly one choice (boltz2's by a term, not a choice).
  const reference = outputs[CASES[0].name];
  const dialectMoves = {};
  for (const { name } of CASES.slice(1)) {
    dialectMoves[name] = Number(
      relativeRms(outputs[name], reference, `moves.${name}`).toPrecision(3));
    if (!(dialectMoves[name] > 1e-4)) {
      failures.push(`${name} is INERT: it agrees with AF3's to ${dialectMoves[name]}`);
    }
  }

  for (const tensor of held) execution.release(tensor);
  if (execution.stats().live !== 0) {
    failures.push(`${execution.stats().live} tensors left live`);
  }
  const red = arms.filter((row) => !row.ok).length;
  return {
    shape: { subsets, queries, keys, channels, condChannels, heads, dimension },
    paddedKeySlots: keysMask.length - keysMask.reduce((a, b) => a + b, 0),
    paddedQueryRows: queriesMask.length - queriesMask.reduce((a, b) => a + b, 0),
    layout: "synthetic and irregular; the real batch's gathers are not on this box",
    bar,
    falsify,
    arms,
    dialectMoves,
    memory: execution.stats(),
    verdict: falsify
      ? (red === arms.length
        ? `ok: all ${arms.length} arms go red on a 1% perturbation`
        : `VACUOUS: only ${red} of ${arms.length} arms noticed it`)
      : (failures.length === 0
        ? `ok: ${arms.length} dialects against the f64 reference, each moving the answer`
        : `FAILED: ${failures.join("; ")}`),
  };
}
