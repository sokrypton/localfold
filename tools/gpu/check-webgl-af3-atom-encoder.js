/**
 * AF3's atom cross-attention encoder in WebGL2 against
 * `cpu/af3/diffusion/atom-encoder.js`.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-af3-atom-encoder.js [--falsify]
 *
 * This is where 384 of `target_feat`'s 447 columns come from, and the diffusion
 * head runs the same module again with the trunk conditioning added - so both
 * forms are arms: `--trunk` adds the trunk's single and pair conditioning and
 * the noisy positions, which is the diffusion's configuration.
 *
 * 🔴 THE CONDITIONING IS MASKED BEFORE THE KEYS GATHER FROM IT, AND THAT ORDER
 * IS ONLY VISIBLE WITH THE TRUNK TERM. "This is a no-op for the trunk's encoder
 * - _per_atom_conditioning already masked its output, so the two orders agree -
 * and it is NOT a no-op once the trunk conditioning is added, because that puts
 * non-zero values into padded atom slots... measured 8.4e-2 on the encoder's
 * output, which reads like a subtly wrong kernel rather than a line in the wrong
 * order." So the `--trunk` arm is the one that can see it.
 *
 * 🔴 AND THE VALIDITY RULE LETS PADDED KEYS IN UNDER STOCK AF3, which is a
 * property of the WEIGHTS: "a padded key slot gathers a zero reference space,
 * and the FIRST reference conformer's uid is also zero - so every padded key
 * reads as a valid neighbour of token 0... AF3 was released this way and
 * OpenFold3 was trained with the mask." Both are arms, and the gate asserts they
 * DIFFER - with every key real they would not.
 *
 * 🔴 AND STOCK AF3 GATHERS THE KEYS' REFERENCE SPACE FROM THE **TOKEN** LAYOUT
 * with indices written for the QUERIES layout, clamped. `keysUidFromQueriesLayout`
 * is the other reading; both are arms and they differ wherever the two layouts
 * do - which this layout makes sure they do.
 *
 * 🔴 AND THE LAYOUT IS SYNTHETIC, which `check-af3-atom-encoder.js` explains is
 * a real limitation: "a hand-built layout would be regular in ways a real one is
 * not, and the padding is where the interesting disagreements live". Its dump is
 * not on this box. This gate makes the five gathers deliberately irregular and
 * says what it cannot claim.
 */
import {
  atomCrossAttentionEncoder as cpuEncoder,
} from "../../cpu/af3/diffusion/atom-encoder.js";
import {
  atomCrossAttentionEncoder,
} from "../../webgl/af3/diffusion/atom-encoder.js";
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
  const tokens = Number(option(args, "tokens", "4"));
  const dense = Number(option(args, "dense", "4"));
  const subsets = Number(option(args, "subsets", "2"));
  const queries = Number(option(args, "queries", "8"));
  const keys = Number(option(args, "keys", "12"));
  const channels = Number(option(args, "channels", "16"));
  const pairChannels = Number(option(args, "pair-channels", "8"));
  const perTokenChannels = Number(option(args, "per-token", "12"));
  const heads = 2;
  const dimension = 8;
  const blocks = Number(option(args, "blocks", "2"));
  const withTrunk = flag(args, "trunk");
  const trunkSingleChannels = 12;
  const trunkPairChannels = 8;
  // 🔴 THE ATOM STACK'S WIDTH MUST BE SQUARE, BECAUSE THE REFERENCE'S
  // `adaptiveZeroInit` READS THE GATHERED ATTENTION AT `channels`. At
  // `--channels=12` with two heads of eight every stage up to the pair MLP is
  // exact and `encoder.stackOut` reads 1.2e-2 - the reference being asked the
  // wrong question, not a wrong kernel, which `check-webgl-af3-atom-block.js`
  // refuses by name for the same reason. The real model is square here (128 =
  // 4 x 32); only the TOKEN transformer's conditioning differs from its
  // activation, and that axis is `seq-channels` against `per-token`.
  if (heads * dimension !== channels) {
    throw new RangeError(
      `af3 atom encoder gate: heads * dimension is ${heads * dimension} and the atom`
      + ` channels ${channels}. The f64 reference's \`adaptiveZeroInit\` reads the`
      + " gathered attention at `channels`, so the two must match for it to be"
      + " answering the same question.");
  }
  const queryRows = subsets * queries;
  const keyRows = subsets * keys;
  const atoms = tokens * dense;

  const random = generator(1732050);
  const noise = (n, spread, centre = 0) => {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i += 1) out[i] = centre + (random() * 2 - 1) * spread;
    return out;
  };
  const adaptiveGroup = (prefix) => ({
    [`${prefix}SingleCondLayerNormScale`]: noise(channels, 0.2, 1),
    [`${prefix}SingleCondScaleWeights`]: noise(channels * channels, 1 / Math.sqrt(channels)),
    [`${prefix}SingleCondScaleBias`]: noise(channels, 0.1),
    [`${prefix}SingleCondBias`]: noise(channels * channels, 0.2 / Math.sqrt(channels)),
  });
  const width = heads * dimension;
  const intermediate = channels * 2;
  const blockWeights = () => ({
    ...adaptiveGroup("q"),
    ...adaptiveGroup("k"),
    ...adaptiveGroup("ffw"),
    qProjection: noise(channels * width, 1 / Math.sqrt(channels)),
    qBias: noise(width, 0.1),
    kProjection: noise(channels * width, 1 / Math.sqrt(channels)),
    vProjection: noise(channels * width, 1 / Math.sqrt(channels)),
    gatingQuery: noise(channels * width, 1 / Math.sqrt(channels)),
    Transition2: noise(width * channels, 0.3 / Math.sqrt(width)),
    AdaptiveZeroCondWeights: noise(channels * channels, 1 / Math.sqrt(channels)),
    AdaptiveZeroCondBias: noise(channels, 0.1),
    ffwTransition1: noise(channels * intermediate * 2, 1 / Math.sqrt(channels)),
    ffwTransition2: noise(intermediate * channels, 0.3 / Math.sqrt(intermediate)),
    ffwAdaptiveZeroCondWeights: noise(channels * channels, 1 / Math.sqrt(channels)),
    ffwAdaptiveZeroCondBias: noise(channels, 0.1),
  });
  const weights = {
    channels, pairChannels, perTokenChannels, heads, dimension,
    trunkSingleChannels, trunkPairChannels,
    singleToPairCondRow: noise(channels * pairChannels, 1 / Math.sqrt(channels)),
    singleToPairCondCol: noise(channels * pairChannels, 1 / Math.sqrt(channels)),
    embedPairOffsets: noise(3 * pairChannels, 0.3),
    embedPairDistances: noise(pairChannels, 0.3),
    embedPairOffsetsValid: noise(pairChannels, 0.3),
    pairMlp1: noise(pairChannels * pairChannels, 1 / Math.sqrt(pairChannels)),
    pairMlp2: noise(pairChannels * pairChannels, 1 / Math.sqrt(pairChannels)),
    pairMlp3: noise(pairChannels * pairChannels, 1 / Math.sqrt(pairChannels)),
    pairInputLayerNormScale: noise(pairChannels, 0.2, 1),
    pairLogitsProjection: noise(pairChannels * blocks * heads, 1 / Math.sqrt(pairChannels)),
    projectAtomFeaturesForAggr: noise(
      channels * perTokenChannels, 1 / Math.sqrt(channels)),
    atomPositionsToFeatures: noise(3 * channels, 0.3),
    lnormTrunkSingleCondScale: noise(trunkSingleChannels, 0.2, 1),
    lnormTrunkSingleCondOffset: noise(trunkSingleChannels, 0.1),
    embedTrunkSingleCond: noise(
      trunkSingleChannels * channels, 1 / Math.sqrt(trunkSingleChannels)),
    lnormTrunkPairCondScale: noise(trunkPairChannels, 0.2, 1),
    lnormTrunkPairCondOffset: noise(trunkPairChannels, 0.1),
    embedTrunkPairCond: noise(
      trunkPairChannels * pairChannels, 1 / Math.sqrt(trunkPairChannels)),
    blocks: Array.from({ length: blocks }, blockWeights),
  };

  // 🔴 FIVE IRREGULAR GATHERS, with padding on both layouts and a reference
  // space that is NOT constant - so the validity rule and the two uid readings
  // all have something to disagree about.
  const atomMask = new Float32Array(atoms).fill(1);
  for (let index = 0; index < atoms; index += 1) {
    if (index % 5 === 4) atomMask[index] = 0;
  }
  const refSpaceUid = new Float32Array(atoms);
  for (let index = 0; index < atoms; index += 1) {
    refSpaceUid[index] = Math.floor(index / dense);
  }
  const tokenAtomsToQueries = {
    indices: new Int32Array(queryRows), mask: new Float32Array(queryRows), count: queryRows,
  };
  // 🔴 SCATTERED, NOT THE IDENTITY. With `indices[i] = i` the two readings of
  // the keys' reference space coincide - the gathered uid and the directly
  // indexed one are the same array - and the dialect arm reported INERT: 0. A
  // layout has to be irregular in the axis the convention lives on, which is
  // the general form of what check-af3-atom-encoder.js says about real batches.
  for (let index = 0; index < queryRows; index += 1) {
    tokenAtomsToQueries.indices[index] = (index * 7 + 2) % atoms;
    tokenAtomsToQueries.mask[index] = index < atoms ? 1 : 0;
  }
  const queriesToKeys = {
    indices: new Int32Array(keyRows), mask: new Float32Array(keyRows), count: keyRows,
  };
  for (let subset = 0; subset < subsets; subset += 1) {
    for (let key = 0; key < keys; key += 1) {
      const index = subset * keys + key;
      queriesToKeys.indices[index] = (subset * queries + (key * 5 + 3) % queries) % queryRows;
      queriesToKeys.mask[index] = key % 3 === 0 ? 1 : 0;
    }
  }
  const queriesToTokenAtoms = {
    indices: new Int32Array(atoms), mask: new Float32Array(atoms), count: atoms,
  };
  for (let index = 0; index < atoms; index += 1) {
    queriesToTokenAtoms.indices[index] = index % queryRows;
    queriesToTokenAtoms.mask[index] = atomMask[index];
  }
  const tokensToQueries = {
    indices: new Int32Array(queryRows), mask: new Float32Array(queryRows), count: queryRows,
  };
  for (let index = 0; index < queryRows; index += 1) {
    tokensToQueries.indices[index] = Math.floor(
      tokenAtomsToQueries.indices[index] / dense);
    tokensToQueries.mask[index] = index < atoms ? 1 : 0;
  }
  const tokensToKeys = {
    indices: new Int32Array(keyRows), mask: new Float32Array(keyRows), count: keyRows,
  };
  for (let index = 0; index < keyRows; index += 1) {
    tokensToKeys.indices[index] = index % tokens;
    // ...the KEY's own mask, not a derived one: see the note in the module.
    tokensToKeys.mask[index] = queriesToKeys.mask[index];
  }

  const conditioning = noise(atoms * channels, 1);
  const refPos = noise(atoms * 3, 3);
  const trunkSingleCond = withTrunk ? noise(tokens * trunkSingleChannels, 1) : undefined;
  const trunkPairCond = withTrunk
    ? noise(tokens * tokens * trunkPairChannels, 1) : undefined;
  const tokenAtomsAct = withTrunk ? noise(atoms * 3, 2) : undefined;

  const gl = createWebGL2Context();
  const execution = new Execution(gl);
  const held = [];
  const upload = (values, label) => {
    const tensor = execution.upload(Float32Array.from(values), label);
    held.push(tensor);
    return tensor;
  };
  const uploadTree = (value, label) => {
    if (value instanceof Float32Array) return upload(value, label);
    if (Array.isArray(value)) return value.map((v, i) => uploadTree(v, `${label}.${i}`));
    if (typeof value === "number") return value;
    return Object.fromEntries(Object.entries(value).map(
      ([key, v]) => [key, uploadTree(v, `${label}.${key}`)]));
  };
  const uploadedWeights = uploadTree(weights, "w.af3.encoder");
  const uploadGather = (g, name) => ({
    indices: upload(Float32Array.from(g.indices), `${name}.indices`),
    mask: upload(g.mask, `${name}.mask`),
  });
  const gathers = {
    tokenAtomsToQueries: uploadGather(tokenAtomsToQueries, "af3.encoder.ta2q"),
    queriesToKeys: uploadGather(queriesToKeys, "af3.encoder.q2k"),
    queriesToTokenAtoms: uploadGather(queriesToTokenAtoms, "af3.encoder.q2ta"),
    tokensToQueries: uploadGather(tokensToQueries, "af3.encoder.t2q"),
    tokensToKeys: uploadGather(tokensToKeys, "af3.encoder.t2k"),
  };
  // The control skews the per-atom conditioning, which every output reads.
  const conditioningTensor = upload(falsify
    ? Float32Array.from(conditioning,
      (v, i) => (i % channels < channels / 2 ? v * 1.01 + 0.01 : v))
    : conditioning, "af3.encoder.cond");
  const inputs = {
    conditioning: conditioningTensor,
    atomMask: upload(atomMask, "af3.encoder.atom-mask"),
    refPos: upload(refPos, "af3.encoder.ref-pos"),
    refSpaceUid: upload(refSpaceUid, "af3.encoder.ref-uid"),
    gathers,
    ...(withTrunk
      ? {
        trunkSingleCond: upload(trunkSingleCond, "af3.encoder.trunk-single"),
        trunkPairCond: upload(trunkPairCond, "af3.encoder.trunk-pair"),
        tokenAtomsAct: upload(tokenAtomsAct, "af3.encoder.positions"),
      }
      : {}),
  };

  const arms = [];
  const failures = [];
  const byDialect = {};
  /** @type {{name: string, maskPaddedKeys: boolean, keysUid: boolean}[]} */
  const CASES = [
    { name: "stock AF3", maskPaddedKeys: false, keysUid: false },
    { name: "openfold3 (masked padded keys)", maskPaddedKeys: true, keysUid: false },
    { name: "the keys' uid from the queries layout", maskPaddedKeys: false, keysUid: true },
  ];
  const ourStages = {};
  const theirStages = {};
  // `--combine=1`: the encoder's parts first met in a warm-up run, then drawn from combined
  // programs (Execution.combine), as a page's fold draws them.
  if ((args ?? []).includes("--combine=1")) {
    const warm = atomCrossAttentionEncoder(execution, {
      ...inputs, shape: { tokens, dense, subsets, queries, keys }, weights: uploadedWeights,
      maskPaddedKeys: CASES[0].maskPaddedKeys, keysUidFromQueriesLayout: CASES[0].keysUid,
      keyMaskedAtomAttention: false, chainedAtomLayerNorm: false, diffusionNoResidual: false,
      pairNormPerBlock: false, maskAtomActPerBlock: false,
    });
    for (const tensor of Object.values(warm)) execution.release(tensor);
    execution.combine([...execution.parts.keys()]);
  }
  for (const { name, maskPaddedKeys, keysUid } of CASES) {
    const out = atomCrossAttentionEncoder(execution, {
      ...inputs,
      shape: { tokens, dense, subsets, queries, keys },
      weights: uploadedWeights,
      maskPaddedKeys, keysUidFromQueriesLayout: keysUid,
      keyMaskedAtomAttention: false, chainedAtomLayerNorm: false,
      diffusionNoResidual: false, pairNormPerBlock: false,
      maskAtomActPerBlock: false,
      onStage: (stage, tensor) => {
        ourStages[stage] = Float32Array.from(execution.read(tensor));
      },
    });
    const mine = Float32Array.from(execution.read(out.tokenAct));
    byDialect[name] = mine;
    for (const tensor of Object.values(out)) execution.release(tensor);

    const dialect = {
      maskPaddedKeys, keysUidFromQueriesLayout: keysUid,
    };
    const expected = cpuEncoder({
      shape: { tokens, dense, subsets, queries, keys },
      dialect,
      // 🔴 THE UNPERTURBED CONDITIONING, BECAUSE A CONTROL THAT FEEDS BOTH
      // ARMS IS NOT A CONTROL. Passing the skew here too reported "0 of 15" -
      // this repository's "comparing something with itself" trap wearing the
      // clothes of a control, which its own kernels gate has a note about.
      conditioning,
      atomMask, refPos, refSpaceUid,
      tokenAtomsToQueries, queriesToKeys, queriesToTokenAtoms,
      tokensToQueries, tokensToKeys,
      ...(withTrunk ? { trunkSingleCond, trunkPairCond, tokenAtomsAct } : {}),
    }, {
      ...weights,
      keyMaskedAtomAttention: false, chainedAtomLayerNorm: false,
      diffusionNoResidual: false, pairNormPerBlock: false,
      blocks: weights.blocks.map((b) => ({
        ...b, keyMaskedAtomAttention: false, chainedAtomLayerNorm: false,
        diffusionNoResidual: false, maskAtomActPerBlock: false,
      })),
    }, (stage, values) => { theirStages[stage] = Float32Array.from(values); });
    // 🔴 THE STAGE TAPS, which is what turns "1.7e-3 somewhere" into a line
    // number. AF3 records its own at these five points and the reference passes
    // them through `onStage`; this port now does too.
    if (name === "stock AF3") {
      for (const stage of Object.keys(ourStages)) {
        const theirs = theirStages[stage];
        if (theirs === undefined) continue;
        const stageRms = relativeRms(ourStages[stage], theirs, `stage.${stage}`);
        // 🔴 THREE OF THESE TAPS ARE PURE INDEX GATHERS and cannot move under a
        // perturbation of the CONDITIONING: the two reference-space uids and the
        // key mask are functions of the layout alone. Declaring that beats
        // inventing a control that appears to move them - and these three are
        // exactly where the two uid readings are compared, which is worth having
        // whether or not a control can reach them.
        const indexOnly = ["encoder.queriesUid", "encoder.keysUid", "encoder.keysMask"]
          .includes(stage);
        arms.push({ arm: `stage ${stage}`,
          relativeRms: Number(stageRms.toPrecision(4)), bar, ok: stageRms <= bar,
          ...(indexOnly ? {
            falsifiable: false,
            reason: "a gather of the layout alone; the conditioning does not reach it",
          } : {}) });
        if (!(stageRms <= bar)) {
          failures.push(`stage ${stage}: relRMS ${stageRms.toExponential(3)} over ${bar}`);
        }
      }
    }
    const relative = relativeRms(mine, expected.tokenAct, name);
    const ok = relative <= bar;
    arms.push({ arm: `${name} vs the f64 reference`,
      relativeRms: Number(relative.toPrecision(4)), bar, ok });
    if (!ok) failures.push(`${name}: relRMS ${relative.toExponential(3)} over ${bar}`);
  }

  // 🔴 AND THE TWO DIALECT CHOICES MUST MOVE THE ANSWER, or this layout cannot
  // see them - which it could not if every key were real or every atom shared
  // one reference space.
  const dialectMoves = {};
  for (const { name } of CASES.slice(1)) {
    dialectMoves[name] = Number(relativeRms(
      byDialect[name], byDialect["stock AF3"], `moves.${name}`).toPrecision(3));
    if (!(dialectMoves[name] > 1e-4)) {
      failures.push(`${name} is INERT on this layout: ${dialectMoves[name]}`);
    }
  }

  for (const tensor of held) execution.release(tensor);
  if (execution.stats().live !== 0) {
    failures.push(`${execution.stats().live} tensors left live`);
  }
  const falsifiable = arms.filter((row) => row.falsifiable !== false);
  const red = falsifiable.filter((row) => !row.ok).length;
  return {
    shape: {
      tokens, dense, subsets, queries, keys, channels, pairChannels,
      perTokenChannels, heads, dimension, blocks,
    },
    withTrunk,
    paddedAtoms: atoms - atomMask.reduce((a, b) => a + b, 0),
    paddedKeySlots: keyRows - queriesToKeys.mask.reduce((a, b) => a + b, 0),
    layout: "synthetic and irregular; the real batch's gathers are not on this box",
    bar,
    falsify,
    arms,
    dialectMoves,
    memory: execution.stats(),
    verdict: falsify
      ? (red === falsifiable.length
        ? `ok: all ${falsifiable.length} falsifiable arms go red on a 1% perturbation`
        : `VACUOUS: only ${red} of ${falsifiable.length} falsifiable arms noticed it`)
      : (failures.length === 0
        ? `ok: ${arms.length} dialects against the f64 reference, each moving the answer`
        : `FAILED: ${failures.join("; ")}`),
  };
}
