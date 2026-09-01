/**
 * The eight-iteration structure core: what it computes, and what it costs.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-structure-core.js
 *     node tools/gpu-chrome.mjs tools/gpu/check-structure-core.js --length=221 --repeats=5
 *
 * 🔴 THIS EXISTS BECAUSE THE STRUCTURE MODULE HAS NO RUNNABLE CHECK ON THIS
 * MACHINE. test/structure-core.gpu.test.js compares eight iterations against
 * AlphaFold's own recorded stage-7 state, which is the reference that matters -
 * but it runs through Dawn, which does not load on this macOS, AND its fixture
 * lives under test/fixtures/evoformer/, which is gitignored and only partly
 * present. So neither half of it can run here. Nothing should be restructured
 * under that, and the loop is about to be.
 *
 * What this can do instead is hold a CHANGE to the loop against the loop as it
 * stands, on deterministic inputs at a real shape. That is a weaker claim than
 * the Dawn test makes - it cannot tell you the structure module is right, only
 * that it still does what it did - which is exactly the claim a refactor whose
 * point is submission structure rather than arithmetic needs to support.
 *
 * 🔴 AND IT TIMES BOTH PATHS IN ONE PROCESS. This machine drifts up to 3.2x
 * between invocations, so a number here cannot be compared with a number from a
 * separate run. See tools/gpu/bench-ab.js.
 */
import { StructureCoreGpu } from "../../src/structure/core.js";
import { InvariantPointAttentionGpu } from "../../src/structure/ipa.js";
import { StructurePostAttentionGpu } from "../../src/structure/iteration.js";
import { errorMetrics } from "../../src/triangle/types.js";

const HEADS = 12;
const SCALAR_QK = 16;
const SCALAR_V = 16;
const POINT_QK = 4;
const POINT_V = 8;
const CHANNELS = 384;
const PAIR_CHANNELS = 128;

/** The same generator the AF3 checkers use, so shapes are reproducible. */
function deterministic(length, seed) {
  let state = seed >>> 0;
  const output = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    output[index] = ((((value ^ (value >>> 14)) >>> 0) / 0x1_0000_0000) * 2 - 1) * 0.1;
  }
  return output;
}

function buildInput(length) {
  const queryScalarColumns = HEADS * SCALAR_QK;
  const kvScalarColumns = HEADS * (SCALAR_QK + SCALAR_V);
  const queryPointColumns = HEADS * 3 * POINT_QK;
  const kvPointColumns = HEADS * 3 * (POINT_QK + POINT_V);
  const featureChannels = HEADS * SCALAR_V + 4 * HEADS * POINT_V + HEADS * PAIR_CHANNELS;
  let seed = 1;
  const w = (elements) => deterministic(elements, (seed += 977));
  // A layer norm's scale sits around one, not around zero: a scale drawn from
  // the same distribution as a weight makes the normalisation degenerate and
  // hides differences the loop would otherwise show.
  const scale = (elements) => deterministic(elements, (seed += 977)).map((v) => 1 + v);

  // 🔴 THE FRAMES MUST BE UNIT QUATERNIONS. affine is [quaternion(4),
  // translation(3)] per residue, and the invariant point attention rotates by
  // it; a random quaternion scales every point by its norm, which does not
  // fail, it just measures something else.
  const affine = new Float32Array(length * 7);
  const raw = deterministic(length * 4, 31);
  for (let residue = 0; residue < length; residue += 1) {
    let norm = 0;
    for (let k = 0; k < 4; k += 1) norm += raw[residue * 4 + k] ** 2;
    norm = Math.sqrt(norm) || 1;
    for (let k = 0; k < 4; k += 1) affine[residue * 7 + k] = raw[residue * 4 + k] / norm;
    for (let k = 0; k < 3; k += 1) affine[residue * 7 + 4 + k] = deterministic(3, residue + 7)[k] * 10;
  }

  return {
    activations: deterministic(length * CHANNELS, 11),
    pair: deterministic(length * length * PAIR_CHANNELS, 13),
    mask: new Float32Array(length).fill(1),
    affine,
    length, channels: CHANNELS, pairChannels: PAIR_CHANNELS,
    ipaWeights: {
      pairNormScale: scale(PAIR_CHANNELS), pairNormOffset: w(PAIR_CHANNELS),
      queryScalarWeight: w(CHANNELS * queryScalarColumns), queryScalarBias: w(queryScalarColumns),
      keyValueScalarWeight: w(CHANNELS * kvScalarColumns), keyValueScalarBias: w(kvScalarColumns),
      queryPointWeight: w(CHANNELS * queryPointColumns), queryPointBias: w(queryPointColumns),
      keyValuePointWeight: w(CHANNELS * kvPointColumns), keyValuePointBias: w(kvPointColumns),
      trainablePointWeights: w(HEADS),
      attention2dWeight: w(PAIR_CHANNELS * HEADS), attention2dBias: w(HEADS),
      outputWeight: w(featureChannels * CHANNELS), outputBias: w(CHANNELS),
    },
    postAttentionWeights: {
      attentionNormScale: scale(CHANNELS), attentionNormOffset: w(CHANNELS),
      transitionWeights: [w(CHANNELS * CHANNELS), w(CHANNELS * CHANNELS), w(CHANNELS * CHANNELS)],
      transitionBiases: [w(CHANNELS), w(CHANNELS), w(CHANNELS)],
      transitionNormScale: scale(CHANNELS), transitionNormOffset: w(CHANNELS),
      affineWeight: w(CHANNELS * 6), affineBias: w(6),
    },
  };
}

/**
 * The loop as it ran before it shared a command buffer: submit and read back
 * per operation, upload the result again for the next one.
 *
 * 🔴 THIS IS THE REFERENCE, AND IT LIVES HERE RATHER THAN IN src/. The Dawn
 * test against AlphaFold's own state cannot run on this machine, so the only
 * available check on the encoded loop is that it still computes what the
 * iterated one did. Building it from ipa.run and post.run - which are public,
 * unchanged, and still covered by their own tests - keeps that reference honest
 * without leaving a second loop in the production path to rot.
 */
async function iteratedLoop(device, input, iterations = 8) {
  const ipa = new InvariantPointAttentionGpu(device);
  const post = new StructurePostAttentionGpu(device);
  const geometry = {
    pair: input.pair, mask: input.mask, length: input.length, channels: input.channels,
    pairChannels: input.pairChannels, heads: HEADS, scalarQk: SCALAR_QK, scalarV: SCALAR_V,
    pointQk: POINT_QK, pointV: POINT_V, weights: input.ipaWeights,
  };
  let activations = input.activations;
  let affine = input.affine;
  const prepared = await ipa.prepare(geometry);
  try {
    for (let iteration = 0; iteration < iterations; iteration += 1) {
      const attention = await ipa.run({ ...geometry, activations, affine, prepared });
      const update = await post.run({
        activations, attentionUpdate: attention.output, affine,
        length: input.length, channels: input.channels, weights: input.postAttentionWeights,
      });
      activations = update.activations;
      affine = update.affine;
    }
  } finally { prepared.release(); }
  return { activations, affine };
}

export async function main(device, args) {
  const option = (name, fallback) =>
    Number(args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback);
  const length = option("length", 59);
  const repeats = option("repeats", 3);

  const input = buildInput(length);
  const core = new StructureCoreGpu(device);
  const iterated = args.includes("--no-reference") ? false : true;

  // Warm the pipeline cache so the first timing is not a compile.
  const reference = await core.run(input);
  const legacy = iterated ? await iteratedLoop(device, input) : null;
  if (iterated) await iteratedLoop(device, input);

  // 🔴 INTERLEAVED IN ONE PROCESS. This machine drifts up to 3.2x between
  // invocations, so timing the encoded loop in one run and the iterated loop in
  // another measures the machine, not the change. A whole round of AF3
  // profiling was thrown away to that once - see AF3.md.
  const times = [];
  const legacyTimes = [];
  let latest = reference;
  for (let repeat = 0; repeat < repeats; repeat += 1) {
    const started = performance.now();
    latest = await core.run(input);
    times.push(performance.now() - started);
    if (!iterated) continue;
    const legacyStarted = performance.now();
    await iteratedLoop(device, input);
    legacyTimes.push(performance.now() - legacyStarted);
  }
  times.sort((a, b) => a - b);
  legacyTimes.sort((a, b) => a - b);
  const median = times[Math.floor(times.length / 2)];
  const legacyMedian = legacyTimes.length === 0 ? null
    : legacyTimes[Math.floor(legacyTimes.length / 2)];

  // 🔴 THE LOOP MUST BE DETERMINISTIC RUN TO RUN, and this is what says so.
  // Once the iterations share one command buffer, a missing barrier between
  // dispatches shows up as a result that changes between runs rather than as an
  // error - so a repeat that disagrees with the first is the signal.
  const drift = errorMetrics(latest.activations, reference.activations);
  const affineDrift = errorMetrics(latest.affine, reference.affine);
  const stable = drift.maxAbsoluteError === 0 && affineDrift.maxAbsoluteError === 0;

  let finite = true;
  for (const value of latest.activations) if (!Number.isFinite(value)) { finite = false; break; }
  for (const value of latest.affine) if (!Number.isFinite(value)) { finite = false; break; }

  console.log(`structure core   ${length} residues x ${CHANNELS} channels, 8 iterations`);
  console.log(`  act    [${latest.activations.length}]  mean `
    + `${(latest.activations.reduce((s, v) => s + Math.abs(v), 0) / latest.activations.length).toExponential(3)}`);
  console.log(`  affine [${latest.affine.length}]`);
  console.log(`  ${stable ? "ok  " : "FAIL"} identical across ${repeats + 1} runs`
    + `   (act ${drift.maxAbsoluteError.toExponential(1)},`
    + ` affine ${affineDrift.maxAbsoluteError.toExponential(1)})`);
  console.log(`  ${finite ? "ok  " : "FAIL"} all finite`);
  if (legacy !== null) {
    const act = errorMetrics(latest.activations, legacy.activations);
    const aff = errorMetrics(latest.affine, legacy.affine);
    const same = act.maxAbsoluteError === 0 && aff.maxAbsoluteError === 0;
    console.log(`  ${same ? "ok  " : "FAIL"} identical to the iterated loop`
      + `   (act ${act.maxAbsoluteError.toExponential(1)},`
      + ` affine ${aff.maxAbsoluteError.toExponential(1)})`);
  }
  console.log(`  encoded  ${median.toFixed(1)} ms median of ${repeats}`
    + `   [${times.map((t) => t.toFixed(1)).join(", ")}]`);
  if (legacyMedian !== null) {
    console.log(`  iterated ${legacyMedian.toFixed(1)} ms median of ${repeats}`
      + `   [${legacyTimes.map((t) => t.toFixed(1)).join(", ")}]`);
    console.log(`  ${(legacyMedian / median).toFixed(2)}x, interleaved in one process`);
  }

  return {
    ok: stable && finite, medianMs: median, times, length,
    // Returned so a later run can be diffed against this one by value, which is
    // how the refactor is held to "same numbers, fewer submissions".
    actDigest: Array.from(latest.activations.slice(0, 8)),
    affineDigest: Array.from(latest.affine.slice(0, 8)),
  };
}
