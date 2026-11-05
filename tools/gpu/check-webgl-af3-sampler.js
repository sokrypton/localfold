/**
 * AF3's diffusion sampler in WebGL2: a whole walk, against the f64 reference's.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-af3-sampler.js [--falsify]
 *
 * 🔴 WHAT A GATE CAN ASK HERE IS NOT "DOES IT MATCH AF3" - the reference says
 * why: "Every step draws from JAX's threefry PRNG, which this does not
 * implement, so our sample is a different draw from the same distribution." The
 * arithmetic is held to AF3 by `tools/oracle/check_af3_sampler.js`, which feeds
 * it AF3's own noisy positions and denoised answer. So this gate asks the two
 * questions a backend port can answer:
 *
 *  1. **A WHOLE TRAJECTORY, TWO DENOISERS.** The same loop with the same draws,
 *     once denoising on the device and once in f64 on the host. This is the only
 *     measurement of what a WALK does to the head's 2e-7: the step divides by
 *     `tHat` and multiplies by 1.5, so the question is whether that compounds.
 *  2. **TWO LOOPS, ONE DENOISER.** `sampleOnWebGl2` against the reference's own
 *     `sample()`, both driving the device head, on the plain path - which must be
 *     **EXACTLY** equal, because the arithmetic is imported rather than restated
 *     and the draws are consumed in the same order. That is what makes this
 *     port's extra options (a warm start, a prefix, a restraint, a motif)
 *     falsifiable: an off-by-one in the levels, a gamma threshold on the wrong
 *     side or a normal drawn out of turn all show up as a non-zero here.
 *
 * 🔴 AND THE OPTIONS ARE CHECKED WITH A STUB DENOISER, ON PURPOSE. Each of them
 * is loop arithmetic, so a cheap analytic denoiser makes the arm sharp and fast;
 * driving the head would spend 963 draws a step to measure a multiplication. The
 * separations assert each option MOVES the answer, which is what says the arm
 * reached the code path at all - this repository's own rule: "A control arm that
 * cannot vary the thing under test is not a control."
 *
 * 🔴 THE CHURN IS THE ONE WORTH READING. `gamma_0` is 0.8 above `gamma_min` 1.0
 * and zero below, so "a run that churns everywhere, or nowhere, is wrong in a
 * way that still produces a protein-shaped answer". The schedule here crosses
 * that threshold, which the gate reports: with four steps from 2560 A the levels
 * are 2560, 325, 29.5, 1.03 and 0.0064, so the last step alone has gamma 0.
 */
import {
  atomCrossAttentionEncoder as cpuEncoder,
} from "../../cpu/af3/diffusion/atom-encoder.js";
import { diffusionHead as cpuHead } from "../../cpu/af3/diffusion/diffusion.js";
import { noiseLevels, sample } from "../../cpu/af3/diffusion/diffusion-sampler.js";
import { denoiserFromHead, sampleOnWebGl2 } from "../../webgl/af3/diffusion/sampler.js";
import { RELATIVE_WIDTH, tokenIdentity } from "../../webgl/af3/trunk/embedder.js";
import { Execution } from "../../webgl/runtime/execution.js";
import { createWebGL2Context } from "../../webgl/index.js";
import { generator } from "./af2-synthetic-block.js";
import { syntheticHead } from "./af3-synthetic-head.js";
import { relativeRms } from "./relative-rms.js";

const flag = (args, name) => (args ?? []).includes(`--${name}`);
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

/**
 * One standard gaussian a call, from this repository's own generator by
 * Box-Muller - deterministic, so two arms consume the same sequence.
 */
export function gaussians(seed) {
  const random = generator(seed);
  let spare = null;
  return () => {
    if (spare !== null) {
      const value = spare;
      spare = null;
      return value;
    }
    const u = Math.max(1e-12, random());
    const v = random();
    const radius = Math.sqrt(-2 * Math.log(u));
    spare = radius * Math.sin(2 * Math.PI * v);
    return radius * Math.cos(2 * Math.PI * v);
  };
}

export async function main(device, args) {
  const falsify = flag(args, "falsify");
  // 🔴 THE BAR IS THE MEASURED FLOOR, AND MY FIRST GUESS OF 2e-5 WAS LOOSE ENOUGH
  // TO SWALLOW A CONTROL. The walks read 7.4e-8 to 1.7e-7, and under --falsify
  // the geometric-schedule walk reads 1.95e-5 - so at 2e-5 that arm came back
  // green on a 1% perturbation of the denoiser and the gate said "VACUOUS: only 4
  // of 5". A WALK ATTENUATES: the last step lands on the prediction, so a 1%
  // error in the head reaches the sample as 2e-5 to 5e-3, two to three orders
  // down. That is the reason the per-stage head gate exists - a whole-walk gate
  // is a blunt instrument for the denoiser, and only a floor-tight bar makes it
  // an instrument at all.
  const bar = Number(option(args, "bar", "1e-6"));
  const steps = Number(option(args, "steps", "4"));
  const seed = Number(option(args, "seed", "271828"));
  const fixture = syntheticHead({ padDna: flag(args, "pad-unknown-dna") });
  const { tokens, dense, subsets, queries, keys } = fixture.shape;
  const { weights: headWeights, fourierWeight, fourierBias, padDna } = fixture;
  const { atomMask, refSpaceUid, features } = fixture.layout;
  const {
    tokenAtomsToQueries, queriesToKeys, queriesToTokenAtoms, tokensToQueries, tokensToKeys,
  } = fixture.layout;
  const {
    conditioning, refPos, trunkSingle, trunkPair, targetFeat, seqMask,
  } = fixture.values;
  const { trunkPairChannels, targetFeatWidth, channels } = fixture.widths;
  const atoms = tokens * dense;

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
    if (typeof value === "number" || typeof value === "boolean") return value;
    return Object.fromEntries(Object.entries(value).map(
      ([key, v]) => [key, uploadTree(v, `${label}.${key}`)]));
  };
  const uploaded = uploadTree(headWeights, "w.af3.head");
  // 🔴 THE SKEW GOES TO THE DEVICE ALONE, because a control that feeds both
  // sides is not a control - and it has to reach the pair track, the single
  // track and the atom stack, which no single input does.
  const skew = (values, period) => Float32Array.from(values,
    (v, i) => (i % period < period / 2 ? v * 1.01 + 0.01 : v));
  const uploadGather = (g, name) => ({
    indices: upload(Float32Array.from(g.indices), `${name}.indices`),
    mask: upload(g.mask, `${name}.mask`),
  });
  const gathers = {
    tokenAtomsToQueries: uploadGather(tokenAtomsToQueries, "af3.sampler.ta2q"),
    queriesToKeys: uploadGather(queriesToKeys, "af3.sampler.q2k"),
    queriesToTokenAtoms: uploadGather(queriesToTokenAtoms, "af3.sampler.q2ta"),
    tokensToQueries: uploadGather(tokensToQueries, "af3.sampler.t2q"),
    tokensToKeys: uploadGather(tokensToKeys, "af3.sampler.t2k"),
  };
  const dialect = {
    maskPaddedKeys: false, keysUidFromQueriesLayout: false,
    keyMaskedAtomAttention: false, chainedAtomLayerNorm: false,
    diffusionNoResidual: false, pairNormPerBlock: false, maskAtomActPerBlock: false,
  };
  const deviceInput = {
    conditioning: upload(
      falsify ? skew(conditioning, channels) : conditioning, "af3.sampler.cond"),
    atomMask: upload(atomMask, "af3.sampler.atom-mask"),
    seqMask: upload(seqMask, "af3.sampler.seq-mask"),
    refPos: upload(refPos, "af3.sampler.ref-pos"),
    refSpaceUid: upload(refSpaceUid, "af3.sampler.ref-uid"),
    trunkSingle: upload(trunkSingle, "af3.sampler.trunk-single"),
    trunkPair: upload(
      falsify ? skew(trunkPair, trunkPairChannels) : trunkPair, "af3.sampler.trunk-pair"),
    targetFeat: upload(
      falsify ? skew(targetFeat, targetFeatWidth) : targetFeat, "af3.sampler.target"),
    identity: upload(tokenIdentity(features, tokens), "af3.sampler.identity"),
    gathers,
    shape: { tokens, dense, subsets, queries, keys },
    weights: uploaded,
    relativeWidth: RELATIVE_WIDTH,
    fourierWeight, fourierBias, padUnknownDna: padDna,
    ...dialect,
  };
  const gpuDenoise = denoiserFromHead(execution, deviceInput);
  const hostInput = {
    shape: { tokens, dense, subsets, queries, keys },
    dialect: { ...dialect, padSingleCondUnknownDna: padDna },
    features, atomMask, seqMask, trunkSingle, trunkPair, targetFeat,
    conditioning, refPos, refSpaceUid,
    tokenAtomsToQueries, queriesToKeys, queriesToTokenAtoms, tokensToQueries, tokensToKeys,
  };
  const cpuDenoise = (positions, noiseLevel) => cpuHead(
    { ...hostInput, positionsNoisy: positions, noiseLevel }, headWeights, cpuEncoder);

  const arms = [];
  const failures = [];
  const separations = [];
  const counts = [];
  const refusals = [];

  /** @type {{name: string, options: object}[]} */
  const CASES = [
    { name: "AF3's own (karras, euler, churn)", options: {} },
    { name: "the exponential update (Blitz's)", options: { update: "exponential" } },
    { name: "a geometric schedule", options: { schedule: "geometric" } },
    { name: "no churn", options: { churn: false } },
    { name: "boltz2's realigned step", options: { realign: true } },
  ];
  for (const { name, options } of CASES) {
    const common = { steps, mask: atomMask, ...options };
    const mine = sampleOnWebGl2(
      execution, deviceInput, { ...common, normal: gaussians(seed), denoise: gpuDenoise });
    const theirs = sampleOnWebGl2(
      execution, deviceInput, { ...common, normal: gaussians(seed), denoise: cpuDenoise });
    const relative = relativeRms(mine, theirs, name);
    const ok = relative <= bar;
    arms.push({
      arm: `a whole walk, ${name}`,
      relativeRms: Number(relative.toPrecision(4)), bar, ok,
    });
    if (!ok) failures.push(`${name}: relRMS ${relative.toExponential(3)} over ${bar}`);
  }

  // 🔴 TWO LOOPS, ONE DENOISER - and this one is EXACTLY equal or it is a bug.
  {
    const mine = sampleOnWebGl2(execution, deviceInput, {
      steps, mask: atomMask, normal: gaussians(seed), denoise: gpuDenoise,
    });
    const theirs = sample(gpuDenoise, {
      atoms, mask: atomMask, steps, normal: gaussians(seed),
    });
    const relative = relativeRms(mine, theirs, "two loops");
    const ok = relative === 0;
    arms.push({
      arm: "this loop against the reference's own `sample()`, same device denoiser",
      relativeRms: relative, bar: 0, ok,
      falsifiable: false,
      reason: "both arms drive the SAME denoiser, so a skewed head moves neither;"
        + " what it falsifies is the loop, and the separations below are what move that",
    });
    if (!ok) {
      failures.push(
        `two loops over one denoiser differ by ${relative} - the arithmetic is imported,`
        + " so this is an off-by-one in the levels, a gamma threshold on the wrong side,"
        + " or a normal drawn out of turn");
    }
  }

  // 🔴 THE OPTIONS, WITH A STUB DENOISER: each must MOVE the answer.
  // A contraction toward a fixed target, which is what a denoiser is.
  const target = Float32Array.from({ length: atoms * 3 }, (_, i) => Math.sin(i * 0.7) * 4);
  let stubCalls = 0;
  const stub = (positions, noiseLevel) => {
    stubCalls += 1;
    const out = new Float32Array(positions.length);
    const keep = Math.min(0.9, noiseLevel / (noiseLevel + 16));
    for (let i = 0; i < out.length; i += 1) {
      out[i] = keep * positions[i] + (1 - keep) * target[i];
    }
    return out;
  };
  const walk = (options) => {
    stubCalls = 0;
    const values = sampleOnWebGl2(execution, deviceInput, {
      steps, mask: atomMask, normal: gaussians(seed), denoise: stub, ...options,
    });
    return { values, calls: stubCalls };
  };
  const baseline = walk({});
  const motif = {
    slots: Int32Array.from([0, 1, 2, 5]),
    xyz: Float32Array.from([0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3]),
    until: 0,
  };
  /** @type {{name: string, options: object, floor?: number}[]} */
  const OPTIONS = [
    { name: "no churn", options: { churn: false } },
    { name: "an Euler step of 1.0 rather than 1.5", options: { stepScale: 1 } },
    { name: "the exponential update", options: { update: "exponential" } },
    { name: "a geometric schedule", options: { schedule: "geometric" } },
    { name: "boltz2's realigned step", options: { realign: true } },
    { name: "a held motif", options: { motif } },
    { name: "boltz2's gamma_0 of 0.605", options: { gamma0: 0.605 } },
    { name: "rho 8 rather than 7", options: { rho: 8 } },
  ];
  // 🔴 THE RESTRAINT NEEDS SOMETHING TO PULL, AND ITS OWN BASELINE. Its step is
  // clamped by the contact energy - `length = min(rate, largest)` - so against a
  // target where the binder is already in contact it moves 5.4e-4 and reads as
  // inert. With the two groups 80 A apart it has work to do, and the baseline is
  // the SAME stub without the restraint: a separation measured against a
  // different denoiser would be measuring the denoiser.
  const splitTarget = Float32Array.from({ length: atoms * 3 },
    (_, i) => (i % 3 === 0 ? (Math.floor(i / 3) < atoms / 2 ? 40 : -40) : Math.sin(i) * 2));
  const splitStub = (positions, noiseLevel) => {
    stubCalls += 1;
    const out = new Float32Array(positions.length);
    const keep = Math.min(0.9, noiseLevel / (noiseLevel + 16));
    for (let i = 0; i < out.length; i += 1) {
      out[i] = keep * positions[i] + (1 - keep) * splitTarget[i];
    }
    return out;
  };
  // 🔴 AND IT IS READ ON THE FRAME IT DISPLACES, NOT ON THE SAMPLE. "A restraint
  // displaces the PREDICTION, not the walk", and at four steps the sample barely
  // sees it: measured 5.5e-4 on the final positions, because the restraint's own
  // schedule is ZERO over the last quarter of the steps - where `positions` and
  // `denoised` have nearly converged - and on the early steps the update's
  // coefficient on a 2 A translation is small. Reading the sample would have
  // reported the mechanism inert when it is working exactly as written.
  const firstDenoised = (options) => {
    let frame = null;
    walk({ ...options, onStep: (f) => { frame = frame ?? f.denoised; } });
    return frame;
  };
  // 🔴 AND IT IS MEASURED IN ANGSTROMS, NOT AS A RATIO. A relRMS cannot see this
  // at all: the first prediction is a cloud at sigma 2560, so a 2 A translation
  // of half the atoms is 2e-4 of it and reads as inert - and on the sample it is
  // 5.5e-4. What the module claims is geometric and so is the arm: the BINDER
  // moves as one body by at most `weight` angstroms and the target does not move.
  {
    const plain = firstDenoised({ denoise: splitStub });
    const pulled = firstDenoised({
      denoise: splitStub, steer: { binder: [0, 1], pocket: [2, 3], weight: 2 },
    });
    const shift = (tokenList) => {
      let total = 0;
      let count = 0;
      for (const token of tokenList) {
        for (let slot = 0; slot < dense; slot += 1) {
          const flat = token * dense + slot;
          if (!atomMask[flat]) continue;
          total += Math.hypot(
            pulled[flat * 3] - plain[flat * 3],
            pulled[flat * 3 + 1] - plain[flat * 3 + 1],
            pulled[flat * 3 + 2] - plain[flat * 3 + 2]);
          count += 1;
        }
      }
      return count === 0 ? 0 : total / count;
    };
    const binderShift = shift([0, 1]);
    const pocketShift = shift([2, 3]);
    // One body: every live binder atom by the same vector, so the mean IS the
    // translation, and never past the step's own ceiling of `weight`. 🔴 THE
    // CEILING NEEDS AN f32 ALLOWANCE DERIVED FROM THE COORDINATE SCALE, NOT A
    // ROUND NUMBER: these positions run to 40 A, where one ULP is 2.4e-6, and at
    // `--seed=99173` the measured translation came out a few ULPs ABOVE 2.0 and
    // failed a hard `<= 2 + 1e-5`. The allowance is eight ULPs of the largest
    // coordinate either frame holds - which cannot hide a second translation.
    const largestCoordinate = Math.max(
      ...Array.from(plain, Math.abs), ...Array.from(pulled, Math.abs));
    const allowance = 8 * 1.1920929e-7 * largestCoordinate;
    const moves = binderShift > 0.5 && binderShift <= 2 + allowance
      && pocketShift <= allowance;
    separations.push({
      option: "a contact restraint: the binder translates, the target does not",
      binderShiftAngstroms: Number(binderShift.toPrecision(8)),
      pocketShiftAngstroms: Number(pocketShift.toPrecision(4)),
      ceiling: 2,
      f32Allowance: Number(allowance.toPrecision(3)),
      moves,
    });
    if (!moves) {
      failures.push(
        `a contact restraint moved the binder ${binderShift.toPrecision(4)} A and the`
        + ` target ${pocketShift.toPrecision(4)} A - it should translate the binder by`
        + " up to 2 A as one body and leave the target alone");
    }
  }

  for (const { name, options } of OPTIONS) {
    const moved = relativeRms(walk(options).values, baseline.values, `moves.${name}`);
    const moves = moved > 1e-3;
    separations.push({ option: name, relativeRms: Number(moved.toPrecision(3)), moves });
    if (!moves) {
      failures.push(`${name} is INERT on this schedule: ${moved} - the arm did not reach it`);
    }
  }

  // 🔴 THE SCHEDULE'S LENGTH AND THE NUMBER OF DENOISER CALLS ARE NOT THE SAME
  // THING, and each vendor's tail is a different count. The loop runs what the
  // LEVELS hold, so this is the arm that catches the off-by-one.
  for (const scheduleTail of [0, 1, 2, 3]) {
    const levels = noiseLevels(steps, { scheduleTail });
    const run = walk({ scheduleTail });
    const expected = levels.length - 1;
    const ok = run.calls === expected;
    counts.push({
      scheduleTail, levels: levels.length, calls: run.calls, expected, ok,
      last: Number(levels[levels.length - 1].toPrecision(4)),
    });
    if (!ok) failures.push(`scheduleTail ${scheduleTail}: ${run.calls} calls, not ${expected}`);
  }
  {
    const prefix = walk({ stopAfter: 2 });
    const ok = prefix.calls === 2;
    counts.push({ stopAfter: 2, calls: prefix.calls, expected: 2, ok });
    if (!ok) failures.push(`stopAfter 2 ran ${prefix.calls} denoiser calls`);
  }
  {
    // 🔴 A WARM START SKIPS THE LARGE NOISE, IT DOES NOT RESCALE THE SCHEDULE:
    // the walk joins the levels a cold run visits, so the number of calls is the
    // number of levels at or below the one it joins.
    const levels = noiseLevels(steps, {});
    const level = levels[2];
    const first = levels.findIndex((sigma) => sigma <= level);
    const warm = walk({ start: { positions: target, level } });
    const expected = levels.length - 1 - first;
    const ok = warm.calls === expected && first === 2;
    counts.push({
      warmStart: Number(level.toPrecision(4)), joinedAt: first,
      calls: warm.calls, expected, ok,
    });
    if (!ok) failures.push(`a warm start at ${level} ran ${warm.calls} calls, not ${expected}`);
    const moved = relativeRms(warm.values, baseline.values, "moves.warm");
    separations.push({
      option: "a warm start at the third level",
      relativeRms: Number(moved.toPrecision(3)), moves: moved > 1e-3,
    });
    if (!(moved > 1e-3)) failures.push("a warm start is INERT");
  }

  // The refusals, each of which is a wrong answer that would otherwise stay
  // finite and protein-shaped.
  const refuses = (what, pattern, run) => {
    let message = null;
    try {
      run();
    } catch (error) {
      message = String(error.message);
    }
    const ok = message !== null && message.includes(pattern);
    refusals.push({ refusal: what, ok, said: message ?? "did not throw" });
    if (!ok) failures.push(`${what} was NOT refused: ${message ?? "no throw"}`);
  };
  refuses("the atom mask given as a device tensor rather than a host array", "HOST array",
    () => sampleOnWebGl2(execution, deviceInput, {
      steps, mask: deviceInput.atomMask, normal: gaussians(seed), denoise: stub,
    }));
  refuses("a warm start whose coordinates are the wrong length", "coordinates",
    () => walk({ start: { positions: new Float32Array(3), level: 10 } }));
  refuses("a warm start with no noise level", "noise level in angstroms",
    () => walk({ start: { positions: target, level: 0 } }));
  refuses("a warm start below the schedule's last level", "at or below",
    () => walk({ start: { positions: target, level: 1e-9 } }));
  refuses("positions of the wrong length handed to the device denoiser", "not 16 x 3",
    () => gpuDenoise(new Float32Array(9), 4.5));

  for (const tensor of held) execution.release(tensor);
  if (execution.stats().live !== 0) {
    failures.push(`${execution.stats().live} tensors left live`);
  }
  const falsifiable = arms.filter((row) => row.falsifiable !== false);
  const red = falsifiable.filter((row) => !row.ok).length;
  return {
    shape: { tokens, dense, subsets, queries, keys, atoms },
    steps,
    levels: Array.from(noiseLevels(steps, {}), (v) => Number(v.toPrecision(4))),
    churnThreshold: 1.0,
    bar,
    falsify,
    arms,
    separations,
    counts,
    refusals,
    memory: execution.stats(),
    verdict: falsify
      ? (red === falsifiable.length
        ? `ok: all ${falsifiable.length} falsifiable arms go red on a 1% perturbation`
        : `VACUOUS: only ${red} of ${falsifiable.length} falsifiable arms noticed it`)
      : (failures.length === 0
        ? `ok: ${arms.length} walks, ${separations.length} separations,`
          + ` ${counts.length} call counts and ${refusals.length} refusals`
        : `FAILED: ${failures.join("; ")}`),
  };
}
