/**
 * How long the pieces of a fold take, relative to each other.
 *
 * 🔴 THE PROGRESS BAR USED TO GUESS, AND THE GUESS DID NOT MENTION LENGTH.
 * web/af3-model.js split the bar with constants measured once on a 59-residue
 * chain: a trunk pass was "3.7 s" and a denoiser call "0.85", so the trunk was
 * always 4.35 calls' worth of work. Measured, that ratio is 3.1 at 59 tokens
 * and 15.3 at 256, because the trunk grows as L squared and the sampler barely
 * faster than linearly. So the bar raced through the trunk band and stalled on
 * a long chain, and crawled through it on a short one. AF2's bar had the
 * mirror-image fault: it counted an IPA iteration as one step and an evoformer
 * block as one step, when at 512 alignment rows the block is a hundred times
 * the work.
 *
 * 🔴 WHAT IS MODELLED IS A SHAPE, NOT A SPEED, AND THAT IS THE WHOLE TRICK.
 * The constants below are milliseconds on one M2, and on a phone every one of
 * them is a lie by a factor of five to ten. What carries over is the RATIO
 * between the pieces, which is a property of the arithmetic rather than of the
 * device. So the units here are only ever compared with each other: the bar is
 * units done over units planned, which needs no speed at all, and the ETA
 * multiplies the units remaining by a millisecond-per-unit that RuntimeEstimator
 * learns from the run in front of it. A device half this speed reports a
 * correct bar throughout and a correct ETA within a few percent of work done.
 *
 * 🔴 THE FITS ARE IN tools/gpu/bench-runtime.js, WHICH IS ALSO HOW TO REDO
 * THEM. Sweeps of length against depth, each shape run twice because the first
 * compiles its pipelines, all in one process because this machine drifts by up
 * to 3.2x between them. Residuals against the measurements: AF3's trunk within
 * 2.7% over 59-256 tokens and 1-512 rows, its denoiser within 0.1%, AF2's
 * stack within 22%. AF2 is the loose one because its cost steps between 128 and
 * 160 residues - some kernel changes behaviour there - and a smooth fit cannot
 * follow that. Twenty per cent is comfortably inside what a bar needs.
 */

/**
 * A shape argument, or a loud complaint about it.
 *
 * 🔴 A COST MODEL THAT RETURNS NaN TAKES THE FOLD DOWN WITH IT. These feed a
 * progress bar, and a bar is not worth a failed prediction - but the failure
 * they caused was not even a wrong bar: one undefined alignment depth made
 * every unit NaN, which reached HTMLProgressElement as "the provided double
 * value is non-finite" and aborted the run. So a bad input is clamped to
 * something harmless AND named on the console, which is how the next one gets
 * found rather than silently drawn.
 */
function shape(value, fallback, what) {
  if (typeof value === "number" && Number.isFinite(value) && value > 0) return value;
  if (value !== undefined) {
    console.warn(`cost model: ${what} was ${value}; using ${fallback}`);
  }
  return fallback;
}

/**
 * What one trunk pass is made of, as a share of the pass, in the order it runs.
 *
 * 🔴 THE BAR HAD ONE MILESTONE FOR THE WHOLE PASS AND IT WAS THE LAST STAGE.
 * `Af3TrunkGpu.run` is five stages - embedder, template, MSA stack, pairformer,
 * distogram - and only the pairformer reports anything, per block. Everything
 * before it therefore passed in silence, and the status line sat on "Trunk ·
 * 0%" for the whole of it. That was a fair trade when the four other stages
 * were a rounding error; on a large protein they are seconds each, and a bar
 * that does not move for seconds reads as a page that has died.
 *
 * 🔴 MEASURED, AT ONE SHAPE, AND THAT IS THE LIMIT OF IT. From
 * `tools/gpu/bench-trunk.js --profile --tokens=200 --msa=32`, in milliseconds:
 * embedder 34, template 95, msa-stack 427, pairformer 2952, distogram 53. The
 * shares below are those over their total. They are NOT constant in the shape:
 * the MSA stack grows with alignment depth and the pairformer with tokens
 * cubed, so at 24 tokens with four blocks the pairformer is a quarter of the
 * pass rather than five sixths. What they buy is a bar that advances through
 * the silent stages in roughly the right places - which is a different and much
 * lower bar than a clock, and the reason the status line names the running
 * stage as well as showing a number.
 */
export const AF3_TRUNK_STAGE_SHARES = [
  ["embedder", 0.010],
  ["template", 0.027],
  ["msa-stack", 0.120],
  ["pairformer", 0.828],
  ["distogram", 0.015],
];

/**
 * Where each trunk stage begins and ends, as a fraction of one pass.
 *
 * The shares are normalised here rather than in the table, so the table can
 * stay readable as measurements and cannot drift from summing to one.
 */
export function af3TrunkStageSpans() {
  const total = AF3_TRUNK_STAGE_SHARES.reduce((sum, [, share]) => sum + share, 0);
  const spans = new Map();
  let at = 0;
  for (const [name, share] of AF3_TRUNK_STAGE_SHARES) {
    const width = share / total;
    spans.set(name, { from: at, to: at + width });
    at += width;
  }
  return spans;
}

/**
 * One AF3 trunk pass over 48 pairformer blocks and 4 MSA blocks.
 *
 * Fitted as const + L^2 + L^3 + rows*L^2. The pairformer dominates and does not
 * read the alignment at all - measured flat at 1324, 1337 and 1336 ms for 1,
 * 128 and 512 rows - so the depth term is small and belongs to the MSA stack.
 */
export function af3TrunkPassUnits(tokensIn, rowsIn) {
  const tokens = shape(tokensIn, 1, "tokens");
  const rows = shape(rowsIn, 1, "MSA rows");
  const l2 = tokens * tokens;
  return 91.8 + 0.08093 * l2 + 7.782e-5 * l2 * tokens + 2.751e-5 * rows * l2;
}

/**
 * One denoiser call: the atom encoder, the diffusion transformer and the atom
 * decoder, for one step of the sampler.
 *
 * Nearly linear in tokens, because the transformer's cost is dominated by the
 * per-token matmuls rather than by the pair logits.
 */
export function af3DenoiserCallUnits(tokensIn) {
  const tokens = shape(tokensIn, 1, "tokens");
  return 55.7 + 0.8034 * tokens + 1.833e-3 * tokens * tokens;
}

/**
 * One AF2 evoformer block, at a given alignment depth.
 *
 * 🔴 AF2'S COST IS MOSTLY ITS ALIGNMENT, WHICH IS WHY THE OLD BAR WAS SO WRONG
 * HERE. Measured at 128 residues: 1672 ms for a whole stack at one row and
 * 18792 at 512, so the depth term is ten times the length term at the depths
 * the page offers. AF3 carries the MSA in four blocks; AF2 carries it in all
 * forty-eight, and its column attention grows with depth.
 */
export function af2BlockUnits(lengthIn, rowsIn) {
  const length = shape(lengthIn, 1, "length");
  const rows = shape(rowsIn, 1, "MSA rows");
  const l2 = length * length;
  return (438 + 7.469e-4 * l2 * length + 2.068e-3 * rows * l2) / 48;
}

/**
 * One step of the structure module, and one of the confidence heads.
 *
 * 🔴 THESE TWO ARE NOT FITTED, THEY ARE BOUNDED. Both are small beside a block
 * - the structure module is eleven steps over a single track of L residues,
 * where a block is a pair track of L squared and an alignment on top - and what
 * the bar needs from them is not to be counted as equal to a block, which is
 * what it did. Modelled as the pair-shaped read they each make, which puts an
 * eleven-step structure module at a few per cent of one recycle at any length
 * this runs at. If they ever matter, measure them; the shape of the fix is the
 * same as everything else here.
 */
export function af2StructureStepUnits(lengthIn) {
  const length = shape(lengthIn, 1, "length");
  return 2 + 4.0e-4 * length * length;
}

export function af2ConfidenceStepUnits(lengthIn) {
  const length = shape(lengthIn, 1, "length");
  return 2 + 8.0e-4 * length * length;
}

/**
 * Building the input features: the synchronous CPU pass over the atoms.
 *
 * 🔴 IT USED TO BE SECONDS AND IS NOW MILLISECONDS: 84, 139 and 186 ms at
 * 1392, 3072 and 4608 atoms, against the 1.45 s for 449 atoms this term was
 * originally fitted to. Featurisation got about forty times faster and the
 * coefficient did not follow, so the bar handed this band 23% of a fold that
 * spends 2% of its clock here - which is most of why it jumped. It is CPU work
 * while everything else here is GPU work, so this is the one term whose ratio
 * to the rest does NOT carry across devices.
 */
export function af3FeaturiseUnits(atomsIn) {
  const atoms = shape(atomsIn, 1, "atoms");
  return 40.4 + 0.0317 * atoms;
}

/**
 * What the PAGE does per sampler step, beyond the denoiser call: superpose the
 * frame onto the reference, write a PDB, push it to the viewer, and yield.
 *
 * 🔴 IT IS NOW ALMOST NOTHING, AND USED TO BE MOST OF THE SAMPLER BAND. This
 * was fitted when a denoiser call was 124 ms alone and 350 ms as a step of a
 * page fold. Measured now with tools/gpu/probe-progress-bar.js, a page step is
 * 109, 193 and 293 ms at 58, 128 and 192 tokens against 109, 190 and 276 for
 * the call by itself - so the page adds single-digit milliseconds, not
 * hundreds. Left in rather than deleted because it is real and grows; the old
 * coefficient made the sampler 64% of the bar for 40% of the clock.
 * Measured against the atom count, which is what the superposition and
 * the PDB are proportional to.
 */
export function af3FrameUnits(atomsIn) {
  const atoms = shape(atomsIn, 1, "atoms");
  return 0.0026 * atoms;
}

/**
 * Compiling the sampler's pipelines, once, before its first call.
 *
 * 🔴 THE FIRST DENOISER CALL IS NOT LIKE THE OTHERS, and the compile it pays
 * for is now FLAT rather than quadratic in the shape: 740, 669 and 789 ms at
 * 58, 128 and 192 tokens, each in its own cold process. It used to be fitted as
 * 0.17 * L^2, which is right at 58 and four times over at 128.
 *
 * 🔴 MEASURE IT COLD, ONE FOLD PER PROCESS. A second fold at another shape
 * reuses most of the pipelines and reports 155 ms where a cold one reports 669,
 * and the page only ever does the cold one.
 */
export function af3SamplerWarmupUnits(tokensIn) {
  shape(tokensIn, 1, "tokens");
  return 730;
}

/**
 * What an AF3 fold is made of, in units, as a list of stages in order.
 *
 * `atoms` is optional: without it the plan covers the GPU work alone, which is
 * what a bench wants. The page passes it and gets the featurisation, the
 * per-frame work and the sampler's warm-up as well.
 *
 * @param {{tokens: number, rows: number, passes: number, calls: number,
 *          atoms?: number}} shape
 */
export function af3Plan({ tokens, rows, passes, calls, atoms }) {
  const inPage = atoms !== undefined;
  const trunk = af3TrunkPassUnits(tokens, rows);
  const call = af3DenoiserCallUnits(tokens) + (inPage ? af3FrameUnits(atoms) : 0);
  return {
    stages: [
      { name: "features", units: inPage ? af3FeaturiseUnits(atoms) : 5, count: 1 },
      { name: "trunk", units: trunk, count: passes },
      // A stage of its own rather than a fatter first call, so the bar moves
      // through the compile instead of jumping when it ends.
      { name: "sampler-warmup",
        units: inPage && calls > 0 ? af3SamplerWarmupUnits(tokens) : 0, count: 1 },
      { name: "sampler", units: call, count: calls },
    ],
  };
}

/**
 * What an AF2 fold is made of. `extraBlocks` and `mainBlocks` run at different
 * depths, which is most of why they cost different amounts.
 */
export function af2Plan({ length, extraRows, mainRows, extraBlocks, mainBlocks,
                          passes, structureSteps = 11, confidenceSteps = 2 }) {
  return {
    stages: [
      { name: "extra-stack", units: af2BlockUnits(length, extraRows),
        count: extraBlocks * passes },
      { name: "main-stack", units: af2BlockUnits(length, mainRows),
        count: mainBlocks * passes },
      { name: "structure", units: af2StructureStepUnits(length),
        count: structureSteps * passes },
      { name: "confidence", units: af2ConfidenceStepUnits(length),
        count: confidenceSteps * passes },
    ],
  };
}

/** The units a plan adds up to. */
export function planTotal(plan) {
  return plan.stages.reduce(
    (sum, stage) => sum + stage.units * shape(stage.count, 0, `${stage.name} count`), 0);
}

/**
 * A bar and a clock over a plan.
 *
 * 🔴 THE FRACTION NEEDS NO CALIBRATION AND THE ETA LEARNS ITS OWN. Units done
 * over units planned is right on any device, because both sides are in the same
 * invented unit. The remaining TIME is that ratio multiplied by a
 * millisecond-per-unit, which starts at this machine's 1.0 and is replaced by
 * what the run has actually shown as soon as there is enough of it to mean
 * anything - so a device five times slower reports honestly from a few per cent
 * in, without anybody having measured it.
 *
 * 🔴 AND IT NEVER GOES BACKWARDS. A bar that retreats reads as a bug even when
 * the new number is better, so the fraction is clamped monotonic. The ETA is
 * allowed to move, because an estimate that cannot be corrected is worse than
 * one that visibly settles.
 */
export class RuntimeEstimator {
  #total;
  #done = 0;
  #started;
  #furthest = 0;
  #observedPerUnit;

  /**
   * @param {{stages: {name: string, units: number, count: number}[]}} plan
   * @param {number} [now] the clock, injectable so this can be tested
   */
  constructor(plan, now = Date.now()) {
    this.plan = plan;
    this.#total = planTotal(plan);
    this.#started = now;
    if (!(this.#total > 0)) throw new RangeError("a plan with no work in it cannot be estimated");
  }

  /** Work has finished: `count` of the stage named `name`. */
  complete(name, count = 1, now = Date.now()) {
    const stage = this.plan.stages.find((entry) => entry.name === name);
    if (stage === undefined) throw new Error(`no stage named ${name} in this plan`);
    this.#done += stage.units * count;
    this.#observe(now);
  }

  /** Work has finished, as an absolute number of units rather than a stage. */
  completedUnits(units, now = Date.now()) {
    this.#done = units;
    this.#observe(now);
  }

  #observe(now) {
    // 🔴 UNDER A FEW PER CENT THE RATE IS MOSTLY PIPELINE COMPILATION, which is
    // paid once and is not the rate of anything. Believing it early makes the
    // first estimate wildly long, which is the estimate people actually read.
    const fraction = this.#done / this.#total;
    if (fraction >= 0.05) this.#observedPerUnit = (now - this.#started) / this.#done;
  }

  /** How far along, in [0, 1], never retreating. */
  fraction() {
    this.#furthest = Math.max(this.#furthest, Math.min(1, this.#done / this.#total));
    return this.#furthest;
  }

  /** Milliseconds left, or undefined while there is nothing to base it on. */
  remainingMs() {
    const perUnit = this.#observedPerUnit;
    if (perUnit === undefined) return undefined;
    return Math.max(0, (this.#total - this.#done) * perUnit);
  }

  /** Milliseconds the whole thing looks like taking, observed plus remaining. */
  totalMs() {
    const perUnit = this.#observedPerUnit ?? 1;
    return this.#total * perUnit;
  }

  /** What this predicts before anything has run, on the machine that was fitted. */
  referenceMs() {
    return this.#total;
  }
}

/**
 * "1 min 20 s", "12 s", "3 s" - the shape of a number somebody is waiting on.
 *
 * Rounded coarsely on purpose: a countdown that ticks every second invites
 * people to check it against a clock, and this is an estimate.
 */
export function describeRemaining(milliseconds) {
  if (milliseconds === undefined || !Number.isFinite(milliseconds)) return undefined;
  const seconds = Math.round(milliseconds / 1000);
  // ...every case is a duration, so the caller can put one word in front of all
  // of them. "a moment" needed "about a moment left", which reads like an
  // apology rather than an estimate.
  if (seconds < 5) return "5 s";
  if (seconds < 60) return `${Math.round(seconds / 5) * 5} s`;
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round((seconds - minutes * 60) / 15) * 15;
  if (rest === 0 || rest === 60) return `${minutes + (rest === 60 ? 1 : 0)} min`;
  return `${minutes} min ${rest} s`;
}
