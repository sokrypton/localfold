/**
 * The reduced-precision pair track, checked AT A LENGTH THAT CHUNKS.
 *
 *   DISPLAY=:99 node tools/gpu-chrome.mjs tools/gpu/check-rfd4p-length.js [--length=400] [--nsteps=20]
 *
 * 🔴 THE ORACLE CASES ARE 54 AND 100 TOKENS, AND `webgpu/rfd4p/trunk.js` CHANGES SHAPE AT
 * 512. Past there the pair scratch no longer fits one chunk, `triF16` turns off and a
 * different set of kernels runs; below it the f16 single-chunk path runs. So the arm the
 * oracle checks is the only arm the oracle can reach, and every f16 narrowing in the pair
 * track is unverified at exactly the lengths people design at - 100 to 400.
 *
 * This asks the one question an oracle dump is not needed for: GENERATING THE SAME DESIGN
 * TWICE AT THE SAME SEED, ONCE IN EACH ARITHMETIC, HOW FAR APART DO THEY LAND? The f32 arm
 * is the reference - it is the same code path the oracle holds at 100 tokens - and the
 * bound is 0.5 A, which is what `tools/gpu/check-generate.js` already holds its own half
 * arm to. The length comes from a contig alone, so no structure and no dump is involved.
 *
 * 🔴 AND THE ANSWER AT n = 200 IS THAT THEY DO NOT AGREE: **mean CA 2.687 A, worst 7.935 A**
 * over 60 steps at one seed, against the 0.5 A this repository holds its half arm to at the
 * oracle's length. Both arms produce a well-formed chain - medians 3.87 and 3.86, nothing
 * torn - so this is not a broken kernel producing rubbish; it is the SAME DISTRIBUTION
 * SAMPLED DIFFERENTLY, a per-pass difference of order 1e-3 compounded over 120 passes of a
 * chaotic trajectory. `cuda/rfd4p`'s notes say the same thing from the other side: "one ulp
 * in one draw diverges a sixty-step trajectory".
 *
 * 🔴 SO f16 IS NOT VALIDATED BEYOND THE ORACLE'S LENGTH, AND `web/generate.js` STAYS f32.
 * Turning `half` on there is worth 1.56x at n = 200 and about 2.6x at n = 400 - every tool
 * in this directory already defaults to it - but a 2.7 A mean deviation is a DIFFERENT
 * STRUCTURE, and "both of them look like proteins" is not evidence that the faster one is
 * right. What would settle it is an oracle dump at 200-400 tokens, which does not exist
 * here, or an independent re-fold of both designs. Until one of those exists this tool
 * records the gap rather than closing it.
 */

import {
  Rfd4pGeneratorGpu,
  buildBatch,
  fetchShardedWeights,
  resolveRfd4pShardBase,
} from '../../webgpu/rfd4p/index.js';

/** Mean and worst CA separation between two runs, in angstrom (`bb_ca` is nm). */
function caDeviation(a, b) {
  const n = Math.min(a.length, b.length) / 3;
  let sum = 0;
  let worst = 0;
  let nonFinite = 0;
  for (let i = 0; i < n; i += 1) {
    if (!Number.isFinite(a[i * 3]) || !Number.isFinite(b[i * 3])) nonFinite += 1;
    const dx = a[i * 3] - b[i * 3];
    const dy = a[i * 3 + 1] - b[i * 3 + 1];
    const dz = a[i * 3 + 2] - b[i * 3 + 2];
    const d = Math.sqrt(dx * dx + dy * dy + dz * dz) * 10;
    sum += d;
    worst = Math.max(worst, d);
  }
  return { meanA: sum / Math.max(n, 1), worstA: worst, nonFinite, residues: n };
}

/**
 * Consecutive CA-CA spacing, the cheapest evidence that a trajectory produced a CHAIN.
 *
 * A diverged run is not always a run of NaNs; it can be a finite ball of atoms, and the
 * deviation between two balls says nothing. 3.8 A is the peptide bond.
 *
 * 🔴 THE TEST IS THE MEDIAN AND THE ABSENCE OF A BREAK, NOT EVERY BOND IN A WINDOW. The
 * first version of this asked for all of them inside [3.6, 4.0] and rejected an f16 run at
 * n = 200 whose median was 3.867 and whose single offender was **4.025** - 0.025 A out, on
 * 199 independent bonds. A hard window with no tolerance is a coin-flip on a long chain and
 * says nothing about whether the generation worked.
 *
 * It still rejects what it was built to reject: the 20-step run that was not finished being
 * a protein had a median of 3.44, which fails `medianA`, and a genuine break or collapse
 * fails `worstA`/`leastA`. Those are the shapes a failure takes - not a bond 25 thousandths
 * of an angstrom wide of a round number.
 */
function caSpacing(ca) {
  const n = ca.length / 3;
  const gaps = [];
  for (let i = 1; i < n; i += 1) {
    const dx = ca[i * 3] - ca[(i - 1) * 3];
    const dy = ca[i * 3 + 1] - ca[(i - 1) * 3 + 1];
    const dz = ca[i * 3 + 2] - ca[(i - 1) * 3 + 2];
    gaps.push(Math.sqrt(dx * dx + dy * dy + dz * dz) * 10);
  }
  gaps.sort((p, q) => p - q);
  const medianA = gaps[gaps.length >> 1];
  return {
    medianA: Number(medianA.toFixed(3)),
    leastA: Number(gaps[0].toFixed(3)),
    worstA: Number(gaps[gaps.length - 1].toFixed(3)),
    insideBond: gaps.filter((g) => g >= 3.6 && g <= 4.0).length,
    bonds: gaps.length,
    // A chain, and nowhere torn: the median is a peptide bond and nothing is a break.
    isChain: medianA >= 3.6 && medianA <= 4.0 && gaps[0] >= 3.0 && gaps[gaps.length - 1] <= 4.6,
  };
}

export async function main(device, args = []) {
  const opt = Object.fromEntries(args.map((a) => a.replace(/^--/, '').split('=')));
  const length = Number(opt.length ?? 400);
  // 🔴 THE FULL TRAJECTORY, BECAUSE A SHORT ONE HAS NO REFERENCE. At 20 steps the f32 arm
  // itself comes out at a median CA-CA of 3.44 A with 61 of 399 bonds inside [3.6, 4.0] -
  // it has not finished being a protein yet - so the two arms were being compared against
  // each other with nothing to say which, if either, was right. The honest check costs a
  // full generation an arm; shortening it does not make the answer cheaper, it removes it.
  const nsteps = Number(opt.nsteps ?? 60);
  const seed = Number(opt.seed ?? 42);
  const tolA = 0.5;
  // 🔴 ONE ARM A PROCESS, BECAUSE `tools/gpu-chrome.mjs` GIVES A MODULE 600 SECONDS. Both
  // arms of a 60-step generation at n = 400 is about fifteen minutes - the f32 arm is the
  // slow one, and at that length it is also chunked - so asking for both in one run returned
  // `timed out after 600000 ms` and no measurement at all. `--arm=f16` or `--arm=f32` runs
  // one and prints its CA trace; `tools/gpu/compare-rfd4p-arms.mjs` does the comparison on
  // the host. `--arm=both` is still the default and is right up to about n = 250.
  const arm = opt.arm ?? 'both';
  if (!['both', 'f16', 'f32'].includes(arm)) throw new Error(`--arm: ${arm}`);

  const shardBase = await resolveRfd4pShardBase();
  const { tensors: weights, config } = await fetchShardedWeights(shardBase);
  const { batch } = buildBatch({ contig: `${length}-${length}` }, { seed });

  const wanted = arm === 'both' ? [true, false] : [arm === 'f16'];
  const arms = {};
  for (const half of wanted) {
    const gen = new Rfd4pGeneratorGpu(device, weights, config, { half });
    const t0 = performance.now();
    const res = await gen.generate(batch, { nsteps, recycleSchedule: 'low', seed });
    arms[half ? 'f16' : 'f32'] = {
      shape: { triF16: gen.trunk.triF16, chunks: gen.trunk.chunks.length },
      wallMs: Number((performance.now() - t0).toFixed(0)),
      ca: res.x.bb_ca.slice(),
      spacing: caSpacing(res.x.bb_ca),
    };
    gen.destroy();
  }

  // One arm: print what it produced, for the host to compare against the other.
  if (arm !== 'both') {
    const only = arms[arm];
    return {
      length, nsteps, seed, arm, ...only.shape, wallMs: only.wallMs, spacing: only.spacing,
      // 🔴 THE ARM HAS TO BE A PROTEIN ON ITS OWN BEFORE ANY COMPARISON IS WORTH MAKING,
      // and this is the half of the check that does not need the other arm at all.
      isProtein: only.spacing.isChain,
      ca: [...only.ca].map((v) => Number(v.toFixed(5))),
    };
  }

  const deviation = caDeviation(arms.f16.ca, arms.f32.ca);
  // 🔴 THE f32 ARM HAS TO HAVE PRODUCED A CHAIN FOR THE COMPARISON TO MEAN ANYTHING.
  const referenceIsProtein = arms.f32.spacing.isChain;
  const ok = deviation.nonFinite === 0 && deviation.meanA <= tolA && referenceIsProtein;

  const report = {
    length,
    nsteps,
    tolA,
    f16: { ...arms.f16.shape, wallMs: arms.f16.wallMs, spacing: arms.f16.spacing },
    f32: { ...arms.f32.shape, wallMs: arms.f32.wallMs, spacing: arms.f32.spacing },
    deviation: {
      meanA: Number(deviation.meanA.toFixed(3)),
      worstA: Number(deviation.worstA.toFixed(3)),
      nonFinite: deviation.nonFinite,
      residues: deviation.residues,
    },
    referenceIsProtein,
    ok,
  };
  if (!ok) throw new Error(`rfd4p length check failed: ${JSON.stringify(report)}`);
  return report;
}
