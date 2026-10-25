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
  return {
    medianA: Number(gaps[gaps.length >> 1].toFixed(2)),
    insideBond: gaps.filter((g) => g >= 3.6 && g <= 4.0).length,
    bonds: gaps.length,
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

  const shardBase = await resolveRfd4pShardBase();
  const { tensors: weights, config } = await fetchShardedWeights(shardBase);
  const { batch } = buildBatch({ contig: `${length}-${length}` }, { seed });

  const arms = {};
  for (const half of [true, false]) {
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

  const deviation = caDeviation(arms.f16.ca, arms.f32.ca);
  // 🔴 THE f32 ARM HAS TO HAVE PRODUCED A CHAIN FOR THE COMPARISON TO MEAN ANYTHING.
  const referenceIsProtein = arms.f32.spacing.insideBond === arms.f32.spacing.bonds;
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
