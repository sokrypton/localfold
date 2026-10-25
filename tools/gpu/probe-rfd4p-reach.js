/**
 * How long a design the rfd4p trunk can ALLOCATE, and which buffer stops it.
 *
 *   DISPLAY=:99 node tools/gpu-chrome.mjs tools/gpu/probe-rfd4p-reach.js
 *   ... tools/gpu/probe-rfd4p-reach.js --lengths=400,512,513,600,800 --arms=f16,f32
 *
 * 🔴 IT RECORDS AND NEVER FOLDS, which is the whole point: a 60-step generation at n = 600
 * is a quarter of an hour and the question here is not what it costs but whether it fits.
 * `Rfd4pTrunkGpu.record` allocates every activation and nothing else, `Rfd4pEngine.buffer`
 * refuses over `maxStorageBufferBindingSize` NAMING THE BUFFER, and `activationBytes` is the
 * total - so one call per length answers reach, and the refusal says what to narrow.
 *
 * 🔴 AND THE REACH QUESTION IS NOT THE CHUNK PLAN'S. `chunkPlan` bounds the R-indexed
 * scratch at 896 MB, and chunking ALREADY decouples that bound from the length - past the
 * single-chunk threshold R stops growing and the scratch is flat forever. What grows with
 * n^2 and is allocated WHOLE, outside that plan, is six buffers:
 *
 *   zInitStatic, zInit, z    nn * 384      the pair track, three copies
 *   aT, bT                   256 * n*nPad  the triangle's channel-major operands
 *   x                        256 * nn      its contraction output
 *
 * `triF16` halves all six and the scratch with them, so f16 is the LENGTH lever and the
 * chunk plan is not. Measured on an Apple M5 (8-core GPU, 16 GB unified, 4 GB bindings),
 * activations alone, before and after the chunked track learned f16:
 *
 *     n    f16 was   f16 now   f32    chunks now
 *   400       1269      1269  2436    1
 *   512       2056      2056  3395    1
 *   513       3001      2245  3381    2     <- was +46% in ONE residue, now +9.2%
 *   600       3769      2724  4290    2
 *   800          -      4114     -    3
 *  1000       8840      5926 10285    5
 *  1200          -      8103 14403    6
 *
 * 🔴 NOTHING REFUSES UP TO n = 1200, so the ceiling is the 16 GB this machine shares with
 * everything else and not a limit any buffer names. The old f16 column IS the f32 column
 * past 512, because `triF16` used to require one chunk: the arithmetic widened at the same
 * residue the slicing changed, which is what made that row a cliff rather than a curve. At
 * a 4 GB budget the reachable length goes from about 585 to about 798 - **1.36x**, the
 * square root of what halving six n^2 buffers buys, less what `bias` and `pairF` keep in
 * f32.
 */

import {
  Rfd4pTrunkGpu,
  fetchShardedWeights,
  pairCondFeatures,
  resolveRfd4pShardBase,
} from '../../webgpu/rfd4p/index.js';

export async function main(device, args = []) {
  const opt = Object.fromEntries(args.map((a) => a.replace(/^--/, '').split('=')));
  const lengths = (opt.lengths ?? '362,400,512,513,600,700,800,900,1000')
    .split(',').map(Number).filter((v) => Number.isInteger(v) && v > 1);
  const arms = (opt.arms ?? 'f16,f32').split(',');

  const shardBase = await resolveRfd4pShardBase();
  const { tensors: weights, config } = await fetchShardedWeights(shardBase);
  const pairCond = pairCondFeatures({ bb_ca: 0.5, local_latents: 0.5 }, config.nn.feats_pair_cond);

  const limits = {
    maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
    maxBufferSize: device.limits.maxBufferSize,
  };

  const rows = [];
  for (const arm of arms) {
    const trunk = new Rfd4pTrunkGpu(device, weights, config.nn, { half: arm === 'f16' });
    const dims = {
      seq: trunk.seqDim, cond: trunk.condDim, pair: trunk.pairDim, pairCond: pairCond.length,
    };
    for (const n of lengths) {
      const row = { arm, n };
      try {
        await trunk.record(n, dims);
        row.triF16 = trunk.triF16;
        row.chunks = trunk.chunks.length;
        row.chunkRows = trunk.R;
        row.activationMiB = Number((trunk.engine.activationBytes / 1048576).toFixed(1));
        row.ok = true;
      } catch (err) {
        row.ok = false;
        row.refusal = String(err.message ?? err).slice(0, 160);
        // 🔴 A REFUSED RECORD LEAVES HALF THE ACTIVATIONS BEHIND, and the next length would
        // then allocate on top of them and refuse early for the wrong reason.
        trunk.engine.releaseActivations();
        trunk.n = 0;
      }
      rows.push(row);
      if (!row.ok) break;
    }
    trunk.engine.destroy();
  }

  return { limits, rows };
}
