/**
 * End-to-end GPU generation (`webgpu/rfd4p/generate.js`) vs the PyTorch oracle trajectory,
 * replaying recorded noise (`randn` and `randperm` draws).
 *
 *   DISPLAY=:99 node tools/gpu-chrome.mjs tools/gpu/check-generate.js [--case=uncond_short] [--half=0]
 */

import { Rfd4pGeneratorGpu } from '../../webgpu/rfd4p/generate.js';
import { loadOracleCase, loadReleased } from './check-trunk.js';

function compare(got, want) {
  let se = 0;
  let sw = 0;
  let maxAbs = 0;
  let bad = 0;
  for (let i = 0; i < want.length; i += 1) {
    const d = got[i] - want[i];
    if (!Number.isFinite(got[i])) bad += 1;
    se += d * d;
    sw += want[i] * want[i];
    maxAbs = Math.max(maxAbs, Math.abs(d));
  }
  return {
    relRms: Math.sqrt(se / Math.max(sw, 1e-30)),
    rms: Math.sqrt(se / Math.max(want.length / 3, 1)),
    maxAbs,
    nonFinite: bad,
  };
}

function coordRmsdAngstrom(gotCoors, wantCoors, atomMask) {
  let sumSq = 0;
  let count = 0;
  let maxDist = 0;
  for (let a = 0; a < atomMask.length; a += 1) {
    if (!atomMask[a]) continue;
    const dx = gotCoors[a * 3] - wantCoors[a * 3];
    const dy = gotCoors[a * 3 + 1] - wantCoors[a * 3 + 1];
    const dz = gotCoors[a * 3 + 2] - wantCoors[a * 3 + 2];
    const d2 = dx * dx + dy * dy + dz * dz;
    sumSq += d2;
    maxDist = Math.max(maxDist, Math.sqrt(d2));
    count += 1;
  }
  return { rmsdA: Math.sqrt(sumSq / Math.max(count, 1)), maxAtomDistA: maxDist, atoms: count };
}

function recordedOracleRng(tensors, rngLog) {
  const randnDraws = rngLog.filter((r) => r.fn === 'randn');
  let rn = 0;
  return {
    randn(count) {
      const r = randnDraws[rn++];
      const d = r && tensors[`rng.${r.i}`].data;
      if (!d || d.length !== count) {
        throw new Error(`randn replay mismatch at ${rn - 1}: want ${count}, have ${d?.length}`);
      }
      return d;
    },
    randperm(len, call) {
      const draws = rngLog.filter((r) => r.fn === 'randperm' && r.call === call);
      if (draws.length === 0) throw new Error(`no randperm recorded for call ${call}`);
      const d = tensors[`rng.${draws[0].i}`].data;
      if (d.length !== len) throw new Error(`randperm length mismatch at call ${call}: want ${len}, have ${d.length}`);
      return d;
    },
    get randnUsed() {
      return rn;
    },
    randnTotal: randnDraws.length,
  };
}

export async function main(device, args) {
  const opt = Object.fromEntries(args.map((a) => a.replace(/^--/, '').split('=')));
  if (opt['no-submat'] === '1') device.noSubmat = true;
  const caseName = opt.case ?? 'uncond_short';
  const half = opt.half === '1';

  const t0 = performance.now();
  const { cfg, weights } = await loadReleased();
  const { meta, t } = await loadOracleCase(caseName);
  const batch = {};
  for (const [k, v] of Object.entries(t)) if (k.startsWith('batch.')) batch[k.slice(6)] = v;
  const n = batch.padding_mask.shape[1];

  const gen = new Rfd4pGeneratorGpu(device, weights, cfg, { half });
  const tLoad = performance.now();

  const rng = recordedOracleRng(t, meta.rng);
  const stepDrifts = [];
  const tRun0 = performance.now();
  const res = await gen.generate(batch, {
    rng,
    nsteps: meta.steps.length,
    recycleSchedule: meta.recycle_depths,
    onStep: opt.fast === '1' ? undefined : ({ step, xt }) => {
      const nextKeyCa = step + 1 < meta.steps.length ? `step.${step + 1}.x_t.bb_ca` : 'final.x.bb_ca';
      const nextKeyLl = step + 1 < meta.steps.length ? `step.${step + 1}.x_t.local_latents` : 'final.x.local_latents';
      const caCmp = compare(xt.bb_ca, t[nextKeyCa].data);
      const llCmp = compare(xt.local_latents, t[nextKeyLl].data);
      if (step === 0 || step === 9 || step === 29 || step === meta.steps.length - 1) {
        stepDrifts.push({
          step,
          caRelRms: Number(caCmp.relRms.toExponential(2)),
          caMaxAbsNm: Number(caCmp.maxAbs.toExponential(2)),
          llRelRms: Number(llCmp.relRms.toExponential(2)),
        });
      }
    },
  });
  const tRun = performance.now() - tRun0;
  gen.destroy();

  const finalCa = compare(res.x.bb_ca, t['final.x.bb_ca'].data);
  const finalLl = compare(res.x.local_latents, t['final.x.local_latents'].data);
  const wantRt = t['final.residue_type'].data;
  let rtDiff = 0;
  for (let i = 0; i < n; i += 1) if (res.residueType[i] !== wantRt[i]) rtDiff += 1;

  const wantMask = t['final.atom_mask'].data;
  let maskDiff = 0;
  for (let i = 0; i < wantMask.length; i += 1) if (res.atomMask[i] !== wantMask[i]) maskDiff += 1;

  const coorsCmp = compare(res.coors, t['final.coors'].data);
  const rmsd = coordRmsdAngstrom(res.coors, t['final.coors'].data, wantMask);
  const tolRmsdA = half ? 0.5 : 0.05;
  const allOk =
    rng.randnUsed === rng.randnTotal &&
    finalCa.nonFinite === 0 &&
    finalLl.nonFinite === 0 &&
    coorsCmp.nonFinite === 0 &&
    rtDiff === 0 &&
    maskDiff === 0 &&
    rmsd.rmsdA < tolRmsdA;

  return {
    case: caseName,
    n,
    half,
    nsteps: meta.steps.length,
    randnConsumed: `${rng.randnUsed}/${rng.randnTotal}`,
    stepDrifts,
    finalBbCa: {
      relRms: Number(finalCa.relRms.toExponential(2)),
      maxAbsNm: Number(finalCa.maxAbs.toExponential(2)),
      rmsdA: Number((finalCa.rms * 10).toExponential(2)),
    },
    finalLatents: {
      relRms: Number(finalLl.relRms.toExponential(2)),
      maxAbs: Number(finalLl.maxAbs.toExponential(2)),
    },
    finalStructure: {
      rtDiff,
      maskDiff,
      rmsdA: Number(rmsd.rmsdA.toExponential(3)),
      maxAtomDistA: Number(rmsd.maxAtomDistA.toExponential(3)),
      atoms: rmsd.atoms,
    },
    ms: {
      load: Math.round(tLoad - t0),
      generate: Math.round(tRun),
    },
    allOk,
  };
}
