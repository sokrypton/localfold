// Does webgpu/proteinebm compute jproney/ProteinEBM's energy and score - the forward AND the backward pass?
//
//     python3 tools/oracle/dump_proteinebm.py            (writes oracle-dumps/proteinebm-6mrr.json)
//     node tools/gpu-chrome.mjs tools/gpu/check-proteinebm.js [--dump=proteinebm-6mrr] [--model=/model-proteinebm-1-f32]
//
// Both self-conditioning arms of the dump (zeros: what the vendor's scripts run; current: the Langevin notebook's),
// each at its seams, so a residual names the stage it comes from:
//
//   s, z, a        the forward: the single conditioner's output, the pair conditioner's, the transformer's (a_norm)
//   energy         sum |r_update|^2
//   dE/dz          the backward through 16 attention biases' LayerNorms to the pair conditioner's output
//   dE/ds          ... and through every gate and AdaLN, the transformer's input and the pair init, to s
//   score          -dE/dr: through the single conditioner and the coordinate embedding
//
// float32 on both sides; the bound is 1e-4 relative RMS. Measured on 6MRR: 6e-7 forward, 4e-6 for the score. Watched
// failing: the pair init's per-residue scalars dropped from the backward gives dE/ds 3.8e-3 and score 4.1e-3 with every
// forward row still passing. The run time of a forward and of a forward + backward is reported.
import { ProteinEbmGpu } from "../../webgpu/proteinebm/network.js";
import { bundleReader } from "../../shared/proteinebm/model.js";
import { langevinStep } from "../../shared/proteinebm/langevin.js";
import { relativeRms } from "./relative-rms.js";

const option = (args, name, fallback) => (args ?? []).find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;

export async function main(device, args) {
  const dumpName = option(args, "dump", "proteinebm-6mrr");
  const model = option(args, "model", "/model-proteinebm-1-f32");
  const bound = Number(option(args, "bound", "1e-4"));
  const dump = await (await fetch(`/oracle-dumps/${dumpName}.json`)).json();
  const t0 = performance.now();
  const net = await ProteinEbmGpu.load(device, bundleReader(model));
  const loadMs = performance.now() - t0;
  await net.prepare({ aatype: dump.aatype, residueIdx: dump.residueIdx });
  const r = Float32Array.from(dump.ca);
  const rows = [];
  const failures = [];
  for (const [arm, want] of Object.entries(dump.arms)) {
    const sc = arm === "current" ? r : null;
    const got = await net.evaluate(r, { sc, t: dump.t, readSeams: true });
    const check = (stage, ours, theirs) => {
      const residual = relativeRms(ours, Float32Array.from(theirs));
      const ok = residual <= bound;
      rows.push({ arm, stage, relRms: Number(residual.toExponential(3)), ok });
      if (!ok) failures.push(`${arm}/${stage} ${residual.toExponential(2)}`);
    };
    check("s", got.s, want.s);
    check("z", got.z, want.z);
    check("a", got.a, want.a);
    check("rUpdate", got.rUpdate, want.rUpdate);
    check("energy", [got.energy], [want.energy]);
    check("dE/dz", got.dEdz, want.dEdz);
    check("dE/ds", got.dEds, want.dEds);
    check("score", got.score, want.score);
    rows.push({ arm, stage: "energy (values)", ours: got.energy, oracle: want.energy });
  }
  // the Langevin update, replayed with the dump's own noise and translations: positions after each step, and the
  // energy the step was taken at (the trajectory is chaotic, so three steps of dt 1e-3 is the whole of the check)
  for (const arm of ["zeros", "current"]) {
    let pos = Float32Array.from(dump.ca);
    for (const [at, want] of dump.langevin[arm].entries()) {
      const got = await net.evaluate(pos, { sc: arm === "current" ? pos : null, t: dump.t });
      pos = langevinStep(pos, got.score, { t: dump.t, dt: dump.langevin.dt, tempScaling: dump.langevin.tempScaling,
                                           noise: want.noise, shift: want.shift });
      const posResidual = relativeRms(pos, Float32Array.from(want.position));
      const eResidual = Math.abs(got.energy - want.energy) / Math.abs(want.energy);
      const ok = posResidual <= bound && eResidual <= bound;
      rows.push({ arm, stage: `langevin ${at}`, positionRelRms: Number(posResidual.toExponential(3)),
                  energyRel: Number(eResidual.toExponential(3)), ok });
      if (!ok) failures.push(`${arm}/langevin ${at} position ${posResidual.toExponential(2)} energy ${eResidual.toExponential(2)}`);
    }
  }
  // timing: warm, both arms of the work
  const time = async (withScore, reps = 5) => {
    await net.evaluate(r, { withScore });
    const t = performance.now();
    for (let k = 0; k < reps; k += 1) await net.evaluate(r, { withScore });
    return (performance.now() - t) / reps;
  };
  const forwardMs = await time(false);
  const scoreMs = await time(true);
  net.destroy();
  return { length: dump.length, loadMs: Math.round(loadMs), forwardMs: Number(forwardMs.toFixed(1)),
           forwardBackwardMs: Number(scoreMs.toFixed(1)), rows, ok: failures.length === 0, failures };
}
