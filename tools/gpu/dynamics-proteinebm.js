// ProteinEBM Langevin dynamics on WebGPU, from a structure: a trajectory, its energy, and whether it stays a protein.
//
//     node tools/gpu-chrome.mjs tools/gpu/dynamics-proteinebm.js [--pdb=/tools/fixtures/6mrr-crystal.pdb] [--chain=A]
//       [--steps=200] [--t=0.1] [--dt=0.001] [--temp=0.85714] [--selfcond=none|current] [--seed=0] [--every=10]
//       [--model=/model-proteinebm-1-f32] [--show=denoised|noisy]
//
// The update is shared/proteinebm/langevin.js (the notebook's, the dump-checked one); the score is
// webgpu/proteinebm's forward + backward. Reported every `every` steps: the energy, the CA RMSD to the start after
// optimal superposition, the consecutive CA-CA spacing (median and worst; 3.80 A is a chain) and the radius of
// gyration. The trajectory is returned as a multi-model CA PDB (`pdb`), which the harness writes when `--out=` is given.
import { ProteinEbmGpu } from "../../webgpu/proteinebm/network.js";
import { bundleReader, caTraceFromPdb, centre } from "../../shared/proteinebm/model.js";
import { langevin } from "../../shared/proteinebm/langevin.js";
import { weightedRigidAlign } from "../../cpu/af3/diffusion/rigid-align.js";

const option = (args, name, fallback) => (args ?? []).find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;

function kabschRmsd(pos, reference) {
  const n = pos.length / 3;
  const fitted = weightedRigidAlign(pos, reference, new Float32Array(n).fill(1), n);
  let sum = 0;
  for (let k = 0; k < n * 3; k += 1) sum += (fitted[k] - reference[k]) ** 2;
  return Math.sqrt(sum / n);
}

function spacing(pos) {
  const d = [];
  for (let i = 1; i < pos.length / 3; i += 1) {
    d.push(Math.hypot(pos[i * 3] - pos[i * 3 - 3], pos[i * 3 + 1] - pos[i * 3 - 2], pos[i * 3 + 2] - pos[i * 3 - 1]));
  }
  d.sort((a, b) => a - b);
  return { median: d[d.length >> 1], min: d[0], max: d[d.length - 1] };
}

function gyration(pos) {
  const n = pos.length / 3;
  const m = [0, 0, 0];
  for (let i = 0; i < n; i += 1) for (let c = 0; c < 3; c += 1) m[c] += pos[i * 3 + c] / n;
  let s = 0;
  for (let i = 0; i < n; i += 1) for (let c = 0; c < 3; c += 1) s += (pos[i * 3 + c] - m[c]) ** 2;
  return Math.sqrt(s / n);
}

function caModel(pos, sequence, model) {
  const three = { A: "ALA", R: "ARG", N: "ASN", D: "ASP", C: "CYS", Q: "GLN", E: "GLU", G: "GLY", H: "HIS", I: "ILE",
                  L: "LEU", K: "LYS", M: "MET", F: "PHE", P: "PRO", S: "SER", T: "THR", W: "TRP", Y: "TYR", V: "VAL" };
  const lines = [`MODEL     ${String(model).padStart(4)}`];
  for (let i = 0; i < pos.length / 3; i += 1) {
    const f = (v) => v.toFixed(3).padStart(8);
    lines.push(`ATOM  ${String(i + 1).padStart(5)}  CA  ${three[sequence[i]] ?? "UNK"} A${String(i + 1).padStart(4)}    `
      + `${f(pos[i * 3])}${f(pos[i * 3 + 1])}${f(pos[i * 3 + 2])}  1.00  0.00           C`);
  }
  lines.push("ENDMDL");
  return lines.join("\n");
}

export async function main(device, args) {
  const pdbPath = option(args, "pdb", "/tools/fixtures/6mrr-crystal.pdb");
  const model = option(args, "model", "/model-proteinebm-1-f32");
  const steps = Number(option(args, "steps", "200"));
  const every = Number(option(args, "every", "10"));
  const show = option(args, "show", "denoised");
  const options = { steps, t: Number(option(args, "t", "0.1")), dt: Number(option(args, "dt", "0.001")),
                    tempScaling: Number(option(args, "temp", "0.85714")), selfCondition: option(args, "selfcond", "none"),
                    seed: Number(option(args, "seed", "0")) };
  const trace = caTraceFromPdb(await (await fetch(pdbPath)).text(), option(args, "chain", null));
  const start = centre(Float32Array.from(trace.ca));
  const net = await ProteinEbmGpu.load(device, bundleReader(model));
  await net.prepare({ aatype: trace.aatype, residueIdx: trace.residueIdx });
  const evaluate = (pos, sc) => net.evaluate(pos, { sc, t: options.t });
  const rows = [];
  const models = [caModel(start, trace.sequence, 0)];
  const t0 = performance.now();
  let last = null;
  for await (const frame of langevin(evaluate, start, options)) {
    last = frame;
    if ((frame.step + 1) % every === 0 || frame.step === 0) {
      // the NOISY state x_t (what the sampler walks, a sample of the t-noised marginal) and the DENOISED prediction
      // x0 (compute_score's pred_coords - the structure run_dynamics.py logs), each against the start
      const sp = spacing(frame.denoised);
      rows.push({ step: frame.step + 1, energy: Number(frame.energy.toFixed(2)),
                  noisyRmsd: Number(kabschRmsd(frame.pos, start).toFixed(3)),
                  noisyCa: Number(spacing(frame.pos).median.toFixed(2)),
                  rmsd: Number(kabschRmsd(frame.denoised, start).toFixed(3)),
                  caMedian: Number(sp.median.toFixed(3)), caMin: Number(sp.min.toFixed(2)), caMax: Number(sp.max.toFixed(2)),
                  rg: Number(gyration(frame.denoised).toFixed(2)) });
      models.push(caModel(show === "noisy" ? frame.pos : frame.denoised, trace.sequence, frame.step + 1));
    }
  }
  const ms = performance.now() - t0;
  net.destroy();
  return { pdb: pdbPath, length: trace.sequence.length, sequence: trace.sequence, options,
           startRg: Number(gyration(start).toFixed(2)), msPerStep: Number((ms / steps).toFixed(1)),
           finalEnergy: last?.energy, rows, trajectory: `${models.join("\n")}\nEND\n` };
}
