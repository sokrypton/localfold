// Where webgpu/bioemu2's time goes on a real input: the trunk by wall clock, the denoiser call per dispatch.
//
//     node tools/gpu-chrome.mjs tools/gpu/profile-bioemu2.js [--job=/tools/fixtures/bioemu2/6mrr-68.json]
//       [--model=/model-bioemu2-int5] [--samples=5] [--calls=10] [--top=30] [--shapes] [--split=off] [--gemm=1]
//       [--fuse=off] [--fuse-swiglu=off] [--batch-cond=off] [--split-target=N]
//
// check-bioemu2-network.js --profile does the denoiser half too, but on the colab oracle dumps, which are not on every
// box; this featurises the job itself. The per-dispatch numbers are a profile (every dispatch its own timestamped
// pass, which costs more than it measures): read them as proportions, and the wall clock as the speed.
import { Bioemu2Gpu } from "../../webgpu/bioemu2/network.js";
import { bundleReader } from "../../shared/bioemu2/weights.js";
import { featuriseBioemu2, bioemu2JobFromAf3Json } from "../../shared/bioemu2/featurise.js";
import { fetchComponents, smilesTemplates } from "../../shared/bioemu2/inputs.js";

const option = (args, name, fallback) => (args ?? []).find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;

export async function main(device, args) {
  const job = bioemu2JobFromAf3Json(await (await fetch(option(args, "job", "/tools/fixtures/bioemu2/6mrr-68.json"))).json());
  const samples = Number(option(args, "samples", "5")), calls = Number(option(args, "calls", "10"));
  const top = Number(option(args, "top", "30"));
  const ccd = await fetchComponents(job, { local: (code) => `/tools/fixtures/ccd/${code}.cif` });
  const { features, atomKeys } = featuriseBioemu2(job, { ccd, mode: "structure", contacts: null, smilesTemplates: await smilesTemplates(job) });
  const read = bundleReader(option(args, "model", "/model-bioemu2-int5"));
  await read.prefetch();
  // --split=off: the coupling's atom-side mixer unsplit (couplingSplit's control); --fuse=off: its combine a separate
  // pass over a stored Z (fuseCombine's control)
  const net = new Bioemu2Gpu(device, read, { half: true, trunkPrecision: "device", couplingSplit: option(args, "split", "on") !== "off",
    fuseCombine: option(args, "fuse", "on") !== "off", batchConditioning: option(args, "batch-cond", "on") !== "off", fuseWindow: option(args, "fuse-win", "on") !== "off", fuseTokenAttention: option(args, "fuse-tok", "on") !== "off",
    condGroupMiB: Number(option(args, "cond-mib", "96")) });
  net.ops.gemmVersion = Number(option(args, "gemm", "2"));     // (--gemm=1: the original GEMM, the control)
  net.ops.b16Panels = option(args, "b16", "on") !== "off";   // (--b16=off: the weight panels staged in float32)
  net.ops.fuseSwiglu = option(args, "fuse-swiglu", "on") !== "off";   // (--fuse-swiglu=off: the SwiGLUs a pass of their own)
  // (--split-target=N: split-K's workgroup target, ops.js's 128 by default; 0 is no split)
  if (option(args, "split-target", null)) net.ops.splitTarget = Number(option(args, "split-target", null));
  const g = net.upload(features);
  const sIn = await net.inputEmbedder(g);
  const trunk = await net.trunkProgram(g, sIn);
  await trunk.pass();                                    // (compiles; the next four are the trunk's time)
  await net.ops.read(trunk.s);
  let t0 = performance.now();
  for (let k = 0; k < 4; k += 1) await trunk.pass();
  await net.ops.read(trunk.s);
  const trunkMs = performance.now() - t0;
  const sTrunk = await net.ops.read(trunk.s);
  const zHost = await net.ops.read(trunk.z);
  const zTrunk = net.ops.buffer("profile.z", zHost.length, zHost);
  const dn = await net.denoiserProgram(g, sIn, sTrunk, zTrunk, { samples });
  const x = new Float32Array(samples * atomKeys.length * 3).map(() => Math.random() - 0.5);
  await dn.denoise(x, 16);
  t0 = performance.now();
  for (let k = 0; k < calls; k += 1) await dn.denoise(x, 16);
  const callMs = (performance.now() - t0) / calls;
  net.ops.profile = new Map();
  for (let k = 0; k < 3; k += 1) await net.ops.runProfiled(dn.program);
  const rows = [...net.ops.profile.entries()].map(([key, r]) => ({ key, ms: r.ms / 3, calls: r.calls / 3 }));
  const gpu = rows.reduce((t, r) => t + r.ms, 0);
  // the same op over every layer, summed: a key's layer number taken out
  const grouped = new Map();
  for (const r of rows) {
    const k = String(r.key).replace(/\d+/g, "#");
    const row = grouped.get(k) ?? { ms: 0, calls: 0 };
    row.ms += r.ms; row.calls += r.calls;
    grouped.set(k, row);
  }
  if (args.includes("--shapes")) {      // (the GEMMs by their own shape: which module a layer's time is)
    const shapes = new Map();
    for (const r of rows) {
      const k = String(r.key);
      const row = shapes.get(k) ?? { ms: 0, calls: 0 };
      row.ms += r.ms; row.calls += r.calls;
      shapes.set(k, row);
    }
    for (const [key, r] of [...shapes.entries()].sort((a, b) => b[1].ms - a[1].ms).slice(0, top))
      console.log(`  shape ${r.ms.toFixed(3).padStart(8)} ms  x${String(r.calls).padEnd(4)} ${key}`);
  }
  console.log(`${g.T} tokens, ${atomKeys.length} atoms, ${samples} samples`);
  console.log(`trunk: ${(trunkMs / 4).toFixed(1)} ms a pass (wall), ${trunkMs.toFixed(0)} ms for 4`);
  console.log(`denoiser call: ${callMs.toFixed(2)} ms wall; ${gpu.toFixed(2)} ms of profiled GPU over ${dn.program.steps.length} dispatches`);
  for (const [key, r] of [...grouped.entries()].sort((a, b) => b[1].ms - a[1].ms).slice(0, top)) {
    console.log(`  ${r.ms.toFixed(3).padStart(8)} ms ${(100 * r.ms / gpu).toFixed(1).padStart(5)}%  x${String(r.calls).padEnd(4)} ${key}`);
  }
  return { trunkPassMs: trunkMs / 4, callMs, gpuMs: gpu };
}
