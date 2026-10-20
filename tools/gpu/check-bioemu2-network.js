// Does webgpu/bioemu2 compute af3-any-model's BioEmu2, stage by stage, on a REAL structure?
//
//     node tools/gpu-chrome.mjs tools/gpu/check-bioemu2-network.js --dump=bioemu2-6mrr \
//       [--model=/model-bioemu2-f32] [--half=off] [--trunk=f32] [--stages=all|embed|trunk|denoise]
//
// The dump is tools/oracle/dump_bioemu2.py's: colab's own featuriser and network, float32, 'highest' precision.
// Each stage is fed the DUMP's inputs, never this port's previous output, so a residual is that stage's:
//
//   s_inputs              the input embedder (three windowed atom layers and the token mean)
//   z_init, pass1..4      the trunk: z_init, then each of the four recycled passes (single and pair)
//   distogram             the contact head on the dump's final pair
//   ctx.z / ctx.p         the denoiser's static context (pair conditioning, atom window pair)
//   denoise0..3           one denoiser call at sigma 160 / 16 / 1 / 0.05 on a real sample re-noised
//   denoise_dyn           the same at sigma 4, conditioned on the sample at a 1 ns lag (propagation)
//
// 🔴 THE BOUNDS FOLLOW THE ARITHMETIC, NOT A GUESS. --half=off --trunk=f32 is the float32 arm and the one held to
// 1e-4 (5e-4 for the four trunk passes, which run 52 pair-track blocks each); the shipped arm (float16 weights, the
// device's pair-track precision) is reported and held to 2e-2, and an int5 bundle against a float32 dump is the
// quantisation, measured rather than bounded.
import { Bioemu2Gpu } from "../../webgpu/bioemu2/network.js";
import { bundleReader } from "../../shared/bioemu2/weights.js";
import { fetchBioemu2Dump, featuresOf, relRms } from "./bioemu2-dump.js";

const option = (args, name, fallback) => (args ?? []).find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;

export async function main(device, args) {
  const dumpName = option(args, "dump", "bioemu2-6mrr");
  const model = option(args, "model", "/model-bioemu2-f32");
  const half = option(args, "half", "on") !== "off";
  const trunkPrecision = option(args, "trunk", "device");
  const which = option(args, "stages", "all");
  const exact = !half && trunkPrecision === "f32" && !model.includes("int");
  const { stages } = await fetchBioemu2Dump(`/oracle-dumps/${dumpName}.json`);
  const f = featuresOf(stages);
  const net = new Bioemu2Gpu(device, bundleReader(model), { half, trunkPrecision });
  const g = net.upload(f);
  const realAtoms = f.atom_mask.reduce((s, v) => s + v, 0);
  const rows = [];
  const report = (name, ours, want, bound, count) => {
    const residual = relRms(ours, want, count);
    rows.push({ stage: name, relRms: Number(residual.toExponential(3)), bound, ok: residual <= bound });
    console.log(`  ${name.padEnd(14)} ${residual.toExponential(2)}  (bound ${bound})`);
  };
  const tight = (b) => (exact ? b : model.includes("int") ? Infinity : 2e-2);
  const t0 = performance.now();

  if (which === "all" || which === "embed") {
    const sIn = await net.inputEmbedder(g);
    report("s_inputs", sIn, stages.s_inputs, tight(1e-4));
  }
  const sInputs = stages.s_inputs;
  if (which === "all" || which === "trunk") {
    const trunk = await net.trunkProgram(g, sInputs);
    report("z_init", await net.ops.read(trunk.zInit), stages.z_init, tight(1e-4));
    report("s_init", trunk.sInit, stages.s_init, tight(1e-4));
    for (let k = 1; k <= 4; k += 1) {
      // each pass from the DUMP's previous one, so a pass's residual is its own
      if (k > 1) {
        net.ops.write(trunk.zPrev, stages[`pass${k - 1}.pair`]);
        net.ops.write(trunk.sPrev, stages[`pass${k - 1}.single`]);
      }
      await trunk.pass();
      report(`pass${k}.single`, await net.ops.read(trunk.s), stages[`pass${k}.single`], tight(5e-4));
      report(`pass${k}.pair`, await net.ops.read(trunk.z), stages[`pass${k}.pair`], tight(5e-4));
    }
  }
  const zTrunk = net.ops.buffer("check.z", stages["pass4.pair"].length, stages["pass4.pair"]);
  const sTrunk = stages["pass4.single"];
  if (which === "all" || which === "trunk") {
    const { logits } = await net.distogram(g, zTrunk);
    report("distogram", logits, stages.distogram, tight(1e-4));
  }
  if (which === "all" || which === "denoise") {
    const dn = await net.denoiserProgram(g, sInputs, sTrunk, zTrunk);
    report("ctx.z", await net.ops.read(dn.context.z), stages["ctx.z"], tight(1e-4));
    report("ctx.q0", dn.context.q0, stages["ctx.q0"], tight(1e-5));
    report("ctx.p", dn.context.p, stages["ctx.p"], tight(1e-4));
    for (let k = 0; k < 4; k += 1) {
      if (stages[`denoise${k}.x`] === undefined) break;
      const out = await dn.denoise(stages[`denoise${k}.x`], stages[`denoise${k}.sigma`]);
      report(`denoise${k}`, out, stages[`denoise${k}.out`], tight(1e-4), realAtoms * 3);
    }
    if (stages["denoise_dyn.x"] !== undefined) {
      const dyn = await net.denoiserProgram(g, sInputs, sTrunk, zTrunk, {
        dt: stages["denoise_dyn.dt"], dynamics: true, coordsCond: stages["denoise_dyn.coords_cond"] });
      const out = await dyn.denoise(stages["denoise_dyn.x"], stages["denoise_dyn.sigma"]);
      report("denoise_dyn", out, stages["denoise_dyn.out"], tight(1e-4), realAtoms * 3);
    }
  }
  const failed = rows.filter((r) => !r.ok).map((r) => r.stage);
  const result = { dump: dumpName, model, half, trunkPrecision, tokens: g.T, atoms: realAtoms,
                   deviceMiB: Math.round(net.ops.bytes / 1048576), seconds: (performance.now() - t0) / 1000, rows };
  if (failed.length) throw new Error(`${failed.join(", ")} past their bound: ${JSON.stringify(result)}`);
  return result;
}
