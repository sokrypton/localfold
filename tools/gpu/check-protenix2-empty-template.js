/**
 * protenix2's template term as the TRUNK builds it: four EMPTY slots, 68 tokens.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-protenix2-empty-template.js
 *
 * 🔴 THE MODULE IS EXACT AND THE TRUNK'S TERM IS NOT. Standalone against
 * af3-any-model, with a real template and the dump's own 108 columns,
 * `check-af3-template-fused.js` reads 1.52e-7 on the CPU and 3.9e-5 on the GPU.
 * In the trunk, on 6MRR - which has NO template, so four empty slots - the term
 * reads **7.39e-3** and drags `z_after_template` to 3.89e-3 and the whole pair
 * to 3.98e-3, the last blemish on protenix2's trunk.
 *
 * Everything the two runs share is exact, so the difference is in what they do
 * NOT share: one template against four empty slots, 76 tokens against 68, and
 * the dump's pair against the trunk's own. This drives the CPU module with the
 * TRUNK's inputs - the reference's own `z_init_generic` at 68 tokens, four
 * empty slots - and compares against the reference's own
 * `evoformer/template_embedding` from the same trunk run. If the CPU matches,
 * the GPU path is the defect; if it does not, the empty-slot handling is.
 */
import { fusedTemplateEmbedding } from "../../src/af3/template-reference.js";
import { Af3TemplateEmbedderGpu, fusedTemplateFeatures }
  from "../../src/af3/template-webgpu.js";
import { openAf3Store, templateWeights, af3Dialect } from "../../src/af3/weights.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};
const relRms = (ours, expected) => {
  let error = 0, scale = 0;
  for (let i = 0; i < expected.length; i += 1) {
    const d = ours[i] - expected[i];
    error += d * d; scale += expected[i] * expected[i];
  }
  return Math.sqrt(error / Math.max(scale, 1e-30));
};
const rms = (a) => Math.sqrt(a.reduce((t, v) => t + v * v, 0) / a.length);

export async function main(device, args) {
  const path = option(args, "dump", "/oracle-dumps/af3-oracle-tmpl-protenix2.json");
  const response = await fetch(path);
  if (!response.ok) throw new Error(`failed to load ${path}: ${response.status}`);
  const stages = (await response.json()).stages;
  const of = (name) => Float32Array.from(stages[name].data);

  const store = await openAf3Store(option(args, "model", "/model-protenix2-f32/manifest.json"));
  store.prefetch();
  const dialect = af3Dialect(store);
  const weights = await templateWeights(store, dialect);

  const zInit = of("tap.z_init_generic");
  const nativeTerm = of("tap.template_term");
  const tokens = Math.round(Math.sqrt(zInit.length / weights.queryChannels));
  const pairs = tokens * tokens;
  const pairMask = new Float32Array(pairs).fill(1);
  const asymId = new Int32Array(tokens).fill(1);

  const arms = [];
  for (const templates of [1, 2, 4]) {
    // The trunk's own empty features, one per slot - built the way the fold
    // builds them.
    const features = fusedTemplateFeatures(undefined, tokens, weights.featureWidth, dialect);
    const cpu = fusedTemplateEmbedding({
      tokens, pair: zInit, pairMask, templates, templateFeatures: features,
    }, weights, dialect);
    const gpu = await new Af3TemplateEmbedderGpu(device).run(
      { pair: zInit, pairMask, tokens, templates, slots: undefined, asymId },
      weights, dialect);
    const gpuPair = gpu instanceof Float32Array ? gpu : (gpu.pair ?? gpu.output ?? gpu.act);
    arms.push({
      templates,
      cpuVsNative: Number(relRms(cpu, nativeTerm).toExponential(2)),
      gpuVsNative: Number(relRms(gpuPair, nativeTerm).toExponential(2)),
      gpuVsCpu: Number(relRms(gpuPair, cpu).toExponential(2)),
      cpuRms: Number(rms(cpu).toFixed(4)), gpuRms: Number(rms(gpuPair).toFixed(4)),
    });
  }
  return { tokens, nativeRms: Number(rms(nativeTerm).toFixed(4)),
           featureWidth: weights.featureWidth, arms };
}
