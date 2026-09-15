/**
 * The FUSED template embedder against af3-any-model's own, on protenix2.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-af3-template-fused.js
 *
 * 🔴 THIS IS AN ORACLE CHECK AND check-af3-template.js IS NOT. That one compares
 * a GPU path against this repository's CPU reference - useful, and blind to any
 * error both halves share. protenix2's template embedder is a module LocalFold
 * had never run, so writing both halves from a specification would have made
 * them agree with each other and proved nothing. The dump here is
 * af3-any-model's `template_parity.ours` - the reference's own gate entry point,
 * with real protenix2 weights, 34 scopes mapped and 0 unmapped - so this
 * compares against something that was not written here.
 *
 * 🔴 AND IT CHECKS THE FORWARD, NOT THE FEATURISER. The dump carries the 108
 * feature columns separately from the output, which is the seam the reference's
 * own gate uses and documents: a check that derives features on both sides
 * cannot tell a wrong projection from a wrong frame convention. Those 108
 * columns go IN; the featuriser that would build them is not written yet and
 * docs/AF3.md has its specification.
 */
import { fusedTemplateEmbedding } from "../../src/af3/trunk/template-reference.js";
import { Af3TemplateEmbedderGpu, emptyFusedFeatures } from "../../src/af3/trunk/template-webgpu.js";
import { openAf3Store, templateWeights, af3Dialect } from "../../src/af3/weights/weights.js";

const DUMP = "/oracle-dumps/af3-oracle-template-protenix2.json";
const MODEL = "/model-protenix2-f32/manifest.json";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

function relativeRms(actual, expected) {
  let error = 0, scale = 0;
  for (let i = 0; i < expected.length; i += 1) {
    const d = actual[i] - expected[i];
    error += d * d; scale += expected[i] * expected[i];
  }
  return Math.sqrt(error / Math.max(scale, 1e-30));
}

export async function main(device, args) {
  const dumpPath = option(args, "dump", DUMP);
  const response = await fetch(dumpPath);
  if (!response.ok) throw new Error(`failed to load ${dumpPath}: ${response.status}`);
  const dump = await response.json();
  const raw = (name) => Float32Array.from(dump.inputs[name].data);

  const store = await openAf3Store(option(args, "model", MODEL));
  const dialect = af3Dialect(store);
  const weights = await templateWeights(store, dialect);
  if (!weights.fused) throw new Error("this bundle is not the fused embedder");

  const tokens = dump.tokens;
  const pairs = tokens * tokens;
  // 🔴 THE 108 COLUMNS, AND THE i/j ORDER IS THE TRAP THE ORACLE CAUGHT.
  // The reference's source says its concatenate puts the "j-varying block
  // FIRST" and records the other way round as worth corr 0.9985 against
  // 0.999998. Read literally, that gives `restype_j` then `restype_i` here -
  // and that scores **5.77e-2**, which is corr 0.9985 almost exactly. The
  // right order against `our_features`' output is `restype_i` then
  // `restype_j`, for 1.52e-7.
  //
  // Both statements are true: the reference is describing NATIVE's tensor
  // naming, where a name says which index the tensor varies along, and the
  // feature dict this dump records has already resolved that. A specification
  // read off someone else's source cannot settle which convention its words
  // are in. The oracle can, and did - in one run, with the failure landing on
  // the exact number the reference had written down for this mistake.
  const parts = [
    ["feat:template_distogram", 39],
    ["feat:template_pseudo_beta_mask", 1],
    ["feat:restype_i", 32],
    ["feat:restype_j", 32],
    ["feat:template_unit_vector", 3],
    ["feat:template_backbone_frame_mask", 1],
  ];
  const skipOracleEarly = option(args, "oracle", "on") === "off";
  const width = parts.reduce((sum, [, w]) => sum + w, 0);
  // 🔴 THE ORACLE'S COLUMNS ARE protenix2's AND THE GPU ARM DOES NOT NEED THEM.
  // boltz2 runs the same module on 109 columns in a different order, so this
  // construction refuses there - correctly - and `--oracle=off` runs the
  // internal GPU-against-CPU arm, which builds its own empty features from the
  // dialect and needs no dump at all.
  if (!skipOracleEarly && width !== weights.featureWidth) {
    throw new Error(`built ${width} feature columns and a_proj wants `
      + `${weights.featureWidth}`);
  }
  const features = new Float32Array(pairs * width);
  let offset = 0;
  for (const [name, w] of (skipOracleEarly ? [] : parts)) {
    const source = raw(name);
    if (source.length !== pairs * w) {
      throw new Error(`${name} has ${source.length} elements; expected ${pairs * w}`);
    }
    for (let index = 0; index < pairs; index += 1) {
      for (let c = 0; c < w; c += 1) {
        features[index * width + offset + c] = source[index * w + c];
      }
    }
    offset += w;
  }

  const got = skipOracleEarly ? new Float32Array(0) : fusedTemplateEmbedding({
    tokens, pair: raw("pair"), pairMask: raw("pairMask"),
    templates: dump.slots, templateFeatures: features,
  }, weights, dialect);
  const expected = skipOracleEarly ? new Float32Array(0)
    : Float32Array.from(dump.output.data);
  const skipOracle = option(args, 'oracle', 'on') === 'off';
  const relRms = skipOracle ? 0 : relativeRms(got, expected);
  const rms = (a) => Math.sqrt(a.reduce((s, v) => s + v * v, 0) / a.length);
  if (!skipOracleEarly) console.log(`fused template\ttokens=${tokens} slots=${dump.slots}`
    + `\trelRMS ${relRms.toExponential(2)}`
    + `\tours rms ${rms(got).toFixed(4)}\tnative rms ${rms(expected).toFixed(4)}`);

  // 🔴 A SEPARATION CONTROL, because an all-zero output would score perfectly
  // against an all-zero reference and this module is genuinely inert without a
  // template. Native's rms is 12.44 here; a comparison where either side is
  // flat is not a comparison.
  if (!skipOracle && (!(rms(expected) > 1e-3) || !(rms(got) > 1e-3))) {
    throw new Error(`one side is flat: ours ${rms(got)}, native ${rms(expected)}`);
  }
  const bound = Number(option(args, "bound", "2e-3"));
  if (!skipOracle && !(relRms < bound)) {
    throw new Error(`fused template relRMS ${relRms.toExponential(3)} exceeds `
      + `${bound.toExponential(0)}`);
  }
  // 🔴 AND THE GPU PATH, WHICH NOTHING ELSE REACHES. The arm above holds the
  // CPU forward to af3-any-model; the SHADER that a fold actually runs had no
  // check at all, and a protenix2 fold came out scrambled with every stage
  // "working". This compares them on the EMPTY slots a de novo fold builds,
  // which is the configuration that folds - and the features are the same
  // builder the fold uses, so a wrong constant shows up here rather than as a
  // broken chain three stages later.
  const slots = Number(option(args, "slots", "4"));
  const emptyFeatures = emptyFusedFeatures(undefined, tokens, weights.featureWidth, dialect);
  const pairMaskOnes = new Float32Array(pairs).fill(1);
  const cpuEmpty = fusedTemplateEmbedding({
    tokens, pair: raw("pair"), pairMask: pairMaskOnes,
    templates: slots, templateFeatures: emptyFeatures,
  }, weights, dialect);
  const gpuEmpty = await new Af3TemplateEmbedderGpu(device).run(
    { pair: raw("pair"), pairMask: pairMaskOnes, tokens, templates: slots,
      slots: undefined, asymId: new Int32Array(tokens).fill(1) },
    weights, dialect);
  const gpuRms = relativeRms(gpuEmpty.output, cpuEmpty);
  console.log(`fused template GPU	slots=${slots}	relRMS ${gpuRms.toExponential(2)}`
    + `	gpu rms ${rms(gpuEmpty.output).toFixed(4)}	cpu rms ${rms(cpuEmpty).toFixed(4)}`);
  if (!(rms(cpuEmpty) > 1e-3)) throw new Error("the empty-slot reference is flat");
  const gpuBound = Number(option(args, "gpu-bound", "2e-3"));
  if (!(gpuRms < gpuBound)) {
    throw new Error(`the fused template GPU path differs from its own reference by `
      + `${gpuRms.toExponential(3)}, over ${gpuBound.toExponential(0)}`);
  }

  return { tokens, slots: dump.slots, relRms, bound, gpuRms,
           oursRms: rms(got), nativeRms: rms(expected) };
}
