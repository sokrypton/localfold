/**
 * The FUSED template embedder's 108 feature columns, built from a structure and
 * checked against af3-any-model's own.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-fused-template-features.js
 *
 * 🔴 THIS IS THE HALF THAT DID NOT EXIST. `check-af3-template-fused.js` hands
 * the module the 108 columns out of the dump and reads 1.52e-7 - and boltz2 and
 * protenix2 could not take a template AT ALL, because nothing built those
 * columns from a structure ("the fused template embedder has no featuriser
 * yet"). The gate passed and the feature was missing.
 *
 * 🔴 AND THE COLUMNS ARE THE NINE-PROJECTION EMBEDDER'S OWN FEATURES. 39 + 1 +
 * 32 + 32 + 3 + 1 = 108, which is exactly what `templateGeometry` already
 * computes for AF3's path plus the two restype one-hots - so the featuriser is
 * a CONCATENATION of things this port has had all along, not new geometry. That
 * is the claim this checker exists to test rather than assert.
 */
import { templateGeometry, DGRAM_BINS } from "../../src/af3/featurise/template-features.js";

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

export async function main(device, args) {
  const model = option(args, "name", "protenix2");
  const path = option(args, "dump", `/oracle-dumps/af3-oracle-template-${model}.json`);
  const response = await fetch(path);
  if (!response.ok) throw new Error(`failed to load ${path}: ${response.status}`);
  const dump = await response.json();
  const of = (name) => {
    const entry = name === "output" ? dump.output : dump.inputs[name];
    if (entry === undefined) throw new Error(`the dump has no ${name}`);
    return { data: Float32Array.from(entry.data), shape: entry.shape };
  };

  const tokens = dump.tokens;
  const aatypeRaw = of("template_aatype");
  const positions = of("template_atom_positions");
  const atomMask = of("template_atom_mask");
  const slots = positions.shape[1];
  const multichain = of("multichainMask2d").data;

  // 🔴 BOLTZ-2's DUMP HAS NO `feat:` COLUMNS, because its module builds its own
  // 109 channels from RAW geometry - so for it the only check is the FORWARD:
  // build the features, run the embedder, compare the output the dump recorded
  // with real weights.
  // `--forward` takes the forward branch for a model whose dump DOES carry the
  // columns, so the same GPU invocation can be run as a control. A GPU arm that
  // disagrees for one model and not another is a model bug; one that disagrees
  // for both is this checker.
  if (dump.inputs["feat:restype_i"] === undefined || args.includes("--forward")) {
    const { boltz2TemplateFeatures } = await import("../../src/af3/featurise/template-features.js");
    const { openAf3Store, templateWeights, af3Dialect } =
      await import("../../src/af3/weights/weights.js");
    const { fusedTemplateEmbedding } = await import("../../src/af3/trunk/template-reference.js");
    const store = await openAf3Store(option(args, "model", `/model-${model}-f32/manifest.json`));
    store.prefetch();
    const dialect = af3Dialect(store);
    const weights = await templateWeights(store, dialect);
    const { fusedTemplateFeatures } = await import("../../src/af3/trunk/template-webgpu.js");
    const templateForFeatures = {
      aatype: Int32Array.from(aatypeRaw.data),
      atomPositions: positions.data, atomMask: atomMask.data,
    };
    const built = dialect.boltz2TemplateFeatures === true
      ? boltz2TemplateFeatures(templateForFeatures, multichain, tokens)
      : fusedTemplateFeatures(templateForFeatures, tokens, weights.featureWidth,
                              dialect, multichain);
    const got = fusedTemplateEmbedding({
      tokens, pair: of("pair").data, pairMask: of("pairMask").data,
      templates: 1, templateFeatures: built,
      // 🔴 boltz2 SKIPS A SLOT IT CANNOT SEE. `templateVisibilityByCoverage`
      // drops any slot whose `input.slots[slot]` is absent, so a checker that
      // passes the features and not the slot gets rms 0.0000 out - which reads
      // as "our forward computes nothing" and is the checker forgetting to say
      // the template is there.
      slots: [{ aatype: Int32Array.from(aatypeRaw.data),
                atomPositions: positions.data, atomMask: atomMask.data }],
    }, weights, dialect);
    const native = of("output");
    // 🔴 AND THE GPU PATH, WHICH IS WHAT A FOLD RUNS. The CPU reference being
    // exact says the featuriser and the arithmetic are right; it says nothing
    // about the kernels that ship, and boltz2's 5CAJ fold with a template lands
    // at 2.2 A where protenix2 and AF3 reach 0.2.
    const { Af3TemplateEmbedderGpu } = await import("../../src/af3/trunk/template-webgpu.js");
    // 🔴 THE PINS GO THROUGH THE CONSTRUCTOR, NOT `--tune`. This checker builds
    // the embedder itself, so a `--tune=` on the command line reaches nothing
    // here - three arms of it read the identical 5.11e-4 and said only that
    // they had not run. `--pins` sets the four the confidence head carries.
    const pins = args.includes("--pins")
      ? { stagedPrecision: "f32", weightPrecision: "f32",
          accumulatePrecision: "f32", pairMatrixKernels: false }
      : {};
    const gpuOut = await new Af3TemplateEmbedderGpu(device, pins).run(
      { pair: of("pair").data, pairMask: of("pairMask").data, tokens, templates: 1,
        slots: [{ aatype: Int32Array.from(aatypeRaw.data),
                  atomPositions: positions.data, atomMask: atomMask.data }],
        asymId: new Int32Array(tokens).fill(1) },
      weights, dialect);
    // ...the runner returns an object on this path; the pair is the field.
    const gpu = gpuOut instanceof Float32Array ? gpuOut
      : (gpuOut.pair ?? gpuOut.output ?? gpuOut.act);
    if (gpu === undefined) {
      return { model, gpuShape: Object.keys(gpuOut),
               note: "the GPU runner returned no recognised field" };
    }
    // 🔴 THE RESIDUAL, HELD ON BOTH SIDES. boltz2's GPU arm is 0.748 where
    // protenix2's is 3.9e-5 on the same invocation, and `templateStackOuterResidual`
    // is the one convention boltz2 has and protenix2 does not. Forcing it OFF in
    // BOTH says whether it is the residual or something else that differs: if
    // the two agree without it, the residual is the defect; if they still
    // disagree, it is not.
    // 🔴 THE EMBED ALONE, both sides. `z_proj(z_norm(z)) + a_proj(a)` before the
    // pairformer blocks: if the GPU already differs here it is the projection
    // or the features reaching it, and if it does not it is the pair track.
    // The residual and the output LayerNorm are held off on the CPU side so the
    // two are the same quantity.
    const flat = { ...dialect, templateStackOuterResidual: false };
    const cpuEmbedOut = await new Af3TemplateEmbedderGpu(device, pins).run(
      { pair: of("pair").data, pairMask: of("pairMask").data, tokens, templates: 1,
        slots: [{ aatype: Int32Array.from(aatypeRaw.data),
                  atomPositions: positions.data, atomMask: atomMask.data }],
        asymId: new Int32Array(tokens).fill(1) },
      weights, flat, { stopAfterEmbed: true });
    const gpuEmbed = cpuEmbedOut instanceof Float32Array ? cpuEmbedOut
      : (cpuEmbedOut.pair ?? cpuEmbedOut.output ?? cpuEmbedOut.act);
    // ...and the CPU's same quantity: the whole module with NO pairformer
    // blocks, which is what `stopAfterEmbed` leaves the GPU doing.
    const cpuEmbed = fusedTemplateEmbedding({
      tokens, pair: of("pair").data, pairMask: of("pairMask").data,
      templates: 1, templateFeatures: built,
      slots: [{ aatype: Int32Array.from(aatypeRaw.data),
                atomPositions: positions.data, atomMask: atomMask.data }],
    }, { ...weights, blocks: [] }, flat);

    const without = { ...dialect, templateStackOuterResidual: false };
    const cpuNoResidual = fusedTemplateEmbedding({
      tokens, pair: of("pair").data, pairMask: of("pairMask").data,
      templates: 1, templateFeatures: built,
      slots: [{ aatype: Int32Array.from(aatypeRaw.data),
                atomPositions: positions.data, atomMask: atomMask.data }],
    }, weights, without);
    const gpuNoResidualOut = await new Af3TemplateEmbedderGpu(device, pins).run(
      { pair: of("pair").data, pairMask: of("pairMask").data, tokens, templates: 1,
        slots: [{ aatype: Int32Array.from(aatypeRaw.data),
                  atomPositions: positions.data, atomMask: atomMask.data }],
        asymId: new Int32Array(tokens).fill(1) },
      weights, without);
    const gpuNoResidual = gpuNoResidualOut instanceof Float32Array ? gpuNoResidualOut
      : (gpuNoResidualOut.pair ?? gpuNoResidualOut.output ?? gpuNoResidualOut.act);
    return { model, tokens, slots, width: built.length / (tokens * tokens),
             forwardRelRms: Number(relRms(got, native.data).toExponential(2)),
             embedGpuVsCpu: Number(relRms(gpuEmbed, cpuEmbed).toExponential(2)),
             gpuEmbedRms: Math.sqrt(gpuEmbed.reduce((t, v) => t + v * v, 0) / gpuEmbed.length),
             cpuEmbedRms: Math.sqrt(cpuEmbed.reduce((t, v) => t + v * v, 0) / cpuEmbed.length),
             gpuVsCpuWithoutResidual:
               Number(relRms(gpuNoResidual, cpuNoResidual).toExponential(2)),
             gpuVsNative: Number(relRms(gpu, native.data).toExponential(2)),
             gpuVsCpu: Number(relRms(gpu, got).toExponential(2)),
             gpuRms: Math.sqrt(gpu.reduce((t, v) => t + v * v, 0) / gpu.length),
             oursRms: Math.sqrt(got.reduce((t, v) => t + v * v, 0) / got.length),
             nativeRms: Math.sqrt(native.data.reduce((t, v) => t + v * v, 0)
                                  / native.data.length) };
  }

  const geometry = templateGeometry({
    aatype: Int32Array.from(aatypeRaw.data),
    atomPositions: positions.data,
    atomMask: atomMask.data,
  }, multichain, tokens);

  const arms = [];
  const compare = (label, ours, nativeName) => {
    const native = of(nativeName);
    if (ours.length !== native.data.length) {
      arms.push({ feature: label, lengthMismatch: [ours.length, native.data.length] });
      return;
    }
    arms.push({ feature: label, shape: native.shape,
                relRms: Number(relRms(ours, native.data).toExponential(2)) });
  };

  compare("distogram", geometry.distogram, "feat:template_distogram");
  compare("pseudo_beta_mask", geometry.pseudoBetaMask2d, "feat:template_pseudo_beta_mask");
  compare("unit_vector", geometry.unitVector, "feat:template_unit_vector");
  compare("backbone_frame_mask", geometry.backboneMask2d, "feat:template_backbone_frame_mask");

  // The two restype one-hots, which the geometry does not carry: AF3's feature
  // 2 varies along j and feature 3 along i - the order the oracle caught once
  // already, recorded in check-af3-template-fused.js.
  const restypeWidth = of("feat:restype_i").shape[2];
  const restypeI = new Float32Array(tokens * tokens * restypeWidth);
  const restypeJ = new Float32Array(tokens * tokens * restypeWidth);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) {
      const base = (i * tokens + j) * restypeWidth;
      // 🔴 `restype_i` VARIES ALONG j AND `restype_j` ALONG i. The name says
      // which index the tensor varies along in the reference's own naming, not
      // which index selects its value - the same trap
      // check-af3-template-fused.js records for the 108-column ORDER, one level
      // down. Built the other way both score 1.36; built this way they are
      // exact, and the aatype needs no remap at all (the recovered table is the
      // identity).
      const ci = aatypeRaw.data[j], cj = aatypeRaw.data[i];
      if (ci >= 0 && ci < restypeWidth) restypeI[base + ci] = 1;
      if (cj >= 0 && cj < restypeWidth) restypeJ[base + cj] = 1;
    }
  }
  compare("restype_i", restypeI, "feat:restype_i");
  compare("restype_j", restypeJ, "feat:restype_j");

  // 🔴 WHICH COLUMN IS HOT NATIVELY? The template restype is a REMAP, not the
  // batch's aatype - docs/AF3.md says 32 classes where the model's alphabet is
  // 31 - and the mapping is recoverable from the dump rather than guessable.
  const nativeI = of("feat:restype_i").data;
  const observed = new Map();
  // 🔴 SAMPLED ALONG j, NOT i. `feat:restype_i` at (i, 0) is CONSTANT in i -
  // the name says which index the tensor varies along in NATIVE's naming, not
  // which one indexes its value, and reading it the other way returned "every
  // aatype maps to column 12", a table with one entry pretending to be a map.
  for (let j = 0; j < tokens; j += 1) {
    const base = (0 * tokens + j) * restypeWidth;
    let hot = -1;
    for (let c = 0; c < restypeWidth; c += 1) if (nativeI[base + c] === 1) { hot = c; break; }
    const from = aatypeRaw.data[j];
    if (!observed.has(from)) observed.set(from, new Set());
    observed.get(from).add(hot);
  }
  const remap = [...observed.entries()].sort((a, b) => a[0] - b[0])
    .map(([from, to]) => ({ aatype: from, hot: [...to] }));
  const bound = Number(option(args, "bound", "1e-5"));
  const worst = arms.filter((a) => a.relRms !== undefined)
    .reduce((w, a) => (w === null || a.relRms > w.relRms ? a : w), null);
  return {
    model, tokens, slots, restypeWidth, dgramBins: DGRAM_BINS,
    totalWidth: DGRAM_BINS + 1 + restypeWidth * 2 + 3 + 1,
    arms, worst, bound, remap,
    agrees: arms.every((a) => a.relRms !== undefined && a.relRms <= bound),
  };
}
