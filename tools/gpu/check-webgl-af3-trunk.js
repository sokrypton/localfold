/**
 * AF3's whole trunk in WebGL2 against `cpu/af3/trunk/trunk.js`.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-af3-trunk.js [--falsify]
 *
 * 🔴 THIS IS THE ASSEMBLY GATE, AND IT IS THE ONE ARM WITH A SINGLE REFERENCE -
 * ON PURPOSE AND WITH THE REASON. Every OPERATION in this trunk already has a
 * WGSL arm beside its f64 one (`check-webgl-af3-transition`, `-triangle`,
 * `-grid-attention`, `-block`, `-msa-block`, `-embedder`, `-template`), so what
 * is left to establish is the ORDER, which is what the f64 reference states
 * independently: it shares no code with any shader. Driving `Af3TrunkGpu` here
 * too would add a precision opinion on a composition whose every part already
 * has one, at the cost of a bundle this box does not have.
 *
 * What the per-piece gates cannot show: "that the pieces fit in the order AF3
 * runs them, and that fifty-two blocks of a residual stack do not turn a
 * correct block into a wrong answer."
 *
 * 🔴 FOUR SEAMS LIVE HERE AND NOWHERE ELSE:
 *
 *   - the TEMPLATE stack reads the pair as the embedder left it, and its term is
 *     added back to that same pair. Neither module owns that, so this one does.
 *   - the MSA stack hands the pairformer its pair, and boltz2 ADDS THE PRE-MSA
 *     PAIR BACK - "what reaches the pairformer is `2 * z_in + delta`" - which
 *     needs the embedder's pair held across four blocks.
 *   - the pairformer's single track starts from the EMBEDDER's single, not from
 *     anything the MSA stack produced.
 *   - and the distogram head is symmetrised by a SUM, not a mean: "the logits
 *     are twice the size a mean would give - which is not a rescaling once a
 *     softmax sees them".
 *
 * Both the pair and the single are compared, plus the logits and the contact
 * probabilities, because an error in the head is invisible in the pair and an
 * error in the single track never reaches either.
 */
import { runTrunk as cpuTrunk } from "../../cpu/af3/trunk/trunk.js";
import { templateEmbedding as cpuTemplate } from "../../cpu/af3/trunk/template.js";
import { setDeviceTuning, setHalfPrecision } from "../../webgpu/runtime/device-profile.js";
import { af3ContactBins } from "../../shared/af3/featurise/contact-classes.js";
import {
  RELATIVE_WIDTH, embed, releaseEmbedderWeights, tokenIdentity, uploadEmbedderWeights,
} from "../../webgl/af3/trunk/embedder.js";
import {
  RESTYPES, releaseTemplateWeights, slotCodes, slotGeometry, templateEmbedding,
  uploadTemplateWeights,
} from "../../webgl/af3/trunk/template.js";
import { binEdges, runTrunk } from "../../webgl/af3/trunk/trunk.js";
import {
  releaseMsaBlockWeights, uploadMsaBlockWeights,
} from "../../webgl/af3/trunk/msa-block.js";
import {
  releasePairformerBlockWeights, uploadPairformerBlockWeights,
} from "../../webgl/af3/trunk/pairformer-block.js";
import { Execution } from "../../webgl/runtime/execution.js";
import { createWebGL2Context } from "../../webgl/index.js";
import { DGRAM_BINS } from "../../shared/af3/featurise/template-features.js";
import { generator } from "./af2-synthetic-block.js";
import { relativeRms } from "./relative-rms.js";

const flag = (args, name) => (args ?? []).includes(`--${name}`);
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  const falsify = flag(args, "falsify");
  const bar = Number(option(args, "bar", "5e-5"));
  const tokens = Number(option(args, "tokens", "6"));
  const sequences = Number(option(args, "sequences", "3"));
  const msaBlockCount = Number(option(args, "msa-blocks", "2"));
  const pairformerBlockCount = Number(option(args, "pairformer-blocks", "2"));
  const templates = Number(option(args, "templates", "2"));
  const featureWidth = 40;
  const pairChannels = 32;
  const singleChannels = 64;
  const msaChannels = 16;
  const templateChannels = 64;
  const outerChannels = 8;
  const msaDoubleAddPair = flag(args, "msa-double-add-pair");
  const msaUpdateBeforeOuterProduct = flag(args, "msa-first");
  const pairs = tokens * tokens;
  const rows = sequences * tokens;

  const random = generator(1618033);
  const noise = (n, spread, centre = 0) => {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i += 1) out[i] = centre + (random() * 2 - 1) * spread;
    return out;
  };
  const triangleGroup = (channels) => ({
    leftNormInputScale: noise(channels, 0.2, 1),
    leftNormInputOffset: noise(channels, 0.1),
    projection: noise(channels * channels * 2, 0.3 / Math.sqrt(channels)),
    gate: noise(channels * channels * 2, 0.3 / Math.sqrt(channels)),
    centerNormScale: noise(channels, 0.2, 1),
    centerNormOffset: noise(channels, 0.1),
    outputProjection: noise(channels * channels, 0.3 / Math.sqrt(channels)),
    gatingLinear: noise(channels * channels, 0.3 / Math.sqrt(channels)),
  });
  const gridGroup = (channels, heads, dimension) => {
    const width = heads * dimension;
    return {
      heads, dimension,
      actNormScale: noise(channels, 0.2, 1),
      actNormOffset: noise(channels, 0.1),
      pairBiasProjection: noise(channels * heads, 1 / Math.sqrt(channels)),
      qProjection: noise(width * channels, 1 / Math.sqrt(channels)),
      kProjection: noise(width * channels, 1 / Math.sqrt(channels)),
      vProjection: noise(channels * width, 1 / Math.sqrt(channels)),
      gatingQuery: noise(width * channels, 1 / Math.sqrt(channels)),
      outputProjection: noise(width * channels, 0.3 / Math.sqrt(width)),
    };
  };
  const transitionGroup = (channels, factor) => ({
    inputLayerNormScale: noise(channels, 0.2, 1),
    inputLayerNormOffset: noise(channels, 0.1),
    transition1: noise(channels * channels * factor * 2, 1 / Math.sqrt(channels)),
    transition2: noise(channels * factor * channels, 0.3 / Math.sqrt(channels * factor)),
  });
  const pairHalfGroups = (channels, factor) => ({
    triangleMultiplicationOutgoing: triangleGroup(channels),
    triangleMultiplicationIncoming: triangleGroup(channels),
    pairAttention1: gridGroup(channels, 4, 8),
    pairAttention2: gridGroup(channels, 4, 8),
    pairTransition: transitionGroup(channels, factor),
  });

  const embedderWeights = {
    pairChannels, singleChannels, msaChannels, targetFeatWidth: featureWidth,
    relativeWidth: RELATIVE_WIDTH,
    dialect: { pairInitFromSingle: false },
    leftSingle: noise(featureWidth * pairChannels, 1 / Math.sqrt(featureWidth)),
    rightSingle: noise(featureWidth * pairChannels, 1 / Math.sqrt(featureWidth)),
    prevEmbeddingNormScale: noise(pairChannels, 0.2, 1),
    prevEmbeddingNormOffset: noise(pairChannels, 0.1),
    prevEmbedding: noise(pairChannels * pairChannels, 1 / Math.sqrt(pairChannels)),
    positionActivations: noise(RELATIVE_WIDTH * pairChannels, 0.3),
    msaActivations: noise(34 * msaChannels, 0.3),
    extraMsaTargetFeat: noise(featureWidth * msaChannels, 1 / Math.sqrt(featureWidth)),
    singleActivations: noise(featureWidth * singleChannels, 1 / Math.sqrt(featureWidth)),
    prevSingleEmbeddingNormScale: noise(singleChannels, 0.2, 1),
    prevSingleEmbeddingNormOffset: noise(singleChannels, 0.1),
    prevSingleEmbedding: noise(
      singleChannels * singleChannels, 1 / Math.sqrt(singleChannels)),
    bondEmbedding: noise(pairChannels, 0.3),
  };
  const templateWeights = {
    channels: templateChannels, queryChannels: pairChannels,
    queryEmbeddingNormScale: noise(pairChannels, 0.2, 1),
    queryEmbeddingNormOffset: noise(pairChannels, 0.1),
    templatePairEmbedding8: noise(
      pairChannels * templateChannels, 1 / Math.sqrt(pairChannels)),
    templatePairEmbedding2: noise(RESTYPES * templateChannels, 0.3),
    templatePairEmbedding3: noise(RESTYPES * templateChannels, 0.3),
    templatePairEmbedding0: noise(DGRAM_BINS * templateChannels, 0.3),
    templatePairEmbedding1: noise(templateChannels, 0.3),
    templatePairEmbedding4: noise(templateChannels, 0.3),
    templatePairEmbedding5: noise(templateChannels, 0.3),
    templatePairEmbedding6: noise(templateChannels, 0.3),
    templatePairEmbedding7: noise(templateChannels, 0.3),
    outputLayerNormScale: noise(templateChannels, 0.2, 1),
    outputLayerNormOffset: noise(templateChannels, 0.1),
    outputLinear: noise(templateChannels * pairChannels, 0.3 / Math.sqrt(templateChannels)),
    blocks: [pairHalfGroups(templateChannels, 2), pairHalfGroups(templateChannels, 2)],
  };
  const msaBlocks = Array.from({ length: msaBlockCount }, () => ({
    pairChannels, msaChannels,
    outerProductMean: {
      outerChannels,
      layerNormInputScale: noise(msaChannels, 0.2, 1),
      layerNormInputOffset: noise(msaChannels, 0.1),
      leftProjection: noise(msaChannels * outerChannels, 1 / Math.sqrt(msaChannels)),
      rightProjection: noise(msaChannels * outerChannels, 1 / Math.sqrt(msaChannels)),
      outputW: noise(outerChannels * outerChannels * pairChannels, 1 / outerChannels),
      outputB: noise(pairChannels, 0.1),
    },
    msaAttention1: {
      heads: 4, dimension: 8,
      actNormScale: noise(msaChannels, 0.2, 1),
      actNormOffset: noise(msaChannels, 0.1),
      pairNormScale: noise(pairChannels, 0.2, 1),
      pairNormOffset: noise(pairChannels, 0.1),
      pairLogits: noise(pairChannels * 4, 1 / Math.sqrt(pairChannels)),
      vProjection: noise(msaChannels * 32, 1 / Math.sqrt(msaChannels)),
      gatingQuery: noise(msaChannels * 32, 1 / Math.sqrt(msaChannels)),
      outputProjection: noise(32 * msaChannels, 0.3 / Math.sqrt(32)),
    },
    msaTransition: transitionGroup(msaChannels, 4),
    ...pairHalfGroups(pairChannels, 4),
  }));
  const pairformerBlocks = Array.from({ length: pairformerBlockCount }, () => ({
    pairChannels, singleChannels,
    ...pairHalfGroups(pairChannels, 4),
    singlePairLogitsNormScale: noise(pairChannels, 0.2, 1),
    singlePairLogitsNormOffset: noise(pairChannels, 0.1),
    singlePairLogitsProjection: noise(pairChannels * 4, 1 / Math.sqrt(pairChannels)),
    singleAttention: {
      heads: 4, dimension: 16,
      layerNormScale: noise(singleChannels, 0.2, 1),
      layerNormOffset: noise(singleChannels, 0.1),
      qProjection: noise(singleChannels * 64, 1 / Math.sqrt(singleChannels)),
      qBias: noise(64, 0.1),
      kProjection: noise(singleChannels * 64, 1 / Math.sqrt(singleChannels)),
      vProjection: noise(singleChannels * 64, 1 / Math.sqrt(singleChannels)),
      gatingQuery: noise(singleChannels * 64, 1 / Math.sqrt(singleChannels)),
      outputProjection: noise(64 * singleChannels, 0.3 / Math.sqrt(64)),
    },
    singleTransition: transitionGroup(singleChannels, 4),
  }));
  const distogramWeights = { halfLogits: noise(pairChannels * 64, 0.3) };

  // Two chains, and a template on the first.
  const asymId = new Int32Array(tokens);
  const entityId = new Int32Array(tokens);
  const symId = new Int32Array(tokens);
  const residueIndex = new Int32Array(tokens);
  const tokenIndex = new Int32Array(tokens);
  const half = Math.ceil(tokens / 2);
  for (let t = 0; t < tokens; t += 1) {
    asymId[t] = t < half ? 0 : 1;
    entityId[t] = asymId[t];
    symId[t] = 0;
    residueIndex[t] = t - (t < half ? 0 : half);
    tokenIndex[t] = residueIndex[t];
  }
  const seqMask = new Float32Array(tokens).fill(1);
  if (tokens > 2) seqMask[tokens - 1] = 0;
  const pairMask = new Float32Array(pairs);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) pairMask[i * tokens + j] = seqMask[i] * seqMask[j];
  }
  const msaMask = new Float32Array(rows);
  for (let s = 0; s < sequences; s += 1) {
    for (let t = 0; t < tokens; t += 1) {
      msaMask[s * tokens + t] = seqMask[t] > 0 && (s + t) % 4 !== 0 ? 1 : 0;
    }
  }
  const DENSE = 24;
  const slots = [];
  for (let slot = 0; slot < templates; slot += 1) {
    if (slot > 0) { slots.push(null); continue; }
    const aatype = new Int32Array(tokens);
    for (let t = 0; t < tokens; t += 1) aatype[t] = Math.floor(random() * 20);
    const atomMask = new Float32Array(tokens * DENSE).fill(1);
    for (let a = 0; a < DENSE; a += 1) atomMask[1 * DENSE + a] = 0;
    slots.push({ aatype, atomMask, atomPositions: noise(tokens * DENSE * 3, 11) });
  }
  const msaRows = new Int32Array(rows);
  const deletionMatrix = new Float32Array(rows);
  for (let index = 0; index < rows; index += 1) {
    msaRows[index] = Math.floor(random() * 34) - 1;
    deletionMatrix[index] = random() * 6;
  }
  const targetFeat = noise(tokens * featureWidth, 1);
  const previousPair = noise(pairs * pairChannels, 0.5);
  const previousSingle = noise(tokens * singleChannels, 0.5);
  // One contact class a token, so each pair gets its own counted-bin threshold.
  const contactClasses = new Int32Array(tokens);
  for (let t = 0; t < tokens; t += 1) contactClasses[t] = t % 2;
  const contactBins = af3ContactBins(contactClasses, tokens, binEdges());

  setDeviceTuning(device, { allowHostWeightPacking: true });
  setHalfPrecision(device, false);
  const gl = createWebGL2Context();
  const execution = new Execution(gl);
  const held = [];
  const upload = (values, label) => {
    const tensor = execution.upload(Float32Array.from(values), label);
    held.push(tensor);
    return tensor;
  };
  const uploadedEmbedder = uploadEmbedderWeights(execution, embedderWeights);
  const uploadedTemplate = uploadTemplateWeights(execution, templateWeights);
  const uploadedMsa = msaBlocks.map((b) => uploadMsaBlockWeights(execution, b));
  const uploadedPairformer = pairformerBlocks.map(
    (b) => uploadPairformerBlockWeights(execution, b));
  const halfLogits = upload(distogramWeights.halfLogits, "af3.trunk.half-logits");

  // The control skews half of each target row's channels: target_feat reaches
  // the pair, the MSA and the single track, so it is the one input every arm
  // reads.
  const targetTensor = upload(falsify
    ? Float32Array.from(targetFeat,
      (v, i) => (i % featureWidth < featureWidth / 2 ? v * 1.01 + 0.01 : v))
    : targetFeat, "af3.trunk.target");
  const pairMaskTensor = upload(pairMask, "af3.trunk.pair-mask");
  const embedded = embed(execution, {
    targetFeat: targetTensor,
    identity: upload(
      tokenIdentity({ residueIndex, tokenIndex, asymId, entityId, symId }, tokens),
      "af3.trunk.identity"),
    msaCodes: upload(msaRows, "af3.trunk.codes"),
    deletions: upload(deletionMatrix, "af3.trunk.deletions"),
    previousPair: upload(previousPair, "af3.trunk.previous-pair"),
    previousSingle: upload(previousSingle, "af3.trunk.previous-single"),
    weights: uploadedEmbedder, tokens, sequences, pairInitFromSingle: false,
  });
  const firstEmpty = slots.filter(Boolean).length;
  const templateUpdate = templateEmbedding(execution, {
    pair: embedded.pair, pairMask: pairMaskTensor, tokens, templates,
    weights: uploadedTemplate,
    codes: slots.map((template, slot) => upload(slotCodes({
      template, tokens, slot, firstEmpty, emptyTemplateAatype: null,
    }), `af3.trunk.codes${slot}`)),
    geometry: slots.map((template, slot) => {
      const packed = slotGeometry({ template, tokens, asymId });
      return packed === undefined ? undefined : upload(packed, `af3.trunk.geom${slot}`);
    }),
  });
  const result = runTrunk(execution, {
    embedded, templateUpdate,
    msaMask: upload(msaMask, "af3.trunk.msa-mask"),
    pairMask: pairMaskTensor,
    seqMask: upload(seqMask, "af3.trunk.seq-mask"),
    contactBins: upload(contactBins, "af3.trunk.contact-bins"),
    halfLogits,
    msaBlocks: uploadedMsa, pairformerBlocks: uploadedPairformer,
    tokens, sequences, pairChannels,
    msaUpdateBeforeOuterProduct, msaDoubleAddPair,
  });
  execution.release(templateUpdate);
  const mine = {
    pair: Float32Array.from(execution.read(result.pair)),
    single: Float32Array.from(execution.read(result.single)),
    logits: Float32Array.from(execution.read(result.logits)),
    contactProbs: Float32Array.from(execution.read(result.contactProbs)),
  };
  for (const tensor of Object.values(result)) execution.release(tensor);
  for (const tensor of held) execution.release(tensor);
  releaseEmbedderWeights(execution, uploadedEmbedder);
  releaseTemplateWeights(execution, uploadedTemplate);
  for (const w of uploadedMsa) releaseMsaBlockWeights(execution, w);
  for (const w of uploadedPairformer) releasePairformerBlockWeights(execution, w);

  const dialect = {
    swapTransposedBias: false,
    msaUpdateBeforeOuterProduct,
    ...(msaDoubleAddPair ? { msaDoubleAddPair: true } : {}),
  };
  const expected = cpuTrunk({
    tokens, sequences, targetFeat, msaRows, deletionMatrix,
    features: { residueIndex, tokenIndex, asymId, entityId, symId },
    previousPair, previousSingle, msaMask, pairMask, seqMask, contactClasses,
    // The template term as a FUNCTION of the pair, which is how the reference
    // gets the composition without knowing what a template is.
    templateEmbedding: (pair) => cpuTemplate(
      { tokens, pair, pairMask, templates, slots, asymId }, templateWeights, dialect),
  }, {
    embedder: embedderWeights, msaBlocks, pairformerBlocks, distogram: distogramWeights,
  }, dialect);

  const arms = [];
  const failures = [];
  for (const track of ["pair", "single", "logits", "contactProbs"]) {
    const relative = relativeRms(mine[track], expected[track], track);
    const ok = relative <= bar;
    arms.push({ arm: `${track} vs the f64 reference`,
      relativeRms: Number(relative.toPrecision(4)), bar, ok });
    if (!ok) failures.push(`${track}: relRMS ${relative.toExponential(3)} over ${bar}`);
  }

  if (execution.stats().live !== 0) {
    failures.push(`${execution.stats().live} tensors left live`);
  }
  const red = arms.filter((row) => !row.ok).length;
  return {
    shape: {
      tokens, sequences, msaBlocks: msaBlockCount, pairformerBlocks: pairformerBlockCount,
      templates, pairChannels, singleChannels, msaChannels, templateChannels,
    },
    chains: 2,
    occupiedSlots: firstEmpty,
    msaUpdateBeforeOuterProduct,
    msaDoubleAddPair,
    bar,
    falsify,
    arms,
    memory: execution.stats(),
    verdict: falsify
      ? (red === arms.length
        ? `ok: all ${arms.length} arms go red on a 1% perturbation`
        : `VACUOUS: only ${red} of ${arms.length} arms noticed it`)
      : (failures.length === 0
        ? `ok: the whole trunk, ${arms.length} arms against the f64 reference`
        : `FAILED: ${failures.join("; ")}`),
  };
}
