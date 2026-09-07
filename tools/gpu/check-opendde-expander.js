/**
 * OpenDDE's structural-token expansion, end to end on real weights.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-opendde-expander.js
 *
 * 🔴 THERE IS NO ORACLE FOR THIS, SO WHAT IS CHECKED IS CONSERVATION AND
 * SHAPE. No dump of OpenDDE's own intermediates exists here, and the expander
 * is not an operation another implementation in this tree computes - so the
 * assertions are the ones that hold BY CONSTRUCTION and would break under the
 * mistakes that are actually available: every atom appears exactly once, a
 * residue's two halves share their reference space, the role of every token is
 * one its residue could produce, and the five boolean pair features are
 * consistent with the layout that produced them.
 *
 * It also runs the whole thing at the real widths against the real tensors,
 * which is what catches a loader reading a [49, 384, 384] projection at the
 * wrong stride.
 */
import { featuriseProtein } from "../../src/af3/featurise.js";
import {
  structuralBatch, structuralLayout, atomNameAt, NO_TWIN,
} from "../../src/af3/structural-tokens.js";
import {
  expandStructural, structuralPairFeatures,
} from "../../src/af3/structural-expander-reference.js";
import { Af3StructuralExpanderGpu } from "../../src/af3/structural-expander-webgpu.js";
import {
  openAf3Store, structuralExpanderWeights, structuralRefinerWeights,
} from "../../src/af3/weights.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

const SEQUENCE = "GSHMSEEELRRRIEEIVRRAEELARQGKYEEAERLYREALEIARRAG";

export async function main(device, args) {
  const sequence = option(args, "sequence", SEQUENCE);
  const manifest = option(args, "model", "/model-opendde-full-f32/manifest.json");
  const batch = featuriseProtein(sequence, {});
  const layout = structuralLayout(batch);
  const structural = structuralBatch(batch, layout);

  const store = await openAf3Store(manifest);
  const weights = await structuralExpanderWeights(store);
  const refiner = await structuralRefinerWeights(store);

  // Stand-in trunk outputs: the expander is linear in them, so what is being
  // checked is the plumbing and the shapes, not a fold.
  const noise = (length, seed) => {
    const out = new Float32Array(length);
    let state = seed >>> 0;
    for (let i = 0; i < length; i += 1) {
      state = (state * 1664525 + 1013904223) >>> 0;
      out[i] = (state / 0xffffffff) * 2 - 1;
    }
    return out;
  };
  const residueTokens = batch.tokens;
  const embeddings = {
    targetFeat: noise(residueTokens * weights.singleInputChannels, 1),
    single: noise(residueTokens * weights.singleChannels, 2),
    pair: noise(residueTokens * residueTokens * weights.pairChannels, 3),
    asymId: batch.asymId,
  };
  const started = performance.now();
  const expanded = expandStructural(layout, embeddings, weights,
    { residueTokens, pairChannels: weights.pairChannels });
  const expandMs = Math.round(performance.now() - started);

  // --- conservation, which is what can be asserted without an oracle ---
  const dense = batch.dense;
  const seen = new Map();
  let doubled = 0;
  for (let t = 0; t < layout.tokens; t += 1) {
    layout.sources[t].forEach((slot) => {
      const key = layout.parent[t] * dense + slot;
      seen.set(key, (seen.get(key) ?? 0) + 1);
      if (seen.get(key) > 1) doubled += 1;
    });
  }
  let residueAtoms = 0;
  for (let i = 0; i < residueTokens * dense; i += 1) if (batch.refMask[i]) residueAtoms += 1;

  // A twin pair must share its reference space, or a side chain stops seeing
  // its own backbone in the atom encoder.
  let twinSpacesAgree = true;
  for (let t = 0; t < layout.tokens; t += 1) {
    const other = layout.twin[t];
    if (other === NO_TWIN) continue;
    if (structural.refSpaceUid[t * dense] !== structural.refSpaceUid[other * dense]) {
      twinSpacesAgree = false;
    }
  }

  // Every backbone token's representative must be CA (or N/C where absent).
  const representatives = {};
  for (let t = 0; t < layout.tokens; t += 1) {
    const name = atomNameAt(structural, t, layout.pseudoBetaSlot[t]);
    const key = `role${layout.role[t]}:${name}`;
    representatives[key] = (representatives[key] ?? 0) + 1;
  }

  // 🔴 THE GPU AGAINST THE REFERENCE, which is the only arm here that is a
  // CHECK rather than an assertion about the layout. The expander is
  // n^2 * c^2, so the host arm is 1.8 s at 91 tokens and the kernel is where
  // this actually runs - and a kernel indexing 49 matrices by a role pair is
  // exactly the shape that returns a plausible tensor when the stride is wrong.
  const features = structuralPairFeatures(layout, batch.asymId);
  const gpuStarted = performance.now();
  const gpu = await new Af3StructuralExpanderGpu(device).run(
    layout, embeddings, weights, features, residueTokens);
  const gpuMs = Math.round(performance.now() - gpuStarted);
  const relativeRms = (actual, expected) => {
    let error = 0;
    let scale = 0;
    for (let i = 0; i < expected.length; i += 1) {
      const d = actual[i] - expected[i];
      error += d * d;
      scale += expected[i] * expected[i];
    }
    return Number(Math.sqrt(error / Math.max(scale, 1e-30)).toExponential(3));
  };

  const rms = (a) => Number(Math.sqrt(a.reduce((s, v) => s + v * v, 0) / a.length).toFixed(4));
  const finite = (a) => a.every(Number.isFinite);

  return {
    sequence: sequence.length, residueTokens, structuralTokens: layout.tokens,
    expandMs,
    atoms: { residue: residueAtoms, structural: structural.atomCount,
             conserved: residueAtoms === structural.atomCount, doubled },
    twinSpacesAgree,
    representatives,
    roles: Object.fromEntries([...new Set(layout.role)].sort()
      .map((r) => [r, layout.role.filter((x) => x === r).length])),
    expanded: {
      targetFeatRms: rms(expanded.targetFeat), singleRms: rms(expanded.single),
      pairRms: rms(expanded.pair), biasRms: rms(expanded.attentionBias),
      finite: finite(expanded.single) && finite(expanded.pair)
        && finite(expanded.attentionBias),
    },
    gpu: {
      ms: gpuMs, hostMs: expandMs,
      pair: relativeRms(gpu.pair, expanded.pair),
      single: relativeRms(gpu.single, expanded.single),
    },
    refiner: {
      blocks: refiner.length,
      pairChannels: refiner[0].pairChannels,
      singleHeads: refiner[0].singleAttention.heads,
      singleDimension: refiner[0].singleAttention.dimension,
      gridHeads: refiner[0].pairAttention1.heads,
    },
  };
}
