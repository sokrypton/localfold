/**
 * Does the sample dimension leave the one-sample path alone?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-difftx-samples.js
 *
 * 🔴 THE POINT OF A SECOND PATH IS THAT THE FIRST ONE DOES NOT MOVE, and the
 * only way to know that is to compare the generated WGSL rather than a fold's
 * pLDDT. A shape that batches has to reach every kernel in this stack - the row
 * axis, the conditioning lookup, the pair bias, the attention bounds - and each
 * of those is a chance to change the one-sample text by accident. A fold gate
 * would catch a WRONG kernel; it would not catch a slower one, and this stack's
 * whole history is of changes that were correct and cost 20%.
 *
 * So this generates every shader at samples 1 against a build with the sample
 * parameter absent entirely, and requires them identical character for
 * character.
 */
import { createDiffusionTransformerShaders } from "../../src/af3/diffusion-transformer-webgpu.js";

const NAMES = ["adaln", "qkvg", "qkvgReduce", "attend", "attentionOutput",
  "attentionOutputReduce", "ffwAdaln", "ffwWide", "ffwWideReduce", "ffwOut",
  "ffwOutReduce", "adalnReduce", "ffwAdalnReduce", "normalisePair", "normaliseCond"];

export async function main(_device, args) {
  void args;
  const offsets = Object.fromEntries([
    "AdaptiveZeroCondBias", "AdaptiveZeroCondWeights", "Transition2",
    "ffwAdaptiveZeroCondBias", "ffwAdaptiveZeroCondWeights", "ffwTransition1",
    "ffwTransition2", "gatingQuery", "kProjection", "qBias", "qProjection",
    "vProjection", "SingleCondLayerNormScale", "SingleCondScaleWeights",
    "SingleCondScaleBias", "SingleCondBias", "ffwSingleCondLayerNormScale",
    "ffwSingleCondScaleWeights", "ffwSingleCondScaleBias", "ffwSingleCondBias",
  ].map((name, i) => [name, i * 1024]));

  const base = {
    tokens: 68, channels: 768, condChannels: 384, pairChannels: 128,
    heads: 16, dimension: 48, factor: 2, pairs: 68 * 68,
    weightPrecision: "f16", splits: 3, outSplits: 3, wideSplits: 6,
    outTile: 1, outChunk: 384, tile: 1,
  };

  // Every combination the device priors can produce, because a path that is
  // identical at the default and not under a split is not identical.
  const arms = [
    { name: "plain", extra: {} },
    { name: "split-k", extra: { kSplits: 16, qkvgTile: 4, wideTile: 4, outKSplits: 4 } },
    { name: "norm-split", extra: { normSplit: true } },
    { name: "everything", extra: {
      kSplits: 16, qkvgTile: 4, wideTile: 4, outKSplits: 4, normSplit: true,
      attnKSplits: 4, normKSplits: 4,
    } },
  ];

  const differences = [];
  for (const arm of arms) {
    const without = createDiffusionTransformerShaders({ ...base, ...arm.extra }, offsets);
    const withOne = createDiffusionTransformerShaders(
      { ...base, ...arm.extra, samples: 1 }, offsets);
    for (const name of NAMES) {
      const a = without[name];
      const b = withOne[name];
      if (typeof a !== "string" && typeof b !== "string") continue;
      if (a !== b) {
        const at = [...(a ?? "")].findIndex((ch, i) => ch !== (b ?? "")[i]);
        differences.push({
          arm: arm.name, shader: name, at,
          absent: (a ?? "").slice(Math.max(0, at - 40), at + 60),
          samplesOne: (b ?? "").slice(Math.max(0, at - 40), at + 60),
        });
      }
    }
  }
  return {
    arms: arms.length, shaders: NAMES.length,
    identical: differences.length === 0,
    differences: differences.slice(0, 4),
  };
}
