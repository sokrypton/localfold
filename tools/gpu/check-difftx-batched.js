/**
 * Does the batched path compute the same thing S times?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-difftx-batched.js
 *
 * 🔴 THE ONLY HONEST TEST OF A BATCH DIMENSION IS DEGENERACY. Given S copies of
 * one activation, every sample must come back byte-identical to the others AND
 * to what one sample alone produces. That catches the two failures a batched
 * kernel actually has - a row axis that indexes a per-token tensor by its row
 * (the conditioning, the pair bias) and an attention that reaches across the
 * sample boundary - neither of which a single-sample fold gate can see, because
 * at one sample they are the same thing.
 *
 * The head cannot drive S > 1 yet; this drives the transformer directly.
 */
import { Af3DiffusionTransformerGpu } from "../../src/af3/diffusion-transformer-webgpu.js";
import { openAf3Store } from "../../src/af3/weights.js";
import { diffusionWeights } from "../../src/af3/diffusion-weights.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

export async function main(device, args) {
  const tokens = Number(option(args, "tokens", "32"));
  const samples = Number(option(args, "samples", "2"));
  const store = await openAf3Store(option(args, "model", "/model-af3-int5/manifest.json"));
  // 🔴 THE SECOND ARGUMENT IS A SUPER-BLOCK COUNT, NOT A DIALECT. Passing an
  // object leaves the loop that builds them comparing 0 < {} and producing
  // none, which surfaces much later as superBlocks[0] being undefined.
  const weights = await diffusionWeights(store);
  const tx = weights.transformer;
  const { channels, condChannels, pairChannels } = tx;

  const noise = (n, seed) => {
    const out = new Float32Array(n);
    let state = seed >>> 0;
    for (let i = 0; i < n; i += 1) {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      out[i] = (state / 4294967296) * 2 - 1;
    }
    return out;
  };

  const act = noise(tokens * channels, 7);
  const cond = noise(tokens * condChannels, 11);
  const pairCond = noise(tokens * tokens * pairChannels, 13);
  const mask = new Float32Array(tokens).fill(1);

  const one = new Af3DiffusionTransformerGpu(device);
  const single = await one.run(act, cond, pairCond, mask, tokens, tx);

  // S copies of the same activation, laid out (sample, token, channel).
  const wide = new Float32Array(samples * tokens * channels);
  for (let s = 0; s < samples; s += 1) wide.set(act, s * tokens * channels);
  const many = new Af3DiffusionTransformerGpu(device);
  const batched = await many.run(wide, cond, pairCond, mask, tokens,
    { ...tx, samples });

  const span = tokens * channels;
  const rel = (a, b) => {
    let num = 0;
    let den = 0;
    for (let i = 0; i < a.length; i += 1) { num += (a[i] - b[i]) ** 2; den += b[i] ** 2; }
    return Math.sqrt(num / Math.max(den, 1e-30));
  };
  const out = new Float32Array(batched.buffer ? batched : batched);
  const perSample = [];
  for (let s = 0; s < samples; s += 1) {
    perSample.push(Number(rel(out.slice(s * span, (s + 1) * span), single).toPrecision(3)));
  }
  // Sample-to-sample, which isolates "the samples differ" from "they all
  // differ from one sample" - a pair-bias indexed by row fails the first, an
  // attention that crosses samples fails both.
  const between = [];
  for (let s = 1; s < samples; s += 1) {
    between.push(Number(rel(out.slice(s * span, (s + 1) * span),
      out.slice(0, span)).toPrecision(3)));
  }
  // What it is worth, once it is known to be right. Medians of repeated runs,
  // because a single call at this size is inside this box's drift.
  const timed = async (runner, input, extra) => {
    const at = [];
    for (let i = 0; i < 5; i += 1) {
      const started = performance.now();
      await runner.run(input, cond, pairCond, mask, tokens, extra);
      await device.queue.onSubmittedWorkDone();
      at.push(performance.now() - started);
    }
    at.sort((a, b) => a - b);
    return at[2];
  };
  const oneMs = await timed(one, act, tx);
  const manyMs = await timed(many, wide, { ...tx, samples });

  return {
    tokens, samples, channels,
    oneSampleMs: Number(oneMs.toFixed(2)),
    batchedMs: Number(manyMs.toFixed(2)),
    unbatchedMs: Number((samples * oneMs).toFixed(2)),
    speedup: Number((samples * oneMs / manyMs).toPrecision(3)),
    againstOneSample: perSample,
    betweenSamples: between,
    ok: perSample.every((v) => v < 1e-5) && between.every((v) => v < 1e-6),
  };
}
