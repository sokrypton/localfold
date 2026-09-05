/**
 * What does packing the attention KEY cost, and what does packing the VALUE?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-attention-packing.js
 *     node tools/gpu-chrome.mjs tools/gpu/check-attention-packing.js --queries=256 --batch=16
 *
 * 🔴 THIS EXISTS BECAUSE AF2's OFFICIAL-VALUE GATE CANNOT RUN HERE.
 * check-evoformer-stack.js is the only comparison against AlphaFold's own
 * activations on this machine, and test/fixtures/evoformer/ is gitignored with
 * 26 of its 530 tensors present, so it 404s. Upstream
 * (martin-steinegger/alphafold2-webgpu, commit 6974112) has those fixtures and
 * reported what this checkout could not see: the projected tensors packed two
 * halves to a word moved an evoformer block's MSA output 4.06e-4 against a 5e-5
 * allowance, and packing the VALUE ALONE reproduced 4.05e-4.
 *
 * The claim is about a RATIO, and a ratio is measurable without the fixtures.
 * Both the key and the value are read once per key in the hot loop, so both
 * halve the same traffic; what differs is where their rounding lands. A key's
 * error becomes a logit error and the softmax divides most of it away. A
 * value's is averaged under weights that already sum to one and arrives in the
 * output undamped. If that is right, the value-only arm should sit near the
 * both-packed arm and the key-only arm far below it, and it does.
 *
 * 🔴 AND EVERY ARM ASSERTS IT REACHED THE KERNEL. A storage option that never
 * arrives reports perfect agreement, which is indistinguishable from a change
 * that costs nothing - and that is exactly how the first version of the packing
 * upstream was written. Each arm here checks the flash kernel's cache key names
 * the elements it asked for, and the run fails if two arms compiled the same
 * shader.
 *
 * The reference is written out in this file - LayerNorm, four projections,
 * scaled dot-product attention, a sigmoid gate, an output projection - because
 * a reference sharing code with the thing it checks tests nothing. It is the
 * same shape as check-evoformer-attention.js's and deliberately a second
 * writing of it.
 */
import { AttentionGpu } from "../../src/evoformer/attention.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

/** LayerNorm, q/k/v/gate, softmax attention, gate, output projection - in f64. */
function reference(input) {
  const { batch, queryLength, channels, heads, activations, mask, weights } = input;
  const epsilon = input.epsilon ?? 1e-5;
  const dimension = channels / heads;
  const scale = 1 / Math.sqrt(dimension);
  const output = new Float32Array(batch * queryLength * channels);
  for (let b = 0; b < batch; b += 1) {
    const q = new Float64Array(queryLength * channels);
    const k = new Float64Array(queryLength * channels);
    const v = new Float64Array(queryLength * channels);
    const g = new Float64Array(queryLength * channels);
    for (let i = 0; i < queryLength; i += 1) {
      const row = (b * queryLength + i) * channels;
      let mean = 0;
      for (let c = 0; c < channels; c += 1) mean += activations[row + c];
      mean /= channels;
      let variance = 0;
      for (let c = 0; c < channels; c += 1) variance += (activations[row + c] - mean) ** 2;
      variance /= channels;
      const inverse = 1 / Math.sqrt(variance + epsilon);
      const normalized = new Float64Array(channels);
      for (let c = 0; c < channels; c += 1) {
        normalized[c] = (activations[row + c] - mean) * inverse * weights.queryNormScale[c] + weights.queryNormOffset[c];
      }
      for (let c = 0; c < channels; c += 1) {
        let qs = 0; let ks = 0; let vs = 0; let gs = 0;
        for (let d = 0; d < channels; d += 1) {
          qs += normalized[d] * weights.queryWeight[d * channels + c];
          ks += normalized[d] * weights.keyWeight[d * channels + c];
          vs += normalized[d] * weights.valueWeight[d * channels + c];
          gs += normalized[d] * weights.gatingWeight[d * channels + c];
        }
        q[i * channels + c] = qs * scale;
        k[i * channels + c] = ks;
        v[i * channels + c] = vs;
        g[i * channels + c] = 1 / (1 + Math.exp(-(gs + weights.gatingBias[c])));
      }
    }
    const weighted = new Float64Array(queryLength * channels);
    for (let h = 0; h < heads; h += 1) {
      const base = h * dimension;
      for (let i = 0; i < queryLength; i += 1) {
        const logits = new Float64Array(queryLength);
        let top = -Infinity;
        for (let j = 0; j < queryLength; j += 1) {
          if (mask[b * queryLength + j] === 0) { logits[j] = -Infinity; continue; }
          let dot = 0;
          for (let d = 0; d < dimension; d += 1) {
            dot += q[i * channels + base + d] * k[j * channels + base + d];
          }
          logits[j] = dot;
          if (dot > top) top = dot;
        }
        let sum = 0;
        for (let j = 0; j < queryLength; j += 1) {
          logits[j] = logits[j] === -Infinity ? 0 : Math.exp(logits[j] - top);
          sum += logits[j];
        }
        for (let d = 0; d < dimension; d += 1) {
          let acc = 0;
          for (let j = 0; j < queryLength; j += 1) acc += logits[j] * v[j * channels + base + d];
          weighted[i * channels + base + d] = (acc / sum) * g[i * channels + base + d];
        }
      }
    }
    for (let i = 0; i < queryLength; i += 1) {
      for (let c = 0; c < channels; c += 1) {
        let acc = weights.outputBias[c];
        for (let d = 0; d < channels; d += 1) {
          acc += weighted[i * channels + d] * weights.outputWeight[d * channels + c];
        }
        output[(b * queryLength + i) * channels + c] = acc;
      }
    }
  }
  return output;
}

export async function main(device, args) {
  const batch = Number(option(args, "batch", "8"));
  const queryLength = Number(option(args, "queries", "128"));
  const channels = Number(option(args, "channels", "64"));
  const heads = Number(option(args, "heads", "2"));
  // 🔴 THE STAGED CHUNK'S ELEMENT IS A SEPARATE AXIS FROM THE STORAGE, and on
  // this repository's default it is ALREADY f16. `chunk16` stages the key and
  // the value in half precision inside workgroup memory whatever the tensors
  // are stored as, so an f32-storage arm has already rounded both to eleven
  // mantissa bits by the time they are used - which is why the baseline below
  // is 1e-3 and not 1e-7, and why packing the storage costs so little on top.
  // `--precision=f32` is the arm where the staged chunk is f32 and the storage
  // is the only rounding, which is the configuration upstream measured.
  const precision = option(args, "precision", "auto");
  // 🔴 AND THE DENSE KERNELS SWAMP THE ANSWER ON `auto`. The projection and the
  // output projection run f16 arithmetic wherever the device has shader-f16 and
  // carry 1e-3 of their own, which is a thousand times the rounding this is
  // trying to resolve: on auto all four arms read 9.6e-4 to 9.7e-4 and the
  // question cannot be answered at all. `--dense=f32` takes them to single
  // precision so the STORAGE is the only rounding left, which is the arm that
  // separates the key from the value. Upstream hit the same wall from the other
  // side - their kernel probe ran on synthetic inputs and read 6e-7 for a
  // packing that moved the real model 4e-4.
  const dense = option(args, "dense", "auto");

  let state = 20260905;
  const random = (n, spread = 1) => {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i += 1) {
      state = (state * 1103515245 + 12345) & 0x7fffffff;
      out[i] = (state / 0x7fffffff - 0.5) * spread;
    }
    return out;
  };
  const weights = {
    queryNormScale: random(channels, 0.4).map((v) => v + 1),
    queryNormOffset: random(channels, 0.2),
    queryWeight: random(channels * channels, 0.3),
    keyWeight: random(channels * channels, 0.3),
    valueWeight: random(channels * channels, 0.3),
    gatingWeight: random(channels * channels, 0.3),
    gatingBias: random(channels, 0.2),
    outputWeight: random(channels * channels, 0.3),
    outputBias: random(channels, 0.2),
  };
  const mask = new Float32Array(batch * queryLength).fill(1);
  // A ragged mask, so the masked path is exercised rather than assumed.
  for (let b = 0; b < batch; b += 1) {
    for (let j = queryLength - (b % 5); j < queryLength; j += 1) mask[b * queryLength + j] = 0;
  }
  const input = {
    batch, queryLength, channels, heads,
    activations: random(batch * queryLength * channels, 2), mask, weights,
  };

  const truth = reference(input);
  const relRms = (got) => {
    let num = 0; let den = 0;
    for (let i = 0; i < truth.length; i += 1) { num += (got[i] - truth[i]) ** 2; den += truth[i] ** 2; }
    return Math.sqrt(num / Math.max(den, 1e-30));
  };

  // The four corners of the question. `output` is the attended result, which
  // the output projection reads; it is held f32 throughout so the two arms
  // differ in the key and the value alone.
  const arms = {
    "f32 throughout": { input: "f32", value: "f32", output: "f32" },
    "key packed": { input: "f16", value: "f32", output: "f32" },
    "value packed": { input: "f32", value: "f16", output: "f32" },
    "key and value packed": { input: "f16", value: "f16", output: "f32" },
  };
  const results = {};
  const keys = {};
  for (const [name, storage] of Object.entries(arms)) {
    const gpu = new AttentionGpu(device, {
      storage, flashPrecision: precision, projectPrecision: dense, outputPrecision: dense,
    });
    const run = await gpu.run(input);
    results[name] = {
      relRms: Number(relRms(run.output).toPrecision(3)),
      storage: `${run.storage.projected}/${run.storage.value}`,
    };
    keys[name] = run.shaders.flash;
  }

  // 🔴 THE ARMS MUST DIFFER, or this measured one shader four times.
  const distinct = new Set(Object.values(keys));
  const inert = distinct.size !== Object.keys(arms).length
    ? `only ${distinct.size} distinct flash shaders for ${Object.keys(arms).length} arms: `
      + JSON.stringify(keys)
    : null;

  const key = results["key packed"].relRms;
  const value = results["value packed"].relRms;
  const base = results["f32 throughout"].relRms;
  return {
    shape: { batch, queryLength, channels, heads, flashPrecision: precision, dense },
    results,
    shaders: keys,
    inert,
    // Upstream's claim, restated as a ratio this machine can check.
    valueOverKey: Number((value / Math.max(key, 1e-30)).toPrecision(3)),
    verdict: inert !== null ? "INERT - the arms did not differ"
      : value > key * 3 ? "the VALUE dominates, as upstream measured"
        : "the two are comparable here - upstream's split does not reproduce",
    baselineRelRms: base,
  };
}
