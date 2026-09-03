/**
 * AF2's evoformer attention on the GPU against a CPU reference.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-evoformer-attention.js
 *     node tools/gpu-chrome.mjs tools/gpu/check-evoformer-attention.js --transpose
 *
 * 🔴 THE VARIANT CHECKER IS NOT THIS. tools/gpu/check-attention-variants.js
 * runs every flash kernel against the same input and reports how far they
 * disagree - which catches a broken VARIANT and cannot catch a broken
 * projection, because all of them share it. The q/k/v/gate projection is 19 ms
 * of AF2's 118 ms block at 512 MSA rows, twice over, and nothing was checking
 * what it computed.
 *
 * The reference is LayerNorm, four projections, scaled dot-product attention
 * with the key mask, a sigmoid gate and an output projection - written out here
 * rather than imported, because a reference sharing code with the thing it
 * checks tests nothing.
 */
import { AttentionGpu } from "../../src/evoformer/attention.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return args.find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

function reference(input) {
  const { batch, queryLength, channels, heads, activations, mask, weights } = input;
  const epsilon = input.epsilon ?? 1e-5;
  const dimension = channels / heads;
  const scale = 1 / Math.sqrt(dimension);
  const output = new Float32Array(batch * queryLength * channels);
  const q = new Float32Array(queryLength * channels);
  const k = new Float32Array(queryLength * channels);
  const v = new Float32Array(queryLength * channels);
  const g = new Float32Array(queryLength * channels);

  for (let b = 0; b < batch; b += 1) {
    for (let i = 0; i < queryLength; i += 1) {
      const row = (b * queryLength + i) * channels;
      let mean = 0;
      for (let c = 0; c < channels; c += 1) mean += activations[row + c];
      mean /= channels;
      let variance = 0;
      for (let c = 0; c < channels; c += 1) {
        const d = activations[row + c] - mean;
        variance += d * d;
      }
      const inverse = 1 / Math.sqrt(variance / channels + epsilon);
      const normalized = new Float32Array(channels);
      for (let c = 0; c < channels; c += 1) {
        normalized[c] = (activations[row + c] - mean) * inverse * weights.queryNormScale[c]
          + weights.queryNormOffset[c];
      }
      for (let o = 0; o < channels; o += 1) {
        let qs = 0, ks = 0, vs = 0, gs = weights.gatingBias[o];
        for (let c = 0; c < channels; c += 1) {
          const x = normalized[c];
          qs += x * weights.queryWeight[c * channels + o];
          ks += x * weights.keyWeight[c * channels + o];
          vs += x * weights.valueWeight[c * channels + o];
          gs += x * weights.gatingWeight[c * channels + o];
        }
        q[i * channels + o] = qs;
        k[i * channels + o] = ks;
        v[i * channels + o] = vs;
        g[i * channels + o] = 1 / (1 + Math.exp(-gs));
      }
    }
    const gathered = new Float32Array(queryLength * channels);
    for (let head = 0; head < heads; head += 1) {
      for (let i = 0; i < queryLength; i += 1) {
        const logits = new Float32Array(queryLength);
        let largest = -Infinity;
        for (let j = 0; j < queryLength; j += 1) {
          let dot = 0;
          for (let d = 0; d < dimension; d += 1) {
            dot += q[i * channels + head * dimension + d] * k[j * channels + head * dimension + d];
          }
          let value = dot * scale;
          if (mask[b * queryLength + j] <= 0) value -= 1e9;
          logits[j] = value;
          largest = Math.max(largest, value);
        }
        let total = 0;
        for (let j = 0; j < queryLength; j += 1) {
          logits[j] = Math.exp(logits[j] - largest);
          total += logits[j];
        }
        for (let d = 0; d < dimension; d += 1) {
          let sum = 0;
          for (let j = 0; j < queryLength; j += 1) {
            sum += logits[j] * v[j * channels + head * dimension + d];
          }
          gathered[i * channels + head * dimension + d] = sum / total;
        }
      }
    }
    for (let i = 0; i < queryLength; i += 1) {
      for (let c = 0; c < channels; c += 1) {
        let value = weights.outputBias[c];
        for (let o = 0; o < channels; o += 1) {
          value += gathered[i * channels + o] * g[i * channels + o]
            * weights.outputWeight[o * channels + c];
        }
        output[(b * queryLength + i) * channels + c] = value;
      }
    }
  }
  return output;
}

export async function main(device, args) {
  const batch = Number(option(args, "batch", "5"));
  const queryLength = Number(option(args, "queries", "13"));
  const heads = Number(option(args, "heads", "4"));
  const dimension = Number(option(args, "dimension", "8"));
  const channels = heads * dimension;

  let state = 90909;
  const random = (count, scale = 1) => {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1) {
      state = (state * 1103515245 + 12345) & 0x7fffffff;
      out[i] = (state / 0x7fffffff - 0.5) * scale;
    }
    return out;
  };
  // 🔴 A RAGGED MASK, so a key that should be excluded showing up in the
  // softmax would move the answer.
  const mask = new Float32Array(batch * queryLength).fill(1);
  for (let i = 0; i < mask.length; i += 1) if (i % 5 === 2) mask[i] = 0;

  const input = {
    batch, queryLength, channels, heads, epsilon: 1e-5,
    activations: random(batch * queryLength * channels, 3), mask,
    weights: {
      queryNormScale: random(channels, 0.4).map((v) => v + 1),
      queryNormOffset: random(channels, 0.2),
      queryWeight: random(channels * channels, 0.4),
      keyWeight: random(channels * channels, 0.4),
      valueWeight: random(channels * channels, 0.4),
      gatingWeight: random(channels * channels, 0.4),
      gatingBias: random(channels, 0.2),
      outputWeight: random(channels * channels, 0.4),
      outputBias: random(channels, 0.1),
    },
  };

  const { output } = await new AttentionGpu(device).run(input);
  const expected = reference(input);
  let error = 0;
  let scale = 0;
  let worst = 0;
  for (let i = 0; i < expected.length; i += 1) {
    error += (output[i] - expected[i]) ** 2;
    scale += expected[i] ** 2;
    worst = Math.max(worst, Math.abs(output[i] - expected[i]));
  }
  const relRms = Math.sqrt(error / scale);
  const bound = 1e-5;
  const ok = relRms <= bound;
  console.log(`${ok ? "PASS" : "FAIL"}\tbatch ${batch} queries ${queryLength} heads ${heads}`
    + ` dim ${dimension}\trelRMS ${relRms.toExponential(2)}\tworst ${worst.toExponential(2)}`);
  if (!ok) throw new Error(`attention relRMS ${relRms.toExponential(2)} exceeds ${bound}`);
  return { batch, queryLength, channels, heads, relRms, worst, ok };
}
