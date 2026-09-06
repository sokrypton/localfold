/**
 * ESMFold2's diffusion conditioning: a trunk pair and a noise level in, the
 * denoiser's two conditioning tensors out.
 *
 *     z = z_proj(z_input_norm([z_trunk | rel_pos]))    512 -> 256
 *     z = z + transition(z), twice
 *     s = s_proj(s_input_norm(s_inputs))               451 -> 768
 *     t = 0.25 * log(max(t_hat / sigma_data, 1e-20))
 *     s = s + noise_proj(noise_norm(cos(2*pi*(t*w + b))))
 *     s = s + transition(s), twice
 *
 * 🔴 THERE IS NO `s_trunk` ANYWHERE IN THIS MODEL. The conditioning takes one
 * and it is handed None: the trunk carries no single track, so the only single
 * representation in the whole graph is `s_inputs` - the 451 channels the inputs
 * embedder produced. An AF3-shaped port that reaches for a trunk single will
 * not find one, and synthesising a zero would be a different model.
 *
 * 🔴 AND `z` IS CACHED ACROSS THE SAMPLER'S STEPS WHILE `s` IS NOT. Only `s`
 * depends on the noise level. Fifteen steps recomputing the pair conditioning
 * would be fifteen times the largest tensor here for one answer; upstream
 * caches it in `inference_cache` and so should any port.
 */

const SIGMA_DATA = 16;

function layerNorm(values, rows, channels, scale, offset, epsilon = 1e-5) {
  const out = new Float32Array(values.length);
  for (let row = 0; row < rows; row += 1) {
    const base = row * channels;
    let mean = 0;
    for (let c = 0; c < channels; c += 1) mean += values[base + c];
    mean /= channels;
    let variance = 0;
    for (let c = 0; c < channels; c += 1) {
      const d = values[base + c] - mean;
      variance += d * d;
    }
    const by = 1 / Math.sqrt(variance / channels + epsilon);
    for (let c = 0; c < channels; c += 1) {
      out[base + c] = (values[base + c] - mean) * by * scale[c] + offset[c];
    }
  }
  return out;
}

function linear(input, rows, inChannels, outChannels, weights) {
  const out = new Float32Array(rows * outChannels);
  for (let row = 0; row < rows; row += 1) {
    const inBase = row * inChannels;
    const outBase = row * outChannels;
    for (let c = 0; c < inChannels; c += 1) {
      const value = input[inBase + c];
      if (value === 0) continue;
      const weightBase = c * outChannels;
      for (let o = 0; o < outChannels; o += 1) out[outBase + o] += value * weights[weightBase + o];
    }
  }
  return out;
}

const silu = (x) => x / (1 + Math.exp(-x));

/**
 * One `TransitionLayer`, added to its input by the caller.
 *
 * 🔴 THE GATE AND THE VALUE ARE SEPARATE TENSORS HERE, NOT ONE FUSED WIDENING.
 * Every other transition in this repository - AF3's, ESMFold2's own pair
 * transition - packs both halves into one `w12` and splits it, so the only
 * question is which half is the gate. This module has `a_proj` and `b_proj` as
 * two Linears and computes `out_proj(silu(a) * b)`, so `a` is the gate. Reading
 * it as a fused pair would index one matrix at half its stride.
 */
export function transitionLayer(input, rows, channels, hidden, weights) {
  const normalised = layerNorm(input, rows, channels,
                               weights.normScale, weights.normOffset);
  const a = linear(normalised, rows, channels, hidden, weights.aProjection);
  const b = linear(normalised, rows, channels, hidden, weights.bProjection);
  const gated = new Float32Array(rows * hidden);
  for (let i = 0; i < gated.length; i += 1) gated[i] = silu(a[i]) * b[i];
  return linear(gated, rows, hidden, channels, weights.outProjection);
}

/** `cos(2 * pi * (t * w + b))`, one row. */
export function fourierEmbedding(t, weights, offsets) {
  const out = new Float32Array(weights.length);
  for (let i = 0; i < weights.length; i += 1) {
    out[i] = Math.cos(2 * Math.PI * (t * weights[i] + offsets[i]));
  }
  return out;
}

/**
 * @param {Float32Array} zTrunk n*n*pairChannels, the trunk's final pair
 * @param {Float32Array} relPos n*n*pairChannels, the SAME encoding z_init used
 * @param {Float32Array} sInputs n*singleChannels
 * @param {number} tHat the noise level for this sampler step
 * @returns {{single: Float32Array, pair: Float32Array}}
 */
export function diffusionConditioning(zTrunk, relPos, sInputs, tHat, shape, weights,
                                      sigmaData = SIGMA_DATA) {
  const { tokens, pairChannels, singleInputs, tokenChannels, multiplier } = shape;
  const pairs = tokens * tokens;

  // 🔴 CONCATENATED, NOT ADDED. The two are the same shape and adding them
  // conforms; the norm is 2 * c_z wide, which is what says so.
  const joined = new Float32Array(pairs * pairChannels * 2);
  for (let pair = 0; pair < pairs; pair += 1) {
    const to = pair * pairChannels * 2;
    const from = pair * pairChannels;
    for (let c = 0; c < pairChannels; c += 1) {
      joined[to + c] = zTrunk[from + c];
      joined[to + pairChannels + c] = relPos[from + c];
    }
  }
  const normalised = layerNorm(joined, pairs, pairChannels * 2,
                               weights.zInputNormScale, weights.zInputNormOffset);
  let pair = linear(normalised, pairs, pairChannels * 2, pairChannels, weights.zProjection);
  for (const block of weights.zTransitions) {
    const delta = transitionLayer(pair, pairs, pairChannels,
                                  pairChannels * multiplier, block);
    for (let i = 0; i < pair.length; i += 1) pair[i] += delta[i];
  }

  let single = linear(
    layerNorm(sInputs, tokens, singleInputs,
              weights.sInputNormScale, weights.sInputNormOffset),
    tokens, singleInputs, tokenChannels, weights.sProjection);

  // 🔴 THE NOISE LEVEL IS LOGGED AND QUARTERED, AND THE CLAMP IS NOT DECORATION.
  // `0.25 * log(clamp(t / sigma_data, min=1e-20))` - the sampler's last step
  // takes t to 4e-4 and a schedule that reached zero would give -Infinity here.
  const tNoise = 0.25 * Math.log(Math.max(tHat / sigmaData, 1e-20));
  const fourier = fourierEmbedding(tNoise, weights.fourierWeights, weights.fourierOffsets);
  const noise = linear(
    layerNorm(fourier, 1, fourier.length, weights.noiseNormScale, weights.noiseNormOffset),
    1, fourier.length, tokenChannels, weights.noiseProjection);
  // ...broadcast over the tokens: one noise level for the whole structure.
  for (let token = 0; token < tokens; token += 1) {
    const base = token * tokenChannels;
    for (let c = 0; c < tokenChannels; c += 1) single[base + c] += noise[c];
  }

  for (const block of weights.sTransitions) {
    const delta = transitionLayer(single, tokens, tokenChannels,
                                  tokenChannels * multiplier, block);
    for (let i = 0; i < single.length; i += 1) single[i] += delta[i];
  }
  return { single, pair };
}
