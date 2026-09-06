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
const sigmoid = (x) => 1 / (1 + Math.exp(-x));

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
                                      sigmaData = SIGMA_DATA, cachedPair = undefined) {
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
  let pair = cachedPair;
  if (pair === undefined) {
    const normalised = layerNorm(joined, pairs, pairChannels * 2,
                                 weights.zInputNormScale, weights.zInputNormOffset);
    pair = linear(normalised, pairs, pairChannels * 2, pairChannels, weights.zProjection);
    for (const block of weights.zTransitions) {
      const delta = transitionLayer(pair, pairs, pairChannels,
                                    pairChannels * multiplier, block);
      for (let i = 0; i < pair.length; i += 1) pair[i] += delta[i];
    }
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

// Cached constants, so a per-block adaLN does not allocate two vectors a call.
// 🔴 SIZE-GENERAL, because the first version seeded the map with 768 alone and
// `map.get(channels)` returns undefined for anything else - which reaches
// layerNorm as a missing scale and throws somewhere unrelated.
const ONES = new Map();
const ZEROS = new Map();
const constant = (map, size, fill) => {
  if (!map.has(size)) map.set(size, new Float32Array(size).fill(fill));
  return map.get(size);
};

/**
 * adaLN-Zero: normalise the activation, and take a scale and a shift from the
 * conditioning single.
 *
 * 🔴 THE TWO LayerNorms ARE NOT THE SAME KIND. The activation's is
 * affine-FREE - `F.layer_norm(a, (d,), None, None)` - and the conditioning's
 * has a learned SCALE and no bias, `F.layer_norm(s, (d,), s_scale, None)`.
 * There is one weight vector between them and it belongs to `s`; giving it to
 * `a`, or giving either an offset it does not have, conforms in shape.
 */
export function adaptiveLayerNorm(activation, single, rows, channels, weights) {
  const normalisedActivation = layerNorm(activation, rows, channels,
    constant(ONES, channels, 1), constant(ZEROS, channels, 0));
  const normalisedSingle = layerNorm(single, rows, channels,
    weights.singleScale, constant(ZEROS, channels, 0));
  const gate = linear(normalisedSingle, rows, channels, channels, weights.gateWeights);
  const shift = linear(normalisedSingle, rows, channels, channels, weights.shiftWeights);
  const out = new Float32Array(activation.length);
  for (let row = 0; row < rows; row += 1) {
    const base = row * channels;
    for (let c = 0; c < channels; c += 1) {
      const i = base + c;
      out[i] = sigmoid(gate[i] + weights.gateBias[c]) * normalisedActivation[i] + shift[i];
    }
  }
  return out;
}


/**
 * Attention biased by the pair representation, gated twice.
 *
 * 🔴 THE SOFTMAX IS OVER THE KEY AXIS OF AN (i, j, head) TENSOR, which is
 * `dim=-2` upstream and not the last axis. A softmax over the heads instead
 * sums to one and returns a plausible tensor.
 *
 * 🔴 AND THERE ARE TWO GATES, FROM DIFFERENT THINGS. `g_proj` gates the
 * per-head context from the ADALN-MODULATED activation, and `out_gate` gates
 * the whole output from the CONDITIONING SINGLE - and only the second has a
 * bias, initialised to -2 upstream so a fresh block starts nearly closed.
 */
export function attentionPairBias(activation, single, pair, tokens, channels,
                                  pairChannels, heads, weights) {
  const headDim = channels / heads;
  const x = adaptiveLayerNorm(activation, single, tokens, channels, weights.adaln);

  const query = linear(x, tokens, channels, channels, weights.queryWeights);
  for (let token = 0; token < tokens; token += 1) {
    for (let c = 0; c < channels; c += 1) query[token * channels + c] += weights.queryBias[c];
  }
  const kv = linear(x, tokens, channels, channels * 2, weights.kvWeights);
  const gate = linear(x, tokens, channels, channels, weights.gateWeights);

  // One scalar per (i, j, head), from the pair representation.
  const pairs = tokens * tokens;
  const bias = linear(
    layerNorm(pair, pairs, pairChannels, weights.pairNormScale, weights.pairNormOffset),
    pairs, pairChannels, heads, weights.pairBiasWeights);

  const scale = 1 / Math.sqrt(headDim);
  const context = new Float32Array(tokens * channels);
  const logits = new Float32Array(tokens);
  for (let head = 0; head < heads; head += 1) {
    for (let i = 0; i < tokens; i += 1) {
      const queryBase = i * channels + head * headDim;
      let largest = -Infinity;
      for (let j = 0; j < tokens; j += 1) {
        const keyBase = j * channels * 2 + head * headDim;
        let total = 0;
        for (let d = 0; d < headDim; d += 1) total += query[queryBase + d] * kv[keyBase + d];
        logits[j] = total * scale + bias[(i * tokens + j) * heads + head];
        if (logits[j] > largest) largest = logits[j];
      }
      let sum = 0;
      for (let j = 0; j < tokens; j += 1) {
        logits[j] = Math.exp(logits[j] - largest);
        sum += logits[j];
      }
      const outBase = i * channels + head * headDim;
      for (let j = 0; j < tokens; j += 1) {
        const weight = logits[j] / sum;
        // ...the value is the SECOND half of kv, so it starts a whole
        // `channels` further along the row.
        const valueBase = j * channels * 2 + channels + head * headDim;
        for (let d = 0; d < headDim; d += 1) {
          context[outBase + d] += weight * kv[valueBase + d];
        }
      }
    }
  }
  for (let i = 0; i < context.length; i += 1) context[i] *= sigmoid(gate[i]);
  const out = linear(context, tokens, channels, channels, weights.outWeights);
  const outGate = linear(single, tokens, channels, channels, weights.outGateWeights);
  for (let token = 0; token < tokens; token += 1) {
    for (let c = 0; c < channels; c += 1) {
      const i = token * channels + c;
      out[i] *= sigmoid(outGate[i] + weights.outGateBias[c]);
    }
  }
  return out;
}

/**
 * The conditioned transition.
 *
 * 🔴 AND THIS ONE FUSES ITS GATE WHERE THE CONDITIONING'S DOES NOT, IN THE SAME
 * MODULE. `lin_swish` is one Linear of `2 * hidden` split in half, gate first;
 * `DiffusionConditioning`'s `TransitionLayer` has `a_proj` and `b_proj` as two
 * separate Linears. Both are "a SwiGLU transition in the diffusion module" and
 * they are packed the two different ways.
 */
export function conditionedTransition(activation, single, tokens, channels, hidden, weights) {
  const x = adaptiveLayerNorm(activation, single, tokens, channels, weights.adaln);
  const wide = linear(x, tokens, channels, hidden * 2, weights.swishWeights);
  const gated = new Float32Array(tokens * hidden);
  for (let token = 0; token < tokens; token += 1) {
    const wideBase = token * hidden * 2;
    const base = token * hidden;
    for (let i = 0; i < hidden; i += 1) {
      gated[base + i] = silu(wide[wideBase + i]) * wide[wideBase + hidden + i];
    }
  }
  const out = linear(gated, tokens, hidden, channels, weights.outWeights);
  const outGate = linear(single, tokens, channels, channels, weights.outGateWeights);
  for (let token = 0; token < tokens; token += 1) {
    for (let c = 0; c < channels; c += 1) {
      const i = token * channels + c;
      out[i] *= sigmoid(outGate[i] + weights.outGateBias[c]);
    }
  }
  return out;
}

/**
 * The denoiser's twelve blocks.
 *
 * 🔴 THE ATTENTIONS AND THE TRANSITIONS ARE TWO SEPARATE ModuleLists, ZIPPED.
 * `attn_blocks` and `transition_blocks` are each `num_blocks` long and the
 * forward pairs them - so block i is `attn_blocks[i]` then
 * `transition_blocks[i]`, not a single list of alternating modules. A flat
 * export that interleaved them would load the right count of the wrong things.
 */
export function tokenTransformer(activation, single, pair, tokens, channels,
                                 pairChannels, heads, hidden, blocks) {
  let a = Float32Array.from(activation);
  for (const block of blocks) {
    const attended = attentionPairBias(a, single, pair, tokens, channels,
                                       pairChannels, heads, block.attention);
    for (let i = 0; i < a.length; i += 1) a[i] += attended[i];
    const transitioned = conditionedTransition(a, single, tokens, channels, hidden,
                                               block.transition);
    for (let i = 0; i < a.length; i += 1) a[i] += transitioned[i];
  }
  return a;
}

export { layerNorm, linear, constant };

/**
 * One EDM denoise step: noisy coordinates in, denoised coordinates out.
 *
 *     s, z    = conditioning(t_hat, s_inputs, z_trunk, rel_pos)
 *     r_noisy = x_noisy / sqrt(t^2 + sigma^2)
 *     a, q, c = atom_encoder(features, r_l = r_noisy)
 *     a       = a + s_to_token(s_step_norm(s))
 *     a       = token_norm(token_transformer(a, s, z))
 *     r       = atom_decoder(a, q, c)
 *     out     = sigma^2/(sigma^2+t^2) * x_noisy
 *             + sigma*t/sqrt(sigma^2+t^2) * r
 *
 * 🔴 THE LAST LINE IS THE EDM PRECONDITIONING AND IT IS NOT A RESIDUAL. The
 * network's output is scaled by `sigma*t/sqrt(sigma^2+t^2)` and the noisy input
 * by `sigma^2/(sigma^2+t^2)`; at a large noise level the second is nearly zero
 * and the answer is almost all network, at a small one almost all input.
 * Writing it as `x_noisy + r` runs, converges to something, and is a different
 * sampler.
 *
 * 🔴 AND `z` IS THE SAME FOR EVERY STEP WHILE `s` IS NOT. The caller may pass
 * `conditioning` back in to skip the pair half; see the note there.
 */
// 🔴 NAMED `denoiseStep`, NOT `denoise`, AND NOT FOR TASTE. `denoise` is a
// PARAMETER of src/af3/diffusion-sampler-reference.js's `sample`, and
// test/module-references.test.js checks the whole project's exported names
// against every file's free identifiers - so exporting it here makes that
// parameter look like a reference to this function. Second time in this
// repository: `block` and `attend` went the same way.
export function denoiseStep(noisy, tHat, features, sInputs, zTrunk, relPos,
                        shape, weights, encoder, cachedPair = undefined) {
  const sigma = shape.sigmaData;
  // 🔴 ONLY THE PAIR IS CACHEABLE. `s` carries the noise level and changes
  // every step; `z` does not depend on `t_hat` at all, which is why upstream
  // keeps it in `inference_cache["z"]` and recomputes `s`. Caching the pair
  // and the single together would freeze the noise level at step zero - eleven
  // steps that all think they are the first, which still converges to
  // something.
  const conditioning = diffusionConditioning(
    zTrunk, relPos, sInputs, tHat, shape, weights.conditioning, sigma, cachedPair);
  const { single, pair } = conditioning;

  const denominator = Math.sqrt(tHat * tHat + sigma * sigma);
  const scaled = new Float32Array(noisy.length);
  for (let i = 0; i < noisy.length; i += 1) scaled[i] = noisy[i] / denominator;

  const embedded = encoder.embed({ ...features, noisyCoordinates: scaled });

  // ...the conditioning single, projected onto the tokens the encoder pooled.
  const stepped = linear(
    layerNorm(single, shape.tokens, shape.tokenChannels,
              weights.stepNormScale, weights.stepNormOffset),
    shape.tokens, shape.tokenChannels, shape.tokenChannels, weights.singleToToken);
  const act = new Float32Array(embedded.tokenAct.length);
  for (let i = 0; i < act.length; i += 1) act[i] = embedded.tokenAct[i] + stepped[i];

  const transformed = tokenTransformer(act, single, pair, shape.tokens,
    shape.tokenChannels, shape.pairChannels, shape.tokenHeads,
    shape.tokenChannels * shape.multiplier, weights.tokenBlocks);
  const normalised = layerNorm(transformed, shape.tokens, shape.tokenChannels,
                               weights.tokenNormScale, weights.tokenNormOffset);

  const update = encoder.decode(normalised, embedded);

  const sigma2 = sigma * sigma;
  const t2 = tHat * tHat;
  const keep = sigma2 / (sigma2 + t2);
  const take = (sigma * tHat) / Math.sqrt(sigma2 + t2);
  const out = new Float32Array(noisy.length);
  for (let i = 0; i < noisy.length; i += 1) out[i] = keep * noisy[i] + take * update[i];
  return { coordinates: out, conditioning };
}
