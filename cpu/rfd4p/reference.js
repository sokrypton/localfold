/**
 * Deterministic float32 CPU reference implementation of RFD4-Proteina:
 *  - FeatureFactory assemblers (Trunk 897-dim seq, 609-dim cond, 107-dim pair,
 *    Decoder 214-dim seq, Encoder 1587-dim seq)
 *  - RMSNorm, LayerNorm, AdaptiveLayerNorm, AdaptiveOutputScale
 *  - PairBiasAttention (with QK-norm, pair bias, and sigmoid gating)
 *  - SwiGLU Transition & TransitionADALN
 *  - PairReprBuilder & PairReprUpdate (with TriangleMultiplicationOutgoing/Incoming + PairTransition)
 *  - LocalLatentsTransformer (with recycling), DecoderTransformer, EncoderTransformer
 *  - End-to-end RDNFlowMatcher trajectory sampler (`generateRfd4pCpu`)
 */

import { decodeLatentsCpu, formatSampleLocalLatents, sequenceForDecoder } from './decoder.js';
import { encodeLatentsCpu } from './encoder.js';
import { pairCondFeatures, pairFeatures, trunkCondFeatures, trunkSeqFeatures } from '../../shared/rfd4p/features.js';
import {
  DEFAULT_SAMPLING,
  flowEmitMasks,
  recycleDepths,
  sampleInitialNoise,
  sampleSchedule,
  seededRng,
  simulationStep,
} from '../../shared/rfd4p/flow.js';
import { processSampleToAtoms, toCifString, toPdbString } from '../../shared/rfd4p/output.js';
import {
  RESAMPLED_FORWARD_MARGINAL,
  buildPartialSchedule,
  partialStepCount,
  corruptBatch,
  resolveInverseFoldingMethod,
  resolvePartialT,
  sampleNoisedTargetBackbone,
} from '../../shared/rfd4p/partial.js';

function getTensorData(weights, name) {
  const entry = weights.tensors ? weights.tensors[name] : weights[name];
  if (!entry) {
    throw new Error(`Missing weight tensor: ${name}`);
  }
  return entry.data ?? entry;
}

function getOptionalTensorData(weights, name) {
  const entry = weights.tensors ? weights.tensors[name] : weights[name];
  if (!entry) return null;
  return entry.data ?? entry;
}

/**
 * Matrix multiplication `Y = X * W^T + bias`, where `X` is `[rows, inDim]`
 * and `W` is `[outDim, inDim]`.
 */
export function linearLayer(x, rows, inDim, weight, outDim, bias = null) {
  const out = new Float32Array(rows * outDim);
  for (let r = 0; r < rows; r += 1) {
    const xBase = r * inDim;
    const outBase = r * outDim;
    for (let o = 0; o < outDim; o += 1) {
      const wBase = o * inDim;
      let sum = bias ? bias[o] : 0.0;
      for (let k = 0; k < inDim; k += 1) {
        sum += x[xBase + k] * weight[wBase + k];
      }
      out[outBase + o] = sum;
    }
  }
  return out;
}

/**
 * RMSNorm over the last dimension `[rows, dim]`.
 */
export function rmsNorm(x, rows, dim, weight = null, eps = 1e-5) {
  const out = new Float32Array(rows * dim);
  for (let r = 0; r < rows; r += 1) {
    const base = r * dim;
    let sumSq = 0.0;
    for (let d = 0; d < dim; d += 1) {
      const v = x[base + d];
      sumSq += v * v;
    }
    const invRms = 1.0 / Math.sqrt(sumSq / dim + eps);
    if (weight) {
      for (let d = 0; d < dim; d += 1) {
        out[base + d] = x[base + d] * invRms * weight[d];
      }
    } else {
      for (let d = 0; d < dim; d += 1) {
        out[base + d] = x[base + d] * invRms;
      }
    }
  }
  return out;
}

/**
 * LayerNorm over the last dimension `[rows, dim]`.
 */
export function layerNorm(x, rows, dim, weight = null, bias = null, eps = 1e-5) {
  const out = new Float32Array(rows * dim);
  for (let r = 0; r < rows; r += 1) {
    const base = r * dim;
    let mean = 0.0;
    for (let d = 0; d < dim; d += 1) {
      mean += x[base + d];
    }
    mean /= dim;
    let varSum = 0.0;
    for (let d = 0; d < dim; d += 1) {
      const diff = x[base + d] - mean;
      varSum += diff * diff;
    }
    const invStd = 1.0 / Math.sqrt(varSum / dim + eps);
    for (let d = 0; d < dim; d += 1) {
      const nVal = (x[base + d] - mean) * invStd;
      const scaled = weight ? nVal * weight[d] : nVal;
      out[base + d] = bias ? scaled + bias[d] : scaled;
    }
  }
  return out;
}

export function applyNorm(x, rows, dim, normType, weight = null, bias = null, eps = 1e-5) {
  if (normType === 'layer_norm') {
    return layerNorm(x, rows, dim, weight, bias, eps);
  }
  return rmsNorm(x, rows, dim, weight, eps);
}

function sigmoid(v) {
  return 1.0 / (1.0 + Math.exp(-v));
}

function silu(v) {
  return v / (1.0 + Math.exp(-v));
}

/**
 * AdaptiveLayerNorm (`rfproteina/nn/modules/adaptive_ln_scale.py`).
 * Supports both per-row conditioning (`condRows === rows`) and broadcast conditioning (`condRows === 1`).
 */
export function adaptiveLayerNorm(x, cond, mask, rows, dim, condRows, dimCond, normType, weights, prefix, eps = 1e-5) {
  const normedX = applyNorm(x, rows, dim, normType, null, null, eps);
  const normCondW = getTensorData(weights, `${prefix}.norm_cond.weight`);
  const normCondB = getOptionalTensorData(weights, `${prefix}.norm_cond.bias`);
  const normedCond = applyNorm(cond, condRows, dimCond, normType, normCondW, normCondB, eps);

  const gammaW = getTensorData(weights, `${prefix}.to_gamma.0.weight`);
  const gammaB = getTensorData(weights, `${prefix}.to_gamma.0.bias`);
  const betaW = getTensorData(weights, `${prefix}.to_beta.weight`);

  const gammaLinear = linearLayer(normedCond, condRows, dimCond, gammaW, dim, gammaB);
  const beta = linearLayer(normedCond, condRows, dimCond, betaW, dim, null);

  const out = new Float32Array(rows * dim);
  for (let r = 0; r < rows; r += 1) {
    const active = !mask || mask[r] !== 0;
    if (!active) continue;
    const cRow = condRows === 1 ? 0 : r;
    const xBase = r * dim;
    const cBase = cRow * dim;
    for (let d = 0; d < dim; d += 1) {
      const g = sigmoid(gammaLinear[cBase + d]);
      out[xBase + d] = normedX[xBase + d] * g + beta[cBase + d];
    }
  }
  return out;
}

/**
 * AdaptiveOutputScale (`rfproteina/nn/modules/adaptive_ln_scale.py`).
 */
export function adaptiveOutputScale(x, cond, mask, rows, dim, dimCond, weights, prefix) {
  const w = getTensorData(weights, `${prefix}.to_adaln_zero_gamma.0.weight`);
  const b = getTensorData(weights, `${prefix}.to_adaln_zero_gamma.0.bias`);
  const gammaLinear = linearLayer(cond, rows, dimCond, w, dim, b);
  const out = new Float32Array(rows * dim);
  for (let r = 0; r < rows; r += 1) {
    if (mask && mask[r] === 0) continue;
    const base = r * dim;
    for (let d = 0; d < dim; d += 1) {
      out[base + d] = x[base + d] * sigmoid(gammaLinear[base + d]);
    }
  }
  return out;
}

/**
 * SwiGLU Transition (`rfproteina/nn/modules/seq_transition_af3.py`).
 * `swish_linear.0.weight` has shape `[dimInner * 2, dim]`, chunked into `(xPart, gates)`,
 * `silu(gates) * xPart`, then projected by `linear_out.weight` `[dim, dimInner]`.
 */
export function swiGluTransition(x, mask, rows, dim, expansionFactor, weights, prefix) {
  const dimInner = dim * expansionFactor;
  const swishW = getTensorData(weights, `${prefix}.swish_linear.0.weight`);
  const outW = getTensorData(weights, `${prefix}.linear_out.weight`);

  const hidden2 = linearLayer(x, rows, dim, swishW, dimInner * 2, null);
  const swigluOut = new Float32Array(rows * dimInner);
  for (let r = 0; r < rows; r += 1) {
    const inBase = r * dimInner * 2;
    const outBase = r * dimInner;
    for (let d = 0; d < dimInner; d += 1) {
      const xVal = hidden2[inBase + d];
      const gate = hidden2[inBase + dimInner + d];
      swigluOut[outBase + d] = silu(gate) * xVal;
    }
  }

  const out = linearLayer(swigluOut, rows, dimInner, outW, dim, null);
  if (mask) {
    for (let r = 0; r < rows; r += 1) {
      if (mask[r] === 0) {
        out.fill(0, r * dim, (r + 1) * dim);
      }
    }
  }
  return out;
}

/**
 * Pair-Biased Multi-Head Self-Attention (`PairBiasAttention` in `pair_bias_attn.py`).
 */
export function pairBiasAttention(
  nodeFeats,
  pairFeats,
  pairMask,
  n,
  nodeDim,
  pairDim,
  heads,
  usePairBias,
  useQkln,
  normType,
  weights,
  prefix,
  eps = 1e-5
) {
  const dimHead = Math.floor(nodeDim / heads);
  const nodeNormW = getTensorData(weights, `${prefix}.node_norm.weight`);
  const nodeNormB = getOptionalTensorData(weights, `${prefix}.node_norm.bias`);
  const normedNodes = applyNorm(nodeFeats, n, nodeDim, normType, nodeNormW, nodeNormB, eps);

  const qkvW = getTensorData(weights, `${prefix}.to_qkv.weight`);
  const qkvB = getOptionalTensorData(weights, `${prefix}.to_qkv.bias`);
  const qkv = linearLayer(normedNodes, n, nodeDim, qkvW, nodeDim * 3, qkvB);

  const qRaw = new Float32Array(n * nodeDim);
  const kRaw = new Float32Array(n * nodeDim);
  const v = new Float32Array(n * nodeDim);
  for (let i = 0; i < n; i += 1) {
    const base3 = i * nodeDim * 3;
    const base1 = i * nodeDim;
    qRaw.set(qkv.subarray(base3, base3 + nodeDim), base1);
    kRaw.set(qkv.subarray(base3 + nodeDim, base3 + 2 * nodeDim), base1);
    v.set(qkv.subarray(base3 + 2 * nodeDim, base3 + 3 * nodeDim), base1);
  }

  const q = useQkln
    ? applyNorm(
        qRaw,
        n,
        nodeDim,
        normType,
        getTensorData(weights, `${prefix}.q_layer_norm.weight`),
        getOptionalTensorData(weights, `${prefix}.q_layer_norm.bias`),
        eps
      )
    : qRaw;
  const k = useQkln
    ? applyNorm(
        kRaw,
        n,
        nodeDim,
        normType,
        getTensorData(weights, `${prefix}.k_layer_norm.weight`),
        getOptionalTensorData(weights, `${prefix}.k_layer_norm.bias`),
        eps
      )
    : kRaw;

  const gW = getTensorData(weights, `${prefix}.to_g.weight`);
  const gB = getOptionalTensorData(weights, `${prefix}.to_g.bias`);
  const g = linearLayer(normedNodes, n, nodeDim, gW, nodeDim, gB);

  let attnBias = null; // [n * n * heads]
  if (usePairBias && pairFeats) {
    const pairNormW = getTensorData(weights, `${prefix}.pair_norm.weight`);
    const pairNormB = getOptionalTensorData(weights, `${prefix}.pair_norm.bias`);
    const normedPair = applyNorm(pairFeats, n * n, pairDim, normType, pairNormW, pairNormB, eps);
    const biasW = getTensorData(weights, `${prefix}.to_bias.weight`);
    attnBias = linearLayer(normedPair, n * n, pairDim, biasW, heads, null);
  }

  const scale = 1.0 / Math.sqrt(dimHead);
  const gatedOut = new Float32Array(n * nodeDim);
  const scores = new Float32Array(n);

  for (let h = 0; h < heads; h += 1) {
    const hOffset = h * dimHead;
    for (let i = 0; i < n; i += 1) {
      let anyValid = false;
      for (let j = 0; j < n; j += 1) {
        if (!pairMask || pairMask[i * n + j] !== 0) {
          anyValid = true;
          break;
        }
      }
      let maxScore = -Infinity;
      for (let j = 0; j < n; j += 1) {
        let dot = 0.0;
        const qiBase = i * nodeDim + hOffset;
        const kjBase = j * nodeDim + hOffset;
        for (let d = 0; d < dimHead; d += 1) {
          dot += q[qiBase + d] * k[kjBase + d];
        }
        let s = dot * scale;
        if (anyValid) {
          const validPair = !pairMask || pairMask[i * n + j] !== 0;
          if (!validPair) {
            s = -1e4;
          } else if (attnBias) {
            s += attnBias[(i * n + j) * heads + h];
          }
        }
        scores[j] = s;
        if (s > maxScore) maxScore = s;
      }

      let sumExp = 0.0;
      for (let j = 0; j < n; j += 1) {
        const wExp = Math.exp(scores[j] - maxScore);
        scores[j] = wExp;
        sumExp += wExp;
      }
      const invSum = sumExp > 0 ? 1.0 / sumExp : 0.0;

      const outBase = i * nodeDim + hOffset;
      for (let d = 0; d < dimHead; d += 1) {
        let acc = 0.0;
        for (let j = 0; j < n; j += 1) {
          acc += scores[j] * invSum * v[j * nodeDim + hOffset + d];
        }
        gatedOut[outBase + d] = sigmoid(g[outBase + d]) * acc;
      }
    }
  }

  const outW = getTensorData(weights, `${prefix}.to_out_node.weight`);
  const outB = getOptionalTensorData(weights, `${prefix}.to_out_node.bias`);
  return linearLayer(gatedOut, n, nodeDim, outW, nodeDim, outB);
}

/**
 * One `MultiheadAttnAndTransition` block (`attn_n_transition.py`).
 */
export function multiheadAttnAndTransition(
  x,
  pairRep,
  cond,
  mask,
  n,
  tokenDim,
  pairDim,
  nheads,
  condDim,
  usePairBias,
  useQkln,
  normType,
  weights,
  prefix,
  eps = 1e-5
) {
  const pairMask = new Uint8Array(n * n);
  for (let i = 0; i < n; i += 1) {
    if (!mask[i]) continue;
    for (let j = 0; j < n; j += 1) {
      if (mask[j]) pairMask[i * n + j] = 1;
    }
  }

  // 1. MHBA with AdaLN + AdaptiveOutputScale + residual
  const xAdaMha = adaptiveLayerNorm(
    x,
    cond,
    mask,
    n,
    tokenDim,
    n,
    condDim,
    normType,
    weights,
    `${prefix}.mhba.adaln`,
    eps
  );
  const xAttn = pairBiasAttention(
    xAdaMha,
    usePairBias ? pairRep : null,
    pairMask,
    n,
    tokenDim,
    pairDim,
    nheads,
    usePairBias,
    useQkln,
    normType,
    weights,
    `${prefix}.mhba.mha`,
    eps
  );
  const xAttnScaled = adaptiveOutputScale(
    xAttn,
    cond,
    mask,
    n,
    tokenDim,
    condDim,
    weights,
    `${prefix}.mhba.scale_output`
  );
  const afterMha = new Float32Array(n * tokenDim);
  for (let i = 0; i < n; i += 1) {
    if (!mask[i]) continue;
    const base = i * tokenDim;
    for (let d = 0; d < tokenDim; d += 1) {
      afterMha[base + d] = x[base + d] + xAttnScaled[base + d];
    }
  }

  // 2. TransitionADALN + residual
  const xAdaTr = adaptiveLayerNorm(
    afterMha,
    cond,
    mask,
    n,
    tokenDim,
    n,
    condDim,
    normType,
    weights,
    `${prefix}.transition.adaln`,
    eps
  );
  const xTr = swiGluTransition(
    xAdaTr,
    mask,
    n,
    tokenDim,
    4,
    weights,
    `${prefix}.transition.transition`
  );
  const xTrScaled = adaptiveOutputScale(
    xTr,
    cond,
    mask,
    n,
    tokenDim,
    condDim,
    weights,
    `${prefix}.transition.scale_output`
  );
  const out = new Float32Array(n * tokenDim);
  for (let i = 0; i < n; i += 1) {
    if (!mask[i]) continue;
    const base = i * tokenDim;
    for (let d = 0; d < tokenDim; d += 1) {
      out[base + d] = afterMha[base + d] + xTrScaled[base + d];
    }
  }
  return out;
}

/**
 * TriangleMultiplicativeUpdate (`Outgoing` or `Incoming`, `triangular_multiplicative_update.py`).
 */
export function triangleMultiplicativeUpdate(z, pairMask, n, cZ, cHidden, direction, weights, prefix, eps = 1e-5) {
  const rows = n * n;
  const lnInW = getTensorData(weights, `${prefix}.layer_norm_in.weight`);
  const lnInB = getOptionalTensorData(weights, `${prefix}.layer_norm_in.bias`);
  const zNorm = layerNorm(z, rows, cZ, lnInW, lnInB, eps);

  const aPW = getTensorData(weights, `${prefix}.linear_a_p.weight`);
  const aPB = getOptionalTensorData(weights, `${prefix}.linear_a_p.bias`);
  const aGW = getTensorData(weights, `${prefix}.linear_a_g.weight`);
  const aGB = getOptionalTensorData(weights, `${prefix}.linear_a_g.bias`);
  const bPW = getTensorData(weights, `${prefix}.linear_b_p.weight`);
  const bPB = getOptionalTensorData(weights, `${prefix}.linear_b_p.bias`);
  const bGW = getTensorData(weights, `${prefix}.linear_b_g.weight`);
  const bGB = getOptionalTensorData(weights, `${prefix}.linear_b_g.bias`);

  const aP = linearLayer(zNorm, rows, cZ, aPW, cHidden, aPB);
  const aG = linearLayer(zNorm, rows, cZ, aGW, cHidden, aGB);
  const bP = linearLayer(zNorm, rows, cZ, bPW, cHidden, bPB);
  const bG = linearLayer(zNorm, rows, cZ, bGW, cHidden, bGB);

  const a = new Float32Array(rows * cHidden);
  const b = new Float32Array(rows * cHidden);
  for (let r = 0; r < rows; r += 1) {
    if (pairMask && pairMask[r] === 0) continue;
    const base = r * cHidden;
    for (let c = 0; c < cHidden; c += 1) {
      a[base + c] = aP[base + c] * sigmoid(aG[base + c]);
      b[base + c] = bP[base + c] * sigmoid(bG[base + c]);
    }
  }

  // Contract over k without materializing O(n^3) tensors
  const x = new Float32Array(rows * cHidden);
  if (direction === 'outgoing') {
    // x[i, j, c] = sum_k a[i, k, c] * b[j, k, c]
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < n; j += 1) {
        const outBase = (i * n + j) * cHidden;
        for (let k = 0; k < n; k += 1) {
          const aBase = (i * n + k) * cHidden;
          const bBase = (j * n + k) * cHidden;
          for (let c = 0; c < cHidden; c += 1) {
            x[outBase + c] += a[aBase + c] * b[bBase + c];
          }
        }
      }
    }
  } else {
    // incoming: x[i, j, c] = sum_k a[k, i, c] * b[k, j, c]
    for (let k = 0; k < n; k += 1) {
      for (let i = 0; i < n; i += 1) {
        const aBase = (k * n + i) * cHidden;
        for (let j = 0; j < n; j += 1) {
          const bBase = (k * n + j) * cHidden;
          const outBase = (i * n + j) * cHidden;
          for (let c = 0; c < cHidden; c += 1) {
            x[outBase + c] += a[aBase + c] * b[bBase + c];
          }
        }
      }
    }
  }

  const lnOutW = getTensorData(weights, `${prefix}.layer_norm_out.weight`);
  const lnOutB = getOptionalTensorData(weights, `${prefix}.layer_norm_out.bias`);
  const xNorm = layerNorm(x, rows, cHidden, lnOutW, lnOutB, eps);

  const zW = getTensorData(weights, `${prefix}.linear_z.weight`);
  const zB = getOptionalTensorData(weights, `${prefix}.linear_z.bias`);
  const gW = getTensorData(weights, `${prefix}.linear_g.weight`);
  const gB = getOptionalTensorData(weights, `${prefix}.linear_g.bias`);

  const xProj = linearLayer(xNorm, rows, cHidden, zW, cZ, zB);
  const gProj = linearLayer(zNorm, rows, cZ, gW, cZ, gB);

  const out = new Float32Array(rows * cZ);
  for (let i = 0; i < rows * cZ; i += 1) {
    out[i] = xProj[i] * sigmoid(gProj[i]);
  }
  return out;
}

/**
 * PairReprUpdate (`rfproteina/nn/modules/pair_update.py`).
 */
export function pairReprUpdate(
  x,
  pairRep,
  mask,
  n,
  tokenDim,
  pairDim,
  triMultC,
  useTriMult,
  pairWrite,
  weights,
  prefix,
  eps = 1e-5
) {
  const rows = n * n;
  const pairMask = new Uint8Array(rows);
  for (let i = 0; i < n; i += 1) {
    if (!mask[i]) continue;
    for (let j = 0; j < n; j += 1) {
      if (mask[j]) pairMask[i * n + j] = 1;
    }
  }

  const xMasked = new Float32Array(n * tokenDim);
  for (let i = 0; i < n; i += 1) {
    if (mask[i]) {
      xMasked.set(x.subarray(i * tokenDim, (i + 1) * tokenDim), i * tokenDim);
    }
  }

  const lnInW = getTensorData(weights, `${prefix}.layer_norm_in.weight`);
  const lnInB = getTensorData(weights, `${prefix}.layer_norm_in.bias`);
  const xNorm = layerNorm(xMasked, n, tokenDim, lnInW, lnInB, eps);
  const linXW = getTensorData(weights, `${prefix}.linear_x.weight`);
  const pij = linearLayer(xNorm, n, tokenDim, linXW, pairDim * 2, null);

  let z = new Float32Array(rows * pairDim);
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      const r = i * n + j;
      if (!pairMask[r]) continue;
      // Note in pair_update.py: pi[:, None, :, :] has shape [b, 1, n, p] (j-indexed)
      // and pj[:, :, None, :] has shape [b, n, 1, p] (i-indexed)
      const piBase = j * pairDim * 2;
      const pjBase = i * pairDim * 2 + pairDim;
      const zBase = r * pairDim;
      if (pairWrite === 'sum') {
        for (let d = 0; d < pairDim; d += 1) {
          z[zBase + d] = pairRep[zBase + d] + pij[piBase + d] + pij[pjBase + d];
        }
      } else {
        for (let d = 0; d < pairDim; d += 1) {
          z[zBase + d] = pairRep[zBase + d] + pij[piBase + d] * pij[pjBase + d];
        }
      }
    }
  }

  if (useTriMult) {
    const triOut = triangleMultiplicativeUpdate(
      z,
      pairMask,
      n,
      pairDim,
      triMultC,
      'outgoing',
      weights,
      `${prefix}.tri_mult_out`,
      eps
    );
    for (let r = 0; r < rows; r += 1) {
      if (!pairMask[r]) continue;
      const base = r * pairDim;
      for (let d = 0; d < pairDim; d += 1) z[base + d] += triOut[base + d];
    }

    const triIn = triangleMultiplicativeUpdate(
      z,
      pairMask,
      n,
      pairDim,
      triMultC,
      'incoming',
      weights,
      `${prefix}.tri_mult_in`,
      eps
    );
    for (let r = 0; r < rows; r += 1) {
      if (!pairMask[r]) continue;
      const base = r * pairDim;
      for (let d = 0; d < pairDim; d += 1) z[base + d] += triIn[base + d];
    }
  }

  // PairTransition (`transition_out`)
  const trPrefix = `${prefix}.transition_out`;
  const trLnW = getTensorData(weights, `${trPrefix}.layer_norm.weight`);
  const trLnB = getTensorData(weights, `${trPrefix}.layer_norm.bias`);
  const zTrNorm = layerNorm(z, rows, pairDim, trLnW, trLnB, eps);
  const l1W = getTensorData(weights, `${trPrefix}.linear_1.weight`);
  const l1B = getTensorData(weights, `${trPrefix}.linear_1.bias`);
  const l2W = getTensorData(weights, `${trPrefix}.linear_2.weight`);
  const l2B = getTensorData(weights, `${trPrefix}.linear_2.bias`);

  const hiddenDim = l1B.length;
  const h = linearLayer(zTrNorm, rows, pairDim, l1W, hiddenDim, l1B);
  for (let i = 0; i < h.length; i += 1) {
    if (h[i] < 0) h[i] = 0;
  }
  const trOut = linearLayer(h, rows, hiddenDim, l2W, pairDim, l2B);
  for (let r = 0; r < rows; r += 1) {
    if (!pairMask[r]) continue;
    const base = r * pairDim;
    for (let d = 0; d < pairDim; d += 1) {
      z[base + d] += trOut[base + d];
    }
  }

  return z;
}

function tensorData(batch, key) {
  const v = batch[key];
  if (!v) throw new Error(`batch is missing '${key}'`);
  return v.data ?? v;
}

/**
 * Run the RFD4-Proteina Trunk (`LocalLatentsTransformer`) on CPU over pre-assembled feature vectors `f`.
 * Mirrors `Rfd4pTrunkGpu.forward(n, f, nRecycle, opts)`.
 *
 * @param {number} n
 * @param {{seq: Float32Array, cond0: Float32Array, cond1?: Float32Array, pair: Float32Array, pairCond: Float32Array}} f
 * @param {ArrayLike<number>} mask `[n]`
 * @param {object} weights
 * @param {object} cfg bundle `config.nn`
 * @param {number} [nRecycle]
 * @param {{insertion?: boolean}} [opts]
 */
export function forwardTrunkCpu(n, f, mask, weights, cfg, nRecycle = 1, opts = {}) {
  const D = cfg.token_dim ?? 768;
  const P = cfg.pair_repr_dim ?? 384;
  const C = cfg.dim_cond ?? 256;
  const PC = cfg.pair_cond_factory_dim ?? 256;
  const TRI = cfg.tri_mult_c ?? 256;
  const H = cfg.nheads ?? 12;
  const norm = 'rmsnorm_torch';
  const W = (k) => getTensorData(weights, k);

  const pairMask = new Uint8Array(n * n);
  for (let i = 0; i < n; i += 1) {
    if (!mask[i]) continue;
    for (let j = 0; j < n; j += 1) {
      if (mask[j]) pairMask[i * n + j] = 1;
    }
  }

  function buildCondC(condFeats) {
    const fdim = condFeats.length / n;
    let c = linearLayer(condFeats, n, fdim, W('nn.cond_factory.linear_out.weight'), C);
    c = swiGluTransition(c, mask, n, C, 2, weights, 'nn.transition_c_1');
    c = swiGluTransition(c, mask, n, C, 2, weights, 'nn.transition_c_2');
    return c;
  }

  const c0 = buildCondC(f.cond0);
  const c1 = nRecycle > 1 ? buildCondC(f.cond1 ?? f.cond0) : c0;

  const sInit = linearLayer(f.seq, n, f.seq.length / n, W('nn.init_repr_factory.linear_out.weight'), D);
  for (let i = 0; i < n; i += 1) {
    if (!mask[i]) sInit.fill(0, i * D, (i + 1) * D);
  }

  let zInit = linearLayer(
    f.pair,
    n * n,
    f.pair.length / (n * n),
    W('nn.pair_repr_builder.init_repr_factory.linear_out.weight'),
    P,
  );
  zInit = applyNorm(zInit, n * n, P, norm, W('nn.pair_repr_builder.init_repr_factory.ln_out.weight'));
  for (let r = 0; r < n * n; r += 1) {
    if (!pairMask[r]) zInit.fill(0, r * P, (r + 1) * P);
  }

  let zCond = linearLayer(
    f.pairCond,
    1,
    f.pairCond.length,
    W('nn.pair_repr_builder.cond_factory.linear_out.weight'),
    PC,
  );
  zCond = applyNorm(zCond, 1, PC, norm, W('nn.pair_repr_builder.cond_factory.ln_out.weight'));
  zInit = adaptiveLayerNorm(zInit, zCond, pairMask, n * n, P, 1, PC, norm, weights, 'nn.pair_repr_builder.adaln');

  let s = new Float32Array(n * D);
  let z = new Float32Array(n * n * P);
  const cycles = Math.max(1, nRecycle);

  for (let cycle = 0; cycle < cycles; cycle += 1) {
    const sNormed = applyNorm(s, n, D, norm, W('nn.s_cycle_norm.weight'));
    const sDelta = linearLayer(sNormed, n, D, W('nn.s_cycle_proj.weight'), D);
    const sIn = new Float32Array(n * D);
    for (let k = 0; k < sIn.length; k += 1) sIn[k] = sInit[k] + sDelta[k];

    const zNormed = applyNorm(z, n * n, P, norm, W('nn.z_cycle_norm.weight'));
    const zDelta = linearLayer(zNormed, n * n, P, W('nn.z_cycle_proj.weight'), P);
    const zIn = new Float32Array(n * n * P);
    for (let k = 0; k < zIn.length; k += 1) zIn[k] = zInit[k] + zDelta[k];

    const c = cycle === 0 ? c0 : c1;
    s = sIn;
    z = zIn;

    for (let i = 0; i < cfg.nlayers; i += 1) {
      s = multiheadAttnAndTransition(
        s,
        z,
        c,
        mask,
        n,
        D,
        P,
        H,
        C,
        true,
        true,
        norm,
        weights,
        `nn.transformer_layers.${i}`,
      );
      if (cfg.update_pair_repr && i < cfg.nlayers - 1 && i % (cfg.update_pair_repr_every_n ?? 2) === 0) {
        z = pairReprUpdate(s, z, mask, n, D, P, TRI, true, 'hadamard', weights, `nn.pair_update_layers.${i}`);
      }
    }
  }

  const sCaNorm = applyNorm(s, n, D, norm, W('nn.ca_linear.0.weight'));
  const bbCa = linearLayer(sCaNorm, n, D, W('nn.ca_linear.1.weight'), 3);

  const sLlNorm = applyNorm(s, n, D, norm, W('nn.local_latents_linear.0.weight'));
  const localLatents = linearLayer(sLlNorm, n, D, W('nn.local_latents_linear.1.weight'), 32);

  const sSeqNorm = applyNorm(s, n, D, norm, W('nn.sequence_readout.0.weight'));
  const logits = linearLayer(sSeqNorm, n, D, W('nn.sequence_readout.1.weight'), 147);

  let insertion = null;
  if (opts.insertion && getOptionalTensorData(weights, 'nn.insertion_q_proj.0.weight')) {
    const qNorm = applyNorm(s, n, D, norm, W('nn.insertion_q_proj.0.weight'));
    const q = linearLayer(qNorm, n, D, W('nn.insertion_q_proj.1.weight'), D);
    const kNorm = applyNorm(s, n, D, norm, W('nn.insertion_k_proj.0.weight'));
    const k = linearLayer(kNorm, n, D, W('nn.insertion_k_proj.1.weight'), D);
    const scale = 1 / Math.sqrt(D);
    insertion = new Float32Array(n * n);
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < n; j += 1) {
        let dot = 0;
        for (let d = 0; d < D; d += 1) dot += q[i * D + d] * k[j * D + d];
        insertion[i * n + j] = dot * scale;
      }
    }
  }

  return { bb_ca: bbCa, local_latents: localLatents, logits, insertion, s, z };
}

export const runTrunkForwardCpu = forwardTrunkCpu;
export const runDecoderForwardCpu = decodeLatentsCpu;

/**
 * Run the full RFD4-Proteina flow-matching trajectory and decoder on CPU (`generateRfd4pCpu`),
 * matching `Rfd4pGeneratorGpu.generate(batch, opts)`.
 */
export async function generateRfd4pCpu(batch, weights, cfg, opts = {}) {
  const pad = tensorData(batch, 'padding_mask');
  const n = batch.padding_mask?.shape ? batch.padding_mask.shape.at(-1) : pad.length;
  const sampling = opts.sampling ?? DEFAULT_SAMPLING;
  const baseNsteps = opts.nsteps ?? sampling.nsteps;
  const rng = opts.rng ?? seededRng(opts.seed ?? 0);
  const decodeEvery = opts.decodeEvery ?? 0;
  const centeringScale = sampling.centering?.scale ?? 0.2;
  const coordCondMask = tensorData(batch, 'coord_cond_mask');
  const moleculeIid = tensorData(batch, 'molecule_iid');

  const emit = flowEmitMasks(
    {
      padding_mask: pad,
      is_in_context_conditioning: tensorData(batch, 'is_in_context_conditioning'),
      supports_local_latent: tensorData(batch, 'supports_local_latent'),
    },
    n,
  );

  const inverseFoldingMethod = resolveInverseFoldingMethod(opts);
  const partialTDict = resolvePartialT({ ...opts, nsteps: baseNsteps }, sampling);
  const isPartial = Object.keys(partialTDict).length > 0;
  const frac = opts.partialTrajectoryFractionSteps ?? opts.partial_trajectory_fraction_steps ?? null;
  const nsteps = isPartial && frac != null ? partialStepCount(frac, baseNsteps) : baseNsteps;
  const depths = recycleDepths(opts.recycleSchedule ?? 'medium', nsteps, opts.fullNsteps ?? baseNsteps);

  let sched;
  let x;
  let targetBackbone = null;
  const resampleTargetBackbone = inverseFoldingMethod === RESAMPLED_FORWARD_MARGINAL;

  if (isPartial) {
    if (partialTDict.bb_ca === undefined) partialTDict.bb_ca = 0.0;
    if (partialTDict.local_latents === undefined) partialTDict.local_latents = 0.0;
    let cleanX1 = opts.x1;
    if (!cleanX1) {
      const caNm = new Float32Array(n * 3);
      const coorsNm = batch.coords_nm ? tensorData(batch, 'coords_nm') : null;
      const coorsA = !coorsNm && batch.coords ? tensorData(batch, 'coords') : null;
      for (let i = 0; i < n; i += 1) {
        const base = (i * 37 + 1) * 3;
        for (let k = 0; k < 3; k += 1) {
          const v = coorsNm ? coorsNm[base + k] : coorsA[base + k] / 10;
          caNm[i * 3 + k] = Number.isFinite(v) ? v : 0;
        }
      }
      const enc = encodeLatentsCpu(batch, {
        n,
        weights,
        cfg: cfg.autoencoder_config.nn_ae.encoder,
        rng,
      });
      cleanX1 = { bb_ca: caNm, local_latents: enc.zLatent };
    }
    const corrupted = corruptBatch({
      n,
      x1: cleanX1,
      partialTDict,
      emit,
      coordCondMask,
      moleculeIid,
      rng,
      centeringScale,
    });
    x = corrupted.xt;
    sched = buildPartialSchedule(partialTDict, nsteps, sampling, {
      fullNsteps: frac != null ? baseNsteps : null,
    });
    if (resampleTargetBackbone) {
      targetBackbone = Float32Array.from(corrupted.x1.bb_ca);
      x.bb_ca = sampleNoisedTargetBackbone({
        n,
        targetBackbone,
        tBb: sched.bb_ca.ts[0],
        emitBbCa: emit.bb_ca,
        coordCondMask,
        moleculeIid,
        rng,
        centeringScale,
      });
    }
  } else {
    sched = sampleSchedule({ ...sampling, nsteps });
    x = sampleInitialNoise({
      n,
      emit,
      coordCondMask,
      moleculeIid,
      rng,
      centeringScale,
    });
  }

  const nnCfg = cfg.nn;
  const distCfg = { dim: nnCfg.xt_pair_dist_dim, min: nnCfg.xt_pair_dist_min, max: nnCfg.xt_pair_dist_max };
  const featCache = {};
  let lastLogits = null;
  let lastInsertion = null;

  for (let s = 0; s < nsteps; s += 1) {
    if (opts.signal?.aborted) throw new Error('Generation aborted');
    const tt = {
      bb_ca: Math.fround(sched.bb_ca.ts[s]),
      local_latents: Math.fround(sched.local_latents.ts[s]),
    };
    const f = {
      seq: trunkSeqFeatures(batch, {
        n,
        xt: x,
        feats: nnCfg.feats_seq,
        weights,
        randperm: (len) => rng.randperm(len, s),
        noiseAlignChirality: nnCfg.noise_align_chirality,
      }).data,
      cond0: trunkCondFeatures(batch, {
        n, t: tt, recycleRemaining: null, feats: nnCfg.feats_cond_seq, weights,
      }).data,
      cond1: trunkCondFeatures(batch, {
        n, t: tt, recycleRemaining: 1, feats: nnCfg.feats_cond_seq, weights,
      }).data,
      pair: pairFeatures(batch, {
        n, caNm: x.bb_ca, feats: nnCfg.feats_pair_repr, dist: distCfg, seqSepDim: nnCfg.seq_sep_dim, cache: featCache,
      }).data,
      pairCond: pairCondFeatures(tt, nnCfg.feats_pair_cond),
    };

    const out = forwardTrunkCpu(n, f, pad, weights, nnCfg, depths[s], { insertion: s === nsteps - 1 });
    for (let i = 0; i < n; i += 1) {
      if (!emit.bb_ca[i]) out.bb_ca.fill(0, i * 3, i * 3 + 3);
      if (!emit.local_latents[i]) out.local_latents.fill(0, i * 32, i * 32 + 32);
    }
    lastLogits = out.logits;
    if (out.insertion) lastInsertion = out.insertion;

    const x1 = {};
    for (const dm of ['bb_ca', 'local_latents']) {
      const ts = sched[dm].ts;
      x1[dm] = simulationStep({
        x: x[dm],
        v: out[dm],
        t: Math.fround(ts[s]),
        dt: ts[s + 1] - ts[s],
        gt: sched[dm].gt[s],
        emit: emit[dm],
        dim: dm === 'bb_ca' ? 3 : 32,
        step: sched[dm].step,
        rng,
      });
    }

    if (resampleTargetBackbone) {
      if (s === nsteps - 1) {
        x.bb_ca = Float32Array.from(targetBackbone);
      } else {
        x.bb_ca = sampleNoisedTargetBackbone({
          n,
          targetBackbone,
          tBb: sched.bb_ca.ts[s + 1],
          emitBbCa: emit.bb_ca,
          coordCondMask,
          moleculeIid,
          rng,
          centeringScale,
        });
      }
      x1.bb_ca = Float32Array.from(targetBackbone);
    }

    if (opts.onStep) {
      let liveDecoded = null;
      if (decodeEvery > 0 && ((s + 1) % decodeEvery === 0 || s + 1 === nsteps)) {
        const liveSeq = sequenceForDecoder(batch, n, lastLogits);
        const dec = decodeLatentsCpu(batch, {
          n, caNm: x1.bb_ca, z: x1.local_latents, seq: liveSeq, weights, cfg: cfg.autoencoder_config.nn_ae.decoder,
        });
        liveDecoded = formatSampleLocalLatents(batch, { n, caNm: x1.bb_ca, decoded: dec });
      }
      await opts.onStep({
        step: s,
        nsteps,
        t: tt,
        depth: depths[s],
        x1,
        xt: { bb_ca: Float32Array.from(x.bb_ca), local_latents: Float32Array.from(x.local_latents) },
        logits: lastLogits,
        decoded: liveDecoded,
      });
    }
  }

  const seq = sequenceForDecoder(batch, n, lastLogits);
  const decoded = decodeLatentsCpu(batch, {
    n, caNm: x.bb_ca, z: x.local_latents, seq, weights, cfg: cfg.autoencoder_config.nn_ae.decoder,
  });
  const formatted = formatSampleLocalLatents(batch, { n, caNm: x.bb_ca, decoded });
  const inputAtoms = opts.inputAtoms ?? batch.__atoms ?? null;
  const structure = processSampleToAtoms({
    coors: formatted.coors,
    atomMask: formatted.atomMask,
    residueType: formatted.residueType,
    paddingMask: pad,
    insertionLogits: lastInsertion,
    inputAtoms,
    batch,
  });
  return {
    n,
    x,
    seq,
    logits: lastLogits,
    insertionLogits: lastInsertion,
    decoded,
    coors: formatted.coors,
    atomMask: formatted.atomMask,
    residueType: formatted.residueType,
    structure,
    cif: toCifString(structure),
    pdb: toPdbString(structure),
  };
}
