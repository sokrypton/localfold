// Dense-tensor primitives for the MPNN engine.
//
// Everything is a flat Float32Array plus an explicit shape. The model only ever
// needs a handful of operations -- an affine map, GELU, LayerNorm, and neighbour
// gathers -- so there is no general broadcasting machinery here on purpose.
//
// Weight layout follows PyTorch: a Linear's weight is [out, in], row major, so
// `out[n][o] = dot(x[n], W[o]) + b[o]` reads two contiguous rows.

/** @typedef {{shape: number[], data: Float32Array}} Tensor */

export function numel(shape) {
  let n = 1;
  for (const d of shape) n *= d;
  return n;
}

/** @returns {Tensor} */
export function zeros(shape) {
  return { shape, data: new Float32Array(numel(shape)) };
}

/** @returns {Tensor} */
export function tensor(shape, data) {
  if (data.length !== numel(shape)) {
    throw new Error(`shape ${shape} does not match ${data.length} elements`);
  }
  return { shape, data };
}

/**
 * y[n, o] = sum_i x[n, i] * W[o, i] + b[o]
 *
 * @param {Float32Array} x    [n, cin]
 * @param {Float32Array} w    [cout, cin]
 * @param {Float32Array|null} b [cout]
 * @param {Float32Array} [out] destination, allocated if omitted
 */
/**
 * Optional accelerator. Set by `useAccelerator()`; when present, `linear`
 * offers each call to it and falls back here if it declines (small shapes) or
 * is absent entirely.
 */
let accel = null;

/** @param {import("./accel.js").Accelerator|null} a */
export function useAccelerator(a) {
  accel = a;
}

export function acceleratorInUse() {
  return accel !== null;
}

/**
 * gelu -> W2 -> gelu -> W3 over [n, hidden], run in one accelerator call when
 * one is available. Returns false if the caller must do it itself.
 */
export function tryTail2(h1, w2, b2, w3, b3, n, hidden, out) {
  return accel !== null && accel.tail2(h1, w2, b2, w3, b3, n, hidden, out);
}

/** A whole node update in one accelerator call; false if unavailable. */
export function tryMessageBlock(h1, maskAttend, hV, w, rows, k, hidden, ff, scale, out) {
  return accel !== null
    && accel.messageBlock(h1, maskAttend, hV, w, rows, k, hidden, ff, scale, out);
}

/** The whole edge featurisation for a chunk; false if unavailable. */
export function tryEdgeFeatures(pos16, xyz, eidx, dCaCa, rowOf, w, g, b, rows, k, hidden, out) {
  return accel !== null
    && accel.edgeFeatures(pos16, xyz, eidx, dCaCa, rowOf, w, g, b, rows, k, hidden, out);
}

/** The encoder's edge half in one call; false if unavailable. */
export function tryEdgeBlock(h1, hE, w, n, hidden, out) {
  return accel !== null && accel.edgeBlock(h1, hE, w, n, hidden, out);
}

/** W_in -> gelu -> W_out, likewise. */
export function tryFeedForward(x, wIn, bIn, wOut, bOut, n, hidden, ff, out) {
  return accel !== null && accel.ff(x, wIn, bIn, wOut, bOut, n, hidden, ff, out);
}

export function linear(x, w, b, n, cin, cout, out) {
  out = out ?? new Float32Array(n * cout);
  if (accel !== null && accel.linear(x, w, b, n, cin, cout, out)) return out;
  let i = 0;
  // Eight input rows share each pass over the weight matrix. Without this the
  // whole of W is streamed from L2 once per row and the kernel is bandwidth
  // bound; with it, throughput roughly doubles.
  for (; i + 8 <= n; i += 8) {
    const x0 = i * cin;
    const y0 = i * cout;
    for (let o = 0; o < cout; o++) {
      const wo = o * cin;
      let a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
      for (let k = 0; k < cin; k++) {
        const wv = w[wo + k];
        a0 += x[x0 + k] * wv;
        a1 += x[x0 + cin + k] * wv;
        a2 += x[x0 + 2 * cin + k] * wv;
        a3 += x[x0 + 3 * cin + k] * wv;
        a4 += x[x0 + 4 * cin + k] * wv;
        a5 += x[x0 + 5 * cin + k] * wv;
        a6 += x[x0 + 6 * cin + k] * wv;
        a7 += x[x0 + 7 * cin + k] * wv;
      }
      const bv = b === null ? 0 : b[o];
      out[y0 + o] = a0 + bv;
      out[y0 + cout + o] = a1 + bv;
      out[y0 + 2 * cout + o] = a2 + bv;
      out[y0 + 3 * cout + o] = a3 + bv;
      out[y0 + 4 * cout + o] = a4 + bv;
      out[y0 + 5 * cout + o] = a5 + bv;
      out[y0 + 6 * cout + o] = a6 + bv;
      out[y0 + 7 * cout + o] = a7 + bv;
    }
  }
  for (; i < n; i++) {
    const xo = i * cin;
    const yo = i * cout;
    for (let o = 0; o < cout; o++) {
      const wo = o * cin;
      let acc = b === null ? 0 : b[o];
      for (let k = 0; k < cin; k++) acc += x[xo + k] * w[wo + k];
      out[yo + o] = acc;
    }
  }
  return out;
}

/**
 * `out[i, :] += scale[i] * (x[src[i], :] · Wᵀ)` for a precomputed projection.
 *
 * Used by the fused message path, where a Linear applied to a concatenation is
 * decomposed into per-block projections that are computed once per node and
 * then gathered onto edges.
 */
export function addScaledGather(out, proj, src, scale, rows, c) {
  for (let i = 0; i < rows; i++) {
    const s = scale[i];
    if (s === 0) continue;
    const from = src[i] * c;
    const to = i * c;
    for (let d = 0; d < c; d++) out[to + d] += s * proj[from + d];
  }
  return out;
}

/** `out[i, :] = a[row(i), :]` broadcast down the neighbour axis, then `+= b`. */
export function broadcastRowInto(out, a, rows, k, c) {
  for (let i = 0; i < rows; i++) {
    const ao = i * c;
    for (let j = 0; j < k; j++) {
      const to = (i * k + j) * c;
      for (let d = 0; d < c; d++) out[to + d] = a[ao + d];
    }
  }
  return out;
}

// Abramowitz & Stegun 7.1.26; |error| < 1.5e-7, comfortably below float32 eps.
const ERF_P = 0.3275911;
const ERF_A = [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429];

export function erf(x) {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + ERF_P * ax);
  const poly = ((((ERF_A[4] * t + ERF_A[3]) * t + ERF_A[2]) * t + ERF_A[1]) * t + ERF_A[0]) * t;
  return sign * (1 - poly * Math.exp(-ax * ax));
}

const [A0, A1, A2, A3, A4] = ERF_A;

/**
 * Exact (erf-based) GELU, matching `torch.nn.GELU()`'s default -- *not* the
 * tanh approximation, which drifts by ~1e-3 and would show up in the logits.
 *
 * `erf` is inlined rather than called. This is the second hottest loop in the
 * engine after the matmul, and at a few hundred thousand elements per call the
 * call overhead was costing about as much as the arithmetic.
 */
export function gelu(a, n = a.length) {
  for (let i = 0; i < n; i++) {
    const v = a[i];
    const x = v * Math.SQRT1_2;
    const ax = x < 0 ? -x : x;
    const t = 1 / (1 + ERF_P * ax);
    const poly = ((((A4 * t + A3) * t + A2) * t + A1) * t + A0) * t;
    const e = 1 - poly * Math.exp(-ax * ax);
    a[i] = 0.5 * v * (1 + (x < 0 ? -e : e));
  }
  return a;
}

/** Row-wise LayerNorm over the trailing axis of length `c`. */
export function layerNorm(x, gamma, beta, n, c, out, eps = 1e-5) {
  out = out ?? new Float32Array(n * c);
  for (let i = 0; i < n; i++) {
    const off = i * c;
    let mean = 0;
    for (let k = 0; k < c; k++) mean += x[off + k];
    mean /= c;
    let variance = 0;
    for (let k = 0; k < c; k++) {
      const d = x[off + k] - mean;
      variance += d * d;
    }
    variance /= c;
    const inv = 1 / Math.sqrt(variance + eps);
    for (let k = 0; k < c; k++) out[off + k] = (x[off + k] - mean) * inv * gamma[k] + beta[k];
  }
  return out;
}

/** In-place `a += b`. */
export function addInto(a, b, n = a.length) {
  for (let i = 0; i < n; i++) a[i] += b[i];
  return a;
}

/**
 * Gather node features onto edges.
 *
 * out[i, k, :] = nodes[idx[i, k], :]
 *
 * @param {Float32Array} nodes [l, c]
 * @param {Int32Array}   idx   [rows, k]
 */
export function gatherNodes(nodes, idx, rows, k, c, out) {
  out = out ?? new Float32Array(rows * k * c);
  let w = 0;
  for (let i = 0; i < rows * k; i++) {
    const src = idx[i] * c;
    for (let d = 0; d < c; d++) out[w++] = nodes[src + d];
  }
  return out;
}

/**
 * Concatenate `[edges, nodes[idx]]` along the feature axis -- the reference
 * implementation's `cat_neighbors_nodes`.
 *
 * @param {Float32Array} nodes [l, cn]
 * @param {Float32Array} edges [rows, k, ce]
 * @returns {Float32Array} [rows, k, ce + cn]
 */
export function catNeighborsNodes(nodes, edges, idx, rows, k, ce, cn, out) {
  const c = ce + cn;
  out = out ?? new Float32Array(rows * k * c);
  for (let i = 0; i < rows * k; i++) {
    const dst = i * c;
    const eo = i * ce;
    for (let d = 0; d < ce; d++) out[dst + d] = edges[eo + d];
    const src = idx[i] * cn;
    for (let d = 0; d < cn; d++) out[dst + ce + d] = nodes[src + d];
  }
  return out;
}

/**
 * Prepend each row's own node feature to its edge block:
 *   out[i, k, :] = [h_V[i, :], h_E[i, k, :]]
 *
 * @param {Float32Array} hV [rows, cv]
 * @param {Float32Array} hE [rows, k, ce]
 */
export function expandNodeOntoEdges(hV, hE, rows, k, cv, ce, out) {
  const c = cv + ce;
  out = out ?? new Float32Array(rows * k * c);
  for (let i = 0; i < rows; i++) {
    const vo = i * cv;
    for (let j = 0; j < k; j++) {
      const dst = (i * k + j) * c;
      for (let d = 0; d < cv; d++) out[dst + d] = hV[vo + d];
      const eo = (i * k + j) * ce;
      for (let d = 0; d < ce; d++) out[dst + cv + d] = hE[eo + d];
    }
  }
  return out;
}

/**
 * Sum messages over the neighbour axis and divide by `scale`.
 *
 * @param {Float32Array} messages [rows, k, c]
 * @param {Float32Array|null} maskAttend [rows, k] or null
 */
export function reduceMessages(messages, maskAttend, rows, k, c, scale, out) {
  out = out ?? new Float32Array(rows * c);
  const inv = 1 / scale;
  for (let i = 0; i < rows; i++) {
    const dst = i * c;
    out.fill(0, dst, dst + c);
    for (let j = 0; j < k; j++) {
      const m = maskAttend === null ? 1 : maskAttend[i * k + j];
      if (m === 0) continue;
      const src = (i * k + j) * c;
      for (let d = 0; d < c; d++) out[dst + d] += m * messages[src + d];
    }
    for (let d = 0; d < c; d++) out[dst + d] *= inv;
  }
  return out;
}

/** Multiply each row of `x` [n, c] by `m[n]`, in place. */
export function maskRows(x, m, n, c) {
  for (let i = 0; i < n; i++) {
    const v = m[i];
    if (v === 1) continue;
    const off = i * c;
    for (let d = 0; d < c; d++) x[off + d] *= v;
  }
  return x;
}

export function logSoftmax(logits, out) {
  const n = logits.length;
  out = out ?? new Float32Array(n);
  let max = -Infinity;
  for (let i = 0; i < n; i++) if (logits[i] > max) max = logits[i];
  let sum = 0;
  for (let i = 0; i < n; i++) sum += Math.exp(logits[i] - max);
  const logZ = max + Math.log(sum);
  for (let i = 0; i < n; i++) out[i] = logits[i] - logZ;
  return out;
}

export function softmax(logits, out) {
  const n = logits.length;
  out = out ?? new Float32Array(n);
  let max = -Infinity;
  for (let i = 0; i < n; i++) if (logits[i] > max) max = logits[i];
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const e = Math.exp(logits[i] - max);
    out[i] = e;
    sum += e;
  }
  for (let i = 0; i < n; i++) out[i] /= sum;
  return out;
}

/**
 * Indices of the `k` smallest values in `values` (length `n`), ascending.
 * Ties break towards the lower index, matching `torch.topk(largest=False)` on
 * the CPU backend -- this matters because the neighbour graph feeds every
 * downstream feature.
 */
export function argTopKSmallest(values, n, k, out) {
  out = out ?? new Int32Array(k);
  const order = new Int32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  // Partial selection sort. k is 32-48 while n is the chain length, so this
  // beats a full sort; the explicit index tie-break keeps the result stable
  // when several distances are exactly equal (masked residues all collapse to
  // the same adjusted distance).
  for (let s = 0; s < k; s++) {
    let best = s;
    let bestVal = values[order[s]];
    let bestIdx = order[s];
    for (let i = s + 1; i < n; i++) {
      const cand = order[i];
      const v = values[cand];
      if (v < bestVal || (v === bestVal && cand < bestIdx)) {
        best = i;
        bestVal = v;
        bestIdx = cand;
      }
    }
    order[best] = order[s];
    order[s] = bestIdx;
    out[s] = bestIdx;
  }
  return out;
}
