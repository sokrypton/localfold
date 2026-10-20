/**
 * A small reverse-mode tape for the BioEmu2 CPU reference: just the operations its denoiser uses, each with its
 * vector-Jacobian product written next to its forward pass.
 *
 * Values are Float32Array (as the reference computes), every reduction accumulates in a double, and gradients flow
 * into ACTIVATIONS only - weights are constants - which is what edm_fkc asks of a denoiser: dU(D(x, sigma))/dx and
 * /dsigma. A node is {data, shape, grad, back}; `backward(node, seed)` runs the tape in reverse creation order.
 */

/**
 * The storage type: Float32Array, as the reference computes - or Float64Array, the arm that says whether a
 * difference from the reference is the tape or float32 itself.
 */
let Store = Float32Array;
export function setTapeStorage(type) { Store = type; }

export class Tape {
  constructor() { this.nodes = []; }

  /** A leaf: an input whose gradient is wanted (track) or a constant (no grad kept). */
  leaf(data, shape, track = false) {
    const node = { data: data instanceof Store ? data : Store.from(data), shape, grad: null, back: null, track };
    if (track) this.nodes.push(node);
    return node;
  }

  #node(data, shape, parents, back) {
    const track = parents.some((p) => p.track);
    const node = { data, shape, grad: null, back: track ? back : null, track, parents };
    if (track) this.nodes.push(node);
    return node;
  }

  static gradOf(node) {
    if (!node.grad) node.grad = new Store(node.data.length);
    return node.grad;
  }

  backward(output, seed) {
    output.grad = Store.from(seed);
    for (let i = this.nodes.length - 1; i >= 0; i -= 1) {
      const node = this.nodes[i];
      if (node.back && node.grad) node.back(node.grad);
    }
  }

  // --- linear algebra ---------------------------------------------------------------------------------------------

  /** y[n, o] = x[n, i] @ W[o, i]^T (+ b[o]): nn.Linear with a torch-layout weight. */
  linear(x, weight, bias = null, outFeatures = null) {
    const [n, inF] = x.shape;
    const outF = outFeatures ?? weight.length / inF;
    const y = new Store(n * outF);
    const xd = x.data;
    for (let r = 0; r < n; r += 1) {
      const xr = r * inF;
      for (let o = 0; o < outF; o += 1) {
        let s = bias ? bias[o] : 0;
        const wo = o * inF;
        for (let k = 0; k < inF; k += 1) s += xd[xr + k] * weight[wo + k];
        y[r * outF + o] = s;
      }
    }
    return this.#node(y, [n, outF], [x], (gy) => {
      if (!x.track) return;
      const gx = Tape.gradOf(x);
      for (let r = 0; r < n; r += 1) {
        const row = new Float64Array(inF);
        for (let o = 0; o < outF; o += 1) {
          const g = gy[r * outF + o];
          if (g === 0) continue;
          const wo = o * inF;
          for (let k = 0; k < inF; k += 1) row[k] += g * weight[wo + k];
        }
        for (let k = 0; k < inF; k += 1) gx[r * inF + k] += row[k];
      }
    });
  }

  /** LayerNorm over the last axis (biased variance, eps 1e-5), optional affine. */
  layerNorm(x, weight = null, bias = null) {
    const [n, c] = x.shape;
    const y = new Store(n * c);
    const xhat = new Float64Array(n * c);
    const inv = new Float64Array(n);
    for (let r = 0; r < n; r += 1) {
      let mu = 0;
      for (let k = 0; k < c; k += 1) mu += x.data[r * c + k];
      mu /= c;
      let v = 0;
      for (let k = 0; k < c; k += 1) { const d = x.data[r * c + k] - mu; v += d * d; }
      inv[r] = 1 / Math.sqrt(v / c + 1e-5);
      for (let k = 0; k < c; k += 1) {
        xhat[r * c + k] = (x.data[r * c + k] - mu) * inv[r];
        y[r * c + k] = xhat[r * c + k] * (weight ? weight[k] : 1) + (bias ? bias[k] : 0);
      }
    }
    return this.#node(y, [n, c], [x], (gy) => {
      if (!x.track) return;
      const gx = Tape.gradOf(x);
      for (let r = 0; r < n; r += 1) {
        let m1 = 0;
        let m2 = 0;
        const g = new Float64Array(c);
        for (let k = 0; k < c; k += 1) {
          g[k] = gy[r * c + k] * (weight ? weight[k] : 1);
          m1 += g[k];
          m2 += g[k] * xhat[r * c + k];
        }
        m1 /= c;
        m2 /= c;
        for (let k = 0; k < c; k += 1) gx[r * c + k] += inv[r] * (g[k] - m1 - xhat[r * c + k] * m2);
      }
    });
  }

  // --- elementwise ------------------------------------------------------------------------------------------------

  #unary(x, f, df) {
    const y = Store.from(x.data, f);
    return this.#node(y, x.shape, [x], (gy) => {
      if (!x.track) return;
      const gx = Tape.gradOf(x);
      for (let i = 0; i < y.length; i += 1) gx[i] += gy[i] * df(x.data[i], y[i]);
    });
  }

  silu(x) {
    return this.#unary(x, (v) => v / (1 + Math.exp(-v)), (v) => { const s = 1 / (1 + Math.exp(-v)); return s * (1 + v * (1 - s)); });
  }

  sigmoid(x) { return this.#unary(x, (v) => 1 / (1 + Math.exp(-v)), (v, y) => y * (1 - y)); }

  /** a + b, b of a's shape or one row broadcast over a's rows. */
  add(a, b) {
    const rowBroadcast = b.data.length !== a.data.length;
    const c = a.shape[a.shape.length - 1];
    const y = Store.from(a.data, (v, i) => v + b.data[rowBroadcast ? i % c : i]);
    return this.#node(y, a.shape, [a, b], (gy) => {
      if (a.track) { const ga = Tape.gradOf(a); for (let i = 0; i < gy.length; i += 1) ga[i] += gy[i]; }
      if (b.track) { const gb = Tape.gradOf(b); for (let i = 0; i < gy.length; i += 1) gb[rowBroadcast ? i % c : i] += gy[i]; }
    });
  }

  /** a * b elementwise, same shapes. */
  mul(a, b) {
    const y = Store.from(a.data, (v, i) => v * b.data[i]);
    return this.#node(y, a.shape, [a, b], (gy) => {
      if (a.track) { const ga = Tape.gradOf(a); for (let i = 0; i < gy.length; i += 1) ga[i] += gy[i] * b.data[i]; }
      if (b.track) { const gb = Tape.gradOf(b); for (let i = 0; i < gy.length; i += 1) gb[i] += gy[i] * a.data[i]; }
    });
  }

  /** x * s for a one-element node s (a function of sigma). */
  scale(x, s) {
    const y = Store.from(x.data, (v) => v * s.data[0]);
    return this.#node(y, x.shape, [x, s], (gy) => {
      if (x.track) { const gx = Tape.gradOf(x); for (let i = 0; i < gy.length; i += 1) gx[i] += gy[i] * s.data[0]; }
      if (s.track) { let acc = 0; for (let i = 0; i < gy.length; i += 1) acc += gy[i] * x.data[i]; Tape.gradOf(s)[0] += acc; }
    });
  }

  /** f(s) of a one-element node, with its derivative df(s). */
  scalar(s, f, df) {
    const y = Store.of(f(s.data[0]));
    return this.#node(y, [1], [s], (gy) => { if (s.track) Tape.gradOf(s)[0] += gy[0] * df(s.data[0]); });
  }

  /** where(mask[row], x, 0) - a row mask, constant. */
  maskRows(x, mask) {
    const c = x.shape[1];
    const y = Store.from(x.data, (v, i) => (mask[Math.floor(i / c)] ? v : 0));
    return this.#node(y, x.shape, [x], (gy) => {
      if (!x.track) return;
      const gx = Tape.gradOf(x);
      for (let i = 0; i < gy.length; i += 1) if (mask[Math.floor(i / c)]) gx[i] += gy[i];
    });
  }

  // --- layout -----------------------------------------------------------------------------------------------------

  concat(parts) {
    const n = parts[0].shape[0];
    const widths = parts.map((p) => p.shape[1]);
    const c = widths.reduce((s, w) => s + w, 0);
    const y = new Store(n * c);
    for (let r = 0; r < n; r += 1) {
      let at = 0;
      parts.forEach((p, k) => { y.set(p.data.subarray(r * widths[k], (r + 1) * widths[k]), r * c + at); at += widths[k]; });
    }
    return this.#node(y, [n, c], parts, (gy) => {
      let at = 0;
      parts.forEach((p, k) => {
        if (p.track) {
          const gp = Tape.gradOf(p);
          for (let r = 0; r < n; r += 1) for (let j = 0; j < widths[k]; j += 1) gp[r * widths[k] + j] += gy[r * c + at + j];
        }
        at += widths[k];
      });
    });
  }

  /** columns [from, from + width). */
  slice(x, from, width) {
    const [n, c] = x.shape;
    const y = new Store(n * width);
    for (let r = 0; r < n; r += 1) for (let j = 0; j < width; j += 1) y[r * width + j] = x.data[r * c + from + j];
    return this.#node(y, [n, width], [x], (gy) => {
      if (!x.track) return;
      const gx = Tape.gradOf(x);
      for (let r = 0; r < n; r += 1) for (let j = 0; j < width; j += 1) gx[r * c + from + j] += gy[r * width + j];
    });
  }

  /** rows index[i] of x, or zeros where index[i] < 0 (or the row mask is off). */
  gatherRows(x, index) {
    const c = x.shape[1];
    const y = new Store(index.length * c);
    index.forEach((src, i) => { if (src >= 0) y.set(x.data.subarray(src * c, (src + 1) * c), i * c); });
    return this.#node(y, [index.length, c], [x], (gy) => {
      if (!x.track) return;
      const gx = Tape.gradOf(x);
      index.forEach((src, i) => { if (src >= 0) for (let j = 0; j < c; j += 1) gx[src * c + j] += gy[i * c + j]; });
    });
  }

  // --- attention --------------------------------------------------------------------------------------------------

  /**
   * softmax(q k^T / sqrt(hd) + bias, masked) v over explicit (query, key) index lists:
   * q, k, v are [n, H * hd] rows; `keysOf(i)` lists query i's key rows (-1 = an out-of-range slot: zero key, zero
   * value, masked); `biasOf(h, i, slot)` the constant pair bias; `live(key)` the key mask. -1e6 for a masked logit,
   * as the reference writes it.
   */
  attention(q, k, v, heads, keysOf, biasOf, live) {
    const [n, d] = q.shape;
    const hd = d / heads;
    const out = new Store(n * d);
    const probs = [];
    const scale = 1 / Math.sqrt(hd);
    for (let i = 0; i < n; i += 1) {
      const keys = keysOf(i);
      const p = new Float64Array(heads * keys.length);
      for (let h = 0; h < heads; h += 1) {
        let max = -Infinity;
        for (let s = 0; s < keys.length; s += 1) {
          const kj = keys[s];
          let logit;
          if (kj >= 0 && live(kj)) {
            let dotp = 0;
            for (let e = 0; e < hd; e += 1) dotp += q.data[i * d + h * hd + e] * k.data[kj * d + h * hd + e];
            logit = Math.fround(Math.fround(dotp * scale) + biasOf(h, i, s));
          } else {
            let dotp = 0;
            if (kj >= 0) for (let e = 0; e < hd; e += 1) dotp += q.data[i * d + h * hd + e] * k.data[kj * d + h * hd + e];
            logit = -1e6;
          }
          p[h * keys.length + s] = logit;
          if (logit > max) max = logit;
        }
        let sum = 0;
        for (let s = 0; s < keys.length; s += 1) { const e = Math.exp(p[h * keys.length + s] - max); p[h * keys.length + s] = e; sum += e; }
        for (let s = 0; s < keys.length; s += 1) p[h * keys.length + s] /= sum;
        for (let e = 0; e < hd; e += 1) {
          let acc = 0;
          for (let s = 0; s < keys.length; s += 1) if (keys[s] >= 0) acc += p[h * keys.length + s] * v.data[keys[s] * d + h * hd + e];
          out[i * d + h * hd + e] = acc;
        }
      }
      probs.push(p);
    }
    return this.#node(out, [n, d], [q, k, v], (gy) => {
      const gq = q.track ? Tape.gradOf(q) : null;
      const gk = k.track ? Tape.gradOf(k) : null;
      const gv = v.track ? Tape.gradOf(v) : null;
      for (let i = 0; i < n; i += 1) {
        const keys = keysOf(i);
        const p = probs[i];
        for (let h = 0; h < heads; h += 1) {
          // dP = gy . v ; dlogit = P (dP - sum P dP)
          const dp = new Float64Array(keys.length);
          let dot = 0;
          for (let s = 0; s < keys.length; s += 1) {
            if (keys[s] < 0) continue;
            let acc = 0;
            for (let e = 0; e < hd; e += 1) acc += gy[i * d + h * hd + e] * v.data[keys[s] * d + h * hd + e];
            dp[s] = acc;
            dot += p[h * keys.length + s] * acc;
          }
          for (let s = 0; s < keys.length; s += 1) {
            const kj = keys[s];
            if (kj < 0) continue;
            const pw = p[h * keys.length + s];
            if (gv) for (let e = 0; e < hd; e += 1) gv[kj * d + h * hd + e] += pw * gy[i * d + h * hd + e];
            if (!live(kj)) continue;            // a masked logit is a constant
            const dl = pw * (dp[s] - dot) * scale;
            if (dl === 0) continue;
            for (let e = 0; e < hd; e += 1) {
              if (gq) gq[i * d + h * hd + e] += dl * k.data[kj * d + h * hd + e];
              if (gk) gk[kj * d + h * hd + e] += dl * q.data[i * d + h * hd + e];
            }
          }
        }
      }
    });
  }

  /** cos(2 pi (t w + b)) for a one-element t: the Fourier embedding of the noise level. */
  fourier(t, w, b) {
    const y = Store.from(w, (wk, k) => Math.cos(2 * Math.PI * (t.data[0] * wk + b[k])));
    return this.#node(y, [1, w.length], [t], (gy) => {
      if (!t.track) return;
      let acc = 0;
      for (let k = 0; k < w.length; k += 1) acc += gy[k] * -Math.sin(2 * Math.PI * (t.data[0] * w[k] + b[k])) * 2 * Math.PI * w[k];
      Tape.gradOf(t)[0] += acc;
    });
  }

  /** one row repeated n times ([1, c] -> [n, c]). */
  repeatRow(x, n) {
    const c = x.shape[1];
    const y = new Store(n * c);
    for (let r = 0; r < n; r += 1) y.set(x.data, r * c);
    return this.#node(y, [n, c], [x], (gy) => {
      if (!x.track) return;
      const gx = Tape.gradOf(x);
      for (let r = 0; r < n; r += 1) for (let j = 0; j < c; j += 1) gx[j] += gy[r * c + j];
    });
  }
}
