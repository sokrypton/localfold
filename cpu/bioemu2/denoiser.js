/**
 * BioEmu2's denoiser on the CPU, with its vector-Jacobian product: the reference every backend's edm_fkc is held to.
 *
 * A transcription of af3-any-model's network/bioemu2.py BioEmu2.denoise - the dual-stream stack of 16 layers, each a
 * windowed atom transformer and a full token transformer (DiT adaLN-Zero) coupled through every token's centre atom
 * - on the reverse-mode tape in tape.js, so the same code gives D(x, sigma) and, through `vjp`, the gradient of
 * w . D(x, sigma) with respect to x AND sigma. sigma reaches the whole conditioning path (c_in, the Fourier embedding
 * of t = log(sigma / sigma_data) / 4, every layer's adaLN, c_skip and c_out), which is why the second half exists.
 * The static context (ctx.z, ctx.q0, ctx.p), s_inputs and the trunk single are inputs and constants: what
 * `static_context` computes once a fold.
 *
 * Held to that port's own denoise and jax.grad by tools/check-bioemu2-denoiser.js. Slow - minutes at 600 atoms - and
 * written to be read, not run in a fold.
 */
import { Tape } from "./tape.js";

export const SIGMA_DATA = 16.0;
const W_Q = 32;
const W_K = 128;
const CTX = (W_K - W_Q) / 2;
const SCOPE = "diffuser/~/bioemu2/";

/** Every structure_module weight the denoiser reads, layer stacks split into per-layer views. */
export async function denoiserWeights(store) {
  const names = Object.keys(store.manifest.tensors).filter((n) => n.startsWith(`${SCOPE}structure_module/`));
  const out = {};
  for (const name of names) {
    const data = await store.tensor(name);
    const key = name.slice(SCOPE.length + "structure_module/".length);
    out[key] = { data: data instanceof Float32Array ? data : Float32Array.from(data), shape: store.shape(name) };
  }
  return out;
}

const layerView = (w, name, layer) => {
  const t = w[name];
  if (!t) throw new Error(`no weight ${name}`);
  const per = t.data.length / t.shape[0];
  return t.data.subarray(layer * per, (layer + 1) * per);
};
const plain = (w, name) => {
  const t = w[name];
  if (!t) throw new Error(`no weight ${name}`);
  return t.data;
};

/** The vendor's key window of atom k-window `win`: atoms win*W_Q - CTX ... + W_K, -1 out of range. */
const keyWindow = (win, n) => Array.from({ length: W_K }, (_, j) => { const a = win * W_Q - CTX + j; return a >= 0 && a < n ? a : -1; });

/**
 * D(x, sigma), on a tape. `f` the features (atom_mask, atom_pad_mask, token_idx_at_atom, token_center_atom_index,
 * token_pad_mask, is_dynamics_conditioned, dt, coords_cond), `ctx` {z [T*T*256], q0 [N*128], p [NW*32*128*16]},
 * `sInputs` [T * 455], `sTrunk` [T * 384]; x and sigma are tape leaves the caller made (track them for a VJP).
 */
export function denoiseOnTape(tape, w, f, ctx, sInputs, sTrunk, x, sigma, dims) {
  const { N, T } = dims;
  const atomTok = f.token_idx_at_atom;
  const atomPad = f.atom_pad_mask;
  const centre = f.token_center_atom_index;
  const tokenPad = f.token_pad_mask;
  const L = w["layers/token_block/cond_0/weight"].shape[0];
  const sd = SIGMA_DATA;
  const cIn = tape.scalar(sigma, (s) => 1 / Math.sqrt(s * s + sd * sd), (s) => -s / (s * s + sd * sd) ** 1.5);
  const r = tape.maskRows(tape.scale(x, cIn), f.atom_mask);
  const t = tape.scalar(sigma, (s) => 0.25 * Math.log(s / sd), (s) => 0.25 / s);

  // the single conditioning
  // (the widths are the data's: the trunk single is 384 here and s_inputs 455, and a typed-in width misreads both)
  const cs = sTrunk.length / T;
  const ci = sInputs.length / T;
  const sCat = tape.leaf(concatRows(sTrunk, cs, sInputs, ci, T), [T, cs + ci]);
  let s = tape.linear(tape.layerNorm(sCat, plain(w, "single_conditioner/norm_single/weight"), plain(w, "single_conditioner/norm_single/bias")),
                      plain(w, "single_conditioner/single_embed/weight"), plain(w, "single_conditioner/single_embed/bias"));
  const four = tape.layerNorm(tape.fourier(t, plain(w, "single_conditioner/fourier_embed/proj/weight"), plain(w, "single_conditioner/fourier_embed/proj/bias")),
                              plain(w, "single_conditioner/norm_fourier/weight"), plain(w, "single_conditioner/norm_fourier/bias"));
  s = tape.add(s, tape.linear(four, plain(w, "single_conditioner/fourier_to_single/weight")));
  if (f.is_dynamics_conditioned) {
    const taus = plain(w, "single_conditioner/dt_embed/taus");
    const qf = plain(w, "single_conditioner/dt_embed/quality_factors");
    const emb = dampedHarmonic(f.dt, taus, qf);
    const dtTerm = tape.leaf(linearConst(emb, plain(w, "single_conditioner/dt_embed_to_single/weight")), [1, s.shape[1]]);
    s = tape.add(s, dtTerm);
  }
  for (let i = 0; i < 2; i += 1) s = tape.add(s, transition(tape, w, `single_conditioner/transitions_${i}`, s, true));
  const cond = s;

  // the atom stream's initial state
  let q = tape.add(tape.leaf(ctx.q0, [N, 128]), tape.linear(r, plain(w, "atom_feature_embedder/r_to_q_trans/weight")));
  if (f.is_dynamics_conditioned && f.coords_cond) {
    const contrib = linearConst(f.coords_cond, plain(w, "atom_feature_embedder/cond_to_q_trans/weight"), 3);
    const taus = plain(w, "atom_feature_embedder/dt_gate_embed/taus");
    const logw = plain(w, "atom_feature_embedder/dt_gate_log_w");
    let gate = 0;
    for (let k = 0; k < taus.length; k += 1) gate += Math.exp(-f.dt / taus[k]) * Math.log1p(Math.exp(logw[k]));
    q = tape.add(q, tape.leaf(Float32Array.from(contrib, (v) => v * gate), [N, 128]));
  }
  let a = tape.linear(tape.layerNorm(tape.leaf(sTrunk, [T, cs]), plain(w, "s_trunk_to_a_0/weight"), plain(w, "s_trunk_to_a_0/bias")),
                      plain(w, "s_trunk_to_a_1/weight"));
  a = tape.add(a, tape.linear(tape.maskRows(tape.gatherRows(r, centre), tokenPad), plain(w, "r_token_to_a/weight")));
  const atomIndex = Array.from(atomTok, (tk, i) => (atomPad[i] ? tk : -1));
  const cToQ = tape.gatherRows(cond, atomIndex);

  for (let layer = 0; layer < L; layer += 1) {
    q = conditionedLayer(tape, w, "layers/atom_block/transformer_layer", layer, q, cToQ, {
      windowed: true, heads: 4, pair: ctx.p, pad: atomPad, N });
    a = conditionedLayer(tape, w, "layers/token_block", layer, a, cond, {
      windowed: false, heads: 16, pair: ctx.z, pad: tokenPad, T });
    const dA = a.shape[1];
    const dQ = q.shape[1];
    const qc = tape.linear(tape.silu(tape.layerNorm(cToQ, layerView(w, "layers/coupling/q_coupling_cond_0/weight", layer),
                                                    layerView(w, "layers/coupling/q_coupling_cond_0/bias", layer))),
                           layerView(w, "layers/coupling/q_coupling_cond_2/weight", layer),
                           layerView(w, "layers/coupling/q_coupling_cond_2/bias", layer));
    const sc = tape.linear(tape.silu(tape.layerNorm(cond, layerView(w, "layers/coupling/s_coupling_cond_0/weight", layer),
                                                    layerView(w, "layers/coupling/s_coupling_cond_0/bias", layer))),
                           layerView(w, "layers/coupling/s_coupling_cond_2/weight", layer),
                           layerView(w, "layers/coupling/s_coupling_cond_2/bias", layer));
    const width = dA + dQ;
    const sToQIn = tape.concat([tape.gatherRows(a, atomIndex), q]);
    const sToQ = tape.mul(tape.slice(qc, 2 * width, qc.shape[1] - 2 * width),
      transitionLayer(tape, w, "layers/coupling/s_to_q_mixer", layer, adaLn(tape, sToQIn, tape.slice(qc, 0, width), tape.slice(qc, width, width))));
    const centreIndex = Array.from(centre, (c, i) => (tokenPad[i] ? c : -1));
    const qToSIn = tape.concat([tape.gatherRows(q, centreIndex), a]);
    const qToS = tape.mul(tape.slice(sc, 2 * width, sc.shape[1] - 2 * width),
      transitionLayer(tape, w, "layers/coupling/q_to_s_mixer", layer, adaLn(tape, qToSIn, tape.slice(sc, 0, width), tape.slice(sc, width, width))));
    a = tape.add(a, qToS);
    q = tape.add(q, sToQ);
  }

  const h = tape.add(tape.linear(q, plain(w, "readout/q_to_hidden/weight")),
                     tape.gatherRows(tape.linear(a, plain(w, "readout/s_to_hidden/weight")), atomIndex));
  const out = tape.linear(tape.layerNorm(h, plain(w, "readout/to_out_0/weight")), plain(w, "readout/to_out_1/weight"));
  const cSkip = tape.scalar(sigma, (v) => (sd * sd) / (v * v + sd * sd), (v) => (-2 * v * sd * sd) / (v * v + sd * sd) ** 2);
  const cOut = tape.scalar(sigma, (v) => (v * sd) / Math.sqrt(sd * sd + v * v), (v) => (sd ** 3) / (sd * sd + v * v) ** 1.5);
  return tape.add(tape.scale(x, cSkip), tape.scale(out, cOut));
}

function concatRows(a, ca, b, cb, n) {
  const out = new Float32Array(n * (ca + cb));
  for (let r = 0; r < n; r += 1) { out.set(a.subarray(r * ca, (r + 1) * ca), r * (ca + cb)); out.set(b.subarray(r * cb, (r + 1) * cb), r * (ca + cb) + ca); }
  return out;
}

function linearConst(x, weight, inF = x.length) {
  const outF = weight.length / inF;
  const n = x.length / inF;
  const y = new Float32Array(n * outF);
  for (let r = 0; r < n; r += 1) for (let o = 0; o < outF; o += 1) {
    let s = 0;
    for (let k = 0; k < inF; k += 1) s += x[r * inF + k] * weight[o * inF + k];
    y[r * outF + o] = s;
  }
  return y;
}

/** DampedHarmonicTimeEmbedding for a scalar lag: [envelope, sin, cos], timescale-major. */
function dampedHarmonic(dt, taus, q) {
  const n = taus.length;
  const env = Array.from(taus, (tau) => Math.exp(-dt / tau));
  const sin = [];
  const cos = [];
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < q.length; j += 1) {
      const phase = 2 * Math.PI * q[j] * (dt / taus[i]);
      sin.push(env[i] * Math.sin(phase));
      cos.push(env[i] * Math.cos(phase));
    }
  }
  return Float32Array.from([...env, ...sin, ...cos]);
}

function adaLn(tape, x, alpha, beta) {
  const n = tape.layerNorm(x);
  const one = tape.leaf(new Float32Array(alpha.data.length).fill(1), alpha.shape);
  return tape.add(tape.mul(n, tape.add(alpha, one)), beta);
}

/** SwiGLU Transition: fc3(silu(fc1(LN x)) * fc2(LN x)), a plain (non-layer) weight prefix. */
function transition(tape, w, prefix, x, norm) {
  const h = norm ? tape.layerNorm(x, plain(w, `${prefix}/norm/weight`), plain(w, `${prefix}/norm/bias`)) : x;
  return tape.linear(tape.mul(tape.silu(tape.linear(h, plain(w, `${prefix}/fc1/weight`))), tape.linear(h, plain(w, `${prefix}/fc2/weight`))),
                     plain(w, `${prefix}/fc3/weight`));
}

/** The same Transition, without its norm, at one layer of a stack. */
function transitionLayer(tape, w, prefix, layer, x) {
  return tape.linear(tape.mul(tape.silu(tape.linear(x, layerView(w, `${prefix}/fc1/weight`, layer))),
                              tape.linear(x, layerView(w, `${prefix}/fc2/weight`, layer))),
                     layerView(w, `${prefix}/fc3/weight`, layer));
}

/** ConditionedTransformerLayer: x + gamma * op((1 + alpha) LN x + beta), for the attention and then the transition. */
function conditionedLayer(tape, w, prefix, layer, x, c, o) {
  const d = x.shape[1];
  const cond = tape.linear(tape.silu(tape.layerNorm(c, layerView(w, `${prefix}/cond_0/weight`, layer), layerView(w, `${prefix}/cond_0/bias`, layer))),
                           layerView(w, `${prefix}/cond_2/weight`, layer), layerView(w, `${prefix}/cond_2/bias`, layer));
  const piece = (k) => tape.slice(cond, k * d, d);
  let h = adaLn(tape, x, piece(0), piece(1));
  const attnPrefix = o.windowed ? `${prefix}/attention/attention_layer/pair_bias_attn` : `${prefix}/attention/pair_bias_attn`;
  h = o.windowed ? windowAttention(tape, w, attnPrefix, layer, h, o) : fullAttention(tape, w, attnPrefix, layer, h, o);
  x = tape.add(x, tape.mul(piece(2), h));
  h = transitionLayer(tape, w, `${prefix}/transition`, layer, adaLn(tape, x, piece(3), piece(4)));
  return tape.add(x, tape.mul(piece(5), h));
}

/** LN(pair) (no offset) -> a per-head bias, for a constant pair of `rows` x `cz`. */
function pairBias(pair, rows, cz, normW, projZ, heads) {
  const out = new Float32Array(rows * heads);
  for (let r = 0; r < rows; r += 1) {
    let mu = 0;
    for (let k = 0; k < cz; k += 1) mu += pair[r * cz + k];
    mu /= cz;
    let v = 0;
    for (let k = 0; k < cz; k += 1) { const d = pair[r * cz + k] - mu; v += d * d; }
    const inv = 1 / Math.sqrt(v / cz + 1e-5);
    for (let h = 0; h < heads; h += 1) {
      let s = 0;
      for (let k = 0; k < cz; k += 1) s += Math.fround((pair[r * cz + k] - mu) * inv * normW[k]) * projZ[h * cz + k];
      out[r * heads + h] = s;
    }
  }
  return out;
}

function projections(tape, w, prefix, layer, x, d) {
  const qkv = tape.linear(x, layerView(w, `${prefix}/proj_qkv/weight`, layer));
  const q = tape.add(tape.slice(qkv, 0, d), tape.leaf(layerView(w, `${prefix}/proj_q_bias`, layer), [d]));
  const g = tape.sigmoid(tape.linear(x, layerView(w, `${prefix}/proj_g/weight`, layer)));
  return { qkv, q, g };
}

function fullAttention(tape, w, prefix, layer, x, o) {
  const T = o.T;
  const d = x.shape[1];
  const { qkv, q, g } = projections(tape, w, prefix, layer, x, d);
  const k = tape.slice(qkv, d, d);
  const v = tape.slice(qkv, 2 * d, d);
  const cz = o.pair.length / (T * T);
  const bias = pairBias(o.pair, T * T, cz, layerView(w, `${prefix}/norm_z/weight`, layer), layerView(w, `${prefix}/proj_z/weight`, layer), o.heads);
  const keys = Array.from({ length: T }, (_, j) => j);
  const att = tape.attention(q, k, v, o.heads, () => keys, (h, i, s) => bias[(i * T + s) * o.heads + h], (j) => !!o.pad[j]);
  return tape.linear(tape.mul(g, att), layerView(w, `${prefix}/proj_o/weight`, layer));
}

function windowAttention(tape, w, prefix, layer, x, o) {
  const N = o.N;
  const d = x.shape[1];
  const heads = o.heads;
  const hd = d / heads;
  const { qkv, q, g } = projections(tape, w, prefix, layer, x, d);
  // the vendor's fused key/value layout: kv_raw reshaped (..., 2, heads, hd) - so the key is columns h*hd.. of the
  // first half PER HEAD and the value the second: [k_h0 .. k_h(H-1), v_h0 .. v_h(H-1)] across d .. 3d
  const kv = tape.slice(qkv, d, 2 * d);
  const k = tape.slice(kv, 0, d);
  const v = tape.slice(kv, d, d);
  const NW = N / W_Q;
  const cz = o.pair.length / (NW * W_Q * W_K);
  const bias = pairBias(o.pair, NW * W_Q * W_K, cz, layerView(w, `${prefix}/norm_z/weight`, layer), layerView(w, `${prefix}/proj_z/weight`, layer), heads);
  const windows = Array.from({ length: NW }, (_, win) => keyWindow(win, N));
  const att = tape.attention(q, k, v, heads, (i) => windows[Math.floor(i / W_Q)],
    (h, i, s) => bias[((Math.floor(i / W_Q) * W_Q + (i % W_Q)) * W_K + s) * heads + h], (j) => !!o.pad[j]);
  void hd;
  return tape.linear(tape.mul(g, att), layerView(w, `${prefix}/proj_o/weight`, layer));
}

/** D(x, sigma) alone. */
export function bioemu2Denoise(w, f, ctx, sInputs, sTrunk, x, sigma, dims) {
  const tape = new Tape();
  const out = denoiseOnTape(tape, w, f, ctx, sInputs, sTrunk, tape.leaf(x, [dims.N, 3]), tape.leaf([sigma], [1]), dims);
  return out.data;
}

/** {out, gx, gs}: D(x, sigma) and the gradient of sum(cotangent * D) w.r.t. x and sigma. */
export function bioemu2DenoiseVjp(w, f, ctx, sInputs, sTrunk, x, sigma, cotangent, dims) {
  const tape = new Tape();
  const xs = tape.leaf(Float32Array.from(x), [dims.N, 3], true);
  const sg = tape.leaf([sigma], [1], true);
  const out = denoiseOnTape(tape, w, f, ctx, sInputs, sTrunk, xs, sg, dims);
  tape.backward(out, cotangent);
  return { out: out.data, gx: xs.grad ?? new Float32Array(x.length), gs: sg.grad ? sg.grad[0] : 0 };
}
