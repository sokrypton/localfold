// ESMFold2's trunk glue for training (cuda/ef2/src/shim.cuh and trunk.cuh are the forward it follows):
//
//   the shim:   single = (sum_k softmax(combine)_k LN(h_k)) @ projection @ downproject + b     (the 37 states of
//               ESM-C, cuda/grad/plm.cuh's Run; a token the tower never saw takes the state of a zero row)
//               lm_z   = LN(W2 gelu(W1 [s_i * s_j | s_i - s_j] + b1) + b2)
//   z_init    = zInit1(s_inputs)[i] + zInit2(s_inputs)[j] + onehot(rel) @ relPos + bonds * tokenBonds (+ lm_z)
//   the recycle, the experimental tier:  z = 0;  loops times:  z = Linear(LN(z)) + z_init;  z = blocks(z)
//   the parcae recycle (ESMFold2, -Fast):  z = z0;  loops times:  inj = lmEncoder(dropout(lm_z)) + z_init;
//               z = decay * z + Linear(LN(inj));  z = blocks(z);   then z = coda(z @ readout)
//   distogram = (z + z^T) @ W + b
//
// The blocks are the AF3 lineage's pairformer without triangle attention or a single track (af3.cuh). Saved for the
// backward: each loop's z before the recycle and the blocks' input, two pairs a loop; a stack's block inputs are
// replayed from those, and each block's forward again before its backward. The shim's pair MLP is never kept:
// its rows are rebuilt a block at a time in the backward, as the forward makes them.
#pragma once
#include "ef2diff.cuh"
#include "af3.cuh"

namespace grad {
namespace ef2 {

// ---------------------------------------------------------------- the language-model shim
struct Shim {
  Param combine;
  LayerNorm norm, pairNorm;
  Linear proj, down, mlp1, mlp2;
  int Cm = 0, P = 0, S = 0;          // the tower's width, the pair's, the number of hidden states mixed
};
inline void forEachParam(Shim& s, int Cm, int P, int S, const std::function<void(const std::string&, Param&, size_t)>& f) {
  s.Cm = Cm; s.P = P; s.S = S;
  auto ln = [&](const std::string& n, LayerNorm& l, int C) { l.C = C; f(n + "/scale", l.g, C); f(n + "/offset", l.b, C); };
  auto lin = [&](const std::string& n, Linear& l, int K, int N, bool bias) {
    l.K = K; l.N = N; l.hasBias = bias; f(n + "/weights", l.w, (size_t)K * N); if (bias) f(n + "/bias", l.b, N);
  };
  f("lm/combine", s.combine, S);
  ln("lm/norm", s.norm, Cm);
  lin("lm/projection", s.proj, Cm, P, false);
  lin("lm/downproject", s.down, P, P, true);
  lin("lm/pair_mlp_1", s.mlp1, 2 * P, P, true);
  lin("lm/pair_mlp_2", s.mlp2, P, P, true);
  ln("lm/pair_norm", s.pairNorm, P);
}
struct ShimSave {
  std::vector<const real*> states;
  std::vector<double> mix;
  real *xmix = nullptr, *acc = nullptr, *single = nullptr, *z1 = nullptr, *zero = nullptr, *s = nullptr;
  real* zeroMask = nullptr;          // [T]: 1 where the token has no tower row
  int* rowTok = nullptr;             // [R]: the token a row lands on, or -1
  const int* tokenToRow = nullptr;
  int R = 0, T = 0;
};
__global__ void scatterTokensK(const real* rows, const int* tokenToRow, const real* zero, real* out, int T, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)T * C) return;
  int token = (int)(t / C), c = (int)(t % C), r = tokenToRow[token];
  out[t] = r < 0 ? zero[c] : rows[(size_t)r * C + c];
}
__global__ void gatherTokensK(const real* ds, const int* rowTok, real* out, int R, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)R * C) return;
  int r = (int)(t / C), c = (int)(t % C), token = rowTok[r];
  out[t] = token < 0 ? 0 : ds[(size_t)token * C + c];
}
// [b*T, 2C] = [s_i * s_j | s_i - s_j] for rows i0..i0+b
__global__ void pairJoinK(const real* s, real* out, int T, int i0, int b, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)b * T * C) return;
  int c = (int)(t % C); size_t ij = t / C; int i = i0 + (int)(ij / T), j = (int)(ij % T);
  real a = s[(size_t)i * C + c], v = s[(size_t)j * C + c];
  out[ij * 2 * C + c] = a * v;
  out[ij * 2 * C + C + c] = a - v;
}
// the join's backward, without atomics: the rows' half (one thread per (i, c) over j) and the columns' (per (j, c) over i)
__global__ void joinBwdRowsK(const real* dj, const real* s, real* ds, int T, int i0, int b, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)b * C) return;
  int il = (int)(t / C), c = (int)(t % C);
  real acc = 0;
  for (int j = 0; j < T; ++j) {
    size_t ij = (size_t)il * T + j;
    acc += dj[ij * 2 * C + c] * s[(size_t)j * C + c] + dj[ij * 2 * C + C + c];
  }
  ds[(size_t)(i0 + il) * C + c] += acc;
}
__global__ void joinBwdColsK(const real* dj, const real* s, real* ds, int T, int i0, int b, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)T * C) return;
  int j = (int)(t / C), c = (int)(t % C);
  real acc = 0;
  for (int il = 0; il < b; ++il) {
    size_t ij = (size_t)il * T + j;
    acc += dj[ij * 2 * C + c] * s[(size_t)(i0 + il) * C + c] - dj[ij * 2 * C + C + c];
  }
  ds[t] += acc;
}
// the pair MLP's rows a block at a time: 32 MB of join, cuda/ef2's lmPairBlock
inline int shimPairBlock(int T, int P) { return std::max(1, std::min(T, (int)(((size_t)32 << 20) / ((size_t)T * 2 * P * sizeof(real))))); }
struct ShimRows { real *join, *pre, *h, *o; LnSave ln; };
inline ShimRows shimPairRows(Shim& m, const real* s, int T, int i0, int b, real* out) {
  const int P = m.P; size_t cells = (size_t)b * T;
  ShimRows k;
  k.join = ARENA.get(cells * 2 * P);
  pairJoinK<<<blocks(cells * P), 256, 0, STREAM>>>(s, k.join, T, i0, b, P);
  k.pre = ARENA.get(cells * P); m.mlp1.forward(k.join, k.pre, cells);
  k.h = ARENA.get(cells * P); geluK<<<blocks(cells * P), 256, 0, STREAM>>>(k.h, k.pre, cells * P);
  k.o = ARENA.get(cells * P); m.mlp2.forward(k.h, k.o, cells);
  k.ln = lnForward(m.pairNorm, k.o, out, cells);
  return k;
}
// states [S] of [R, Cm] (plm Run's xs[0..L-1] and out), tokenToRow [T] (device; -1: no row) -> lmZ [T, T, P] (null:
// only the token states, s.s)
inline ShimSave shimForward(Shim& m, const std::vector<const real*>& states, int R, const int* tokenToRow, int T, real* lmZ) {
  const int Cm = m.Cm, P = m.P, S = m.S;
  ShimSave s; s.states = states; s.R = R; s.T = T; s.tokenToRow = tokenToRow;
  if ((int)states.size() != S) { fprintf(stderr, "the shim mixes %d states, %zu given\n", S, states.size()); exit(1); }
  {
    auto c = host(m.combine.w, S);
    double mx = -1e300, sum = 0; s.mix.resize(S);
    for (int k = 0; k < S; ++k) mx = std::max(mx, (double)c[k]);
    for (int k = 0; k < S; ++k) sum += s.mix[k] = std::exp((double)c[k] - mx);
    for (auto& v : s.mix) v /= sum;
  }
  std::vector<int> t2r(T); GK(cudaMemcpy(t2r.data(), tokenToRow, T * sizeof(int), cudaMemcpyDeviceToHost));
  std::vector<int> r2t(std::max(R, 1), -1); std::vector<real> zm(T, 0);
  for (int t = 0; t < T; ++t) {
    if (t2r[t] < 0) { zm[t] = 1; continue; }
    if (t2r[t] >= R || r2t[t2r[t]] >= 0) { fprintf(stderr, "token %d: row %d is not one token's\n", t, t2r[t]); exit(1); }
    r2t[t2r[t]] = t;
  }
  s.zeroMask = ARENA.get(T); GK(cudaMemcpyAsync(s.zeroMask, zm.data(), T * sizeof(real), cudaMemcpyHostToDevice, STREAM));
  s.rowTok = (int*)ARENA.get(std::max(R, 1));
  GK(cudaMemcpyAsync(s.rowTok, r2t.data(), r2t.size() * sizeof(int), cudaMemcpyHostToDevice, STREAM));
  s.single = ARENA.get((size_t)std::max(R, 1) * P);
  if (R > 0) {
    s.xmix = ARENA.zeros((size_t)R * Cm);
    {
      Scratch sc;
      real* xn = ARENA.get((size_t)R * Cm);
      for (int k = 0; k < S; ++k) {
        lnForward(m.norm, states[k], xn, R);
        axpyR<<<blocks((size_t)R * Cm), 256, 0, STREAM>>>(s.xmix, xn, (real)s.mix[k], (size_t)R * Cm);
      }
    }
    s.acc = ARENA.get((size_t)R * P); m.proj.forward(s.xmix, s.acc, R);
    m.down.forward(s.acc, s.single, R);
  }
  // a row-less token's state is zero: LN(0) is the offset and the mix sums to one
  s.z1 = ARENA.get(P); m.proj.forward(m.norm.b.w, s.z1, 1);
  s.zero = ARENA.get(P); m.down.forward(s.z1, s.zero, 1);
  s.s = ARENA.get((size_t)T * P);
  scatterTokensK<<<blocks((size_t)T * P), 256, 0, STREAM>>>(s.single, tokenToRow, s.zero, s.s, T, P);
  if (!lmZ) return s;
  const int bi = shimPairBlock(T, P);
  for (int i0 = 0; i0 < T; i0 += bi) {
    Scratch sc;
    shimPairRows(m, s.s, T, i0, std::min(bi, T - i0), lmZ + (size_t)i0 * T * P);
  }
  return s;
}
// dlmZ [T, T, P] (null: none), dTok [T, P] (an extra cotangent on the token states; null: none) -> every parameter;
// dstates[k] [R, Cm] written (a null entry: that state's gradient not wanted)
inline void shimBackward(Shim& m, const ShimSave& s, const real* dlmZ, const real* dTok, const std::vector<real*>& dstates) {
  Scratch scratch;
  const int Cm = m.Cm, P = m.P, S = m.S, T = s.T, R = s.R;
  real* ds = ARENA.zeros((size_t)T * P);
  if (dTok) add(ds, dTok, (size_t)T * P);
  if (dlmZ) {
    const int bi = shimPairBlock(T, P);
    for (int i0 = 0; i0 < T; i0 += bi) {
      Scratch sc;
      const int b = std::min(bi, T - i0); size_t cells = (size_t)b * T;
      ShimRows k = shimPairRows(m, s.s, T, i0, b, ARENA.get(cells * P));
      real* dO = ARENA.get(cells * P);
      lnBackward(m.pairNorm, k.ln, dlmZ + (size_t)i0 * T * P, dO, cells, false);
      real* dh = ARENA.get(cells * P); m.mlp2.backward(k.h, dO, dh, cells);
      geluBwdK<<<blocks(cells * P), 256, 0, STREAM>>>(dh, k.pre, cells * P);
      real* dj = ARENA.get(cells * 2 * P); m.mlp1.backward(k.join, dh, dj, cells);
      joinBwdRowsK<<<blocks((size_t)b * P), 256, 0, STREAM>>>(dj, s.s, ds, T, i0, b, P);
      joinBwdColsK<<<blocks((size_t)T * P), 256, 0, STREAM>>>(dj, s.s, ds, T, i0, b, P);
    }
  }
  // the zero row: its offset and both projections
  real* dzero = ARENA.zeros(P); mmAT(s.zeroMask, ds, dzero, T, 1, P, 1);
  real* dz1 = ARENA.get(P); m.down.backward(s.z1, dzero, dz1, 1);
  real* doff = ARENA.get(Cm); m.proj.backward(m.norm.b.w, dz1, doff, 1);
  add(m.norm.b.g, doff, Cm);
  if (R == 0) return;
  real* dsingle = ARENA.get((size_t)R * P);
  gatherTokensK<<<blocks((size_t)R * P), 256, 0, STREAM>>>(ds, s.rowTok, dsingle, R, P);
  real* dacc = ARENA.get((size_t)R * P); m.down.backward(s.acc, dsingle, dacc, R);
  real* dxmix = ARENA.get((size_t)R * Cm); m.proj.backward(s.xmix, dacc, dxmix, R);
  std::vector<double> dmix(S);
  real *xn = ARENA.get((size_t)R * Cm), *dxn = ARENA.get((size_t)R * Cm);
  for (int k = 0; k < S; ++k) {
    Scratch sc;
    LnSave ls = lnForward(m.norm, s.states[k], xn, R);
    dmix[k] = dot(xn, dxmix, (size_t)R * Cm);
    GK(cudaMemsetAsync(dxn, 0, (size_t)R * Cm * sizeof(real), STREAM));
    axpyR<<<blocks((size_t)R * Cm), 256, 0, STREAM>>>(dxn, dxmix, (real)s.mix[k], (size_t)R * Cm);
    real* out = dstates[k];
    if (!out) out = ARENA.get((size_t)R * Cm);          // (the norm's weights still want its backward)
    lnBackward(m.norm, ls, dxn, out, R, false);
  }
  // the mix's softmax
  double dot_ = 0; for (int k = 0; k < S; ++k) dot_ += s.mix[k] * dmix[k];
  auto g = host(m.combine.g, S);
  for (int k = 0; k < S; ++k) g[k] += (real)(s.mix[k] * (dmix[k] - dot_));
  GK(cudaMemcpy(m.combine.g, g.data(), S * sizeof(real), cudaMemcpyHostToDevice));
}

// ---------------------------------------------------------------- z_init
struct ZInit {
  Linear z1, z2, rel;
  Param bond;
  int Si = 0, C = 0;
};
inline void forEachParam(ZInit& z, int Si, int C, const std::function<void(const std::string&, Param&, size_t)>& f) {
  z.Si = Si; z.C = C;
  auto lin = [&](const std::string& n, Linear& l, int K, int N) { l.K = K; l.N = N; l.hasBias = false; f(n, l.w, (size_t)K * N); };
  lin("featuriser/zInit1", z.z1, Si, C);
  lin("featuriser/zInit2", z.z2, Si, C);
  lin("featuriser/relPos", z.rel, 139, C);
  f("featuriser/tokenBonds", z.bond, C);
}
struct Tokens { const int *ri, *asym, *sym, *ent, *ti; const real* bonds; int T; };   // bonds dense [T, T] (null: none)
__global__ void outerSumK(const real* rows, const real* cols, const real* bonds, const real* wb, const real* lm, real* z,
                          int T, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)T * T * C) return;
  int c = (int)(t % C); size_t ij = t / C; int i = (int)(ij / T), j = (int)(ij % T);
  real v = rows[(size_t)i * C + c] + cols[(size_t)j * C + c];
  if (bonds) v += bonds[ij] * wb[c];
  if (lm) v += lm[t];
  z[t] = v;
}
// drows[i] = sum_j dz[i, j], dcols[j] = sum_i dz[i, j]
__global__ void pairRowSumK(const real* dz, real* out, int T, int C, bool byRow) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)T * C) return;
  int a = (int)(t / C), c = (int)(t % C);
  real acc = 0;
  for (int k = 0; k < T; ++k) acc += dz[((byRow ? (size_t)a * T + k : (size_t)k * T + a)) * C + c];
  out[t] = acc;
}
inline real* relOneHot(const Tokens& t) {
  size_t P = (size_t)t.T * t.T;
  real* oh = ARENA.get(P * 139);
  relOneHotK<<<blocks(P * 139), 256, 0, STREAM>>>(t.ri, t.asym, t.sym, t.ent, t.ti, oh, t.T);
  return oh;
}
// sInputs [T, Si], lmZ (null: the parcae models', which inject it in the recycle) -> z [T, T, C]
inline void zinitForward(ZInit& m, const Tokens& t, const real* sIn, const real* lmZ, real* z) {
  Scratch sc;
  const int T = t.T, C = m.C; size_t P = (size_t)T * T;
  real *rows = ARENA.get((size_t)T * C), *cols = ARENA.get((size_t)T * C);
  m.z1.forward(sIn, rows, T); m.z2.forward(sIn, cols, T);
  outerSumK<<<blocks(P * C), 256, 0, STREAM>>>(rows, cols, t.bonds, m.bond.w, lmZ, z, T, C);
  m.rel.forward(relOneHot(t), z, P, 1);
}
// dz -> every parameter; dsIn, dlm added when not null
inline void zinitBackward(ZInit& m, const Tokens& t, const real* sIn, const real* dz, real* dsIn, real* dlm) {
  Scratch sc;
  const int T = t.T, C = m.C; size_t P = (size_t)T * T;
  real *dr = ARENA.get((size_t)T * C), *dc = ARENA.get((size_t)T * C);
  pairRowSumK<<<blocks((size_t)T * C), 256, 0, STREAM>>>(dz, dr, T, C, true);
  pairRowSumK<<<blocks((size_t)T * C), 256, 0, STREAM>>>(dz, dc, T, C, false);
  m.z1.backward(sIn, dr, dsIn, T, true);
  m.z2.backward(sIn, dc, dsIn, T, true);
  m.rel.backward(relOneHot(t), dz, nullptr, P);
  if (t.bonds) mmAT(t.bonds, dz, m.bond.g, (int)P, 1, C, 1);
  if (dlm) add(dlm, dz, P * C);
}

// ---------------------------------------------------------------- the recycle and the trunk
struct Trunk {
  std::vector<af3::Pairformer> blocks, lmEnc, coda;
  LayerNorm recNorm;
  Linear recProj, readout;
  Param decay;
  bool parcae = false;
  int C = 0;
};
inline void forEachParam(Trunk& t, int C, int pairFactor, int nBlocks, int nLm, int nCoda, bool parcae,
                         const std::function<void(const std::string&, Param&, size_t)>& f) {
  t.C = C; t.parcae = parcae;
  auto stack = [&](std::vector<af3::Pairformer>& s, const std::string& pre, int n) {
    s.resize(n);
    for (int k = 0; k < n; ++k)
      af3::forEachParam(s[k], pre + "/" + std::to_string(k), C, 0, 0, 0, 0, 0, pairFactor, 0, false, false, f, false, false, "/");
  };
  stack(t.blocks, "blocks", nBlocks);
  t.recNorm.C = C; f("recycle/norm/scale", t.recNorm.g, C); f("recycle/norm/offset", t.recNorm.b, C);
  t.recProj.K = t.recProj.N = C; t.recProj.hasBias = false; f("recycle/projection", t.recProj.w, (size_t)C * C);
  if (!parcae) return;
  stack(t.lmEnc, "lmEncoder/blocks", nLm);
  stack(t.coda, "coda/blocks", nCoda);
  f("recycle/decay", t.decay, C);
  t.readout.K = t.readout.N = C; t.readout.hasBias = false; f("readout", t.readout.w, (size_t)C * C);
}
struct TrunkRun {
  std::vector<real*> zPrev, zIn;       // [loops]: z before each loop's recycle, the blocks' input
  real *zLast = nullptr, *zRead = nullptr, *mask = nullptr;
  int n = 0, loops = 0;
  real p = 0;
  uint64_t seed = 0;
};
// cuda/ef2's dropout draw (trunk.cuh's uniform01), so a seed drops what the port's drops
__device__ __forceinline__ uint64_t dropMix64(uint64_t z) {
  z += 0x9E3779B97F4A7C15ull; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
__device__ __forceinline__ bool dropped(uint64_t seed, uint64_t stream, uint64_t i, float p) {
  return (float)(dropMix64(seed ^ dropMix64(stream * 0x100000001B3ull + i)) >> 40) * (1.f / 16777216.f) < p;
}
// out = dropout(in) (forward), or out += dropout'(in) (backward: in is the output's gradient)
__global__ void dropoutK(const real* in, real* out, size_t n, float p, uint64_t seed, uint64_t stream, bool accumulate) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  real v = dropped(seed, stream, t, p) ? (real)0 : in[t] / (real)(1.f - p);
  if (accumulate) out[t] += v; else out[t] = v;
}
__global__ void decayK(real* z, const real* zPrev, const real* y, const real* a, size_t rows, int C) {   // z = a zPrev + y
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * C) z[t] = a[t % C] * zPrev[t] + y[t];
}
__global__ void fillR(real* x, real v, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) x[t] = v;
}
// a stack of pair-only blocks over z in place
inline void stackForward(std::vector<af3::Pairformer>& s, real* z, const real* mask, int n) {
  for (auto& b : s) {
    Scratch sc;
    af3::pairformerForward(b, {}, z, nullptr, mask, mask, nullptr, nullptr, z, nullptr, n);
  }
}
// in: the stack's input; dz: its output's gradient, replaced by its input's
inline void stackBackward(std::vector<af3::Pairformer>& s, const real* in, real* dz, const real* mask, int n, int C) {
  if (s.empty()) return;
  Scratch sc;
  size_t P = (size_t)n * n * C;
  std::vector<real*> ins(s.size());
  ins[0] = const_cast<real*>(in);
  for (size_t k = 1; k < s.size(); ++k) {
    ins[k] = ARENA.get(P);
    GK(cudaMemcpyAsync(ins[k], ins[k - 1], P * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
    Scratch b;
    af3::pairformerForward(s[k - 1], {}, ins[k], nullptr, mask, mask, nullptr, nullptr, ins[k], nullptr, n);
  }
  for (size_t k = s.size(); k-- > 0;) {
    Scratch b;
    real* po = ARENA.get(P);
    auto bs = af3::pairformerForward(s[k], {}, ins[k], nullptr, mask, mask, nullptr, nullptr, po, nullptr, n);
    af3::pairformerBackward(s[k], {}, bs, mask, mask, nullptr, dz, nullptr, dz, nullptr, nullptr);
  }
}
// the parcae injection: inj = lmEncoder(dropout(lmZ)) + zi, and (kept when drop is not null) its input
inline void injection(Trunk& t, const TrunkRun& r, const real* zi, const real* lmZ, int loop, real* inj, real* drop) {
  size_t P = (size_t)r.n * r.n * t.C;
  dropoutK<<<blocks(P), 256, 0, STREAM>>>(lmZ, inj, P, (float)r.p, r.seed, 1000 + loop, false);
  if (drop) GK(cudaMemcpyAsync(drop, inj, P * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  stackForward(t.lmEnc, inj, r.mask, r.n);
  add(inj, zi, P);
}
// zi [n, n, C] (z_init), lmZ (the parcae models'; null for the experimental tier), z0 (the parcae models' initial
// state; null: zeros), p and seed the language model's pair dropout -> out [n, n, C], the trunk's pair
inline void trunkForward(Trunk& t, TrunkRun& r, const real* zi, const real* lmZ, const real* z0, int n, int loops,
                         real p, uint64_t seed, real* out) {
  const int C = t.C; size_t P = (size_t)n * n * C;
  r.n = n; r.loops = loops; r.p = p; r.seed = seed;
  if (t.parcae && !lmZ) { fprintf(stderr, "the parcae recycle needs the language model's pair\n"); exit(1); }
  r.mask = ARENA.get((size_t)n * n); fillR<<<blocks((size_t)n * n), 256, 0, STREAM>>>(r.mask, 1, (size_t)n * n);
  r.zPrev.assign(loops, nullptr); r.zIn.assign(loops, nullptr);
  r.zPrev[0] = z0 ? cloneR(z0, P) : ARENA.zeros(P);
  for (int l = 0; l < loops; ++l) {
    r.zIn[l] = ARENA.get(P);
    {
      Scratch sc;
      real *xn = ARENA.get(P);
      if (t.parcae) {
        real* inj = ARENA.get(P);
        injection(t, r, zi, lmZ, l, inj, nullptr);
        lnForward(t.recNorm, inj, xn, (size_t)n * n);
        real* y = ARENA.get(P); t.recProj.forward(xn, y, (size_t)n * n);
        decayK<<<blocks(P), 256, 0, STREAM>>>(r.zIn[l], r.zPrev[l], y, t.decay.w, (size_t)n * n, C);
      } else {
        lnForward(t.recNorm, r.zPrev[l], xn, (size_t)n * n);
        t.recProj.forward(xn, r.zIn[l], (size_t)n * n);
        add(r.zIn[l], zi, P);
      }
    }
    real* zo = ARENA.get(P);
    GK(cudaMemcpyAsync(zo, r.zIn[l], P * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
    stackForward(t.blocks, zo, r.mask, n);
    if (l + 1 < loops) r.zPrev[l + 1] = zo; else r.zLast = zo;
  }
  if (!t.parcae) { GK(cudaMemcpyAsync(out, r.zLast, P * sizeof(real), cudaMemcpyDeviceToDevice, STREAM)); return; }
  r.zRead = ARENA.get(P); t.readout.forward(r.zLast, r.zRead, (size_t)n * n);
  GK(cudaMemcpyAsync(out, r.zRead, P * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  stackForward(t.coda, out, r.mask, n);
}
// sum over rows of a * b, per channel, into g
__global__ void prodColSumK(const real* a, const real* b, real* g, size_t rows, int C) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  real acc = 0;
  for (size_t r = 0; r < rows; ++r) acc += a[r * C + c] * b[r * C + c];
  g[c] += acc;
}
__global__ void scaleRowsK(real* x, const real* a, size_t rows, int C) {   // x[r, c] *= a[c]
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * C) x[t] *= a[t % C];
}
// dout -> every parameter; dzi added; dlmZ and dz0 added when not null
inline void trunkBackward(Trunk& t, const TrunkRun& r, const real* zi, const real* lmZ, const real* dout, real* dzi,
                          real* dlmZ, real* dz0) {
  Scratch scratch;
  const int n = r.n, C = t.C; size_t rows = (size_t)n * n, P = rows * C;
  real* dz = cloneR(dout, P);
  if (t.parcae) {
    stackBackward(t.coda, r.zRead, dz, r.mask, n, C);
    real* d = ARENA.get(P); t.readout.backward(r.zLast, dz, d, rows); dz = d;
  }
  for (int l = r.loops; l-- > 0;) {
    stackBackward(t.blocks, r.zIn[l], dz, r.mask, n, C);       // dz: the blocks' input's
    Scratch sc;
    real *xn = ARENA.get(P), *dxn = ARENA.get(P);
    if (t.parcae) {
      real *inj = ARENA.get(P), *drop = ARENA.get(P);
      injection(t, r, zi, lmZ, l, inj, drop);
      LnSave ls = lnForward(t.recNorm, inj, xn, rows);
      prodColSumK<<<blocks(C, 128), 128, 0, STREAM>>>(r.zPrev[l], dz, t.decay.g, rows, C);
      t.recProj.backward(xn, dz, dxn, rows);
      real* dinj = ARENA.get(P);
      lnBackward(t.recNorm, ls, dxn, dinj, rows, false);
      add(dzi, dinj, P);
      stackBackward(t.lmEnc, drop, dinj, r.mask, n, C);
      if (dlmZ) dropoutK<<<blocks(P), 256, 0, STREAM>>>(dinj, dlmZ, P, (float)r.p, r.seed, 1000 + l, true);
      scaleRowsK<<<blocks(P), 256, 0, STREAM>>>(dz, t.decay.w, rows, C);
    } else {
      add(dzi, dz, P);
      LnSave ls = lnForward(t.recNorm, r.zPrev[l], xn, rows);
      t.recProj.backward(xn, dz, dxn, rows);
      lnBackward(t.recNorm, ls, dxn, dz, rows, false);
    }
  }
  if (dz0) add(dz0, dz, P);
}

// ---------------------------------------------------------------- the distogram
struct Distogram { Linear lin; int C = 0, B = 0; };
inline void forEachParam(Distogram& d, int C, int B, const std::function<void(const std::string&, Param&, size_t)>& f) {
  d.C = C; d.B = B; d.lin.K = C; d.lin.N = B; d.lin.hasBias = true;
  f("distogram/weights", d.lin.w, (size_t)C * B); f("distogram/bias", d.lin.b, B);
}
__global__ void symK(const real* z, real* out, int n, int C, bool accumulate) {    // out (+)= z + z^T
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)n * n * C) return;
  int c = (int)(t % C); size_t ij = t / C; int i = (int)(ij / n), j = (int)(ij % n);
  real v = z[t] + z[((size_t)j * n + i) * C + c];
  if (accumulate) out[t] += v; else out[t] = v;
}
inline void distogramForward(Distogram& d, const real* z, int n, real* logits) {
  Scratch sc;
  size_t rows = (size_t)n * n;
  real* zs = ARENA.get(rows * d.C);
  symK<<<blocks(rows * d.C), 256, 0, STREAM>>>(z, zs, n, d.C, false);
  d.lin.forward(zs, logits, rows);
}
inline void distogramBackward(Distogram& d, const real* z, int n, const real* dlogits, real* dz) {   // dz added
  Scratch sc;
  size_t rows = (size_t)n * n;
  real* zs = ARENA.get(rows * d.C);
  symK<<<blocks(rows * d.C), 256, 0, STREAM>>>(z, zs, n, d.C, false);
  real* dzs = ARENA.get(rows * d.C);
  d.lin.backward(zs, dlogits, dzs, rows);
  symK<<<blocks(rows * d.C), 256, 0, STREAM>>>(dzs, dz, n, d.C, true);
}

}  // namespace ef2
}  // namespace grad
