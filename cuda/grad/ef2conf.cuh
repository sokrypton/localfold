// ESMFold2's confidence head for training (Synthyra's, which biohub ships without; cuda/ef2/src/confidence.cuh is the
// forward it follows):
//   s = LN(s_inputs);  z = LN(z_trunk) + rel_pos + bonds + sToZ(s)_i + sToZ^T(s)_j + prodOut(in1(s)_i * in2(s)_j)
//                        + distance_embed[#(|x_rep_i - x_rep_j| > boundaries)]
//   z = z + blocks(z)          (four pair-only blocks of the trunk's form, their own weights)
//   single = (softmax_j(z @ attn) z) @ out;  pLDDT logits per atom = LN(single[token]) @ plddtWeight[slot]
//   PAE logits = (LN)(z) @ pae
// rel_pos and the bond weight are z_init's (ef2trunk.cuh's ZInit): one parameter, both gradients. The coordinates are
// an input (the sampler's; the distance bucket takes no gradient).
#pragma once
#include "ef2trunk.cuh"
#include "af3conf.cuh"

namespace grad {
namespace ef2 {

struct Ef2Confidence {
  LayerNorm sInLn, zLn, paeLn, plddtLn;
  Linear sToZ, sToZT, in1, in2, prodOut, poolAttn, poolOut, pae;
  Param distEmb, plddtW;           // [edges + 1, C], [slots, Cs, bins]
  std::vector<real> edges;         // the distance boundaries (host)
  real* dEdges = nullptr;
  std::vector<af3::Pairformer> blocks;
  ZInit* zi = nullptr;             // its rel_pos and bond weight
  bool hasPaeLn = false;
  int Si = 0, C = 0, Cs = 0, slots = 0, bins = 0, pb = 0;
};
inline void forEachParam(Ef2Confidence& h, int nb, int pairFactor, const std::function<bool(const std::string&)>& has,
                         const std::function<void(const std::string&, Param&, size_t)>& f) {
  const std::string P = "confidence/";
  auto ln = [&](const std::string& n, LayerNorm& l, int C) { l.C = C; f(P + n + "/scale", l.g, C); f(P + n + "/offset", l.b, C); };
  auto lin = [&](const std::string& n, Linear& l, int K, int N) { l.K = K; l.N = N; l.hasBias = false; f(P + n, l.w, (size_t)K * N); };
  ln("sInputsNorm", h.sInLn, h.Si); ln("zNorm", h.zLn, h.C);
  lin("sToZ", h.sToZ, h.Si, h.C); lin("sToZTranspose", h.sToZT, h.Si, h.C);
  lin("sToZProdIn1", h.in1, h.Si, h.C); lin("sToZProdIn2", h.in2, h.Si, h.C); lin("sToZProdOut", h.prodOut, h.C, h.C);
  f(P + "distanceEmbedding", h.distEmb, (h.edges.size() + 1) * h.C);
  h.blocks.resize(nb);
  for (int k = 0; k < nb; ++k)
    af3::forEachParam(h.blocks[k], P + "blocks/" + std::to_string(k), h.C, 0, 0, 0, 0, 0, pairFactor, 0, false, false, f, false, false, "/");
  lin("poolingAttention", h.poolAttn, h.C, 1); lin("poolingOutput", h.poolOut, h.C, h.Cs);
  ln("plddtNorm", h.plddtLn, h.Cs);
  f(P + "plddtWeight", h.plddtW, (size_t)h.slots * h.Cs * h.bins);
  h.hasPaeLn = has(P + "paeNorm/scale"); if (h.hasPaeLn) ln("paeNorm", h.paeLn, h.C);
  lin("pae", h.pae, h.C, h.pb);
}
// z[i, j] += r[i] + c[j] + table[bucket(i, j)]
__global__ void confPairAddK(real* z, const real* r, const real* c, const int* bucket, const real* table, int T, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)T * T * C) return;
  int ch = t % C; size_t ij = t / C; int i = ij / T, j = ij % T;
  z[t] += r[(size_t)i * C + ch] + c[(size_t)j * C + ch] + table[(size_t)bucket[ij] * C + ch];
}
__global__ void distBucketK(const real* x, const int* rep, const real* edges, int nEdges, int* bucket, int T) {
  size_t ij = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (ij >= (size_t)T * T) return;
  int i = ij / T, j = ij % T;
  const real *a = x + (size_t)rep[i] * 3, *b = x + (size_t)rep[j] * 3;
  float dx = (float)a[0] - (float)b[0], dy = (float)a[1] - (float)b[1], dz = (float)a[2] - (float)b[2];
  float d = sqrtf(dx * dx + dy * dy + dz * dz);
  int k = 0;
  for (int e = 0; e < nEdges; ++e) k += d > (float)edges[e];
  bucket[ij] = k;
}
__global__ void bucketBwdK(const real* d, const int* bucket, real* dtable, size_t pairs, int C, int rows) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)rows * C) return;
  int k = t / C, ch = t % C;
  real acc = 0;
  for (size_t ij = 0; ij < pairs; ++ij) if (bucket[ij] == k) acc += d[ij * C + ch];
  dtable[t] += acc;
}
// pooled[i] = sum_j softmax_j(score[i, j]) z[i, j]; w kept [T, T]
__global__ void rowPoolFwdK(const real* z, const real* score, real* w, real* pooled, int T, int C) {
  int i = blockIdx.x;
  __shared__ real mx, sum;
  if (threadIdx.x == 0) {
    real m = -1e300; for (int j = 0; j < T; ++j) m = fmax(m, score[(size_t)i * T + j]);
    real s = 0; for (int j = 0; j < T; ++j) s += exp(score[(size_t)i * T + j] - m);
    mx = m; sum = s;
  }
  __syncthreads();
  for (int j = threadIdx.x; j < T; j += blockDim.x) w[(size_t)i * T + j] = exp(score[(size_t)i * T + j] - mx) / sum;
  __syncthreads();
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    real acc = 0;
    for (int j = 0; j < T; ++j) acc += w[(size_t)i * T + j] * z[((size_t)i * T + j) * C + c];
    pooled[(size_t)i * C + c] = acc;
  }
}
// dz[i, j] += w dpooled[i]; dscore[i, j] = w (dw - sum_k w dw), dw = dpooled[i] . z[i, j]   (a block a row)
__global__ void rowPoolBwdK(const real* z, const real* w, const real* dp, real* dz, real* dscore, int T, int C) {
  int i = blockIdx.x;
  extern __shared__ char shm[];
  real* dw = reinterpret_cast<real*>(shm);
  for (int j = threadIdx.x; j < T; j += blockDim.x) {
    real a = 0;
    for (int c = 0; c < C; ++c) a += dp[(size_t)i * C + c] * z[((size_t)i * T + j) * C + c];
    dw[j] = a;
  }
  __syncthreads();
  __shared__ real s;
  if (threadIdx.x == 0) { real a = 0; for (int j = 0; j < T; ++j) a += w[(size_t)i * T + j] * dw[j]; s = a; }
  __syncthreads();
  for (int j = threadIdx.x; j < T; j += blockDim.x) dscore[(size_t)i * T + j] = w[(size_t)i * T + j] * (dw[j] - s);
  for (size_t t = threadIdx.x; t < (size_t)T * C; t += blockDim.x) {
    int j = t / C, c = t % C;
    dz[((size_t)i * T + j) * C + c] += w[(size_t)i * T + j] * dp[(size_t)i * C + c];
  }
}
// logits[a, b] = sum_c sn[token(a), c] W[slot(a), c, b]
__global__ void plddtLogitsK(const real* sn, const int* a2t, const int* slot, const real* W, real* out, int A, int Cs, int bins) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)A * bins) return;
  int a = t / bins, b = t % bins;
  const real* s = sn + (size_t)a2t[a] * Cs; const real* w = W + (size_t)slot[a] * Cs * bins + b;
  real acc = 0;
  for (int c = 0; c < Cs; ++c) acc += s[c] * w[(size_t)c * bins];
  out[t] = acc;
}
__global__ void plddtBwdWK(const real* sn, const int* a2t, const int* slot, const real* d, real* dW, int A, int Cs, int bins, int slots) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)slots * Cs * bins) return;
  int b = t % bins; size_t r = t / bins; int c = r % Cs, sl = r / Cs;
  real acc = 0;
  for (int a = 0; a < A; ++a) if (slot[a] == sl) acc += sn[(size_t)a2t[a] * Cs + c] * d[(size_t)a * bins + b];
  dW[t] += acc;
}
__global__ void plddtBwdSK(const real* W, const int* a2t, const int* slot, const real* d, real* dsn, int A, int Cs, int bins) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)A * Cs) return;
  int a = t / Cs, c = t % Cs;
  const real* w = W + ((size_t)slot[a] * Cs + c) * bins;
  real acc = 0;
  for (int b = 0; b < bins; ++b) acc += w[b] * d[(size_t)a * bins + b];
  atomicAdd(&dsn[(size_t)a2t[a] * Cs + c], acc);
}
struct ConfInputs { const Tokens* tok; const real* x; const int *rep, *atomToToken, *slot; int A; };
struct ConfLogits { real *plddt, *pae; };          // [A, bins], [T T, pb]
struct ConfSave {
  real *s, *r, *c, *l, *rr, *z0, *z, *score, *w, *pooled, *single, *sn, *pn;
  int* bucket;
  LnSave lsIn, lz, lpl, lpae;
  std::vector<real*> ins;
  int T;
};
inline ConfSave confForward(Ef2Confidence& h, const ConfInputs& in, const real* sIn, const real* zTrunk, const real* mask, ConfLogits& o) {
  ConfSave s{}; const int T = in.tok->T, C = h.C; s.T = T;
  const size_t P = (size_t)T * T, PC = P * C;
  s.s = ARENA.get((size_t)T * h.Si); s.lsIn = lnForward(h.sInLn, sIn, s.s, T);
  s.r = ARENA.get((size_t)T * C); h.sToZ.forward(s.s, s.r, T);
  s.c = ARENA.get((size_t)T * C); h.sToZT.forward(s.s, s.c, T);
  s.l = ARENA.get((size_t)T * C); h.in1.forward(s.s, s.l, T);
  s.rr = ARENA.get((size_t)T * C); h.in2.forward(s.s, s.rr, T);
  s.bucket = (int*)ARENA.get(P);
  distBucketK<<<blocks(P), 256, 0, STREAM>>>(in.x, in.rep, h.dEdges, (int)h.edges.size(), s.bucket, T);
  s.z0 = ARENA.get(PC); s.lz = lnForward(h.zLn, zTrunk, s.z0, P);
  confPairAddK<<<blocks(PC), 256, 0, STREAM>>>(s.z0, s.r, s.c, s.bucket, h.distEmb.w, T, C);
  {
    Scratch sc;
    h.zi->rel.forward(relOneHot(*in.tok), s.z0, P, 1);
    if (in.tok->bonds) mm(in.tok->bonds, h.zi->bond.w, s.z0, (int)P, 1, C, 1);
    real* prod = ARENA.get(PC);
    af3::outerRowsK<<<blocks(PC), 256, 0, STREAM>>>(s.l, s.rr, prod, T, C);
    h.prodOut.forward(prod, s.z0, P, 1);
  }
  s.z = cloneR(s.z0, PC);
  stackForward(h.blocks, s.z, mask, T);
  add(s.z, s.z0, PC);
  s.score = ARENA.get(P); h.poolAttn.forward(s.z, s.score, P);
  s.w = ARENA.get(P); s.pooled = ARENA.get((size_t)T * C);
  rowPoolFwdK<<<T, 256, 0, STREAM>>>(s.z, s.score, s.w, s.pooled, T, C);
  s.single = ARENA.get((size_t)T * h.Cs); h.poolOut.forward(s.pooled, s.single, T);
  s.sn = ARENA.get((size_t)T * h.Cs); s.lpl = lnForward(h.plddtLn, s.single, s.sn, T);
  o.plddt = ARENA.get((size_t)in.A * h.bins);
  plddtLogitsK<<<blocks((size_t)in.A * h.bins), 256, 0, STREAM>>>(s.sn, in.atomToToken, in.slot, h.plddtW.w, o.plddt, in.A, h.Cs, h.bins);
  s.pn = s.z;
  if (h.hasPaeLn) { s.pn = ARENA.get(PC); s.lpae = lnForward(h.paeLn, s.z, s.pn, P); }
  o.pae = ARENA.get(P * h.pb); h.pae.forward(s.pn, o.pae, P);
  return s;
}
// the logits' cotangents -> every parameter (rel_pos and the bond weight into z_init's); dsIn, dzTrunk added (null: none)
inline void confBackward(Ef2Confidence& h, const ConfInputs& in, const ConfSave& s, const real* sIn, const real* mask,
                         const ConfLogits& d, real* dsIn, real* dzTrunk) {
  Scratch scratch;
  const int T = s.T, C = h.C; const size_t P = (size_t)T * T, PC = P * C;
  real* dz = ARENA.zeros(PC);
  if (d.pae) {
    real* dpn = ARENA.get(PC); h.pae.backward(s.pn, d.pae, dpn, P);
    if (h.hasPaeLn) lnBackward(h.paeLn, s.lpae, dpn, dz, P, true); else add(dz, dpn, PC);
  }
  if (d.plddt) {
    plddtBwdWK<<<blocks((size_t)h.slots * h.Cs * h.bins), 256, 0, STREAM>>>(s.sn, in.atomToToken, in.slot, d.plddt, h.plddtW.g, in.A,
                                                                              h.Cs, h.bins, h.slots);
    real* dsn = ARENA.zeros((size_t)T * h.Cs);
    plddtBwdSK<<<blocks((size_t)in.A * h.Cs), 256, 0, STREAM>>>(h.plddtW.w, in.atomToToken, in.slot, d.plddt, dsn, in.A, h.Cs, h.bins);
    real* dsg = ARENA.get((size_t)T * h.Cs); lnBackward(h.plddtLn, s.lpl, dsn, dsg, T, false);
    real* dpool = ARENA.get((size_t)T * C); h.poolOut.backward(s.pooled, dsg, dpool, T);
    real* dscore = ARENA.get(P);
    rowPoolBwdK<<<T, 256, T * sizeof(real), STREAM>>>(s.z, s.w, dpool, dz, dscore, T, C);
    h.poolAttn.backward(s.z, dscore, dz, P, true);
  }
  // z = z0 + blocks(z0)
  real* dz0 = cloneR(dz, PC);
  stackBackward(h.blocks, s.z0, dz, mask, T, C);
  add(dz0, dz, PC);
  // z0's terms
  real *dr = ARENA.get((size_t)T * C), *dc = ARENA.get((size_t)T * C);
  af3::confPairInitBwdLRK<<<blocks((size_t)T * C), 256, 0, STREAM>>>(dz0, dc, dr, T, C);   // (left: column j, right: row i)
  bucketBwdK<<<blocks((h.edges.size() + 1) * C), 256, 0, STREAM>>>(dz0, s.bucket, h.distEmb.g, P, C, (int)h.edges.size() + 1);
  h.zi->rel.backward(relOneHot(*in.tok), dz0, nullptr, P);
  if (in.tok->bonds) mmAT(in.tok->bonds, dz0, h.zi->bond.g, (int)P, 1, C);
  {
    real* prod = ARENA.get(PC);
    af3::outerRowsK<<<blocks(PC), 256, 0, STREAM>>>(s.l, s.rr, prod, T, C);
    real* dprod = ARENA.get(PC); h.prodOut.backward(prod, dz0, dprod, P);
    real *dl = ARENA.get((size_t)T * C), *drr = ARENA.get((size_t)T * C);
    af3::outerRowsBwdK<<<blocks((size_t)T * C), 256, 0, STREAM>>>(dprod, s.l, s.rr, dl, drr, T, C);
    real* ds = ARENA.get((size_t)T * h.Si);
    h.sToZ.backward(s.s, dr, ds, T); h.sToZT.backward(s.s, dc, ds, T, true);
    h.in1.backward(s.s, dl, ds, T, true); h.in2.backward(s.s, drr, ds, T, true);
    if (dsIn) lnBackward(h.sInLn, s.lsIn, ds, dsIn, T, true);
    else { real* t = ARENA.get((size_t)T * h.Si); lnBackward(h.sInLn, s.lsIn, ds, t, T, false); }
  }
  if (dzTrunk) lnBackward(h.zLn, s.lz, dz0, dzTrunk, P, true);
  else { real* t = ARENA.get(PC); lnBackward(h.zLn, s.lz, dz0, t, P, false); }
}

}  // namespace ef2
}  // namespace grad
