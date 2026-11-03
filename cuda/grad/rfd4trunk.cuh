// RFD4-Proteina's whole trunk for training, one cycle: the conditioning (cond_factory and its two SwiGLU transitions),
// s_init, z_init (the pair factory's RMS-normed projection adaLN-conditioned on the pair condition), the token layers and
// pair updates (rfd4p.cuh), and the four heads - the CA coordinates, the local latents, the sequence logits and the
// insertion head's q k^T / sqrt(D). cuda/rfd4p/src/trunk.cuh's forward() is what it is held to.
//
// One cycle: what a flow-matching step trains. (A recycled state is the previous cycle's stopped output, and on the
// first cycle the recycled terms are a zero row's projection - zero - so this is the first cycle exactly.)
#pragma once
#include "rfd4p.cuh"

namespace grad {
namespace rfd4p {

struct Trunk {
  int D = 0, P = 0, C = 0, PC = 0, TRI = 0, H = 0, Hd = 0, nlayers = 0, everyN = 2;
  int seqF = 0, condF = 0, pairF = 0, pairCondF = 0, classes = 0, latent = 0;
  bool updatePair = true, insertion = true;
  Linear condIn, seqIn, pairIn, pairCondIn, ca, lat, seq, insQ, insK;
  SwiGluR c1, c2;
  Rms pairInN, pairCondN, caN, latN, seqN;
  AdaLnR pairAda;
  std::vector<TokenLayer> layers;
  std::vector<PairUpdate> updates;           // [nlayers], only those with an update allocated
  bool hasUpdate(int i) const { return updatePair && i < nlayers - 1 && i % everyN == 0; }
};
inline void forEachParam(Trunk& t, const ParamFn& f) {
  auto lin = [&](const std::string& n, Linear& x, int K, int N, bool bias = false) {
    x.K = K; x.N = N; x.trans = true; x.hasBias = bias; f(n + ".weight", x.w, (size_t)K * N); if (bias) f(n + ".bias", x.b, N);
  };
  auto rms = [&](const std::string& n, Rms& r, int C_) { r.C = C_; f(n + ".weight", r.w, C_); };
  auto sw = [&](const std::string& n, SwiGluR& s, int D_) {
    s.D = D_; s.I = 2 * D_; lin(n + ".swish_linear.0", s.w1, D_, 4 * D_); lin(n + ".linear_out", s.out, 2 * D_, D_);
  };
  const std::string N = "w/nn.";
  lin(N + "cond_factory.linear_out", t.condIn, t.condF, t.C);
  sw(N + "transition_c_1", t.c1, t.C); sw(N + "transition_c_2", t.c2, t.C);
  lin(N + "init_repr_factory.linear_out", t.seqIn, t.seqF, t.D);
  lin(N + "pair_repr_builder.init_repr_factory.linear_out", t.pairIn, t.pairF, t.P);
  rms(N + "pair_repr_builder.init_repr_factory.ln_out", t.pairInN, t.P);
  lin(N + "pair_repr_builder.cond_factory.linear_out", t.pairCondIn, t.pairCondF, t.PC);
  rms(N + "pair_repr_builder.cond_factory.ln_out", t.pairCondN, t.PC);
  t.pairAda.D = t.P; t.pairAda.xn.C = t.P; t.pairAda.xn.hasW = false;
  rms(N + "pair_repr_builder.adaln.norm_cond", t.pairAda.cn, t.PC);
  lin(N + "pair_repr_builder.adaln.to_gamma.0", t.pairAda.gamma, t.PC, t.P, true);
  lin(N + "pair_repr_builder.adaln.to_beta", t.pairAda.beta, t.PC, t.P);
  t.layers.assign(t.nlayers, TokenLayer{});
  t.updates.assign(t.nlayers, PairUpdate{});
  for (int i = 0; i < t.nlayers; ++i) {
    tokenParams(t.layers[i], N + "transformer_layers." + std::to_string(i), t.D, t.C, t.P, t.H, f);
    if (t.hasUpdate(i)) pairParams(t.updates[i], N + "pair_update_layers." + std::to_string(i), t.D, t.P, t.TRI, t.Hd, f);
  }
  rms(N + "ca_linear.0", t.caN, t.D); lin(N + "ca_linear.1", t.ca, t.D, 3);
  rms(N + "local_latents_linear.0", t.latN, t.D); lin(N + "local_latents_linear.1", t.lat, t.D, t.latent);
  rms(N + "sequence_readout.0", t.seqN, t.D); lin(N + "sequence_readout.1", t.seq, t.D, t.classes);
  if (t.insertion) { lin(N + "insertion_q_proj", t.insQ, t.D, t.D); lin(N + "insertion_k_proj", t.insK, t.D, t.D); }
}
__global__ void repRowK(const real* row, real* out, size_t rows, int C) {     // out[r] = row
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows * C) out[t] = row[t % C];
}
struct TrunkSave {
  real *c0, *t1, *c, *sInit, *zr, *zn, *zc0, *zc, *zcRep, *zInit, *headN[3], *q, *k;
  RmsSave szn, szc, sh[3];
  SgSave sg1, sg2;
  AdaRSave ada;
  std::vector<real*> s, z;                   // every layer's input (and the last one's output)
  int n;
};
struct TrunkOut { real *ca, *latents, *logits, *insertion; };
// seq [n, seqF], cond0 [n, condF], pairF [n*n, pairF], pairCondF [pairCondF], mask [n], pairMask [n*n]
inline TrunkSave trunkForward(Trunk& t, const real* seq, const real* cond0, const real* pairF, const real* pairCondF,
                              const real* mask, const real* pm, int n, TrunkOut& o) {
  TrunkSave s{}; s.n = n;
  const size_t nn = (size_t)n * n;
  s.c0 = ARENA.get((size_t)n * t.C); t.condIn.forward(cond0, s.c0, n);
  s.t1 = ARENA.get((size_t)n * t.C); s.sg1 = sgForward(t.c1, s.c0, mask, s.t1, n);
  s.c = ARENA.get((size_t)n * t.C); s.sg2 = sgForward(t.c2, s.t1, mask, s.c, n);
  s.sInit = ARENA.get((size_t)n * t.D); t.seqIn.forward(seq, s.sInit, n);
  maskRowsK<<<blocks((size_t)n * t.D), 256, 0, STREAM>>>(s.sInit, mask, n, t.D);
  s.zr = ARENA.get(nn * t.P); t.pairIn.forward(pairF, s.zr, nn);
  s.zn = ARENA.get(nn * t.P); s.szn = rmsForward(t.pairInN, s.zr, t.P, s.zn, t.P, nn);
  maskRowsK<<<blocks(nn * t.P), 256, 0, STREAM>>>(s.zn, pm, nn, t.P);
  s.zc0 = ARENA.get(t.PC); t.pairCondIn.forward(pairCondF, s.zc0, 1);
  s.zc = ARENA.get(t.PC); s.szc = rmsForward(t.pairCondN, s.zc0, t.PC, s.zc, t.PC, 1);
  s.zcRep = ARENA.get(nn * t.PC); repRowK<<<blocks(nn * t.PC), 256, 0, STREAM>>>(s.zc, s.zcRep, nn, t.PC);
  s.zInit = ARENA.get(nn * t.P); s.ada = adaRForward(t.pairAda, s.zn, s.zcRep, pm, s.zInit, nn);
  s.s.assign(t.nlayers + 1, nullptr); s.z.assign(t.nlayers + 1, nullptr);
  s.s[0] = s.sInit; s.z[0] = s.zInit;
  for (int i = 0; i < t.nlayers; ++i) {
    s.s[i + 1] = ARENA.get((size_t)n * t.D);
    s.z[i + 1] = t.hasUpdate(i) ? ARENA.get(nn * t.P) : s.z[i];     // (both outside the layer's scratch scope)
    Scratch sc;
    tokenForward(t.layers[i], s.s[i], s.z[i], s.c, mask, s.s[i + 1], n);
    if (t.hasUpdate(i)) puForward(t.updates[i], s.s[i + 1], s.z[i], mask, pm, s.z[i + 1], n);
  }
  const real* sf = s.s.back();
  Rms* norms[3] = {&t.caN, &t.latN, &t.seqN};
  Linear* heads[3] = {&t.ca, &t.lat, &t.seq};
  real** outs[3] = {&o.ca, &o.latents, &o.logits};
  for (int k = 0; k < 3; ++k) {
    s.headN[k] = ARENA.get((size_t)n * t.D); s.sh[k] = rmsForward(*norms[k], sf, t.D, s.headN[k], t.D, n);
    *outs[k] = ARENA.get((size_t)n * heads[k]->N); heads[k]->forward(s.headN[k], *outs[k], n);
  }
  o.insertion = nullptr;
  if (t.insertion) {
    s.q = ARENA.get((size_t)n * t.D); t.insQ.forward(sf, s.q, n);
    s.k = ARENA.get((size_t)n * t.D); t.insK.forward(sf, s.k, n);
    o.insertion = ARENA.get(nn); mmBT(s.q, s.k, o.insertion, n, n, t.D, 0, 1 / sqrt((real)t.D));
  }
  return s;
}
// the heads' cotangents (any null) -> every parameter (the inputs are data)
inline void trunkBackward(Trunk& t, const TrunkSave& s, const real* seq, const real* cond0, const real* pairF,
                          const real* pairCondF, const real* mask, const real* pm, const TrunkOut& d) {
  Scratch scratch;
  const int n = s.n; const size_t nn = (size_t)n * n;
  const real* sf = s.s.back();
  real* ds = ARENA.zeros((size_t)n * t.D);
  Rms* norms[3] = {&t.caN, &t.latN, &t.seqN};
  Linear* heads[3] = {&t.ca, &t.lat, &t.seq};
  const real* douts[3] = {d.ca, d.latents, d.logits};
  for (int k = 0; k < 3; ++k) {
    if (!douts[k]) continue;
    real* dn = ARENA.get((size_t)n * t.D); heads[k]->backward(s.headN[k], douts[k], dn, n);
    rmsBackward(*norms[k], s.sh[k], dn, t.D, ds, t.D, true);
  }
  if (t.insertion && d.insertion) {
    // ins = q k^T / sqrt(D): dq = dins k, dk = dins^T q
    const real sc = 1 / sqrt((real)t.D);
    real *dq = ARENA.get((size_t)n * t.D), *dk = ARENA.get((size_t)n * t.D);
    mm(d.insertion, s.k, dq, n, n, t.D, 0, sc);
    mmAT(d.insertion, s.q, dk, n, n, t.D, 0, sc);
    t.insQ.backward(sf, dq, ds, n, true); t.insK.backward(sf, dk, ds, n, true);
  }
  real* dz = ARENA.zeros(nn * t.P);
  real* dc = ARENA.zeros((size_t)n * t.C);
  for (int i = t.nlayers; i-- > 0;) {
    Scratch sc;
    real* so = ARENA.get((size_t)n * t.D);
    TokenSave ts = tokenForward(t.layers[i], s.s[i], s.z[i], s.c, mask, so, n);
    if (t.hasUpdate(i)) {
      real* zo = ARENA.get(nn * t.P);
      PuSave us = puForward(t.updates[i], so, s.z[i], mask, pm, zo, n);
      real* dzIn = ARENA.get(nn * t.P);
      puBackward(t.updates[i], us, mask, pm, dz, dzIn, ds);     // ds += the update's route into the layer's output
      GK(cudaMemcpyAsync(dz, dzIn, nn * t.P * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
    }
    real *dsIn = ARENA.get((size_t)n * t.D), *dzt = ARENA.zeros(nn * t.P);
    tokenBackward(t.layers[i], ts, mask, ds, dsIn, dzt, dc, n);
    add(dz, dzt, nn * t.P);
    GK(cudaMemcpyAsync(ds, dsIn, (size_t)n * t.D * sizeof(real), cudaMemcpyDeviceToDevice, STREAM));
  }
  // z_init
  real *dzn = ARENA.zeros(nn * t.P), *dzcRep = ARENA.zeros(nn * t.PC);
  adaRBackward(t.pairAda, s.ada, pm, dz, dzn, dzcRep);
  real* dzc = ARENA.zeros(t.PC); colSum(dzc, dzcRep, nn, t.PC);
  real* dzc0 = ARENA.get(t.PC); rmsBackward(t.pairCondN, s.szc, dzc, t.PC, dzc0, t.PC, false);
  t.pairCondIn.backward(pairCondF, dzc0, nullptr, 1);
  maskRowsK<<<blocks(nn * t.P), 256, 0, STREAM>>>(dzn, pm, nn, t.P);
  real* dzr = ARENA.get(nn * t.P); rmsBackward(t.pairInN, s.szn, dzn, t.P, dzr, t.P, false);
  t.pairIn.backward(pairF, dzr, nullptr, nn);
  // s_init
  maskRowsK<<<blocks((size_t)n * t.D), 256, 0, STREAM>>>(ds, mask, n, t.D);
  t.seqIn.backward(seq, ds, nullptr, n);
  // c
  real* dt1 = ARENA.zeros((size_t)n * t.C); sgBackward(t.c2, s.sg2, mask, dc, dt1);
  real* dc0 = ARENA.zeros((size_t)n * t.C); sgBackward(t.c1, s.sg1, mask, dt1, dc0);
  t.condIn.backward(cond0, dc0, nullptr, n);
}

}  // namespace rfd4p
}  // namespace grad
