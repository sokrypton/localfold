// ESMFold2's sliding-window atom transformer with 3D RoPE (cpu/esmfold2/atom-transformer.js
// is the reading): the inputs embedder, and the stacks the diffusion module's atom encoder and
// decoder reuse.
//
//   c0 = LN(atomFeatures @ linear);  q = c0 (+ coords @ coordsLinear in the diffusion's encoder)
//   3 x adaLN-Zero block, conditioned on c0 throughout:
//       mod = silu(c0) @ adaln -> shift_a scale_a gate_a shift_f scale_f gate_f
//       q  += gate_a * SWA(rms(q) (1 + scale_a) + shift_a)
//       q  += gate_f * swiglu(rms(q) (1 + scale_f) + shift_f)
//   SWA: qkv; q, k rms-normed PER HEAD then rotated; q, k, v narrowed to bfloat16 (the module casts
//   whatever the model's dtype); keys within halfWindow in rank among VALID atoms, the diagonal always;
//   out = (ctx * live * sigmoid(input @ attnGate)) @ attnOut
//   Every rms is affine-free with torch's eps (float32 epsilon, 1.19e-7).
#pragma once
#include "ops.cuh"

constexpr int ATOM_FEATURES = 3 + 1 + 1 + 128 + 4 * 64;   // 389
constexpr float RMS_EPS = 1.1920928955078125e-7f;
// the module narrows q, k, v to bfloat16 whatever the model's dtype; false is the control arm, held to
// an oracle written with oracle.py --float32-attention (the rope table stays bfloat16 either way)
inline bool ATOM_BF16 = true;

__device__ __forceinline__ float bf16Round(float v) {      // round to nearest even, back to float
  unsigned w = __float_as_uint(v);
  w = (w + 0x7fffu + ((w >> 16) & 1u)) & 0xffff0000u;
  return __uint_as_float(w);
}

__global__ void atomFeaturesK(const float* pos, const float* charge, const float* mask, const int* element,
                              const int* nameChars, float* out, int A) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)A * ATOM_FEATURES) return;
  int atom = (int)(t / ATOM_FEATURES), f = (int)(t % ATOM_FEATURES);
  bool live = mask[atom] != 0.f;
  float v = 0.f;
  if (f < 3) v = pos[atom * 3 + f];
  else if (f == 3) v = charge[atom];
  else if (f == 4) v = mask[atom];
  else if (f < 5 + 128) v = live && element[atom] == f - 5 ? 1.f : 0.f;
  else { int i = (f - 133) / 64, ch = (f - 133) % 64; v = live && nameChars[atom * 4 + i] == ch ? 1.f : 0.f; }
  out[t] = v;
}
// the rotary table [A, 16]: x's two pairs (base 20), y's, z's, then the space uid's ten (base 10000);
// spacing 1 / base^(i/n); stored in bfloat16 as the model stores it
__global__ void ropeTableK(const float* pos, const int* uid, float* cosT, float* sinT, int A) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= A * 16) return;
  int atom = t / 16, at = t % 16;
  double angle;
  if (at < 6) { int axis = at / 2, i = at % 2; angle = (double)pos[atom * 3 + axis] * (1.0 / pow(20.0, i / 2.0)); }
  else { int i = at - 6; angle = (double)uid[atom] * (1.0 / pow(10000.0, i / 10.0)); }
  cosT[t] = bf16Round((float)cos(angle));
  sinT[t] = bf16Round((float)sin(angle));
}
__global__ void rankK(const float* mask, int* rank, int A) {      // inclusive scan, one thread (A small)
  if (blockIdx.x || threadIdx.x) return;
  int seen = 0;
  for (int a = 0; a < A; ++a) { seen += mask[a] != 0.f; rank[a] = seen - 1; }
}
__global__ void siluK(const float* x, float* y, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] = siluF(x[t]);
}
// rms(x) * (1 + mod[scale]) + mod[shift], a warp a row (C = 128)
__global__ void rmsModulateK(const float* x, const float* mod, float* y, int rows, int C, int shift, int scale) {
  int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
  if (row >= rows) return;
  const float* xr = x + (size_t)row * C;
  float s = 0;
  for (int c = lane; c < C; c += 32) s += xr[c] * xr[c];
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  float inv = rsqrtf(s / C + RMS_EPS);
  const float* m = mod + (size_t)row * 6 * C;
  for (int c = lane; c < C; c += 32) y[(size_t)row * C + c] = xr[c] * inv * (1.f + m[scale * C + c]) + m[shift * C + c];
}
// q and k (heads of 32, inside the packed [A, 3C] qkv): rms per head, rotate, narrow to bfloat16; v narrowed
__global__ void qkvPrepareK(float* qkv, const float* cosT, const float* sinT, int A, int C, int heads, bool bf16) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;      // one thread per (atom, part, head)
  if (t >= A * 3 * heads) return;
  int head = t % heads, part = (t / heads) % 3, atom = t / (3 * heads);
  float* p = qkv + (size_t)atom * 3 * C + part * C + head * 32;
  if (part == 2) { if (bf16) for (int d = 0; d < 32; ++d) p[d] = bf16Round(p[d]); return; }
  float s = 0;
  for (int d = 0; d < 32; ++d) s += p[d] * p[d];
  float inv = rsqrtf(s / 32 + RMS_EPS);
  float v[32];
  for (int d = 0; d < 32; ++d) v[d] = p[d] * inv;
  for (int i = 0; i < 16; ++i) {
    float c = cosT[atom * 16 + i], sn = sinT[atom * 16 + i], a = v[i], b = v[16 + i];
    float lo = a * c - b * sn, hi = b * c + a * sn;
    p[i] = bf16 ? bf16Round(lo) : lo;
    p[16 + i] = bf16 ? bf16Round(hi) : hi;
  }
}
// the windowed attention itself, f32 throughout (no [heads, A, A] matrix): a valid query's keys are the
// valid atoms within halfWindow in rank, valid[rank - hw .. rank + hw], itself among them. A block takes
// 32 consecutive valid queries of one head and walks the union of their windows in tiles of SWA_KT keys
// staged in shared memory; a warp a query at a time, online softmax, a lane a key for the scores and a
// lane a channel for the output. An invalid atom's row is zero (gateLiveK zeroes it in any case).
constexpr int SWA_KT = 144, SWA_Q = 16;     // 16 queries (4 warps) and their whole +-64 window in one tile
__global__ void __launch_bounds__(128) swaWindowK(const float* qkv, const int* valid, int nValid, float* ctx, int C,
                                                  int halfWindow, float scale) {
  __shared__ float Ks[SWA_KT][33];
  __shared__ float Vs[SWA_KT][32];
  int head = blockIdx.y, warp = threadIdx.x / 32, lane = threadIdx.x & 31;
  int r0 = blockIdx.x * SWA_Q, r1 = min(nValid, r0 + SWA_Q) - 1;
  int klo = max(0, r0 - halfWindow), khi = min(nValid - 1, r1 + halfWindow);
  // each warp's four consecutive queries, together: one key read serves all four
  int rq = r0 + 4 * warp;
  float qd[4][32], m[4], l[4], acc[4];
  #pragma unroll
  for (int u = 0; u < 4; ++u) {
    m[u] = -INFINITY; l[u] = 0.f; acc[u] = 0.f;
    const float* q = qkv + (size_t)valid[min(rq + u, nValid - 1)] * 3 * C + head * 32;
    #pragma unroll
    for (int d = 0; d < 32; ++d) qd[u][d] = q[d] * scale;
  }
  for (int t0 = klo; t0 <= khi; t0 += SWA_KT) {
    int n = min(SWA_KT, khi - t0 + 1);
    __syncthreads();
    for (int e = threadIdx.x; e < n * 8; e += blockDim.x) {
      int j = e / 8, part = e % 8;
      const float* row = qkv + (size_t)valid[t0 + j] * 3 * C + head * 32 + part * 4;
      float4 k = *(const float4*)(row + C), v = *(const float4*)(row + 2 * C);
      Ks[j][part * 4] = k.x; Ks[j][part * 4 + 1] = k.y; Ks[j][part * 4 + 2] = k.z; Ks[j][part * 4 + 3] = k.w;
      *(float4*)&Vs[j][part * 4] = v;
    }
    __syncthreads();
    if (rq > r1) continue;
    // the union of the four windows inside this tile
    int lo = max(rq - halfWindow, t0) - t0, hi = min(min(rq + 3, r1) + halfWindow, t0 + n - 1) - t0;
    for (int j0 = lo; j0 <= hi; j0 += 32) {
      int j = j0 + lane, key = t0 + j;
      float s[4] = {0.f, 0.f, 0.f, 0.f};
      if (j <= hi) {
        #pragma unroll
        for (int d = 0; d < 32; ++d) {
          float k = Ks[j][d];
          #pragma unroll
          for (int u = 0; u < 4; ++u) s[u] += qd[u][d] * k;
        }
      }
      float p[4];
      #pragma unroll
      for (int u = 0; u < 4; ++u) {
        bool ok = j <= hi && abs(key - (rq + u)) <= halfWindow;
        float v = ok ? s[u] : -INFINITY, cm = v;
        for (int o = 16; o; o >>= 1) cm = fmaxf(cm, __shfl_xor_sync(~0u, cm, o));
        float mn = fmaxf(m[u], cm);
        if (mn == -INFINITY) { p[u] = 0.f; continue; }      // nothing of this query's in the chunk yet
        float corr = expf(m[u] - mn);
        p[u] = ok ? expf(v - mn) : 0.f;
        float ps = p[u];
        for (int o = 16; o; o >>= 1) ps += __shfl_xor_sync(~0u, ps, o);
        l[u] = l[u] * corr + ps; acc[u] *= corr; m[u] = mn;
      }
      int cnt = min(32, hi - j0 + 1);
      for (int k = 0; k < cnt; ++k) {
        float v = Vs[j0 + k][lane];
        #pragma unroll
        for (int u = 0; u < 4; ++u) acc[u] += __shfl_sync(~0u, p[u], k) * v;
      }
    }
  }
  #pragma unroll
  for (int u = 0; u < 4; ++u)
    if (rq + u <= r1) ctx[(size_t)valid[rq + u] * C + head * 32 + lane] = acc[u] / l[u];
}
__global__ void gateLiveK(float* ctx, const float* gate, const float* mask, size_t A, int C) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= A * C) return;
  ctx[t] *= (mask[t / C] != 0.f ? 1.f : 0.f) / (1.f + expf(-gate[t]));
}
__global__ void gatedAddK(float* x, const float* mod, const float* d, int A, int C, int which) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)A * C) return;
  size_t a = t / C; int c = (int)(t % C);
  x[t] += mod[a * 6 * C + which * C + c] * d[t];
}

struct AtomCtx { int A, C, heads, hidden; const float* mask; const int* rank; const float* cosT; const float* sinT;
                 const int* valid; int nValid;                  // the valid atoms in order (rank -> atom)
                 const int* tokenStart; const int* tokenAtoms;  // each token's valid atoms (CSR)
               };

inline void dumpOnce(const char* name, const float* d, size_t n) {
  static std::set<std::string> done;
  const char* dir = getenv("EF2_DUMPDIR");
  if (!dir || done.count(name)) return;
  done.insert(name);
  auto h = download(d, n); FILE* f = fopen((std::string(dir) + "/" + name + ".bin").c_str(), "wb");
  fwrite(h.data(), 4, h.size(), f); fclose(f);
}
inline void swaBlock(const AtomCtx& a, float* x, const float* cond, const std::string& B, int halfWindow) {
  size_t A = a.A; int C = a.C;
  float* sc = scratch<float>("atom.silu", A * C); float* mod = scratch<float>("atom.mod", A * 6 * C);
  siluK<<<blocks(A * C), 256, 0, STREAM>>>(cond, sc, A * C);
  gemm(sc, F(B + "adaln"), mod, A, C, 6 * C);
  float* xm = scratch<float>("atom.xm", A * C);
  rmsModulateK<<<(unsigned)((A + 7) / 8), 256, 0, STREAM>>>(x, mod, xm, (int)A, C, 0, 1);
  float* qkv = scratch<float>("atom.qkv", A * 3 * C);
  gemm(xm, F(B + "qkv"), qkv, A, C, 3 * C);
  dumpOnce("mod", mod, A * 6 * C); dumpOnce("xm", xm, A * C); dumpOnce("qkv_raw", qkv, A * 3 * C);
  qkvPrepareK<<<blocks(A * 3 * a.heads, 128), 128, 0, STREAM>>>(qkv, a.cosT, a.sinT, (int)A, C, a.heads, ATOM_BF16);
  dumpOnce("qkv", qkv, A * 3 * C); dumpOnce("cos", a.cosT, A * 16);
  float* ctx = scratch<float>("atom.ctx", A * C);
  CK(cudaMemsetAsync(ctx, 0, A * C * 4, STREAM));
  if (a.nValid)
    swaWindowK<<<dim3((unsigned)((a.nValid + SWA_Q - 1) / SWA_Q), (unsigned)a.heads), SWA_Q / 4 * 32, 0, STREAM>>>(
        qkv, a.valid, a.nValid, ctx, C, std::min(halfWindow, (int)A), 1.f / sqrtf(32.f));
  float* gate = scratch<float>("atom.gate", A * C);
  gemm(xm, F(B + "attnGate"), gate, A, C, C);
  dumpOnce("ctx", ctx, A * C);
  gateLiveK<<<blocks(A * C), 256, 0, STREAM>>>(ctx, gate, a.mask, A, C);
  float* att = scratch<float>("atom.att", A * C);
  gemm(ctx, F(B + "attnOut"), att, A, C, C);
  gatedAddK<<<blocks(A * C), 256, 0, STREAM>>>(x, mod, att, (int)A, C, 2);
  rmsModulateK<<<(unsigned)((A + 7) / 8), 256, 0, STREAM>>>(x, mod, xm, (int)A, C, 3, 4);
  float* h = scratch<float>("atom.h", A * 2 * a.hidden); float* g = scratch<float>("atom.g", A * a.hidden);
  gemm(xm, F(B + "ffnUp"), h, A, C, 2 * a.hidden);
  swigluK<<<blocks(A * a.hidden), 256, 0, STREAM>>>(h, g, A, a.hidden);
  gemm(g, F(B + "ffnDown"), att, A, a.hidden, C);
  gatedAddK<<<blocks(A * C), 256, 0, STREAM>>>(x, mod, att, (int)A, C, 5);
}
inline void swaStack(const AtomCtx& a, float* x, const float* cond, const std::string& prefix, int blocksN, int halfWindow) {
  for (int b = 0; b < blocksN; ++b) swaBlock(a, x, cond, prefix + "/blocks/" + std::to_string(b) + "/", halfWindow);
}

__global__ void reluK(float* x, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) x[t] = fmaxf(x[t], 0.f);
}
// mean over each token's atoms, weighted by the mask, into s_inputs' first C columns (row stride ld)
__global__ void scatterMeanK(const float* v, const int* tokenStart, const int* tokenAtoms, const float* mask,
                             float* out, int T, int C, int ld) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;   // one thread per (token, channel)
  if (t >= (size_t)T * C) return;
  int token = (int)(t / C), c = (int)(t % C);
  float s = 0, w = 0;
  for (int k = tokenStart[token]; k < tokenStart[token + 1]; ++k) {
    int a = tokenAtoms[k];
    s += v[(size_t)a * C + c] * mask[a]; w += mask[a];
  }
  out[(size_t)token * ld + c] = s / fmaxf(w, 1e-9f);
}
__global__ void tailInputsK(const float* aatype, const float* profile, const float* delMean, float* out, int T, int C,
                            int K, int ld) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)T * (2 * K + 1)) return;
  int token = (int)(t / (2 * K + 1)), f = (int)(t % (2 * K + 1));
  float v = f < K ? aatype[token * K + f] : !profile ? 0.f : f < 2 * K ? profile[token * K + f - K] : delMean[token];
  out[(size_t)token * ld + C + f] = v;
}

// the atom side of an input, built once: features -> c0, the rope table, ranks
struct Atoms { AtomCtx ctx; float* c0; };
inline Atoms prepareAtoms(int A, const std::string& prefix) {
  Atoms at{};
  int C = (int)M.meta("meta/atomChannels");
  at.ctx.A = A; at.ctx.C = C; at.ctx.heads = (int)M.meta("meta/atomHeads");
  at.ctx.hidden = (int)dimOf("f/" + prefix + "/blocks/0/ffnDown", 0);
  at.ctx.mask = W("atom_mask");
  int* rank = dallocT<int>(A); rankK<<<1, 1, 0, STREAM>>>(at.ctx.mask, rank, A); at.ctx.rank = rank;
  {   // the valid atoms in order, and each token's (for the windowed attention and the pooling)
    std::vector<float> mask = download(at.ctx.mask, A);
    std::vector<int> toToken(A), valid;
    CK(cudaMemcpy(toToken.data(), Idev("atom_to_token"), (size_t)A * 4, cudaMemcpyDeviceToHost));
    int T = (int)M.meta("meta/tokens");
    std::vector<std::vector<int>> byToken(T);
    for (int a = 0; a < A; ++a) if (mask[a] != 0.f) { valid.push_back(a); byToken.at(toToken[a]).push_back(a); }
    std::vector<int> start{0}, atoms;
    for (auto& v : byToken) { atoms.insert(atoms.end(), v.begin(), v.end()); start.push_back((int)atoms.size()); }
    at.ctx.nValid = (int)valid.size();
    at.ctx.valid = upload(valid.data(), std::max<size_t>(valid.size(), 1));
    at.ctx.tokenStart = upload(start.data(), start.size());
    at.ctx.tokenAtoms = upload(atoms.data(), std::max<size_t>(atoms.size(), 1));
  }
  float* cosT = dalloc((size_t)A * 16); float* sinT = dalloc((size_t)A * 16);
  ropeTableK<<<blocks((size_t)A * 16), 256, 0, STREAM>>>(W("ref_pos"), Idev("ref_space_uid"), cosT, sinT, A);
  at.ctx.cosT = cosT; at.ctx.sinT = sinT;
  float* feat = scratch<float>("atom.features", (size_t)A * ATOM_FEATURES);
  atomFeaturesK<<<blocks((size_t)A * ATOM_FEATURES), 256, 0, STREAM>>>(W("ref_pos"), W("ref_charge"), at.ctx.mask,
    Idev("ref_element"), Idev("ref_atom_name_chars"), feat, A);
  at.c0 = dalloc((size_t)A * C);
  gemm(feat, F(prefix + "/linear"), at.c0, A, ATOM_FEATURES, C);
  layerNorm(at.c0, at.c0, A, C, F(prefix + "/norm/scale"), F(prefix + "/norm/offset"));
  return at;
}

inline void freeAtoms(Atoms& at) {
  CK(cudaStreamSynchronize(STREAM));
  for (const void* p : {(const void*)at.c0, (const void*)at.ctx.rank, (const void*)at.ctx.cosT, (const void*)at.ctx.sinT,
                        (const void*)at.ctx.valid, (const void*)at.ctx.tokenStart, (const void*)at.ctx.tokenAtoms})
    CK(cudaFree((void*)p));
  at = Atoms{};
}
// the inputs embedder: s_inputs [T, 451] = [pool(relu(stack(c0) @ toToken)) | aatype | profile | deletion mean]
// halfWindow 64 is biohub's esm package - the vendor's code, which windows every atom stack (flash-attn
// window_size=(64, 64) on CUDA, the rank mask on the CPU). Dense is the page's reading, from Synthyra's
// fastplms, which never windows this stage (docs/EF2FAST.md); --inputs-window=0 is that arm. Measured
// natively against the crystals, neither reading wins everywhere: 6MRR 0.92 A windowed against 1.59
// dense (five seeds), 1QYS 0.89 against 0.94, 5CAJ 2.10 against 1.86 (three)
inline int INPUTS_HALF_WINDOW = 64;
inline void inputsEmbedder(int T, int A, float* sInputs, int sWidth, bool check = false) {
  Atoms at = prepareAtoms(A, "atom");
  if (check) checkOracle("atom norm (c0)", at.c0, (size_t)A * at.ctx.C, "o/atom/norm");
  int C = at.ctx.C, Ct = (int)dimOf("f/atom/toToken", 1), K = (int)M.meta("meta/classes");
  float* x = scratch<float>("embed.x", (size_t)A * C);
  CK(cudaMemcpyAsync(x, at.c0, (size_t)A * C * 4, cudaMemcpyDeviceToDevice, STREAM));
  for (int b = 0; b < (int)M.meta("meta/atomBlocks"); ++b) {
    swaBlock(at.ctx, x, at.c0, "atom/blocks/" + std::to_string(b) + "/", INPUTS_HALF_WINDOW);
    if (check) checkOracle(("atom block " + std::to_string(b)).c_str(), x, (size_t)A * C, "o/atom/block" + std::to_string(b));
  }
  float* tok = scratch<float>("embed.tok", (size_t)A * Ct);
  gemm(x, F("atom/toToken"), tok, A, C, Ct);
  reluK<<<blocks((size_t)A * Ct), 256, 0, STREAM>>>(tok, (size_t)A * Ct);
  scatterMeanK<<<blocks((size_t)T * Ct), 256, 0, STREAM>>>(tok, at.ctx.tokenStart, at.ctx.tokenAtoms, at.ctx.mask, sInputs, T, Ct, sWidth);
  // the alignment's profile and mean deletion only where the checkpoint reads them: the released models do; the
  // experimental tier (disable_msa_features) zeroes both, and its bundles predate the field
  const bool msaFeatures = M.has("meta/msaFeatures") && M.meta("meta/msaFeatures") > 0;
  tailInputsK<<<blocks((size_t)T * (2 * K + 1)), 256, 0, STREAM>>>(W("aatype"), msaFeatures ? W("profile") : nullptr,
                                                                 msaFeatures ? W("deletion_mean") : nullptr, sInputs,
                                                                 T, Ct, K, sWidth);
  freeAtoms(at);
}
