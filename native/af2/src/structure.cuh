// The structure module and the confidence heads - af3-any-model's AF2 (alphafold3/af2/model/folding.py:
// generate_monomer_rigids, FoldIteration, InvariantPointAttention, MultiRigidSidechain; all_atom.py),
// float32. A rigid is 12 floats: the rotation row-major (xx xy xz yx yy yz zx zy zz), then the
// translation.
#pragma once
#include "ops.cuh"

__device__ inline void rapply(const float* r, const float* p, float* o) {      // R p + t
  for (int a = 0; a < 3; ++a) o[a] = r[a * 3] * p[0] + r[a * 3 + 1] * p[1] + r[a * 3 + 2] * p[2] + r[9 + a];
}
__device__ inline void rapplyInv(const float* r, const float* p, float* o) {   // R^T (p - t)
  float d[3] = {p[0] - r[9], p[1] - r[10], p[2] - r[11]};
  for (int a = 0; a < 3; ++a) o[a] = r[a] * d[0] + r[3 + a] * d[1] + r[6 + a] * d[2];
}
__device__ inline void rcompose(const float* A, const float* B, float* C) {    // A @ B
  float R[9], t[3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) R[i * 3 + j] = A[i * 3] * B[j] + A[i * 3 + 1] * B[3 + j] + A[i * 3 + 2] * B[6 + j];
  rapply(A, B + 9, t);
  for (int k = 0; k < 9; ++k) C[k] = R[k];
  for (int k = 0; k < 3; ++k) C[9 + k] = t[k];
}

__global__ void identityRigidK(float* r, int L) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= L) return;
  float* x = r + i * 12;
  for (int k = 0; k < 12; ++k) x[k] = 0;
  x[0] = x[4] = x[8] = 1;
}
// points [L, H*3*P] from the projection (per head: x[0:P] y[P:2P] z[2P:3P]) -> global [L, H, P, 3]
__global__ void pointsToGlobalK(const float* proj, const float* rig, float* out, int L, int Hh, int Pp) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * Hh * Pp) return;
  int p = (int)(t % Pp); size_t r = t / Pp; int h = (int)(r % Hh); int i = (int)(r / Hh);
  const float* s = proj + (size_t)i * Hh * 3 * Pp + h * 3 * Pp;
  float loc[3] = {s[p], s[Pp + p], s[2 * Pp + p]}, g[3];
  rapply(rig + i * 12, loc, g);
  for (int a = 0; a < 3; ++a) out[t * 3 + a] = g[a];
}
// IPA logits -> attention weights, a warp a (query, head): logits[q, k, h] then softmax over k
__global__ void ipaWeightsK(const float* qs, const float* ks, const float* qp, const float* kp, const float* b2d,
                            const float* pw, const float* seqMask, float* attn, int L, int Hh, int Cs, int Pq) {
  size_t wid = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  if (wid >= (size_t)L * Hh) return;
  int h = (int)(wid % Hh), q = (int)(wid / Hh);
  float* row = attn + ((size_t)q * Hh + h) * L;     // [q, h, k]
  float mx = -INFINITY;
  float w = pw[h];
  for (int k = lane; k < L; k += 32) {
    float d2 = 0;
    for (int p = 0; p < Pq; ++p)
      for (int a = 0; a < 3; ++a) {
        float d = qp[(((size_t)q * Hh + h) * Pq + p) * 3 + a] - kp[(((size_t)k * Hh + h) * Pq + p) * 3 + a];
        d2 += d * d;
      }
    float s = -0.5f * w * d2;
    float sc = 0;
    for (int c = 0; c < Cs; ++c) sc += qs[((size_t)q * Hh + h) * Cs + c] * ks[((size_t)k * Hh + h) * Cs + c];
    s += sc + b2d[((size_t)q * L + k) * Hh + h];
    s -= 1e5f * (1.f - seqMask[q] * seqMask[k]);
    s *= sqrtf(1.f / 3.f);
    row[k] = s;
    mx = fmaxf(mx, s);
  }
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, o));
  float sum = 0;
  for (int k = lane; k < L; k += 32) { float e = __expf(row[k] - mx); row[k] = e; sum += e; }
  for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o);
  for (int k = lane; k < L; k += 32) row[k] /= sum;
}
// IPA's outputs, a block a (query, head): [result_scalar | point local x | y | z | norms | over 2d]
// written into final [L, 2112] at the reference's concatenation offsets
__global__ void ipaOutputsK(const float* attn, const float* vs, const float* vp, const float* act2d, const float* rig,
                            float* final_, int L, int Hh, int Cs, int Pv, int C2) {
  int q = blockIdx.x / Hh, h = blockIdx.x % Hh;
  const float* a = attn + ((size_t)q * Hh + h) * L;
  int Fw = Hh * Cs + 4 * Hh * Pv + Hh * C2;
  float* f = final_ + (size_t)q * Fw;
  for (int c = threadIdx.x; c < Cs; c += blockDim.x) {
    float s = 0;
    for (int k = 0; k < L; ++k) s += a[k] * vs[((size_t)k * Hh + h) * Cs + c];
    f[h * Cs + c] = s;
  }
  for (int p = threadIdx.x; p < Pv; p += blockDim.x) {
    float g[3] = {0, 0, 0};
    for (int k = 0; k < L; ++k)
      for (int x = 0; x < 3; ++x) g[x] += a[k] * vp[(((size_t)k * Hh + h) * Pv + p) * 3 + x];
    float loc[3];
    rapplyInv(rig + q * 12, g, loc);
    int base = Hh * Cs, idx = h * Pv + p;
    f[base + idx] = loc[0];
    f[base + Hh * Pv + idx] = loc[1];
    f[base + 2 * Hh * Pv + idx] = loc[2];
    f[base + 3 * Hh * Pv + idx] = sqrtf(fmaxf(loc[0] * loc[0] + loc[1] * loc[1] + loc[2] * loc[2], 1e-16f));
  }
  for (int c = threadIdx.x; c < C2; c += blockDim.x) {
    float s = 0;
    for (int k = 0; k < L; ++k) s += a[k] * act2d[((size_t)q * L + k) * C2 + c];
    f[Hh * Cs + 4 * Hh * Pv + h * C2 + c] = s;
  }
}
// the backbone update: QuatRigid's (qx qy qz tx ty tz), qw = 1, normalised; rigid = rigid @ update
__global__ void rigidUpdateK(float* rig, const float* upd, int L) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= L) return;
  const float* u = upd + i * 6;
  float w = 1, x = u[0], y = u[1], z = u[2];
  float inv = rsqrtf(fmaxf(1e-6f, w * w + x * x + y * y + z * z));
  w *= inv; x *= inv; y *= inv; z *= inv;
  float U[12] = {1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                 2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                 2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y), u[3], u[4], u[5]};
  float C[12];
  rcompose(rig + i * 12, U, C);
  for (int k = 0; k < 12; ++k) rig[i * 12 + k] = C[k];
}
__global__ void reluCopyK(const float* x, float* y, size_t n) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] = fmaxf(x[t], 0.f);
}
// angles (unnormalised [L, 7, 2]) -> frames -> atom14 -> atom37, a thread a residue
__global__ void sidechainAtomsK(const float* unnorm, const float* rig, float positionScale, const int* aatypeIn,
                                const float* defaultFrames, const int* atom14Group, const float* litPos,
                                const float* atom14Mask, const int* atom37To14, const float* atom37Mask,
                                const float* seqMask, float* angles, float* pos14, float* pos37, int L) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= L) return;
  int aa = min(max(aatypeIn[i], 0), 19);    // the reference clips X to the 20 standard types
  float bb[12];
  for (int k = 0; k < 12; ++k) bb[k] = rig[i * 12 + k];
  for (int k = 9; k < 12; ++k) bb[k] *= positionScale;
  float sinA[8], cosA[8];
  sinA[0] = 0; cosA[0] = 1;
  for (int a = 0; a < 7; ++a) {
    float s = unnorm[(i * 7 + a) * 2], c = unnorm[(i * 7 + a) * 2 + 1];
    float inv = 1.f / sqrtf(fmaxf(s * s + c * c, 1e-12f));
    s *= inv; c *= inv;
    angles[(i * 7 + a) * 2] = s; angles[(i * 7 + a) * 2 + 1] = c;
    sinA[a + 1] = s; cosA[a + 1] = c;
  }
  // all_frames[g] = default_frame[aa, g] composed with the rotation about x by the g-th torsion
  float frames[8][12];
  for (int g = 0; g < 8; ++g) {
    const float* m = defaultFrames + ((size_t)aa * 8 + g) * 16;
    float D[12] = {m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10], m[3], m[7], m[11]};
    float Rx[12] = {1, 0, 0, 0, cosA[g], -sinA[g], 0, sinA[g], cosA[g], 0, 0, 0};
    float R[12];
    rcompose(D, Rx, R);
    for (int k = 9; k < 12; ++k) R[k] = D[k];       // compose_rotation keeps the translation
    for (int k = 0; k < 12; ++k) frames[g][k] = R[k];
  }
  // chi2..4 chain onto chi1
  float tmp[12];
  rcompose(frames[4], frames[5], tmp); for (int k = 0; k < 12; ++k) frames[5][k] = tmp[k];
  rcompose(frames[5], frames[6], tmp); for (int k = 0; k < 12; ++k) frames[6][k] = tmp[k];
  rcompose(frames[6], frames[7], tmp); for (int k = 0; k < 12; ++k) frames[7][k] = tmp[k];
  float global[8][12];
  for (int g = 0; g < 8; ++g) rcompose(bb, frames[g], global[g]);
  for (int a = 0; a < 14; ++a) {
    int g = atom14Group[aa * 14 + a];
    float p[3];
    rapply(global[g], litPos + ((size_t)aa * 14 + a) * 3, p);
    float m = atom14Mask[aa * 14 + a];
    for (int x = 0; x < 3; ++x) pos14[((size_t)i * 14 + a) * 3 + x] = p[x] * m;
  }
  for (int a = 0; a < 37; ++a) {
    int s = atom37To14[aa * 37 + a];
    float m = atom37Mask[aa * 37 + a] * seqMask[i];
    for (int x = 0; x < 3; ++x) pos37[((size_t)i * 37 + a) * 3 + x] = pos14[((size_t)i * 14 + s) * 3 + x] * m;
  }
}
__global__ void softplusScaleK(const float* raw, float* out, int Hh, float base) {
  int h = threadIdx.x;
  if (h < Hh) out[h] = base * logf(1.f + expf(raw[h]));
}

struct StructureOut { float* act; float* rigid; float* pos37; float* pos14; float* angles; };
inline StructureOut structureModule(const float* single, const float* pair, int L, float positionScale) {
  const std::string S = "structure_module/", F = S + "fold_iteration/", I = F + "invariant_point_attention/";
  const int C = 384, C2 = 128;
  size_t pairs = (size_t)L * L;
  float* initial = scratch<float>("sm.initial", (size_t)L * C);
  layerNorm(single, initial, L, C, S + "single_layer_norm");
  float* act = scratch<float>("sm.act", (size_t)L * C);
  linearB(initial, S + "initial_projection", -1, act, L, C, C);
  float* act2d = scratch<float>("sm.act2d", pairs * C2);
  layerNorm(pair, act2d, pairs, C2, S + "pair_layer_norm");
  int Hh = (int)dimW(I + "q_scalar_projection/weights", 1), Cs = (int)dimW(I + "q_scalar_projection/weights", 2);
  int Pq = (int)dimW(I + "q_point_projection/point_projection/weights", 2) / 3;
  int Pv = (int)dimW(I + "v_point_projection/point_projection/weights", 2) / 3;
  float* b2d = scratch<float>("sm.b2d", pairs * Hh);
  linearB(act2d, I + "attention_2d", -1, b2d, pairs, C2, Hh);    // the same every iteration (shared weights)
  float* pw = scratch<float>("sm.pw", Hh);
  softplusScaleK<<<1, 32, 0, STREAM>>>(P(I + "trainable_point_weights"), pw, Hh, sqrtf(1.f / (Pq * 9.f / 2.f)));
  float* rig = dalloc((size_t)L * 12);
  identityRigidK<<<blocks(L), 256, 0, STREAM>>>(rig, L);
  float* qs = scratch<float>("sm.qs", (size_t)L * Hh * Cs); float* ks = scratch<float>("sm.ks", (size_t)L * Hh * Cs);
  float* vs = scratch<float>("sm.vs", (size_t)L * Hh * Cs);
  float* proj = scratch<float>("sm.pproj", (size_t)L * Hh * 3 * std::max(Pq, Pv));
  float* qp = scratch<float>("sm.qp", (size_t)L * Hh * Pq * 3); float* kp = scratch<float>("sm.kp", (size_t)L * Hh * Pq * 3);
  float* vp = scratch<float>("sm.vp", (size_t)L * Hh * Pv * 3);
  float* attn = scratch<float>("sm.attn", (size_t)L * Hh * L);
  int Fw = Hh * Cs + 4 * Hh * Pv + Hh * C2;
  float* fin = scratch<float>("sm.final", (size_t)L * Fw);
  float* upd = scratch<float>("sm.upd", (size_t)L * C);
  float* t1 = scratch<float>("sm.t1", (size_t)L * C); float* t2 = scratch<float>("sm.t2", (size_t)L * C);
  float* rq = scratch<float>("sm.rq", (size_t)L * 6);
  const float* seqMask = W("seq_mask");
  int layers = 8;
  for (int it = 0; it < layers; ++it) {
    // IPA
    linearB(act, I + "q_scalar_projection", -1, qs, L, C, Hh * Cs);
    linearB(act, I + "k_scalar_projection", -1, ks, L, C, Hh * Cs);
    linearB(act, I + "v_scalar_projection", -1, vs, L, C, Hh * Cs);
    {
      size_t n = (size_t)L * Hh * Cs;
      // q_scalar *= sqrt(1 / num_scalar_qk)
      float s = sqrtf(1.f / Cs);
      CB(cublasSscal(H, (int)n, &s, qs, 1));
    }
    linearB(act, I + "q_point_projection/point_projection", -1, proj, L, C, Hh * 3 * Pq);
    pointsToGlobalK<<<blocks((size_t)L * Hh * Pq), 256, 0, STREAM>>>(proj, rig, qp, L, Hh, Pq);
    linearB(act, I + "k_point_projection/point_projection", -1, proj, L, C, Hh * 3 * Pq);
    pointsToGlobalK<<<blocks((size_t)L * Hh * Pq), 256, 0, STREAM>>>(proj, rig, kp, L, Hh, Pq);
    linearB(act, I + "v_point_projection/point_projection", -1, proj, L, C, Hh * 3 * Pv);
    pointsToGlobalK<<<blocks((size_t)L * Hh * Pv), 256, 0, STREAM>>>(proj, rig, vp, L, Hh, Pv);
    ipaWeightsK<<<(unsigned)(((size_t)L * Hh + 7) / 8), 256, 0, STREAM>>>(qs, ks, qp, kp, b2d, pw, seqMask, attn, L, Hh, Cs, Pq);
    ipaOutputsK<<<L * Hh, 128, 0, STREAM>>>(attn, vs, vp, act2d, rig, fin, L, Hh, Cs, Pv, C2);
    linearB(fin, I + "output_projection", -1, upd, L, Fw, C);
    addK2<<<blocks((size_t)L * C), 256, 0, STREAM>>>(act, upd, (size_t)L * C);
    layerNorm(act, t1, L, C, F + "attention_layer_norm");
    // the transition: three layers, ReLU between, a residual round them
    linearB(t1, F + "transition", -1, t2, L, C, C, true);
    linearB(t2, F + "transition_1", -1, upd, L, C, C, true);
    linearB(upd, F + "transition_2", -1, t2, L, C, C);
    addK2<<<blocks((size_t)L * C), 256, 0, STREAM>>>(t2, t1, (size_t)L * C);
    layerNorm(t2, act, L, C, F + "transition_layer_norm");
    linearB(act, F + "quat_rigid/rigid", -1, rq, L, C, 6);
    rigidUpdateK<<<blocks(L), 256, 0, STREAM>>>(rig, rq, L);
  }
  // the side chains, from the last iteration (the only one the outputs keep)
  const std::string R = F + "rigid_sidechain/";
  float* ra = scratch<float>("sm.ra", (size_t)L * C); float* sc = scratch<float>("sm.sc", (size_t)L * 128);
  float* sb = scratch<float>("sm.sb", (size_t)L * 128); float* sc2 = scratch<float>("sm.sc2", (size_t)L * 128);
  reluCopyK<<<blocks((size_t)L * C), 256, 0, STREAM>>>(act, ra, (size_t)L * C);
  linearB(ra, R + "input_projection", -1, sc, L, C, 128);
  reluCopyK<<<blocks((size_t)L * C), 256, 0, STREAM>>>(initial, ra, (size_t)L * C);
  linearB(ra, R + "input_projection_1", -1, sb, L, C, 128);
  addK2<<<blocks((size_t)L * 128), 256, 0, STREAM>>>(sc, sb, (size_t)L * 128);
  for (int r = 0; r < 2; ++r) {
    std::string suffix = r == 0 ? "" : "_1";
    reluCopyK<<<blocks((size_t)L * 128), 256, 0, STREAM>>>(sc, sb, (size_t)L * 128);
    linearB(sb, R + "resblock1" + suffix, -1, sc2, L, 128, 128, true);
    linearB(sc2, R + "resblock2" + suffix, -1, sb, L, 128, 128);
    addK2<<<blocks((size_t)L * 128), 256, 0, STREAM>>>(sc, sb, (size_t)L * 128);
  }
  reluCopyK<<<blocks((size_t)L * 128), 256, 0, STREAM>>>(sc, sb, (size_t)L * 128);
  float* un = scratch<float>("sm.un", (size_t)L * 14);
  linearB(sb, R + "unnormalized_angles", -1, un, L, 128, 14);
  StructureOut o{act, rig, dalloc((size_t)L * 37 * 3), dalloc((size_t)L * 14 * 3), dalloc((size_t)L * 14)};
  sidechainAtomsK<<<blocks(L), 128, 0, STREAM>>>(un, rig, positionScale, Idev("aatype"), W("c/rigid_group_default_frame"),
                                                 Idev("c/atom14_to_rigid_group"), W("c/atom14_rigid_group_positions"),
                                                 W("c/atom14_mask"), Idev("c/atom37_to_atom14"), W("c/atom37_mask"),
                                                 seqMask, o.angles, o.pos14, o.pos37, L);
  return o;
}
