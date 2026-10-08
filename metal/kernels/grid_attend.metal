// AF3's grid attention (`grid.attend`): for one row of the pair and one head,
// every query position i attends over every key position j of that row.
// The tensors are laid out as webgpu/af3/trunk/grid-attention.js lays them out,
// so the two backends can be held to each other:
//
//   q, k, v, out   ((row * N + j) * H + h) * D + d      half
//   bias           (h * N + i) * N + j                  float  (shared by every row)
//   mask           row * N + j                          float  (the KEY's mask)
//
// Measured on an M2 at 256 tokens (metal/check/check-kernels): 4.3 ms a call
// against WebGPU's 12.2 for the same work, ~2.0 TFLOP/s.
//
// HOW. One threadgroup is (32 queries, row, head): four simdgroups of eight
// queries each. Keys arrive 16 at a time through threadgroup memory, and both
// products - S = Q K^T and O += P V - run on the 8x8 simdgroup matrix units.
// The online softmax never leaves registers: `thread_elements()` hands each lane
// two adjacent columns of one row of an 8x8 tile, and the four lanes that share
// a row differ in lane bits 0 and 3, so a row's max and sum are two shuffles.
//
// 🔴 EVERY TILE LOOP IS FULLY UNROLLED, AND THAT IS NOT A STYLE CHOICE. Without
// it the arrays of simdgroup matrices are indexed by a loop variable, go to
// stack memory, and the kernel runs 5-10x SLOWER than the scalar one - correct
// and catastrophically slow, which reads as "the matrix units do not pay".
//
// Measured and NOT taken (M2, 256 tokens): a half-precision bias (flat), double
// buffering the key tile (7% slower - the second buffer costs occupancy), two or
// three row tiles a simdgroup and 32-key tiles (4-20% slower, registers again),
// and the softmax through threadgroup memory instead of registers (2x slower).
// Pre-transposing K while staging it and working in base 2 are worth ~5%.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
#define UNROLL _Pragma("clang loop unroll(full)")

struct GridAttendShape { uint N; uint H; float scale; };

constant float LOG2E = 1.4426950408889634f;

// The 1e9 penalty is the WebGPU kernel's, subtracted from the logit rather than
// replacing it; see the note on the mask in webgpu/af3/trunk/grid-attention.js.
constant float MASKED = 1.0e9f;

template <uint D, uint SG, uint BK>
void grid_attend_impl(device const half* q, device const half* k, device const half* v,
                      device const float* bias, device const float* mask, device half* out,
                      constant GridAttendShape& s, uint3 group, uint lid, uint sg, uint lane,
                      threadgroup half* Qs, threadgroup half* Ks, threadgroup half* Vs) {
  const uint N = s.N, H = s.H, QB = 8 * SG, T = 32 * SG;
  const uint row = group.y, h = group.z, i0 = group.x * QB;
  // Base 2 throughout: log2(e) is folded into the scale and the bias once, and
  // every exponential is an exp2.
  const float sc = s.scale * LOG2E;

  for (uint idx = lid; idx < QB * D; idx += T) {
    uint r = idx / D, d = idx % D, i = i0 + r;
    Qs[idx] = i < N ? q[((row * N + i) * H + h) * D + d] : half(0);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_half8x8 Qm[D / 8];
  simdgroup_float8x8 O[D / 8];
  UNROLL for (uint dt = 0; dt < D / 8; ++dt) {
    simdgroup_load(Qm[dt], Qs + sg * 8 * D, D, ulong2(dt * 8, 0));
    O[dt] = simdgroup_float8x8(0);
  }
  // This lane's place in every 8x8 tile: row sm, columns sn and sn + 1.
  const uint sm = (lane / 16) * 4 + (lane % 8) / 2;
  const uint sn = ((lane / 8) % 2) * 4 + (lane % 2) * 2;
  const uint i = i0 + sg * 8 + sm;
  const uint ib = i < N ? i : 0;
  float m = -3.0e38f, l = 0;

  for (uint j0 = 0; j0 < N; j0 += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // K is staged TRANSPOSED ([d][key]) so S = Q K^T loads it without a
    // transposing simdgroup_load; V stays [key][d]. A key past N is clamped to
    // the last real one and its probability zeroed below.
    for (uint idx = lid * 4; idx < BK * D; idx += T * 4) {
      uint jj = idx / D, d = idx % D, j = min(j0 + jj, N - 1);
      uint src = ((row * N + j) * H + h) * D + d;
      half4 kv = *(device const half4*)(k + src);
      Ks[(d + 0) * BK + jj] = kv.x; Ks[(d + 1) * BK + jj] = kv.y;
      Ks[(d + 2) * BK + jj] = kv.z; Ks[(d + 3) * BK + jj] = kv.w;
      *(threadgroup half4*)(Vs + idx) = *(device const half4*)(v + src);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 S[BK / 8];
    float tmax = -3.0e38f;
    UNROLL for (uint kt = 0; kt < BK / 8; ++kt) {
      S[kt] = simdgroup_float8x8(0);
      UNROLL for (uint dt = 0; dt < D / 8; ++dt) {
        simdgroup_half8x8 Kt;
        simdgroup_load(Kt, Ks, BK, ulong2(kt * 8, dt * 8));
        simdgroup_multiply_accumulate(S[kt], Qm[dt], Kt, S[kt]);
      }
      thread auto& e = S[kt].thread_elements();
      uint j = j0 + kt * 8 + sn;
      if (j + 1 < N) {
        float2 b = *(device const float2*)(bias + (h * N + ib) * N + j);
        float2 mk = *(device const float2*)(mask + row * N + j);
        e[0] = e[0] * sc + b.x * LOG2E - (mk.x <= 0 ? MASKED : 0.0f);
        e[1] = e[1] * sc + b.y * LOG2E - (mk.y <= 0 ? MASKED : 0.0f);
      } else {
        e[0] = j < N ? e[0] * sc + bias[(h * N + ib) * N + j] * LOG2E - (mask[row * N + j] <= 0 ? MASKED : 0.0f)
                     : -3.0e38f;
        e[1] = -3.0e38f;
      }
      tmax = max(tmax, max(e[0], e[1]));
    }
    tmax = max(tmax, simd_shuffle_xor(tmax, 1));
    tmax = max(tmax, simd_shuffle_xor(tmax, 8));
    float nm = max(m, tmax), alpha = exp2(m - nm), sum = 0;
    simdgroup_half8x8 P[BK / 8];
    UNROLL for (uint kt = 0; kt < BK / 8; ++kt) {
      thread auto& e = S[kt].thread_elements();
      // A key past N carries -3e38, so its exp2 underflows to exactly zero.
      float p0 = exp2(e[0] - nm), p1 = exp2(e[1] - nm);
      sum += p0 + p1;
      thread auto& pe = P[kt].thread_elements();
      pe[0] = half(p0); pe[1] = half(p1);
    }
    sum += simd_shuffle_xor(sum, 1);
    sum += simd_shuffle_xor(sum, 8);
    l = l * alpha + sum; m = nm;
    // O's elements sit on the same rows as S's, so the rescale is per lane.
    UNROLL for (uint dt = 0; dt < D / 8; ++dt) {
      thread auto& oe = O[dt].thread_elements();
      oe[0] *= alpha; oe[1] *= alpha;
    }
    UNROLL for (uint kt = 0; kt < BK / 8; ++kt) UNROLL for (uint dt = 0; dt < D / 8; ++dt) {
      simdgroup_half8x8 Vm;
      simdgroup_load(Vm, Vs, D, ulong2(dt * 8, kt * 8));
      simdgroup_multiply_accumulate(O[dt], P[kt], Vm, O[dt]);
    }
  }
  if (i >= N) return;
  float inv = 1.0f / l;
  UNROLL for (uint dt = 0; dt < D / 8; ++dt) {
    thread auto& oe = O[dt].thread_elements();
    *(device half2*)(out + ((row * N + i) * H + h) * D + dt * 8 + sn) = half2(oe[0] * inv, oe[1] * inv);
  }
}

// Dispatch: threadgroups (ceil(N / 32), N, H) of 128 threads.
kernel void grid_attend_d32(device const half* q [[buffer(0)]], device const half* k [[buffer(1)]],
                            device const half* v [[buffer(2)]], device const float* bias [[buffer(3)]],
                            device const float* mask [[buffer(4)]], device half* out [[buffer(5)]],
                            constant GridAttendShape& s [[buffer(6)]],
                            uint3 group [[threadgroup_position_in_grid]],
                            uint lid [[thread_index_in_threadgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
  threadgroup half Qs[8 * 4 * 32];
  threadgroup half Ks[16 * 32];
  threadgroup half Vs[16 * 32];
  grid_attend_impl<32, 4, 16>(q, k, v, bias, mask, out, s, group, lid, sg, lane, Qs, Ks, Vs);
}
