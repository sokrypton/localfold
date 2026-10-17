// gated = SwiGLU(xn W1) in one kernel: a GEMM whose tile holds a block of the gate's columns and the same block of the
// value's, the gate applied in registers and only the gated f16 rows stored - the [rows][2I] widening never reaches
// memory. Shaped as cuBLAS's own kernel for this GEMM (Nsight Compute: 128 x 128 x 32, 4 warps of 64 x 64, 5 stages,
// 2 blocks an SM, the tensor pipe 72% busy): 4 warps of 64 rows x (32 gate + 32 value) columns, K in steps of 32 through
// a 5-stage cp.async pipeline, the stages XOR-swizzled rather than padded (80 KB a block), the next k step's fragments
// loaded while this one's MMAs issue.
// xn [rows][C] f16 (C a multiple of 8; the K tail zero-filled), W1i: W1 [C][2I] re-laid by swigluInterleave - each
// 64-column block of the gate followed by the same block of the value - and gated [rows][I] f16 (I a multiple of 64)
#pragma once

constexpr int SG_BM = 128, SG_BN = 64, SG_KT = 32, SG_STAGES = 5, SG_NTH = 128;
constexpr int SG_ACH = SG_KT / 8, SG_BCH = 2 * SG_BN / 8;          // 16-byte chunks a row: A 4, B 16
constexpr size_t SG_STAGE = (size_t)(SG_BM * SG_KT + SG_KT * 2 * SG_BN) * 2;
constexpr size_t SG_SMEM = SG_STAGES * SG_STAGE;
// A tile [128][32]: 4 chunks a row, chunk ^ ((row >> 1) & 3) - eight ldmatrix rows at one chunk land in 8 banks groups
__device__ __forceinline__ int sgA(int r, int c) { return r * SG_KT + ((((c >> 3) ^ (r >> 1)) & 3) << 3) + (c & 7); }
// B tile [32][128]: 16 chunks a row, chunk ^ (row & 7)
__device__ __forceinline__ int sgB(int r, int c) { return r * 2 * SG_BN + ((((c >> 3) ^ r) & 7) | ((c >> 3) & 8)) * 8 + (c & 7); }

__global__ void __launch_bounds__(SG_NTH, 2) swigluUpK(const half* __restrict__ xn, const half* __restrict__ W1i,
                                                        half* __restrict__ gated, int rows, int C, int I) {
  extern __shared__ __align__(16) unsigned char smem[];
  auto As = [&](int s) { return (half*)(smem + s * SG_STAGE); };
  auto Bs = [&](int s) { return As(s) + SG_BM * SG_KT; };
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wm = warp & 1, wn = warp >> 1;
  const int row0 = blockIdx.y * SG_BM, n0 = blockIdx.x * SG_BN;        // n0: the gate's column (W1i's block 2 n0)
  const int tiles = (C + SG_KT - 1) / SG_KT;
  // every per-thread offset once: the copies' sources and swizzled destinations, the fragments' swizzled addresses -
  // recomputed inside the loop they were most of the kernel's instructions (5.5 a tensor-core one)
  const int AIT = SG_BM * SG_ACH / SG_NTH, BIT = SG_KT * SG_BCH / SG_NTH;
  int aDst[AIT], bDst[BIT], aCol[AIT], bRow[BIT]; const half* aSrc[AIT]; bool aRow[AIT];
#pragma unroll
  for (int it = 0; it < AIT; ++it) {
    const int q = it * SG_NTH + threadIdx.x, r = q / SG_ACH, c = (q % SG_ACH) * 8;
    aDst[it] = sgA(r, c); aCol[it] = c; aRow[it] = row0 + r < rows; aSrc[it] = xn + (size_t)(aRow[it] ? row0 + r : 0) * C + c;
  }
  const half* bSrc0 = W1i + (size_t)2 * n0;
  int bOffs[BIT];
#pragma unroll
  for (int it = 0; it < BIT; ++it) {
    const int q = it * SG_NTH + threadIdx.x, r = q / SG_BCH, c = (q % SG_BCH) * 8;
    bDst[it] = sgB(r, c); bRow[it] = r; bOffs[it] = r * 2 * I + c;
  }
  auto issue = [&](int t, int st) {
    const int k0 = t * SG_KT;
    half *a = As(st), *b = Bs(st);
    const bool tail = k0 + SG_KT > C;
#pragma unroll
    for (int it = 0; it < AIT; ++it) {
      const bool ok = aRow[it] && (!tail || k0 + aCol[it] < C);
      cpAsync16(a + aDst[it], ok ? aSrc[it] + k0 : xn, ok);
    }
    const half* bk = bSrc0 + (size_t)k0 * 2 * I;
#pragma unroll
    for (int it = 0; it < BIT; ++it) {
      const bool ok = !tail || k0 + bRow[it] < C;
      cpAsync16(b + bDst[it], ok ? bk + bOffs[it] : W1i, ok);
    }
    cpCommit();
  };
  // a warp: rows wm 64 .. +64 (4 m16 tiles), the gate's columns wn 32 .. +32 (4 n8 tiles) and the value's the same
  float acc[4][4][2][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int v = 0; v < 2; ++v)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[i][j][v][e] = 0.f;
#pragma unroll
  for (int t = 0; t < SG_STAGES - 1; ++t) { if (t < tiles) issue(t, t); else cpCommit(); }
  int aFr[2][4], bFr[2][2][2];                                           // [ks][m16]; [ks][gate/value][n16], halves
#pragma unroll
  for (int ks = 0; ks < 2; ++ks) {
#pragma unroll
    for (int mt = 0; mt < 4; ++mt) aFr[ks][mt] = sgA(wm * 64 + mt * 16 + (lane & 15), ks * 16 + (lane >> 4) * 8);
    const int k = ks * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
#pragma unroll
    for (int v = 0; v < 2; ++v)
#pragma unroll
      for (int n2 = 0; n2 < 2; ++n2) bFr[ks][v][n2] = SG_BM * SG_KT + sgB(k, v * SG_BN + wn * 32 + n2 * 16 + (lane >> 4) * 8);
  }
  uint32_t fa[2][4][4], fb[2][2][2][4];                                  // [buf][m16][4]; [buf][gate/value][n16][4]
  auto load = [&](const half* stage, int ks, int buf) {
#pragma unroll
    for (int mt = 0; mt < 4; ++mt) ldsm4(fa[buf][mt], stage + aFr[ks][mt]);
#pragma unroll
    for (int v = 0; v < 2; ++v)
#pragma unroll
      for (int n2 = 0; n2 < 2; ++n2) ldsm4t(fb[buf][v][n2], stage + bFr[ks][v][n2]);
  };
  int st = 0, nst = SG_STAGES - 1;
  for (int t = 0; t < tiles; ++t) {
    cpWait<SG_STAGES - 2>();
    __syncthreads();
    if (t + SG_STAGES - 1 < tiles) issue(t + SG_STAGES - 1, nst); else cpCommit();
    nst = nst + 1 == SG_STAGES ? 0 : nst + 1;
    const half* stage = As(st);
    st = st + 1 == SG_STAGES ? 0 : st + 1;
    load(stage, 0, 0);
#pragma unroll
    for (int ks = 0; ks < SG_KT / 16; ++ks) {
      const int cur = ks & 1;
      if (ks + 1 < SG_KT / 16) load(stage, ks + 1, cur ^ 1);
#pragma unroll
      for (int mt = 0; mt < 4; ++mt)
#pragma unroll
        for (int v = 0; v < 2; ++v)
#pragma unroll
          for (int n2 = 0; n2 < 2; ++n2) {
            mma16816(acc[mt][2 * n2][v], fa[cur][mt], fb[cur][v][n2][0], fb[cur][v][n2][1]);
            mma16816(acc[mt][2 * n2 + 1][v], fa[cur][mt], fb[cur][v][n2][2], fb[cur][v][n2][3]);
          }
    }
  }
  cpWait<0>();
  __syncthreads();                                      // (the stages are free: the gated tile goes through them)
  constexpr int LDO = SG_BN + 8;
  half* out = (half*)smem;
  const int g = lane >> 2, tig = lane & 3;
#pragma unroll
  for (int mt = 0; mt < 4; ++mt)
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int r = wm * 64 + mt * 16 + g, c = wn * 32 + nt * 8 + tig * 2;
      auto gate = [&](int e) { float x = acc[mt][nt][0][e]; return x * sigmH(x) * acc[mt][nt][1][e]; };   // (sigmH: tanh.approx, as cuda/ef2's transitions)
      *reinterpret_cast<uint32_t*>(out + r * LDO + c) = pack2(gate(0), gate(1));
      *reinterpret_cast<uint32_t*>(out + (r + 8) * LDO + c) = pack2(gate(2), gate(3));
    }
  __syncthreads();
#pragma unroll
  for (int it = 0; it < SG_BM * SG_BN / 8 / SG_NTH; ++it) {
    const int q = it * SG_NTH + threadIdx.x, r = q / (SG_BN / 8), c = (q % (SG_BN / 8)) * 8;
    if (row0 + r < rows) *reinterpret_cast<uint4*>(gated + (size_t)(row0 + r) * I + n0 + c) = *reinterpret_cast<const uint4*>(out + r * LDO + c);
  }
}
// W1 [C][2I] (the gate's I columns, then the value's) -> [C][2I] in 64-column blocks: gate block j, value block j, ...
__global__ void swigluInterleaveK(const half* W1, half* out, int C, int I) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)C * 2 * I) return;
  const int k = (int)(t / (2 * I)), col = (int)(t % (2 * I)), blk = col / (2 * SG_BN), w = col % (2 * SG_BN);
  out[t] = W1[(size_t)k * 2 * I + (w < SG_BN ? 0 : I) + blk * SG_BN + (w % SG_BN)];
}
inline void swigluUp(const half* xn, const half* W1i, half* g, size_t rows, int C, int I) {
  static bool attr = false;
  if (!attr) { smemAttr(swigluUpK, (int)SG_SMEM); attr = true; }
  swigluUpK<<<dim3(I / SG_BN, (unsigned)((rows + SG_BM - 1) / SG_BM)), SG_NTH, SG_SMEM, STREAM>>>(xn, W1i, g, (int)rows, C, I);
}
