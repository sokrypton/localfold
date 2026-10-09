// cuda/ef2/src/atoms.cuh's swaWindowK in 96-key tiles: its 144 (a 16-query block's whole +-64 window at once)
// take 37 KB of shared memory, past Apple's 32. The kernel already walks the window tile by tile with an online
// softmax, so a smaller tile is the same attention in more steps.
{
  constexpr int KT_ = 96;
  __shared__ float Ks[KT_][33];
  __shared__ float Vs[KT_][32];
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
  for (int t0 = klo; t0 <= khi; t0 += KT_) {
    int n = min(KT_, khi - t0 + 1);
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
