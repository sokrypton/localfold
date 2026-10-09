// cuda/af3/src/flash.cuh's flashGridF32 (the precise path's grid attention) with 32-key tiles past a 32-wide head: its
// 64-key tiles at D 64 take 33.5 KB of shared memory, past Apple's 32. The key loop is already tiled: the same attention.
{
  constexpr int BQ = 64, BK = D > 32 ? 32 : 64;   // (Metal: 64-key tiles at D 64 are 33.5 KB, past 32)
  __shared__ float Ks[BK][D + 1], Vs[BK][D + 1], Ms[BK];
  size_t b = blockIdx.y; int h = (int)(b % heads); size_t rl = b / heads, r = r0 + rl;
  const int Wd = heads * D, W4 = 4 * Wd;
  const float* base = qkvg + rl * (size_t)n * W4 + h * D;
  int i = blockIdx.x * BQ + threadIdx.x;
  bool live = i < n;
  float q[D], o[D];
  for (int e = 0; e < D; ++e) { q[e] = live ? base[(size_t)i * W4 + e] * scale : 0.f; o[e] = 0.f; }
  float m = -INFINITY, l = 0.f;
  const float* brow = bias + ((size_t)h * n + (live ? i : 0)) * biasStride;
  for (int j0 = 0; j0 < n; j0 += BK) {
    __syncthreads();
    for (int t = threadIdx.x; t < BK * D; t += BQ) {
      int jj = t / D, e = t % D, j = j0 + jj;
      Ks[jj][e] = j < n ? base[(size_t)j * W4 + Wd + e] : 0.f;
      Vs[jj][e] = j < n ? base[(size_t)j * W4 + 2 * Wd + e] : 0.f;
    }
    if (threadIdx.x < BK) {
      int j = j0 + threadIdx.x;
      Ms[threadIdx.x] = j < n ? (mask[tr ? ((size_t)j * n + r) : (r * n + j)] > 0 ? 0.f : -1e9f) : -INFINITY;
    }
    __syncthreads();
    for (int jj = 0; jj < BK && j0 + jj < n; ++jj) {
      float dot = 0.f;
      for (int e = 0; e < D; ++e) dot += q[e] * Ks[jj][e];
      float v = dot + brow[j0 + jj] + Ms[jj];
      float mNew = fmaxf(m, v), c = expf(m - mNew), p = expf(v - mNew);
      l = l * c + p;
      for (int e = 0; e < D; ++e) o[e] = o[e] * c + p * Vs[jj][e];
      m = mNew;
    }
  }
  if (live) {
    const float* gp = base + (size_t)i * W4 + 3 * Wd;
    float* op = out + (rl * n + i) * Wd + h * D;
    for (int e = 0; e < D; ++e) op[e] = o[e] / l * (1.f / (1.f + expf(-gp[e])));
  }
}
