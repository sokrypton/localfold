// cuda/af3/src/atom.cuh's atomAttentionFusedK with K and V staged in half: the CUDA kernel's float tiles are
// 38,912 bytes at 128 keys and 8 warps, past Apple's 32 KB of threadgroup memory; half tiles are 22 KB.
// (the host's smem request is the float layout's; this body lays out its own within the cap)
{
  int s = blockIdx.x, ss = s % subsets, h = blockIdx.y, warp = threadIdx.x / 32, lane = threadIdx.x & 31;
  int nw = blockDim.x / 32, Wd = heads * D, W2 = 2 * Wd;
  extern __shared__ float sm[];
  half* Ks = (half*)sm; half* Vs = Ks + keys * (D + 1);
  float* P = sm + (keys * (D + 1) + 3) / 4 * 4;     // (the two half tiles are keys * (D + 1) floats' worth)
  float* Q = P + nw * keys;
  for (int t = threadIdx.x; t < keys * D; t += blockDim.x) {
    int key = t / D, e = t % D; size_t row = (size_t)s * keys + key;
    Ks[key * (D + 1) + e] = (half)toF(kv[row * W2 + h * D + e]);
    Vs[key * (D + 1) + e] = (half)toF(kv[row * W2 + Wd + h * D + e]);
  }
  __syncthreads();
  float scale = 1.f / sqrtf((float)D);
  for (int qi = warp; qi < queries; qi += nw) {
    size_t qrow = (size_t)s * queries + qi;
    float* Qw = Q + warp * D;
    for (int e = lane; e < D; e += 32) Qw[e] = toF(qg[qrow * W2 + h * D + e]) + qBias[h * D + e];
    __syncwarp();
    float* Pw = P + warp * keys;
    float mx = -INFINITY;
    for (int key = lane; key < keys; key += 32) {
      float dot = 0;
      for (int e = 0; e < D; ++e) dot += Qw[e] * (float)Ks[key * (D + 1) + e];
      size_t krow = (size_t)ss * keys + key, qm = (size_t)ss * queries + qi;
      float maskBias = keyMasked ? -1e9f * ((1.f - qMask[qm]) + (1.f - kMask[krow]))
                                 : 1e9f * (qMask[qm] - 1.f) * (kMask[krow] - 1.f);
      float l = dot * scale + maskBias + pairLogits[(((size_t)ss * heads + h) * queries + qi) * keys + key];
      Pw[key] = l; mx = fmaxf(mx, l);
    }
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, o));
    float sum = 0;
    for (int key = lane; key < keys; key += 32) { float e = expf(Pw[key] - mx); Pw[key] = e; sum += e; }
    for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o);
    __syncwarp();
    for (int e = lane; e < D; e += 32) {
      float acc = 0;
      for (int key = 0; key < keys; ++key) acc += Pw[key] * (float)Vs[key * (D + 1) + e];
      out[qrow * Wd + h * D + e] = fromF<T>(acc / sum * sigm(toF(qg[qrow * W2 + Wd + h * D + e])));
    }
    __syncwarp();
  }
}
