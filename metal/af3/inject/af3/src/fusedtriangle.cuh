// out[h][row] = (LN(x[row]) W)[h] - fusedtriangle.cuh's lnHeadsK (tensor cores, 49 KB of shared memory) as one
// simdgroup a row: LN as lnRowsToShared computes it (eps 1e-5, rounded to half as it stages it), then the N dot
// products of the row with W's columns, each a simdgroup sum.
template <int C, int N, class PT>
__global__ void lnHeadsMetal(const float* x, const float* lnScale, const float* lnOffset, const half* Wp, float* out,
                             size_t rows) {
  constexpr int K = C / 32;
  const int lane = threadIdx.x & 31;
  const size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (row >= rows) return;
  float v[K];
  float s = 0.f;
  for (int k = 0; k < K; ++k) { v[k] = (float)reinterpret_cast<const PT*>(x)[row * C + lane + 32 * k]; s += v[k]; }
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
  float mean = s / C, q = 0.f;
  for (int k = 0; k < K; ++k) { float d = v[k] - mean; q += d * d; }
  for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(~0u, q, o);
  float inv = rsqrtf(q / C + 1e-5f);
  for (int k = 0; k < K; ++k) v[k] = (float)__float2half((v[k] - mean) * inv * lnScale[lane + 32 * k] + lnOffset[lane + 32 * k]);
  for (int h = 0; h < N; ++h) {
    float a = 0.f;
    for (int k = 0; k < K; ++k) a += v[k] * (float)Wp[(size_t)(lane + 32 * k) * N + h];
    for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(~0u, a, o);
    if (lane == 0) out[(size_t)h * rows + row] = a;
  }
}
