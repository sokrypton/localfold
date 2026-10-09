// cuda/af2/src/evoformer.cuh's biasFromProjRowsK with its (i, j, head) found by lf_udiv, not size_t divisions an element
// (Apple's GPUs have no integer divider). The same arithmetic.
{
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= cnt * H) return;
  const unsigned u = (unsigned)t, q = lf_udiv(u, (unsigned)H), h = u - q * (unsigned)H;
  const unsigned ij = (unsigned)r0 + q;
  unsigned i = lf_udiv(ij, (unsigned)L), j = ij - i * (unsigned)L;
  float v = proj[headMajor ? (size_t)h * cnt + q : t] + (pairMask ? 1e9f * (pairMask[ij] - 1.f) : 0.f);
  if (transposed) { unsigned x = i; i = j; j = x; }
  out[((size_t)h * L + i) * stride + j] = __float2half(fmaxf(v * LOG2E, -6e4f));
}
