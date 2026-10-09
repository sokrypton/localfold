// cuda/af2/src/evoformer.cuh's biasFromProjK with its (i, j, head) found by lf_udiv, not size_t divisions an element
// (Apple's GPUs have no integer divider). The same arithmetic.
{
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)L * L * H) return;
  const unsigned u = (unsigned)t, ij = lf_udiv(u, (unsigned)H), h = u - ij * (unsigned)H;
  unsigned i = lf_udiv(ij, (unsigned)L), j = ij - i * (unsigned)L;
  float v = proj[t] + (pairMask ? 1e9f * (pairMask[ij] - 1.f) : 0.f);
  if (transposed) { unsigned x = i; i = j; j = x; }
  out[((size_t)h * L + i) * stride + j] = __float2half(fmaxf(v * LOG2E, -6e4f));
}
