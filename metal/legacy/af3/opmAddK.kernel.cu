// cuda/af3/src/elementwise.cuh's opmAddK with its (i, j, channel) found without an integer division (lf_udiv, the
// runtime's prelude): Apple's GPUs have none, and the two an element (or three in 64 bits, as the port has them) made
// it 6.5 ms a call in an AF2 fold at 261 residues, 1.2 without them. The same arithmetic.
{
  const size_t total = (size_t)Bi * n * C;
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= total) return;
  size_t i, j; int f;
  if (total < (1u << 24)) {
    const unsigned u = (unsigned)t, ij = lf_udiv(u, (unsigned)C), ii = lf_udiv(ij, (unsigned)n);
    f = (int)(u - ij * (unsigned)C); i = i0 + ii; j = ij - ii * (unsigned)n;
  } else {
    f = (int)(t % C); size_t ij = t / C; i = i0 + ij / n; j = ij % n;
  }
  float nv = norm[i * n + j];
  float v = biasAfterNorm ? x[t] / fmaxf(nv, 1.f) + bias[f] : (bias[f] + x[t]) / (1e-3f + nv);
  size_t at = (i * n + j) * C + f;
  if constexpr (std::is_same_v<PT, float>) pair[at] += v;
  else reinterpret_cast<PT*>(pair)[at] = fromF<PT>(toF(reinterpret_cast<PT*>(pair)[at]) + v);
}
