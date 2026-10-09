// cuda/af3/src/pairtrack.cuh's centerNormStreamK with the variance in TWO passes (the mean, then the centred sum of
// squares): the CUDA kernel's one pass, E[x^2] - E[x]^2, can cancel where a triangle product's mean is large against
// its spread. (Written for a protenix2 residual that turned out to be WebGPU's f16 matrix kernels, not this: the f32
// trunk agrees with stock-flag WebGPU to 8.4e-7. Kept as the safer form.)
{
  __shared__ float ps[8][33], mean[32], inv[32];
  size_t i0 = (size_t)blockIdx.x * 32; int tx = threadIdx.x, ty = threadIdx.y;
  size_t local = i0 + tx;
  unsigned p = (unsigned)(r0 + local), i = p / (unsigned)n;
  size_t at = (size_t)i * np + (p - i * (unsigned)n);
  float s = 0;
  if (local < rows) for (int c = ty; c < C; c += 8) s += prod[(size_t)c * pairs + at];
  ps[ty][tx] = s;
  __syncthreads();
  if (ty == 0) { float a = 0; for (int k = 0; k < 8; ++k) a += ps[k][tx]; mean[tx] = a / C; }
  __syncthreads();
  float m = mean[tx], ss = 0;
  if (local < rows) for (int c = ty; c < C; c += 8) { float d = prod[(size_t)c * pairs + at] - m; ss += d * d; }
  __syncthreads();
  ps[ty][tx] = ss;
  __syncthreads();
  if (ty == 0) { float b = 0; for (int k = 0; k < 8; ++k) b += ps[k][tx]; inv[tx] = rsqrtf(b / C + 1e-5f); }
  __syncthreads();
  for (int ry = ty; ry < 32; ry += 8) {
    size_t lr = i0 + ry; if (lr >= rows) continue;
    unsigned q = (unsigned)(r0 + lr), qi = q / (unsigned)n;
    size_t qa = (size_t)qi * np + (q - qi * (unsigned)n);
    for (int c = tx; c < C; c += 32)
      out[lr * C + c] = fromF<TO>((prod[(size_t)c * pairs + qa] - mean[ry]) * inv[ry] * scale[c] + offset[c]);
  }
}
