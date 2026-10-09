// cuda/ef2/src/fast.cuh's swigluHK with 32-bit index math and four channels a thread: its size_t division an element
// is emulated on Apple's GPUs, and was 17 ms a call at 261 tokens (3.2 s of a 19 s trunk). The same arithmetic.
{
  const unsigned q = (unsigned)I / 4u;                               // (I a multiple of 4: ESMFold2's 1024)
  const unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
  if ((I & 3) == 0) {
    if ((size_t)t * 4 >= rows * (size_t)I) return;
    const unsigned r = t / q, c = (t - r * q) * 4u;
    const half* row = h + (size_t)r * 2u * (unsigned)I;
    for (int k = 0; k < 4; ++k) {
      float x = __half2float(row[c + k]);
      g[(size_t)t * 4 + k] = __float2half(x / (1.f + __expf(-x)) * __half2float(row[I + c + k]));
    }
    return;
  }
  for (size_t u = t; u < rows * (size_t)I; u += (size_t)gridDim.x * blockDim.x) {
    size_t r = u / I; int c = (int)(u % I);
    float x = __half2float(h[r * 2 * I + c]);
    g[u] = __float2half(x / (1.f + __expf(-x)) * __half2float(h[r * 2 * I + I + c]));
  }
}
