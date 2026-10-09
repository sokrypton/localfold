// cuda/af3/src/elementwise.cuh's gateMulAddHK with its row divided out in 32 bits where the tensor allows it: the
// size_t division an element is emulated on Apple's GPUs (8.5 ms a call at 261 tokens). The same arithmetic.
{
  const size_t n = pairs * (size_t)C;
  if (n < 0xffffffffull) {
    const unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const unsigned r = t / (unsigned)C, c = t - r * (unsigned)C;
    pair[t] += out[t] / (1.f + __expf(-__half2float(gate[(size_t)r * ld + c])));
    return;
  }
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  size_t r = t / C; int c = (int)(t % C);
  pair[t] += out[t] / (1.f + __expf(-__half2float(gate[r * ld + c])));
}
