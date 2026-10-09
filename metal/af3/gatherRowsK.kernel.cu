// cuda/af3/src/atom.cuh's gatherRowsK with its (row, chunk, slot, copy) found by lf_udiv where it is exact, not four
// size_t divisions an element (Apple's GPUs have no integer divider). The same copy.
{
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * chunks) return;
  size_t r, g, k; int c;
  if (rows * chunks < (1u << 24)) {
    const unsigned rr = lf_udiv((unsigned)t, (unsigned)chunks), kk = lf_udiv(rr, (unsigned)count);
    c = (int)((unsigned)t - rr * (unsigned)chunks); r = rr; k = kk; g = rr - kk * (unsigned)count;
  } else {
    r = t / chunks; c = (int)(t % chunks); g = r % count; k = r / count;
  }
  uint4 v = mask[g] != 0 ? reinterpret_cast<const uint4*>(in)[((size_t)idx[g] + k * srcRows) * chunks + c]
                         : make_uint4(0, 0, 0, 0);
  reinterpret_cast<uint4*>(out)[t] = v;
}
