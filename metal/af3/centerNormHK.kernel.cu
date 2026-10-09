// cuda/af3/src/elementwise.cuh's centerNormHK, the block's 32 rows in two halves of 16 through a [C][17] tile: the
// CUDA kernel's [C][33] float tile is 33 KB at C 256 (ESMFold2's, AF2's), past Apple's 32. The same arithmetic.
{
  extern __shared__ float tile[];               // [C][17]
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  for (int half_ = 0; half_ < 2; ++half_) {
    size_t r0 = (size_t)blockIdx.x * 32 + half_ * 16;
    __syncthreads();
    // (a pair's place in the padded plane divided out once, in 32 bits - not a size_t division a channel, which Apple's
    // GPUs emulate)
    const unsigned rr = (unsigned)(r0 + lane), ii = rr / (unsigned)L;
    const size_t q = (size_t)ii * Lp + (rr - ii * (unsigned)L), plane = (size_t)Lp * Lp;
    const bool live = lane < 16 && r0 + lane < pairs;
    for (int c = warp; c < C; c += nw) {
      if (lane < 16) tile[c * 17 + lane] = live ? prod[(size_t)c * plane + q] : 0.f;
    }
    __syncthreads();
    for (int row = warp; row < 16; row += nw) {
      size_t r = r0 + row;
      if (r >= pairs) break;
      float s = 0;
      for (int c = lane; c < C; c += 32) s += tile[c * 17 + row];
      for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
      float mean = s / C, v = 0;
      for (int c = lane; c < C; c += 32) { float d = tile[c * 17 + row] - mean; v += d * d; }
      for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
      float inv = rsqrtf(v / C + 1e-5f);
      for (int c = lane; c < C; c += 32) out[r * C + c] = __float2half((tile[c * 17 + row] - mean) * inv * scale[c] + offset[c]);
    }
  }
}
