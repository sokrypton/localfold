// cuda/af3/src/elementwise.cuh's centerNormHK. Where C allows it (a multiple of 8, at most 256 channels): a lane a pair,
// eight rows of lanes splitting the channels - the channel-major product read coalesced, every value held in registers,
// the mean and the (two-pass, as the original) variance reduced through 2 KB - and the rows written coalesced through a
// half tile. It was ~30 GB/s (11 ms a call at 261 tokens in ESMFold2) with half its lanes idle in the loads. Otherwise the
// block's 32 rows in two halves of 16 through a [C][17] tile: the CUDA kernel's [C][33] float tile is 33 KB at C 256,
// past Apple's 32. The same arithmetic, summed in another order.
{
  extern __shared__ float tile[];
  if (C % 8 == 0 && C <= 256 && blockDim.x == 256) {
    const int tid = threadIdx.x, lane = tid & 31, grp = tid >> 5, nk = C / 8;
    float* part = tile;                                  // [8][32]
    half* T = (half*)(tile + 256);                       // [32][C + 8]
    const int ld = C + 8;
    const size_t r0 = (size_t)blockIdx.x * 32;
    const unsigned rr = (unsigned)(r0 + lane), ii = rr / (unsigned)L;
    const size_t q = (size_t)ii * Lp + (rr - ii * (unsigned)L), plane = (size_t)Lp * Lp;
    const bool live = r0 + lane < pairs;
    float v[32];
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < 32; ++k) if (k < nk) { v[k] = live ? prod[(size_t)(grp + 8 * k) * plane + q] : 0.f; s += v[k]; }
    part[grp * 32 + lane] = s;
    __syncthreads();
    float S = 0.f;
    for (int g = 0; g < 8; ++g) S += part[g * 32 + lane];
    const float mean = S / C;
    float d2 = 0.f;
#pragma unroll
    for (int k = 0; k < 32; ++k) if (k < nk) { float d = v[k] - mean; d2 += d * d; }
    __syncthreads();
    part[grp * 32 + lane] = d2;
    __syncthreads();
    float V = 0.f;
    for (int g = 0; g < 8; ++g) V += part[g * 32 + lane];
    const float inv = rsqrtf(V / C + 1e-5f);
#pragma unroll
    for (int k = 0; k < 32; ++k) if (k < nk) {
      int c = grp + 8 * k;
      T[lane * ld + c] = __float2half((v[k] - mean) * inv * scale[c] + offset[c]);
    }
    __syncthreads();
    for (int row = grp; row < 32; row += 8) {
      size_t r = r0 + row;
      if (r >= pairs) break;
      for (int c = lane; c < C; c += 32) out[r * C + c] = T[row * ld + c];
    }
    return;
  }
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
