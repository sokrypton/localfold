// The strided grid attention - rows and positions at arbitrary strides, so an attention ACROSS a tensor's
// leading axis (the MSA's columns, the triangle's ending node) reads the tensor where it lies instead of a
// transposed copy - is cuda/af3's flash kernel in its STRIDED form (flash.cuh: flashGridHalf); this file
// is its launchers. (It was a copy of that kernel until the copy had to take every change twice.)
#pragma once
#include "../../af3/src/flash.cuh"

template <int D, int WARPS, bool MASKED, bool REG, int BK = FA_BK>
void flashStridedRun(const half* qkvg, const half* bias, int stride, const float* mask, half* out, int n, int heads,
                     size_t rows, float scale, size_t rowStride, size_t posStride, size_t outRowStride, size_t outPosStride) {
  const int bytes = (REG ? 1 : 2) * faStage<D, WARPS, BK>();
  static bool attr = false;
  if (!attr) {
    smemAttr((flashGridHalf<D, WARPS, MASKED, BK, REG, 1, true>), bytes);
    attr = true;
  }
  dim3 grid((n + 16 * WARPS - 1) / (16 * WARPS), (unsigned)(rows * heads));
  flashGridHalf<D, WARPS, MASKED, BK, REG, 1, true><<<grid, 32 * WARPS, bytes, STREAM>>>(
    qkvg, bias, stride, mask, out, n, heads, 0, false, scale, nullptr, rowStride, posStride, outRowStride, outPosStride);
}
template <int D, int WARPS, bool MASKED>
void flashStridedAt(const half* qkvg, const half* bias, int stride, const float* mask, half* out, int n, int heads,
                    size_t rows, float scale, size_t rowStride, size_t posStride, size_t outRowStride, size_t outPosStride) {
  if (flashRegStaged()) {
    // a T4: flashGrid2R's one-stage form where it applies (unmasked, 32 wide; a null bias - the column attention -
    // loads no bias tile)
    if constexpr (D == 32 && !MASKED) {
      if (!bias ? flash2R1Strided<true>(qkvg, bias, stride, out, n, heads, rows, scale, nullptr, rowStride, posStride, outRowStride, outPosStride)
                : flash2R1Strided<false>(qkvg, bias, stride, out, n, heads, rows, scale, nullptr, rowStride, posStride, outRowStride, outPosStride))
        return;
    }
    flashStridedRun<D, WARPS, MASKED, true>(qkvg, bias, stride, mask, out, n, heads, rows, scale, rowStride, posStride, outRowStride, outPosStride);
  } else if constexpr (D == 32 && !MASKED) {
    // unmasked at 32 wide: the pair track's grid kernel (two tiles a warp, f16 scores, two rows a block)
    // at these strides - the MSA's column attention and the triangle's ending node
    // (a null bias - the column attention has none - takes the form that loads no bias tile)
    if (FLASH_2R && !bias) flashGrid2RRun<D, 2, 48, 2, 2, true>(qkvg, bias, stride, out, n, heads, rows, scale, nullptr, rowStride,
                                                               posStride, outRowStride, outPosStride);
    else if (FLASH_2R) flashGrid2RRun<D, 2, 48, 2, 2>(qkvg, bias, stride, out, n, heads, rows, scale, nullptr, rowStride, posStride,
                                                      outRowStride, outPosStride);
    else flashStridedRun<D, WARPS, MASKED, false, 48>(qkvg, bias, stride, mask, out, n, heads, rows, scale, rowStride, posStride, outRowStride, outPosStride);
  } else if constexpr (D == 32)     // (48-key tiles where cp.async double-buffers: flashGridHalfLaunch's rule)
    flashStridedRun<D, WARPS, MASKED, false, 48>(qkvg, bias, stride, mask, out, n, heads, rows, scale, rowStride, posStride, outRowStride, outPosStride);
  else
    flashStridedRun<D, WARPS, MASKED, false>(qkvg, bias, stride, mask, out, n, heads, rows, scale, rowStride, posStride, outRowStride, outPosStride);
}
// rows x (n positions) of a [.., 4W] qkvg at the given strides (elements); out likewise ([.., W])
inline void flashGridStrided(const half* qkvg, const half* bias, int stride, const float* mask, half* out, int n, int heads,
                             int D, size_t rows, float scale, size_t rowStride, size_t posStride, size_t outRowStride,
                             size_t outPosStride) {
  auto go = [&](auto dTag) {
    constexpr int DD = decltype(dTag)::value;
    if (mask) flashStridedAt<DD, 4, true>(qkvg, bias, stride, mask, out, n, heads, rows, scale, rowStride, posStride, outRowStride, outPosStride);
    else flashStridedAt<DD, 4, false>(qkvg, bias, stride, mask, out, n, heads, rows, scale, rowStride, posStride, outRowStride, outPosStride);
  };
  if (D == 32) go(std::integral_constant<int, 32>{});
  else if (D == 16) go(std::integral_constant<int, 16>{});
  else { fprintf(stderr, "flashGridStrided: no kernel for head width %d\n", D); exit(1); }
}
