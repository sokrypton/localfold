template <bool NB = false>
bool flash2R1Strided(const half* qkvg, const half* bias, int stride, half* out, int n, int heads, size_t rows, float scale,
                     const float* qBias, size_t rowStride = 0, size_t posStride = 0, size_t outRowStride = 0,
                     size_t outPosStride = 0) {
  flashGridMetalRun<32>(qkvg, NB ? nullptr : bias, stride, nullptr, out, n, heads, 0, rows, false, scale, qBias,
                        rowStride, posStride, outRowStride, outPosStride);
  return true;
}
