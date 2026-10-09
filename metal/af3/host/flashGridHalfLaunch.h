template <int D>
void flashGridHalfLaunch(const half* qkvg, const half* bias, int stride, const float* mask, half* out,
                         int n, int heads, size_t r0, size_t rows, bool tr, float scale, const float* qBias = nullptr) {
  flashGridMetalRun<D>(qkvg, bias, stride, mask, out, n, heads, r0, rows, tr, scale, qBias);
}
