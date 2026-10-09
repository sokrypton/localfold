template <int N>
inline void lnHeadsRaw(const float* x, const float* scale, const float* offset, const half* w, float* out, size_t rows) {
  WITH_PAIR_T(lnHeadsMetal<128, N, PT><<<(unsigned)((rows + 7) / 8), 256, 0, STREAM>>>(x, scale, offset, w, out, rows));
}
