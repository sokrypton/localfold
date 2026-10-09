template <int N>
inline void lnHeadsRaw(const float* x, const float* scale, const float* offset, const half* w, float* out, size_t rows) {
  WITH_PAIR_T(lnHeadsMetal<128, N, PT><<<(unsigned)std::min<size_t>((rows + 63) / 64, 1024), 256, 0, STREAM>>>(x, scale, offset, w, out, rows));
}
