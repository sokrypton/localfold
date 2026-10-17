// metal/ef2's shared pieces: a named weight's GEMM, the relative-position indices, derived weights.
#include "ef2.h"
#include <set>

bool HALF_GEMM = true;

void lin(const float* X, const std::string& w, float* Y, size_t rows, int in, int out, float beta, const float* bias, float alpha,
         const float* Yin) {
  Gemm g{}; g.X = X; g.tx = F32; g.Y = Y; g.ty = F32; g.Yin = Yin; g.rows = rows; g.in = in; g.out = out;
  g.beta = beta; g.alpha = alpha; g.bias = bias; g.label = w.c_str();
  if (HALF_GEMM) { g.W = M.h(w); g.tw = F16; g.half = true; }
  else { g.W = M.f(w); g.tw = F32; }
  gemm(g);
}
void lin(const half* X, const std::string& w, float* Y, size_t rows, int in, int out, float beta, const float* bias, float alpha) {
  Gemm g{}; g.X = X; g.tx = F16; g.W = M.h(w); g.tw = F16; g.Y = Y; g.ty = F32; g.rows = rows; g.in = in; g.out = out;
  g.beta = beta; g.alpha = alpha; g.bias = bias; g.label = w.c_str();
  gemm(g);
}
void linH(const half* X, const std::string& w, half* Y, size_t rows, int in, int out, const float* bias, bool gelu) {
  Gemm g{}; g.X = X; g.tx = F16; g.W = M.h(w); g.tw = F16; g.Y = Y; g.ty = F16; g.rows = rows; g.in = in; g.out = out;
  g.bias = bias; g.gelu = gelu; g.label = w.c_str();
  gemm(g);
}
RelIdx relIdx() {
  return {Ii("residue_index"), Ii("asym_id"), Ii("sym_id"), Ii("entity_id"), Ii("token_index"), F("featuriser/relPos")};
}
const half* swigluPairs(const std::string& key, const half* w, int rows, int I) {
  return M.derived<half>("swiglu:" + key, (size_t)rows * 2 * I, [&](half* out) {
    run1d("ef2_interleave8", (size_t)rows * 2 * I, Interleave8Args{w, w + I, out, (uint)rows, (uint)I, (uint)(2 * I), 0});
  });
}
// ...in the tensor's own memory (read by nothing else): interleaved through a temporary the size of one, so the
// original and its interleaving are never both held - the language model's 36 fc1 weights were 0.5 GB twice
const half* swigluPairsInPlace(const std::string& name, int rows, int I) {
  static std::set<std::string> done;
  half* w = const_cast<half*>(M.h(name));
  if (done.insert(name).second) {
    size_t n = (size_t)rows * 2 * I;
    half* tmp = allocT<half>(n);
    run1d("ef2_interleave8", n, Interleave8Args{w, w + I, tmp, (uint)rows, (uint)I, (uint)(2 * I), 0});
    copy(w, tmp, n * 2);
    release(tmp);
  }
  return w;
}
const half* swigluPairs2(const std::string& key, const half* a, const half* b, int rows, int I) {
  return M.derived<half>("swiglu2:" + key, (size_t)rows * 2 * I, [&](half* out) {
    run1d("ef2_interleave8", (size_t)rows * 2 * I, Interleave8Args{a, b, out, (uint)rows, (uint)I, (uint)I, 0});
  });
}
