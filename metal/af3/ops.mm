// metal/af3's shared pieces: a named weight's GEMM, LayerNorm by names, the derived weights.
#include "af3.h"

static void checkSize(const std::string& w, int in, int out) {
  if (lenW(w) != (size_t)in * out) die("%s has %zu elements, not %d x %d", w.c_str(), lenW(w), in, out);
}
void lin(const float* X, const std::string& w, float* Y, size_t rows, int in, int out, float beta, const float* bias, float alpha) {
  checkSize(w, in, out);
  Gemm g{}; g.X = X; g.tx = F32; g.W = Wh(w); g.tw = F16; g.half = true; g.Y = Y; g.rows = rows; g.in = in; g.out = out;
  g.beta = beta; g.alpha = alpha; g.bias = bias; g.label = w.c_str();
  gemm(g);
}
void lin(const half* X, const std::string& w, float* Y, size_t rows, int in, int out, float beta, const float* bias, float alpha) {
  checkSize(w, in, out);
  linW(X, Wh(w), Y, rows, in, out, beta, bias, alpha, w.c_str());
}
void linH(const half* X, const std::string& w, half* Y, size_t rows, int in, int out, const float* bias) {
  checkSize(w, in, out);
  linW(X, Wh(w), Y, rows, in, out, bias, w.c_str());
}
void linH(const float* X, const std::string& w, half* Y, size_t rows, int in, int out, const float* bias) {
  checkSize(w, in, out);
  Gemm g{}; g.X = X; g.tx = F32; g.W = Wh(w); g.tw = F16; g.half = true; g.Y = Y; g.ty = F16; g.rows = rows; g.in = in;
  g.out = out; g.bias = bias; g.accFloat = true; g.label = w.c_str();
  gemm(g);
}
void linW(const half* X, const half* w, float* Y, size_t rows, int in, int out, float beta, const float* bias, float alpha,
          const char* label) {
  Gemm g{}; g.X = X; g.tx = F16; g.W = w; g.tw = F16; g.Y = Y; g.rows = rows; g.in = in; g.out = out; g.beta = beta;
  g.alpha = alpha; g.bias = bias; g.label = label;
  gemm(g);
}
void linW(const half* X, const half* w, half* Y, size_t rows, int in, int out, const float* bias, const char* label) {
  Gemm g{}; g.X = X; g.tx = F16; g.W = w; g.tw = F16; g.Y = Y; g.ty = F16; g.rows = rows; g.in = in; g.out = out;
  g.bias = bias; g.accFloat = true; g.label = label;
  gemm(g);
}
void ln(const float* x, float* y, size_t rows, int C, const std::string& scale, const std::string& offset) {
  layerNorm(x, y, rows, C, scale.empty() ? nullptr : W(scale), offset.empty() ? nullptr : Wopt(offset));
}
void ln(const float* x, half* y, size_t rows, int C, const std::string& scale, const std::string& offset) {
  layerNorm(x, y, rows, C, scale.empty() ? nullptr : W(scale), offset.empty() ? nullptr : Wopt(offset));
}
void scaleRows(float* x, const float* mask, size_t rows, int C, size_t period) {
  run1d("af3_scale_rows", rows * C, ScaleRowsArgs{x, mask, rows, (uint)C, (uint)period});
}
void scale(float* x, size_t n, float s, bool relu) { run1d("af3_scale", n, ScaleArgs{x, n, s, relu ? 1u : 0u}); }

const half* concatColumns(const std::string& key, int C, const std::vector<Part>& parts) {
  size_t total = 0;
  for (auto& p : parts) total += p.width;
  return M.derived<half>("concat:" + key, (size_t)C * total, [&](half* out) {
    size_t off = 0;
    for (auto& p : parts) {
      if (!p.name.empty()) {
        if (lenW(p.name) != (size_t)C * p.width) die("%s has %zu elements, not %d x %d", p.name.c_str(), lenW(p.name), C, p.width);
        run1d("af3_concat_part", (size_t)C * p.width,
              ConcatPartArgs{Wh(p.name), out, (uint)C, (uint)p.width, (uint)total, (uint)off, p.transposed ? 1u : 0u, 0});
      }
      off += p.width;
    }
  });
}
const half* swigluPairs(const std::string& key, const half* w, int rows, int I) {
  return M.derived<half>("swiglu:" + key, (size_t)rows * 2 * I, [&](half* out) {
    run1d("af3_interleave8", (size_t)rows * 2 * I, Interleave8Args{w, w + I, out, (uint)rows, (uint)I, (uint)(2 * I), 0});
  });
}
const half* qkvgWeight(const std::string& pre, int C, int Wd, bool tr) {
  return concatColumns(pre + ".qkvg", C, {{pre + ".qProjection", Wd, tr}, {pre + ".kProjection", Wd, tr},
                                          {pre + ".vProjection", Wd, false}, {pre + ".gatingQuery", Wd, tr}});
}
const float* qkvgBias(const std::string& pre, int Wd) {
  if (!hasW(pre + ".gatingQueryBias")) return nullptr;
  return M.derived<float>("qkvgBias:" + pre, (size_t)4 * Wd, [&](float* out) {
    copy(out + 3 * Wd, W(pre + ".gatingQueryBias"), (size_t)Wd * 4);
  });
}
