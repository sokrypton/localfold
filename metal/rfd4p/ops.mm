// The pieces every stage of this port is written out of: a named weight's GEMM in torch's layout, RMSNorm, and the
// SwiGLU transition.
#include "rfd4p.h"
#include <vector>

void lin(const float* X, const std::string& w, float* Y, size_t rows, int in, int out, const float* bias, bool relu) {
  Gemm g{};
  g.X = X; g.tx = F32;
  g.Y = Y; g.ty = F32;
  g.rows = rows; g.in = in; g.out = out;
  g.bias = bias; g.tbias = F32; g.relu = relu;
  g.label = w.c_str();
  // 🔴 `transW`, BECAUSE EVERY WEIGHT IN THIS CHECKPOINT IS [out][in]. `linearLayer` in the CPU reference reads
  // `weight[o * inDim + k]`, which is torch's `nn.Linear`, and so is the bundle's shape.
  g.transW = true;
  if (HALF && M.has("w/" + w) && M.len("w/" + w) >= 4096) { g.W = M.h("w/" + w); g.tw = F16; g.half = true; }
  else { g.W = M.f("w/" + w); g.tw = F32; }
  gemm(g);
}

const float* opt(const std::string& name) {
  // 🔴 ABSENT IS NOT ZERO FOR A NORM'S SCALE, but it is for every bias here, which is why only biases and the
  // triangle's optional norms go through this. `getOptionalTensorData` in the CPU reference is the same door.
  return M.has("w/" + name) ? M.f("w/" + name) : nullptr;
}

void rmsNorm(const float* x, float* y, size_t rows, int C, const float* w, float eps) {
  RfdRmsArgs a{};
  a.x = x; a.y = y; a.yh = nullptr; a.w = w;
  a.rows = rows; a.C = C; a.eps = eps;
  // a simdgroup a row, 8 rows a threadgroup
  run("rfd4p_rms_norm", grid1d((rows + 7) / 8, 1), 256, a);
}

void swiglu(const float* h, float* g, size_t rows, int I) {
  RfdSwigluArgs a{};
  a.h = h; a.g = g; a.gh = nullptr; a.rows = rows; a.I = (uint)I;
  run1d("rfd4p_swiglu", rows * (size_t)I, a);
}

void transition(const float* x, float* y, size_t rows, int dim, int expansion, const std::string& prefix, float* hid,
                float* gated) {
  const int inner = dim * expansion;
  lin(x, prefix + ".swish_linear.0.weight", hid, rows, dim, 2 * inner);
  swiglu(hid, gated, rows, inner);
  lin(gated, prefix + ".linear_out.weight", y, rows, inner, dim);
}

void adalnRows(const float* x, const float* cond, const std::string& prefix, float* y, size_t rows, int dim,
               int condDim, bool ln, half* yh) {
  const int C = condDim ? condDim : D_.C;
  // 🔴 THE CONDITIONING IS NORMED AGAIN HERE, by this adaLN's own `norm_cond` - `adaptiveLayerNorm` in the CPU
  // reference norms both of its inputs, so c is normed per layer and x is normed on top of whatever normed it last.
  float* cn = scratch<float>("ada.cn", rows * C);
  if (ln) layerNorm(cond, cn, rows, C, M.f("w/" + prefix + ".norm_cond.weight"), opt(prefix + ".norm_cond.bias"));
  else rmsNorm(cond, cn, rows, C, M.f("w/" + prefix + ".norm_cond.weight"));
  float* gamma = scratch<float>("ada.gamma", rows * (size_t)dim);
  float* beta = scratch<float>("ada.beta", rows * (size_t)dim);
  lin(cn, prefix + ".to_gamma.0.weight", gamma, rows, C, dim, M.f("w/" + prefix + ".to_gamma.0.bias"));
  lin(cn, prefix + ".to_beta.weight", beta, rows, C, dim);
  if (ln) {
    RfdAdalnRowsLnArgs a{};
    a.x = x; a.y = y; a.gamma = gamma; a.beta = beta; a.rows = rows; a.C = dim; a.eps = 1e-5f;
    run("rfd4p_adaln_rows_ln", grid1d((rows + 7) / 8, 1), 256, a);
    return;
  }
  RfdAdalnRowsArgs a{};
  a.x = x; a.y = y; a.yh = yh; a.gamma = gamma; a.beta = beta; a.rows = rows; a.C = dim; a.eps = 1e-5f;
  run("rfd4p_adaln_rows", grid1d((rows + 7) / 8, 1), 256, a);
}

void gatherRows(const float* in, float* out, const int* idx, size_t rows, int C) {
  RfdRowsArgs a{}; a.in = in; a.out = out; a.idx = idx; a.rows = rows; a.C = C;
  run1d("rfd4p_gather_rows", rows * (size_t)C, a);
}
void scatterRows(const float* in, float* out, const int* idx, size_t rows, int C) {
  RfdRowsArgs a{}; a.in = in; a.out = out; a.idx = idx; a.rows = rows; a.C = C;
  run1d("rfd4p_scatter_rows", rows * (size_t)C, a);
}

void scaleAdd(float* x, const float* y, const float* cond, const std::string& prefix, size_t rows, int dim,
              int condDim) {
  // 🔴 AND THIS ONE READS THE CONDITIONING RAW. `adaptiveOutputScale` projects `cond` itself, with no norm of any
  // kind - the one adaptive module in this network that does not - so passing it a normed c is a silent 1e-1 error.
  float* gamma = scratch<float>("sc.gamma", rows * (size_t)dim);
  lin(cond, prefix + ".to_adaln_zero_gamma.0.weight", gamma, rows, condDim ? condDim : D_.C, dim,
      M.f("w/" + prefix + ".to_adaln_zero_gamma.0.bias"));
  RfdScaleAddArgs a{};
  a.x = x; a.y = y; a.gamma = gamma; a.rows = rows; a.C = dim;
  run1d("rfd4p_scale_add", rows * (size_t)dim, a);
}

void linH(const half* X, const std::string& w, half* Y, size_t rows, int in, int out, const float* bias, bool relu,
          int ldy) {
  Gemm g{};
  g.X = X; g.tx = F16;
  g.W = M.h("w/" + w); g.tw = F16; g.transW = true;
  g.Y = Y; g.ty = F16; g.ldy = ldy;
  g.rows = rows; g.in = in; g.out = out;
  g.bias = bias; g.tbias = F32; g.relu = relu;
  g.label = w.c_str();
  gemm(g);
}

void linHF(const half* X, const std::string& w, float* Y, size_t rows, int in, int out, const float* bias, bool relu) {
  Gemm g{};
  g.X = X; g.tx = F16;
  g.W = M.h("w/" + w); g.tw = F16; g.transW = true;
  g.Y = Y; g.ty = F32;
  g.rows = rows; g.in = in; g.out = out;
  g.bias = bias; g.tbias = F32; g.relu = relu;
  g.label = w.c_str();
  gemm(g);
}

void transitionH(const half* x, half* y, size_t rows, int dim, int expansion, const std::string& prefix, half* hid,
                 half* gated) {
  const int inner = dim * expansion;
  linH(x, prefix + ".swish_linear.0.weight", hid, rows, dim, 2 * inner);
  RfdSwigluHArgs a{};
  a.h = hid; a.g = gated; a.rows = rows; a.I = (uint)inner;
  run1d("rfd4p_swiglu_h", rows * (size_t)inner, a);
  linH(gated, prefix + ".linear_out.weight", y, rows, inner, dim);
}

const half* weightT(const std::string& name) {
  const std::string key = "w/" + name;
  const int out = (int)M.dim(key, 0), in = (int)M.dim(key, 1);
  return M.derived<half>("gatedT:" + key, (size_t)out * in, [&](half* p) {
    RfdTransposeHArgs a{};
    a.in = M.h(key); a.out = p; a.rows = (size_t)out; a.cols = (uint)in;
    run1d("rfd4p_transpose_h", (size_t)out * in, a);
  });
}

const half* weightStack(const std::string& key, const std::vector<std::string>& names) {
  size_t total = 0;
  for (const std::string& n : names) total += M.len("w/" + n);
  return M.derived<half>("stack:" + key, total, [&](half* p) {
    size_t at = 0;
    for (const std::string& n : names) {
      const size_t len = M.len("w/" + n);
      copy(p + at, M.h("w/" + n), len * 2);
      at += len;
    }
  });
}

int condStride(int layers, int dim) { return layers * 4 * dim; }

const float* precomputeCond(const float* c, size_t rows, int layers, int dim, int condDim) {
  const int ld = condStride(layers, dim);
  // the four projections of every layer, stacked [layers * 4 * dim][condDim], each already carrying its own
  // adaLN's `norm_cond` scale - so the norm below needs no weight and is shared by all of them
  // 🔴 ITS PRECISION FOLLOWS THE ARM. Derived in half unconditionally, this projection was computed in half even
  // under `--f32` - and the f32 arm is the exact one, so the trajectory gate went red on a change that was meant
  // for the fast path only. A shared precompute has to be as exact as the arm that reads it.
  std::vector<float> stackHost;
  auto fillStack = [&]() {
    std::vector<float>& host = stackHost;
    host.assign((size_t)ld * condDim, 0.f);
    for (int l = 0; l < layers; ++l) {
      const std::string L = "nn.transformer_layers." + std::to_string(l) + ".";
      for (int which = 0; which < 2; ++which) {
        const std::string A = L + (which == 0 ? "mhba.adaln." : "transition.adaln.");
        const float* wn = M.hostF("w/" + A + "norm_cond.weight");
        for (int part = 0; part < 2; ++part) {
          const float* w = M.hostF("w/" + A + (part == 0 ? "to_gamma.0.weight" : "to_beta.weight"));
          const size_t base = (size_t)((l * 4 + which * 2 + part) * dim) * condDim;
          for (int r = 0; r < dim; ++r)
            for (int k = 0; k < condDim; ++k)
              host[base + (size_t)r * condDim + k] = w[(size_t)r * condDim + k] * wn[k];
        }
      }
    }
  };
  const float* Wf = HALF ? nullptr : M.derived<float>("condStackF", (size_t)ld * condDim, [&](float* p) {
    fillStack();
    upload(p, stackHost.data(), stackHost.size() * 4);
  });
  const half* W = !HALF ? nullptr : M.derived<half>("condStack", (size_t)ld * condDim, [&](half* p) {
    fillStack();
    std::vector<half> h(stackHost.size());
    for (size_t i = 0; i < stackHost.size(); ++i) h[i] = (half)stackHost[i];
    upload(p, h.data(), h.size() * 2);
  });
  // ...and the biases the same way (to_beta has none, which is a block of zeros)
  const float* B = M.derived<float>("condBias", (size_t)ld, [&](float* p) {
    std::vector<float> host((size_t)ld, 0.f);
    for (int l = 0; l < layers; ++l) {
      const std::string L = "nn.transformer_layers." + std::to_string(l) + ".";
      for (int which = 0; which < 2; ++which) {
        const std::string A = L + (which == 0 ? "mhba.adaln." : "transition.adaln.");
        const float* b = M.hostF("w/" + A + "to_gamma.0.bias");
        for (int r = 0; r < dim; ++r) host[(size_t)(l * 4 + which * 2) * dim + r] = b[r];
      }
    }
    upload(p, host.data(), host.size() * 4);
  });
  float* rms0 = scratch<float>("cond.rms0", rows * condDim);
  rmsNorm(c, rms0, rows, condDim, nullptr);
  float* gb = scratch<float>("cond.gb", rows * (size_t)ld);
  Gemm g{};
  g.X = rms0; g.tx = F32;
  if (HALF) { g.W = W; g.tw = F16; g.half = true; } else { g.W = Wf; g.tw = F32; }
  g.transW = true;
  g.Y = gb; g.ty = F32;
  g.rows = rows; g.in = condDim; g.out = ld;
  g.bias = B; g.tbias = F32;
  g.label = "cond-precompute";
  gemm(g);
  return gb;
}

void adalnPre(const float* x, const float* gb, int layer, int which, int layers, float* y, size_t rows, int dim,
              half* yh) {
  const int ld = condStride(layers, dim);
  RfdAdalnRowsArgs a{};
  a.x = x; a.y = y; a.yh = yh;
  a.gamma = gb + (size_t)(layer * 4 + which * 2) * dim;
  a.beta = gb + (size_t)(layer * 4 + which * 2 + 1) * dim;
  a.rows = rows; a.C = dim; a.eps = 1e-5f; a.ldg = ld;
  run("rfd4p_adaln_rows", grid1d((rows + 7) / 8, 1), 256, a);
}
