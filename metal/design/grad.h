// metal/design: the backward of what the folding network is built from, written against metal/core.
//
// Design is gradient descent on a sequence through the network, so it needs a backward every other backend here
// lacks. The convention throughout: a cotangent is named d<forward name>, every array is a device pointer from
// metal/core's allocator, and a parameter's gradient is ACCUMULATED only where a function says so - otherwise it is
// written, so a caller zeroes nothing.
#pragma once
#include "core.h"

namespace mt {

// ---------------------------------------------------------------- LayerNorm
// The forward is metal/core's layerNorm: y[c] = ((x[c] - mean) * inv) * scale[c] + offset[c]. This computes the row
// statistics itself (the forward saves none), then dx, and optionally the parameters' gradients.
//
//   x, dy    the forward's input and the incoming cotangent, [rows][C] (ldx / ldy if padded)
//   dx       [rows][C], written
//   dscale / doffset   [C], written; either may be null
// `scale` must be the forward's, or dx is the gradient of a DIFFERENT function - a null scale means the forward had
// none, not "ignore it".
void layerNormBwd(const float* x, const float* dy, float* dx, size_t rows, int C, const float* scale,
                  float* dscale = nullptr, float* doffset = nullptr, float eps = 1e-5f,
                  int ldx = 0, int ldy = 0, int lddx = 0);

// ---------------------------------------------------------------- a linear layer
// The forward is Y[rows][out] = X[rows][in] W[in][out] (+ bias). Both products below are metal/core's GEMM with one
// operand transposed, which it already serves, so there is no kernel here:
//   dX[rows][in] = dY[rows][out] W^T        (W read as stored [in][out], i.e. transposed for this product)
//   dW[in][out]  = X^T dY                   (X read as stored [rows][in], i.e. transposed for this product)
//   dbias[out]   = sum over rows of dY
// Any of dX, dW, dbias may be null. Nothing is accumulated: each is written.
void linearBwd(const float* X, const float* W, const float* dY, float* dX, float* dW, float* dbias,
               size_t rows, int in, int out);

// The column sums on their own - a bias's gradient, and what doffset is.
void columnSum(const float* dy, float* out, size_t rows, int C, int ldy = 0);

}  // namespace mt
