// LocalFold's native Metal core: the device, its memory, its kernels and its GEMM, which every native port
// (metal/ef2, metal/af2, metal/af3) is written against.
//
// Memory: every allocation is a shared MTLBuffer - Apple silicon's memory is the host's - and a device pointer is
// its GPU address, so pointers carry offsets and live in kernel argument structs as they are (metal/core/args.h).
// host() maps one back to the bytes. Order: one queue, one serial compute encoder at a time - dispatches run in the
// order they are issued. A read on the host (download) waits for everything issued before it.
#pragma once
#include "args.h"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <initializer_list>
#include <map>
#include <string>
#include <vector>

namespace mt {

[[noreturn]] void die(const char* fmt, ...) __attribute__((format(printf, 1, 2)));

// ---------------------------------------------------------------- memory
void* alloc(size_t bytes);                  // zero-filled
void release(const void* p);                // (kept alive until the work issued so far is done)
template <class T> T* allocT(size_t n) { return (T*)alloc(n * sizeof(T)); }
void* host(const void* p);                  // the bytes behind a device pointer (read them after a sync)
bool isDevice(const void* p);
size_t allocated();                         // bytes held now
size_t peakAllocated();
size_t peakLive();           // (and the released buffers work in flight kept)
void resetPeak();
// named working buffers, grown as asked and kept until released (a fold asks for the same ones every block)
void* scratchBytes(const std::string& name, size_t bytes);
template <class T> T* scratch(const std::string& name, size_t n) { return (T*)scratchBytes(name, n * sizeof(T)); }
void releaseScratch(std::initializer_list<const char*> prefixes = {});
size_t scratchHeld();
std::vector<std::pair<std::string, size_t>> scratchList();

void fill(void* p, int byte, size_t bytes);                       // in stream order
void padZero(half* planes, size_t count, int n, int np);   // zero [count] [np][np] planes' padding past n x n
void copy(void* dst, const void* src, size_t bytes);              // device to device, in stream order
// rows of `width` bytes, `height` of them, pitches apart (a strided copy, in stream order)
void copy2d(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height);
void upload(void* dst, const void* src, size_t bytes);            // host to device, in stream order
template <class T> T* uploadNew(const T* src, size_t n) { T* d = allocT<T>(n ? n : 1); if (n) upload(d, src, n * sizeof(T)); return d; }
void download(void* dst, const void* src, size_t bytes);          // waits for the queue
template <class T> std::vector<T> download(const T* p, size_t n) { std::vector<T> v(n); if (n) download(v.data(), p, n * sizeof(T)); return v; }
void sync();                                                      // everything issued so far, done

// ---------------------------------------------------------------- kernels
// The kernels' Metal source: the core's and the port's (metal/build.sh embeds them). Pipelines are made by name on
// first use; a template's instance (the GEMM) is compiled with its declaration, and remembered for the next run.
void setSource(const char* portName, const char* source);
struct Grid { uint32_t x = 1, y = 1, z = 1; };
void dispatch(const char* kernel, const void* args, size_t argBytes, Grid groups, uint32_t threads, size_t smem = 0,
              const char* label = nullptr);
template <class A> void run(const char* kernel, Grid groups, uint32_t threads, const A& a, size_t smem = 0) {
  dispatch(kernel, &a, sizeof a, groups, threads, smem);
}
// n threads of 256 as a grid (kernels read their index with LF_INDEX: x and y folded)
Grid grid1d(size_t n, uint32_t threads = 256);
template <class A> void run1d(const char* kernel, size_t n, const A& a) {
  if (n) dispatch(kernel, &a, sizeof a, grid1d(n), 256);
}
// a template instance: `decl` explicitly instantiates it under `name`; source = the core's GEMM source by default
void dispatchInstance(const std::string& name, const std::string& decl, const void* args, size_t argBytes, Grid groups,
                      uint32_t threads, size_t smem = 0, const char* label = nullptr);
void precompile();                          // the instances a previous run used, compiled now in parallel
// the name of the list of GEMM instances a port's folds use (~/.cache/localfold/metal/<name>; a wheel ships one beside
// the binaries, in metal-specs/): LOCALFOLD_METAL_SPECS_NAME=1 localfold-<port> prints it
std::string specsName(const char* port);

// ---------------------------------------------------------------- GEMM
// Y[rows, out] = alpha X[rows, in] W[in, out] (+ beta Y) (+ bias) (relu / gelu), row-major. X transposed: stored
// [in][rows]; W transposed: stored [out][in]. Accumulated in float, or in half where every operand is half (the
// trunk's f16 GEMMs; accFloat keeps float). Operands are multiplied in half where both are half, or where `half`
// asks for it (an f32 operand rounded as it is staged), in float otherwise.
enum DT { F32, F16 };
struct Gemm {
  const void* X; DT tx = F32; int ldx = 0; bool transX = false;
  const void* W; DT tw = F32; int ldw = 0; bool transW = false;
  void* Y; DT ty = F32; int ldy = 0;
  const void* Yin = nullptr;                 // beta's input, if not Y itself (same type and leading dimension)
  size_t rows; int in, out;
  float alpha = 1.f, beta = 0.f;
  const void* bias = nullptr; DT tbias = F32; bool relu = false, gelu = false;
  int batch = 1; int64_t sx = 0, sw = 0, sy = 0;
  bool half = false, accFloat = false;
  const char* label = nullptr;
  // the next LayerNorm (lnScale, lnOffset), emitted into lnOut [rows][out] half where the matrix units run a float output
  // 128 or 256 wide (64 x 128 or 32 x 256 tiles) - gemm() returns true; false: not written, the caller's LayerNorm still to run
  ::half* lnOut = nullptr; const float* lnScale = nullptr; const float* lnOffset = nullptr; float lnEps = 1e-5f;
  ::half* lnOut2 = nullptr; const float* lnScale2 = nullptr; const float* lnOffset2 = nullptr;   // (a second norm, same eps)
};
bool gemm(const Gemm& g);
extern int GEMM_EXTRA_EP;     // (an arm: bits ORed into every GEMM instance's EP - metal/bench)
// shorthands: f32 or f16 X, a weight, Y f32 or f16
inline void gemm(const float* X, const half* W, float* Y, size_t rows, int in, int out, float beta = 0.f) {
  Gemm g{}; g.X = X; g.tx = F32; g.W = W; g.tw = F16; g.Y = Y; g.ty = F32; g.rows = rows; g.in = in; g.out = out;
  g.beta = beta; g.half = true; gemm(g);
}
inline void gemm(const half* X, const half* W, float* Y, size_t rows, int in, int out, float beta = 0.f) {
  Gemm g{}; g.X = X; g.tx = F16; g.W = W; g.tw = F16; g.Y = Y; g.ty = F32; g.rows = rows; g.in = in; g.out = out;
  g.beta = beta; gemm(g);
}
inline void gemm(const half* X, const half* W, half* Y, size_t rows, int in, int out) {
  Gemm g{}; g.X = X; g.tx = F16; g.W = W; g.tw = F16; g.Y = Y; g.ty = F16; g.rows = rows; g.in = in; g.out = out; gemm(g);
}
inline void gemm(const float* X, const float* W, float* Y, size_t rows, int in, int out, float beta = 0.f) {
  Gemm g{}; g.X = X; g.W = W; g.Y = Y; g.rows = rows; g.in = in; g.out = out; g.beta = beta; gemm(g);
}
// fused epilogues: pair[r][o] += aux[r][o] sigmoid(X W) (aux f16, pair f32); gated[r][k] = silu(a) b with W's
// columns interleaved in blocks of 8 (a_0..a_7 b_0..b_7 a_8..: swigluPairs); the triangle's projection and gate -
// W [C][4C], channel c's (pa ga pb gb) in blocks of 8 (triGatePairs) - writing a and b channel-major into padded
// planes [C][np * np] (rows r0.. of the n * n pairs)
// (lnOut: as Gemm::lnOut - the updated pair's next LayerNorm, where the matrix units take it; true if written)
bool gemmGatedAdd(const half* X, const half* W, const half* aux, float* pair, size_t rows, int in, int out,
                  const float* bias = nullptr, half* lnOut = nullptr, const float* lnScale = nullptr,
                  const float* lnOffset = nullptr, float lnEps = 1e-5f);
// the triangle's tail in one pass: pair += (Xv Wv + biasV, rounded to half) sigmoid(Xg Wg + biasG) - on the matrix
// units one kernel, the value never written; elsewhere (or LOCALFOLD_GATED_DUAL=0) Xv Wv into vTmp, then gemmGatedAdd
// lnOut: where the fused kernel runs and out is 128, it also writes LayerNorm(the updated pair) (lnScale, lnOffset) to lnOut
// in half - the next update's input - and returns true (false: the caller's LayerNorm still to run)
bool gemmGatedAddDual(const half* Xg, const half* Wg, const half* Xv, const half* Wv, float* pair, size_t rows, int in,
                      int out, const float* biasG, const float* biasV, half* vTmp, const char* label = nullptr,
                      half* lnOut = nullptr, const float* lnScale = nullptr, const float* lnOffset = nullptr, float lnEps = 1e-5f);
// the outer product's block P[(i, c)][(j, e)] = sum_s l[s][i0 + i][c] r[s][j][e] (lt, rt transposed: [S][L * O] with
// leading dimensions ldl, ldr), stored permuted into X [(i, j)][(c, e)] by the epilogue - on the matrix units only
// (false: nothing run, the caller's GEMM and permute; LOCALFOLD_OPM_PERMUTED=0 the control)
bool gemmOpmPermuted(const half* lt, int ldl, const half* rt, int ldr, half* X, int bi, int L, int O, int S);
// the outer product's output projection adding into the pair: pair[r] += (bias + X[r] W) / (1e-3 + norm[r]) - on the
// matrix units only (the vector epilogue, EP bit 8192); false: nothing run, the caller's own GEMM and add
bool gemmOpmOut(const half* X, const half* W, float* pair, const float* bias, const float* norm, size_t rows, int in, int out);
void gemmSwiglu(const half* X, const half* Wpairs, half* gated, size_t rows, int in, int hidden, const float* bias = nullptr);
void gemmTriGate(const half* X, const half* W, const float* mask, half* a, half* b, size_t r0, size_t rows, int C,
                 size_t pairs, int n, int np, const float* bias = nullptr, bool quartered = false);
// the matrix units' layout of that weight (and bias): each 128 columns 32 channels' pa, then ga, pb, gb - where
// triQuartersApply(C), a port builds its weight so once (triQuarters, into its own derived tensor) and passes
// quartered; otherwise gemmTriGate rearranges the blocks-of-8 weight itself every call
bool triQuartersApply(int C);
void triQuarters(const half* W8, half* out, const float* bias8, float* outBias, int C);

// ---------------------------------------------------------------- gated flash attention
// out = softmax(q k^T scale + bias + mask) v * sigmoid(g), per batch row and head; D 8, 16, 24, 32, 48 or 64. Dense
// layouts by default ([rows][n][4W] in, [rows][n][W] out); strides for an attention ACROSS a tensor's leading axis
struct Attention {
  const half* qkvg; half* out; int n, heads, D; size_t rows; float scale;
  int64_t rowStride = 0, posStride = 0, outRowStride = 0, outPosStride = 0;     // (0: the dense layout's)
  const half* bias = nullptr; int biasStride = 0;                               // [H][n][biasStride], log2 units
  // the key's mask, mask[(r0 + b) maskB + key maskK] (0 strides: [rows][n], maskB n and maskK 1)
  const float* mask = nullptr; int64_t r0 = 0, maskB = 0, maskK = 0;
  const float* qBias = nullptr;
};
void attention(const Attention& a);

// ---------------------------------------------------------------- common kernels (metal/core/common.metal)
void layerNorm(const float* x, float* y, size_t rows, int C, const float* scale, const float* offset, float eps = 1e-5f,
               int ldx = 0, int ldy = 0);
void layerNorm(const float* x, half* y, size_t rows, int C, const float* scale, const float* offset, float eps = 1e-5f,
               int ldx = 0, int ldy = 0);
void layerNorm(const half* x, half* y, size_t rows, int C, const float* scale, const float* offset, float eps = 1e-5f,
               int ldx = 0, int ldy = 0);
// the triangle's centre LayerNorm: a channel-major product [C][Lp * Lp] (padded planes) to pair rows [L * L][C] in half
void centerNorm(const float* prod, half* out, int L, int Lp, int C, const float* scale, const float* offset);
void toHalf(const float* x, half* y, size_t n);
void toFloat(const half* x, float* y, size_t n);
void add(float* y, const float* x, size_t n, float a = 1.f);      // y += a x
void addBias(float* y, const float* b, size_t rows, int C, int act = 0);   // act: 1 relu, 2 gelu

// ---------------------------------------------------------------- profiling (LOCALFOLD_PROFILE=1)
// every labelled dispatch its own command buffer, its GPU time summed by label: an upper bound, for proportions
bool profiling();
void profileStart();
void profileReport(const char* stage, int top = 20);

// ---------------------------------------------------------------- the clock
double now();                               // seconds, monotonic
void printStats();                          // LOCALFOLD_METAL_STATS: where the host's time went
}  // namespace mt
