// Native ESMFold2 on Metal: the stages' declarations (metal/ef2/*.mm). cuda/ef2 is the reading - the same stages
// and arithmetic, here on metal/core: weights decoded to float16 where they are large (the confidence head's own
// projections stay float32), activations float32 between the f16 GEMMs.
#pragma once
#include "core.h"
#include "kernels.h"
#include "host.h"
#include "model.h"
#include <string>
#include <vector>

using namespace mt;

// the folding bundle's tensors ("f/...") as float32 / float16, the input's as they are named
inline const float* F(const std::string& n) { return M.f("f/" + n); }
inline const half* Fh(const std::string& n) { return M.h("f/" + n); }
inline const float* In(const std::string& n) { return M.f(n); }
inline const int* Ii(const std::string& n) { return M.i(n); }
inline int dimOf(const std::string& n, int k) { return (int)M.dim(n, k); }

// Y = X W for a named weight, X float32: in half (X rounded as it is staged) where the weight is held as float16 or
// `halfAlways`, float otherwise. beta: + beta Y; Yin: beta's input; bias: a named weight, or none
void lin(const float* X, const std::string& w, float* Y, size_t rows, int in, int out, float beta = 0.f,
         const float* bias = nullptr, float alpha = 1.f, const float* Yin = nullptr);
void lin(const half* X, const std::string& w, float* Y, size_t rows, int in, int out, float beta = 0.f,
         const float* bias = nullptr, float alpha = 1.f);
void linH(const half* X, const std::string& w, half* Y, size_t rows, int in, int out, const float* bias = nullptr,
          bool gelu = false);
extern bool HALF_GEMM;            // every float32 GEMM's operands rounded to half (the cuda port's TF32 / FAST_16F)
RelIdx relIdx();
// a weight interleaved for gemmSwiglu: [rows][2I] = [a | b] -> blocks of 8 (a_0..a_7 b_0..b_7 a_8..)
const half* swigluPairs(const std::string& key, const half* w, int rows, int I);
// ...interleaved in the tensor's own memory, for a weight read by nothing else
const half* swigluPairsInPlace(const std::string& name, int rows, int I);
// two weights [rows][I] each -> the same interleaving, a from the first and b from the second
const half* swigluPairs2(const std::string& key, const half* a, const half* b, int rows, int I);

// ---------------------------------------------------------------- the language model (tower.mm)
struct Esmc { int rows, model, heads, ffn, layers, pair; float residualScale = 1.f; };
// the tower and the shim: ids [rows] -> per-token states "shim.tokens" [T, pair], and the pair term lmZ [T, T, pair]
void languageModel(const Esmc& e, const int* ids, const int* seq, const int* tokenToRow, int T, float* lmZ);

// ---------------------------------------------------------------- the trunk (trunk.mm)
void zInit(int T, int C, const float* sInputs, int Si, const float* lmZ, float* z);
// nextFollows: block b + 1 runs next on this pair, nothing between (its first norm is emitted by this block)
void trunkBlock(float* pair, const float* mask, int L, int C, const std::string& prefix, int b, bool nextFollows = false);
void foldingTrunk(int T, int C, const float* zInit, float* z, int loops);
// the page's contact map: the softmax mass under each pair's threshold - [T * T] on the host
std::vector<float> contactMap(const float* z, int T, int C);

// ---------------------------------------------------------------- the atoms (atoms.mm)
struct AtomCtx { int A, C, heads, hidden; const float* mask; const float* cosT; const float* sinT;
                 const int* valid; int nValid; const int* tokenStart; const int* tokenAtoms; };
struct Atoms { AtomCtx ctx; float* c0; };
extern bool ATOM_BF16;
extern int INPUTS_HALF_WINDOW;
Atoms prepareAtoms(int A, const std::string& prefix);
void freeAtoms(Atoms& at);
void swaStack(const AtomCtx& a, float* x, const float* cond, const std::string& prefix, int blocks, int halfWindow);
void inputsEmbedder(int T, int A, float* sInputs, int sWidth);

// ---------------------------------------------------------------- the diffusion module and its sampler (diffusion.mm)
struct SamplerSettings { int steps = 15; double sMax = 160, sMin = 4e-4, p = 8, maxSigma = 256, gamma0 = 0.605,
                         gammaMin = 1.107, noiseScale = 0.901, stepScale = 1.638; };
struct Denoiser {
  int T, A, Cz, Ct, heads, tokenBlocks, Si, atomBlocks;
  float sigma;
  Atoms atoms;
  std::vector<half*> biases;       // a token block's pair bias, [H][T][stride] in log2 units
  int stride;
  const float* sInputs;
  float* single; float* snScaled; float* G; float* scales; float* level;
  int entries;
};
Denoiser makeDenoiser(int T, int A, const float* zTrunk, const float* sInputs);
void freeDenoiser(Denoiser& d);
void denoise(const Denoiser& d, const float* xNoisy, float t, float* xDenoised);
// the whole sampler: the final coordinates [A, 3]; onStep(denoised positions on the device, step, steps)
std::vector<float> sample(const Denoiser& d, const SamplerSettings& s, uint64_t seed, int* stepsRun,
                          const std::function<void(const float*, int, int)>& onStep = nullptr);

// ---------------------------------------------------------------- the confidence head (confidence.mm)
struct Confidence { std::vector<float> plddtAtom, plddtToken, pae; double ptm, iptm, meanPlddt; };
Confidence confidenceHead(int T, int A, const float* zTrunk, const float* sInputs, int Si, const float* xDevice);

// ---------------------------------------------------------------- output (output.mm)
void writePdb(const std::string& templatePath, const std::string& out, const std::vector<float>& x,
              const std::vector<float>* bfactor = nullptr);
void writeConfidences(const std::string& pdb, int T, const Confidence& conf, const std::vector<float>& contacts);
