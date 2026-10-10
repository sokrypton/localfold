// Native AlphaFold 2 on Metal: the stages' declarations (metal/af2/*.mm). cuda/af2 is the reading - af3-any-model's AF2
// multimer graph, one graph for monomer and multimer checkpoints - on metal/core: the Evoformer's projections f16
// with f32 accumulation, attention on the core's flash kernel, residual streams f32.
#pragma once
#include "core.h"
#include "kernels.h"
#include "model.h"
#include <string>
#include <vector>

using namespace mt;

// a haiku parameter, "w/<module>/<param>"; `block` picks one slice of a layer-stacked one
inline int64_t dimW(const std::string& name, int k) { return M.dim("w/" + name, k); }
inline const float* P(const std::string& name, int block = -1) {
  const float* base = M.f("w/" + name);
  return block < 0 ? base : base + M.len("w/" + name) / dimW(name, 0) * block;
}
inline const half* PH(const std::string& name, int block = -1) {
  const half* base = M.h("w/" + name);
  return block < 0 ? base : base + M.len("w/" + name) / dimW(name, 0) * block;
}
inline const float* In(const std::string& n) { return M.f(n); }
inline const int* Ii(const std::string& n) { return M.i(n); }

// Y = X W + bias (relu): a named haiku linear ("<module>" with /weights and /bias), X f32 or f16, its operands
// multiplied in half; beta: + beta Y
void linearB(const float* X, const std::string& w, int block, float* Y, size_t rows, int in, int out, bool relu = false,
             float beta = 0.f);
void linearB(const half* X, const std::string& w, int block, float* Y, size_t rows, int in, int out, bool relu = false,
             float beta = 0.f);
void layerNormW(const float* x, float* y, size_t rows, int C, const std::string& w, int block = -1);
void layerNormW(const float* x, half* y, size_t rows, int C, const std::string& w, int block = -1);

struct Trunk {
  int L, N, E;                  // residues, MSA rows, extra MSA rows
  int T = 0;                    // template rows appended to the MSA, after N
  float* msa;                   // [N + T, L, 256]
  float* extra;                 // [E, L, 64]
  float* pair;                  // [L, L, 128]
  float* msaMask;               // [N + T, L]
  float* extraMask;             // [E, L]
  float* pairMask;              // [L, L]
  bool opmFirst;
  bool msaOnes = false, extraOnes = false, pairOnes = false;     // all ones: no mask read
};
void embed(Trunk& t, int pass, const float* prevMsaRow, const float* prevPair, const float* prevPos);
void evoformerBlock(Trunk& t, bool extraStack, int blk);
// the pair modules, shared with the template stacks
// nextNorm: the next update's pair norm, emitted by this one's last GEMM where the matrix units run it
void triangleMultiplication(float* pair, const float* pairMask, int L, int C, const std::string& S, int blk, bool outgoing,
                            const std::string& nextNorm = "");
void triangleAttention(float* pair, const float* pairMask, int L, int C, const std::string& S, int blk, bool starting, bool pairOnes,
                       const std::string& nextNorm = "");
void transition(float* x, size_t rows, int C, const std::string& T, int blk);

// templates (templates.mm)
void templateEmbedding(float* pair, const float* pairMask, int L);
void templateEmbeddingMonomer(float* pair, const float* pairMask, int L, int T);
void templateRows(int L, int T, bool multimer, float* rows, float* rowMask);

// the structure module (structure.mm)
struct StructureOut { float* act; float* rigid; float* pos37; float* pos14; float* angles; };
StructureOut structureModule(const float* single, const float* pair, int L, float positionScale);
