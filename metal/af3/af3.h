// Native AlphaFold 3 lineage on Metal: the stages' declarations (metal/af3/*.mm). cuda/af3 is the reading - the same
// stages and arithmetic, its --fast path, on metal/core: GEMMs in half with float accumulation, attention on the core's
// flash kernel, residual streams float32. One code path for every family; the bundle's dialect (trunk.dialect.*)
// and its weights' presence choose the branches, as cuda/af3's do.
#pragma once
#include "core.h"
#include "kernels.h"
#include "model.h"
#include <functional>
#include <string>
#include <vector>

using namespace mt;

// ---------------------------------------------------------------- names
inline bool hasW(const std::string& k) { return M.has(k); }
inline const float* W(const std::string& k) { return M.f(k); }
inline const float* Wopt(const std::string& k) { return M.has(k) ? M.f(k) : nullptr; }
inline const half* Wh(const std::string& k) { return M.h(k); }
inline size_t lenW(const std::string& k) { return M.len(k); }
inline bool flag(const std::string& k) { return M.meta(k, 0) != 0; }
inline int metaI(const std::string& k) { return (int)M.meta(k); }
inline std::string num(long long k) { return std::to_string(k); }

// ---------------------------------------------------------------- operations (ops.mm)
// Y = alpha X W (+ beta Y) (+ bias) for a named weight [in][out], its size checked; operands multiplied in half
void lin(const float* X, const std::string& w, float* Y, size_t rows, int in, int out, float beta = 0.f,
         const float* bias = nullptr, float alpha = 1.f);
void lin(const half* X, const std::string& w, float* Y, size_t rows, int in, int out, float beta = 0.f,
         const float* bias = nullptr, float alpha = 1.f);
void linH(const half* X, const std::string& w, half* Y, size_t rows, int in, int out, const float* bias = nullptr);
void linH(const float* X, const std::string& w, half* Y, size_t rows, int in, int out, const float* bias = nullptr);
// a weight already on the device ([in][out] half)
void linW(const half* X, const half* w, float* Y, size_t rows, int in, int out, float beta = 0.f, const float* bias = nullptr,
          float alpha = 1.f, const char* label = nullptr);
void linW(const half* X, const half* w, half* Y, size_t rows, int in, int out, const float* bias = nullptr, const char* label = nullptr);
// LayerNorm by weight names (an empty or absent offset: none)
void ln(const float* x, float* y, size_t rows, int C, const std::string& scale, const std::string& offset);
void ln(const float* x, half* y, size_t rows, int C, const std::string& scale, const std::string& offset);
void scaleRows(float* x, const float* mask, size_t rows, int C, size_t period);
void scale(float* x, size_t n, float s, bool relu = false);

// derived weights: columns concatenated (a part [C][width], or stored [width][C] if transposed; no name: zeros)
struct Part { std::string name; int width; bool transposed; };
const half* concatColumns(const std::string& key, int C, const std::vector<Part>& parts);
// [rows][2I] = [a | b] -> gemmSwiglu's interleaving
const half* swigluPairs(const std::string& key, const half* w, int rows, int I);
// an attention's q | k | v | gate as one [C][4W] weight (the grid attention stores q, k and the gate transposed),
// and its GEMM bias where the gate has one (zeros elsewhere)
const half* qkvgWeight(const std::string& pre, int C, int Wd, bool transposedQkg);
const float* qkvgBias(const std::string& pre, int Wd);

// ---------------------------------------------------------------- the pair track (pairtrack.mm)
// pairMask [n * n] (a real buffer; every update reads it), seqMask [n], `ones` every token real (no attention mask)
struct Masks { const float* pair; const float* seq; bool ones; };
void triangle(float* pair, const Masks& m, int n, int C, const std::string& pre, bool outgoing, bool divide);
void gridAttention(float* pair, const Masks& m, int n, int C, const std::string& pre, bool tr, bool swap);
void transition(float* x, size_t rows, int C, const std::string& pre);
void pairUpdates(float* pair, const Masks& m, int n, int C, const std::string& pre);
void singleTrack(float* single, const float* pair, const Masks& m, int n, int C, int Cs, const std::string& B,
                 const float* extraBias = nullptr);
void pairformerBlock(float* pair, float* single, const Masks& m, int n, int C, int Cs, const std::string& B,
                     const float* extraBias = nullptr);
// raw [pairs][heads] -> [heads][n][stride] half, log2-scaled (swap: (i, j) read at (j, i))
half* biasLayout(const float* raw, const std::string& name, int n, int heads, int stride, bool swap);

// ---------------------------------------------------------------- the trunk (trunk.mm)
struct Trunk {
  int n, S, C, Cs, Cm, F;                 // tokens, MSA rows, pair, single, msa channels, target_feat width
  float *pair, *single, *msa, *targetFeat, *pairMask, *seqMask, *msaMask, *deletion;
  int* msaRows;
  float *prevPair, *prevSingle;
  Masks masks;
  int pass = 0;
};
Trunk makeTrunk(const float* targetFeat, int msaCap);
void freeTrunk(Trunk& t);
void runTrunk(Trunk& t);
// the distogram's logits [n * n * bins] (symmetrised), and the page's contact map [n * n] on the host
void distogram(Trunk& t, float* logits);
std::vector<float> contactProbabilities(Trunk& t);

// ---------------------------------------------------------------- target_feat (atom.mm)
float* buildTargetFeat();

// ---------------------------------------------------------------- the atoms (atom.mm)
// Diffusion samples in flight through the denoiser's per-step path: every per-sample tensor is NS copies,
// sample-major; what the samples share (conditioning, masks, biases) is read at row % one sample's rows
extern int NS;
extern bool ADA_RAW;                       // chai-1's adaptive LayerNorm form
struct Gather { const int* idx; const float* mask; int count; };
Gather gatherOf(const std::string& name);
struct AtomShape { int tokens, dense, subsets, queries, keys; };
AtomShape atomShape();
void convert(const Gather& g, const float* src, float* out, int C, size_t srcRows = 0, int ns = 1);
// everything a cross-attention block derives from the conditioning, the same at every denoiser step
struct AtomBlockCache { float *qScale, *qShift, *kScale, *kShift, *zg, *ffwScale, *ffwShift, *tg, *pairLogits; bool chained; };
struct EncoderOut {
  float* tokenAct;        // [NS tokens][perToken]
  float* skip;            // [NS queryRows][C]
  float *qMask, *kMask, *qCond, *kCond, *pair, *qUid, *kUid, *qStart;
  std::vector<AtomBlockCache> blocks;
  int C, heads, D, perToken;
  bool keyMasked, noResidual;
};
EncoderOut prepareEncoder(const std::string& E, const std::string& ref, const float* trunkSingle, const float* trunkPair);
void encoderStep(const std::string& E, EncoderOut& o, const float* atomPositions);
std::vector<float*> atomPairLogits(const std::string& P, const float* pair, size_t pairRows, int Cp, int nblocks, int heads,
                                   const AtomShape& sh);
AtomBlockCache prepareAtomBlock(const std::string& B, const float* qCond, size_t qRows, int C, float* pairLogits);
struct AtomStep { Gather queriesToKeys; const float *qMask, *kMask; bool keyMasked, noResidual; };
void crossAttentionBlock(float* act, const AtomStep& st, const AtomBlockCache& bc, const AtomShape& sh, int C, int heads, int D,
                         const std::string& B);
// the adaptive LayerNorm: ADA(scale) LN(x) + shift, scale and shift shared every `period` rows
void adaLn(const float* x, const float* scale, const float* shift, half* out, float* outF, size_t rows, int C, size_t period,
           int ld = 0);

// ---------------------------------------------------------------- the diffusion head and its sampler (diffusion.mm)
struct SamplerOptions { bool flow = false; double sigmaMax = 0; };
extern SamplerOptions SAMPLER;
void prepareDiffusion(const float* trunkSingle, const float* trunkPair, const float* targetFeat, const Masks& masks, int n);
// every sample's final positions [ns][atoms][3]; onStep(the denoised positions on the device, step, steps)
std::vector<float> sample(int steps, const std::vector<uint64_t>& seeds, const std::vector<float>& mask,
                          const std::function<void(const float*, int, int)>& onStep = nullptr);
void freeDiffusion();
void reportStages();     // (AF3_STAGES=1: the denoiser's stages timed, a sync between them)

// ---------------------------------------------------------------- the confidence head (confidence.mm)
struct ConfidenceOut { std::vector<float> plddt, pae, pde, tmTerm; double meanPlddt, ptm, iptm; };
ConfidenceOut confidenceHead(const Trunk& t, const float* pseudoBeta);
