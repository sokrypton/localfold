// Native BioEmu2 on Metal: the stages' declarations (metal/bioemu2/*.mm). cuda/bioemu2 is the reading - the same
// stages and arithmetic, here on metal/core - and the weights are the PAGE'S int5 bundle (model-bioemu2-int5), so this
// port and webgpu/bioemu2 read the same numbers and tools/dump-bioemu2-webgpu.mjs is an oracle with the quantisation
// cancelled (docs/BIOEMU2.md).
//
// The bundle's layouts (tools/export_bioemu2_model.py): the PAIR TRACK - the MSA module's 4 and the PairMixer's 48
// triangle multiplications and pair transitions - is AF3's, (in, out), under msa/<l>/ and pairmixer/<l>/: the triangle's
// a and b interleaved by channel and the INCOMING one's halves swapped (so AF3's incoming einsum computes Boltz's), the
// transition's [gate | value] blocked. Everything else keeps the vendor's name and TORCH layout, (out, in), with the
// stacked layers unstacked into <stack>/<l>/<rest>.
#pragma once
#include "core.h"
#include "kernels.h"
#include "host.h"
#include "model.h"
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace mt;

constexpr int WQ = 32, WK = 128, CTX = (WK - WQ) / 2;   // atom query / key windows; keys centred on queries
constexpr float SIGMA_DATA = 16.f;

// ---------------------------------------------------------------- weights and inputs (ops.mm)
// af3-any-model's int8 blob (fetch.h's `bioemu2`, what cuda/bioemu2 folds with) read AS the page's bundle - the same
// names and layouts under "w/" (weights.mm); asHalf: which tensors to decode to float16
void loadBlobAsBundle(const std::string& dir, const std::function<bool(const std::string&, size_t)>& asHalf);
// the bundle's tensors under "w/"
bool hasW(const std::string& n);
const float* Wf(const std::string& n);
const float* WfOpt(const std::string& n);      // (null where the bundle has none)
int dimW(const std::string& n, int k);
size_t lenW(const std::string& n);
inline const float* Fin(const std::string& n) { return M.f(n); }
inline const int* Iin(const std::string& n) { return M.i(n); }
inline std::string num(int k) { return std::to_string(k); }
// run1d for a kernel that splits its index with 32-bit arithmetic (bioemu2.metal's BE2_SPLIT): refused past 2^32 elements
template <class A> void be2Run1d(const char* kernel, size_t n, const A& a) {
  if (n >> 32) die("BioEmu2: %s over %zu elements (its index is 32-bit)", kernel, n);
  run1d(kernel, n, a);
}

// Y [rows][out] (= beta Y +) X [rows][in] W for a named weight: linT a torch (out, in) one, linA an AF3 (in, out) one.
// Its shape is checked against in and out - a transposed weight read as square is the bug this catches
// HALF (the fast arm, the default; --f32 is the reference): float16 weights, X rounded as it is staged, float accumulation -
// `wide` keeps an input that may pass float16's range in float
extern bool HALF;
void prepareWeights();
void linT(const float* X, const std::string& w, float* Y, size_t rows, int in, int out, float beta = 0.f, int ldx = 0, int ldy = 0,
          bool wide = false, const float* bias = nullptr);     // (bias: added in the GEMM's epilogue)
bool linT(const half* X, const std::string& w, float* Y, size_t rows, int in, int out, float beta = 0.f,
          const float* bias = nullptr, half* lnOut = nullptr, const float* lnScale = nullptr, const float* lnOffset = nullptr);
void linA(const float* X, const std::string& w, float* Y, size_t rows, int in, int out, float beta = 0.f, int ldx = 0, int ldy = 0,
          bool wide = false);
void relu(const float* x, float* y, size_t n);
void silu(const float* x, float* y, size_t n);
// the vendor's SwiGLU transition (fc1 / fc2 / fc3, torch): out (+)= fc3(silu(fc1 x') fc2 x'), x' = LN(x) under `norm`
// (p + "norm/weight", "norm/bias"); ldx: x's row stride
void transition(const float* x, float* out, size_t rows, int C, const std::string& p, bool residual = true, bool norm = true,
                int ldx = 0, const float* nextScale = nullptr, const float* nextOffset = nullptr,
                const float* nextScale2 = nullptr, const float* nextOffset2 = nullptr);
// ...on the fast arm, from an input already in half (an adaLN that wrote it so): no norm
// (gate: out += gate[map(r)] fc3(...) - the adaLN-gated residual in fc3's epilogue where metal/core takes it, else fc3
// and be2_gated_add; gate rows ldg apart)
void transitionHalf(const half* xh, float* out, size_t rows, int C, const std::string& p, bool residual,
                    const float* gate = nullptr, int ldg = 0, const int* map = nullptr, const float** deferH = nullptr,
                    const float* nextScale = nullptr, const float* nextOffset = nullptr,
                    const float* nextScale2 = nullptr, const float* nextOffset2 = nullptr);
// (its two halves, for the coupling split: the [C][2I] blocks-of-8 [fc1 | fc2 / 64] weight, and fc3 over the gated
// product scaled back by 64)
const half* swigluPairs(const std::string& p, int C, int I);
const half* triGateWeight(const std::string& p, int C);
const half* pairTransitionWeight(const std::string& p, int C, int I);
const half* pwaMgWeight(const std::string& p, int Cm, int HCh);
const half* qkvgWeight(const std::string& a, int D);
float swigluScale();      // (fc2's divisor in every derived SwiGLU weight: 64)
void transitionFc3(const half* g, float* out, size_t rows, int I, const std::string& p, bool residual,
                   const float* gate = nullptr, int ldg = 0, const int* map = nullptr, const float** deferH = nullptr,
                   const float* nextScale = nullptr, const float* nextOffset = nullptr,
                   const float* nextScale2 = nullptr, const float* nextOffset2 = nullptr);
// AF3's pair transition (inputLayerNorm, transition1 [C][2I] blocked [gate | value], transition2 [I][C]), into z
void pairTransition(float* z, size_t rows, int C, const std::string& p,
                    const float* nextScale = nullptr, const float* nextOffset = nullptr);
// AF3's triangle multiplication on the bundle's layout (leftNormInput, projection / gate [C][2C], centerNorm,
// outputProjection, gatingLinear), into z
void triangle(float* z, int T, int C, const std::string& p, bool outgoing,
              const float* nextScale = nullptr, const float* nextOffset = nullptr);
half* takeEmitted(const void* x, const float* scale, const float* offset);
half* lnTarget(size_t P, int C, const half* inUse = nullptr, const char* tag = "be2.ftri.xn");
void recordEmitted(const void* x, const float* scale, const float* offset, half* buf);
void forgetEmitted(const void* x = nullptr);
// the relative position encoding (p + "embed_rel_pos/weight" ...) over the tokens, into out (+= under add)
void relPos(const std::string& p, float* out, int T, int C, bool add, int ldo = 0);

using Tap = std::function<void(const char*, const float*, size_t)>;
struct Dims { int T = 0, N = 0, S = 0, nValidRows = 1; };
extern Dims D_;

// ---------------------------------------------------------------- the atoms (embed.mm)
void atomFeatures(const std::string& p, float* c, size_t N);
void pairGeometry(const std::string& p, float* pr, int N);
void pairTail(const std::string& p, float* pr, const float* c, int N);
void windowBias(const float* pr, const float* normW, const float* projZ, float* out, int N, int H, int layers = 1);
// ConditionedTransformerLayer (adaLN-Zero), in place on x [R][D]: `p` the layer's prefix, `att` its attention's;
// windowed: the sliding-window atom attention over the windowed pair's bias, else the full token attention over [H][T][T].
// The conditioning [Rc][6D] is reached through `map` (null: the identity); perSample rows a stacked sample
// adaLN: y = LN(x) (1 + alpha) + beta, alpha / beta the conditioning's rows map(r) (stride ldc; map null: the identity)
// (yh: the output in half instead - the fast arm's GEMM input, with no conversion pass between; resH / resGate: an
// upstream x += resGate resH fused in place before the norm)
void adaLn(float* x, float* y, size_t rows, int C, const float* alpha, const float* beta, int ldc, const int* map,
           half* yh = nullptr, const float* resH = nullptr, const float* resGate = nullptr, int ldrg = 0);
struct CondLayer { std::string p, att; bool windowed; int heads; };
float* layerCond(const CondLayer& L, const float* c, int Rc, int Cc, int D, const char* name);
// biasH: the token attention's bias in half, log2 units - the fast arm runs it on metal/core's flash attention
void condLayer(const CondLayer& L, float* x, int R, int D, const float* cond, const int* map, const float* bias, int perSample,
               const half* biasH = nullptr, const float* inResH = nullptr, const float* inResGate = nullptr, int inResLdg = 0,
               const float** outResH = nullptr, bool preNorm = false);
void inputEmbedder(float* sInputs, int Si, const Tap& tap);

// ---------------------------------------------------------------- the trunk (trunk.mm)
extern int* TOK_START;                 // each token's atoms: [start[t], start[t + 1])
void buildTokenStarts(int T, int N);
void trunkInit(const float* sInputs, int Si, float* sInit, float* zInit, const Tap& tap);
void trunkPass(const float* sInputs, int Si, const float* sInit, const float* zInit, float* s, float* z, bool nextFollows = false);
void distogram(const float* z, float* logits);
// the contact probabilities [T * T] on the host: the distogram's mass up to 8 A
std::vector<float> contactProbs(const float* logits, int T, int bins);

// ---------------------------------------------------------------- the denoiser (denoise.mm)
struct Denoiser {
  int T = 0, N = 0, Si = 0, layers = 0;
  float* z = nullptr;          // [P, 256] the pairwise conditioner's pair
  float* q0 = nullptr;         // [N, 128]
  float* p = nullptr;          // [N, 128, 16] the windowed pair
  float* wbias = nullptr;      // [layers][NW, 4, 32, 128]
  float* tbias = nullptr;      // [layers][16, T, T] (the reference arm)
  half* tbiasH = nullptr;      // [layers][16, T, T] in half, log2 units: metal/core's flash attention's (the fast arm)
  float* sc0 = nullptr;        // [T, 768] single_embed(LN [s_trunk | s_inputs]) (+ the lag's time embedding)
  float* a0 = nullptr;         // [T, 768] s_trunk_to_a(LN s_trunk)
  float dynGate = 0.f;         // the conditioning frame's gate (zero unconditioned)
  bool dyn = false;
  const float* coordsCond = nullptr;
};
Denoiser staticContext(const float* sTrunk, const float* zTrunk, const float* sInputs, int Si, const Tap& tap);
void setDynamics(Denoiser& d, float dt, const float* coordsCond);
void freeDenoiser(Denoiser& d);
void forgetRowMaps();             // (every fold: the row maps are an input's)
u64* pointerArray(const std::string& key, const std::vector<u64>& v);
void setSigmaSchedule(const float* sigmas, int n);
// x [samples][N][3] (noisy, angstrom) at sigma -> den; cond: each sample's conditioning frame (null: coords_cond)
void denoise(const Denoiser& d, const float* x, float sigma, float* den, const float* cond = nullptr, int samples = 1);

// ---------------------------------------------------------------- the sampler and the output (sampler.mm)
struct SamplerSettings { int steps = 200; double sMax = 160, sMin = 4e-4, rho = 7, gamma0 = 0.8, gammaMin = 1.0,
                         noiseScale = 1.0, stepScale = 1.5; };
extern int BATCH_SAMPLES;
// onStep(step, steps, the first particle's denoised estimate [N][3] on the host): the page's live frames
using StepHook = std::function<void(int, int, const std::vector<float>&)>;
std::vector<std::vector<float>> sample(const Denoiser& d, const SamplerSettings& s, uint64_t seed, int n,
                                       std::vector<std::vector<float>> companions = {}, bool centreEnd = true,
                                       const StepHook& onStep = nullptr);
std::vector<std::vector<std::vector<float>>> propagate(const Denoiser& d, const SamplerSettings& s, uint64_t seed, int n,
                                                       int chain, const std::vector<float>& x0);
void centreMasked(std::vector<float>& x, const std::vector<float>& mask);
// a uniform random rotation (a normalised Gaussian quaternion) and a N(0, 1) translation: the sampler's augmentation
void randomRigid(std::mt19937_64& rng, double R[9], double sh[3]);
// the steered samplers (steer.mm: cuda/bioemu2's steer.h over this port's denoiser) - edm_gd and edm_fk; edm_fkc refused.
// potentials: a --potentials file, or empty for the vendor's defaults. Throws a sentence on a refusal
std::vector<std::vector<float>> steeredFold(const Denoiser& d, const SamplerSettings& ss, const std::string& sampler,
                                            const std::string& potentials, uint64_t seed, int n);
void writePdb(const std::string& templatePath, const std::string& out, const std::vector<float>& x);
void writeConfidences(const std::string& dir, const std::string& pdb, int T, const std::vector<float>& contacts);
