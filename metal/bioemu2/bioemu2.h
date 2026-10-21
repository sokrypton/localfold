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
#include <string>
#include <vector>

using namespace mt;

constexpr int WQ = 32, WK = 128, CTX = (WK - WQ) / 2;   // atom query / key windows; keys centred on queries
constexpr float SIGMA_DATA = 16.f;

// ---------------------------------------------------------------- weights and inputs (ops.mm)
// the bundle's tensors under "w/", float32 (every one decoded so on this arm)
bool hasW(const std::string& n);
const float* Wf(const std::string& n);
const float* WfOpt(const std::string& n);      // (null where the bundle has none)
int dimW(const std::string& n, int k);
size_t lenW(const std::string& n);
inline const float* Fin(const std::string& n) { return M.f(n); }
inline const int* Iin(const std::string& n) { return M.i(n); }
inline std::string num(int k) { return std::to_string(k); }

// Y [rows][out] (= beta Y +) X [rows][in] W for a named weight: linT a torch (out, in) one, linA an AF3 (in, out) one.
// Its shape is checked against in and out - a transposed weight read as square is the bug this catches
void linT(const float* X, const std::string& w, float* Y, size_t rows, int in, int out, float beta = 0.f, int ldx = 0, int ldy = 0);
void linA(const float* X, const std::string& w, float* Y, size_t rows, int in, int out, float beta = 0.f, int ldx = 0, int ldy = 0);
void relu(const float* x, float* y, size_t n);
void silu(const float* x, float* y, size_t n);
// the vendor's SwiGLU transition (fc1 / fc2 / fc3, torch): out (+)= fc3(silu(fc1 x') fc2 x'), x' = LN(x) under `norm`
// (p + "norm/weight", "norm/bias"); ldx: x's row stride
void transition(const float* x, float* out, size_t rows, int C, const std::string& p, bool residual = true, bool norm = true,
                int ldx = 0);
// AF3's pair transition (inputLayerNorm, transition1 [C][2I] blocked [gate | value], transition2 [I][C]), into z
void pairTransition(float* z, size_t rows, int C, const std::string& p);
// AF3's triangle multiplication on the bundle's layout (leftNormInput, projection / gate [C][2C], centerNorm,
// outputProjection, gatingLinear), into z
void triangle(float* z, int T, int C, const std::string& p, bool outgoing);
// the relative position encoding (p + "embed_rel_pos/weight" ...) over the tokens, into out (+= under add)
void relPos(const std::string& p, float* out, int T, int C, bool add, int ldo = 0);

using Tap = std::function<void(const char*, const float*, size_t)>;
struct Dims { int T = 0, N = 0, S = 0, nValidRows = 1; };
extern Dims D_;

// ---------------------------------------------------------------- the atoms (embed.mm)
void atomFeatures(const std::string& p, float* c, size_t N);
void pairGeometry(const std::string& p, float* pr, int N);
void pairTail(const std::string& p, float* pr, const float* c, int N);
void windowBias(const float* pr, const float* normW, const float* projZ, float* out, int N, int H);
// ConditionedTransformerLayer (adaLN-Zero), in place on x [R][D]: `p` the layer's prefix, `att` its attention's;
// windowed: the sliding-window atom attention over the windowed pair's bias, else the full token attention over [H][T][T].
// The conditioning [Rc][6D] is reached through `map` (null: the identity); perSample rows a stacked sample
// adaLN: y = LN(x) (1 + alpha) + beta, alpha / beta the conditioning's rows map(r) (stride ldc; map null: the identity)
void adaLn(const float* x, float* y, size_t rows, int C, const float* alpha, const float* beta, int ldc, const int* map);
struct CondLayer { std::string p, att; bool windowed; int heads; };
float* layerCond(const CondLayer& L, const float* c, int Rc, int Cc, int D, const char* name);
void condLayer(const CondLayer& L, float* x, int R, int D, const float* cond, const int* map, const float* bias, int perSample);
void inputEmbedder(float* sInputs, int Si, const Tap& tap);

// ---------------------------------------------------------------- the trunk (trunk.mm)
extern int* TOK_START;                 // each token's atoms: [start[t], start[t + 1])
void buildTokenStarts(int T, int N);
void trunkInit(const float* sInputs, int Si, float* sInit, float* zInit, const Tap& tap);
void trunkPass(const float* sInputs, int Si, const float* sInit, const float* zInit, float* s, float* z);
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
  float* tbias = nullptr;      // [layers][16, T, T]
  float* sc0 = nullptr;        // [T, 768] single_embed(LN [s_trunk | s_inputs]) (+ the lag's time embedding)
  float* a0 = nullptr;         // [T, 768] s_trunk_to_a(LN s_trunk)
  float dynGate = 0.f;         // the conditioning frame's gate (zero unconditioned)
  bool dyn = false;
  const float* coordsCond = nullptr;
};
Denoiser staticContext(const float* sTrunk, const float* zTrunk, const float* sInputs, int Si, const Tap& tap);
void setDynamics(Denoiser& d, float dt, const float* coordsCond);
void freeDenoiser(Denoiser& d);
// x [samples][N][3] (noisy, angstrom) at sigma -> den; cond: each sample's conditioning frame (null: coords_cond)
void denoise(const Denoiser& d, const float* x, float sigma, float* den, const float* cond = nullptr, int samples = 1);

// ---------------------------------------------------------------- the sampler and the output (sampler.mm)
struct SamplerSettings { int steps = 200; double sMax = 160, sMin = 4e-4, rho = 7, gamma0 = 0.8, gammaMin = 1.0,
                         noiseScale = 1.0, stepScale = 1.5; };
extern int BATCH_SAMPLES;
std::vector<std::vector<float>> sample(const Denoiser& d, const SamplerSettings& s, uint64_t seed, int n,
                                       std::vector<std::vector<float>> companions = {}, bool centreEnd = true);
std::vector<std::vector<std::vector<float>>> propagate(const Denoiser& d, const SamplerSettings& s, uint64_t seed, int n,
                                                       int chain, const std::vector<float>& x0);
void centreMasked(std::vector<float>& x, const std::vector<float>& mask);
void writePdb(const std::string& templatePath, const std::string& out, const std::vector<float>& x);
void writeConfidences(const std::string& dir, const std::string& pdb, int T, const std::vector<float>& contacts);
