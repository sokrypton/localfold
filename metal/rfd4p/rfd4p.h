// RFD4-Proteina natively on Metal: the design model `webgpu/rfd4p/` runs in the page and `cuda/rfd4p/` runs on
// NVIDIA, written for Apple's GPUs against metal/core. One port's worth of names and layouts, read from the PAGE's
// published int8 bundle (`model-rfd4p-int8`: manifest.json and its shards), which metal/core's `loadBundle` already
// decodes - its `dtype: int8` with `block: 64` and a float16 scale a block is the same codec the fold bundles use.
//
// 🔴 THE REFERENCE IS THE BUNDLE'S OWN ARITHMETIC, NOT THE VENDOR CHECKPOINT. `--oracle=` reads what
// `cuda/rfd4p/export_input.mjs --oracle=` wrote from `cpu/rfd4p/reference.js` over these same weights
// (tools/rfd4p-bundle-to-safetensors.mjs dequantises the shards for it), so the quantisation is on both sides of
// every comparison and cancels: a residual is this port's, not the weights'. It is the argument metal/bioemu2 makes
// for holding itself to the page's int5 bundle, and it is what makes a Mac with no 1.7 GB checkpoint able to gate.
#pragma once
#include "core.h"
#include "kernels.h"
#include "model.h"
#include <string>

using namespace mt;

// The trunk's widths. Read off the weights rather than from config.json, and checked: a bundle whose shapes disagree
// with what these kernels are built for is refused by name (`checkDims`).
struct Dims {
  int n = 0;            // tokens (the input's meta/n)
  int nRecycle = 1;
  int D = 768;          // token_dim
  int P = 384;          // pair_repr_dim
  int C = 256;          // dim_cond
  int H = 12;           // nheads
  int T = 256;          // tri_mult_c
  int layers = 14;
  int seqDim = 0;       // the sequence features' width (897 at this checkpoint)
  int condDim = 0;      // 609
  int pairDim = 0;      // 107
  int pairCondDim = 0;  // 512
};
extern Dims D_;

// ---------------------------------------------------------------- weights.mm
// the bundle under "w/", every tensor float32 (float16 where `--f16` asks and a GEMM can take it)
void loadWeights(const std::string& dir);
void checkDims();
extern bool HALF;

// ---------------------------------------------------------------- ops.mm
// Y[rows, out] = X[rows, in] W[out, in]^T (+ bias) - torch's layout, which every tensor in this checkpoint uses
void lin(const float* X, const std::string& w, float* Y, size_t rows, int in, int out, const float* bias = nullptr,
         bool relu = false);
// the same with HALF operands and a half result: the pair track's path under `--f16`, where staging an f32
// operand would mean a conversion pass of its own (measured at 18.6% of a step before these existed)
void linH(const half* X, const std::string& w, half* Y, size_t rows, int in, int out, const float* bias = nullptr,
          bool relu = false);
void linHF(const half* X, const std::string& w, float* Y, size_t rows, int in, int out, const float* bias = nullptr,
           bool relu = false);
// a tensor that may not be in this checkpoint (`tri_mult_bias: false` leaves the triangle's biases out, and the
// CPU reference reads every one of them optionally): its float32 bytes, or null
const float* opt(const std::string& name);
// y = x / sqrt(mean(x^2) + eps) * w, w optional (`nn.*_norm.weight`); rmsNormH also writes the halves
void rmsNorm(const float* x, float* y, size_t rows, int C, const float* w, float eps = 1e-5f);
// g[r][i] = silu(h[r][I + i]) * h[r][i], from the one [rows][2I] product `swish_linear.0` makes
void swiglu(const float* h, float* g, size_t rows, int I);
// one SwiGLU transition (`transition_c_1`, `transition_c_2`): dimInner = dim * expansion, no bias, no mask
void transition(const float* x, float* y, size_t rows, int dim, int expansion, const std::string& prefix, float* hid, float* gated);
// AdaptiveLayerNorm against a conditioning row per row: y = RMS(x) * sigmoid(gamma) + beta, where gamma and beta
// are that layer's projections of RMSNorm(cond, norm_cond.weight) - `prefix` is the adaLN's (e.g. "....mhba.adaln")
void adalnRows(const float* x, const float* cond, const std::string& prefix, float* y, size_t rows, int dim,
               int condDim = 0, bool ln = false, half* yh = nullptr);
// one SwiGLU transition with half operands throughout (`--f16`): the product, the gate and the projection
void transitionH(const half* x, half* y, size_t rows, int dim, int expansion, const std::string& prefix, half* hid,
                 half* gated);
// x += y * sigmoid(scale_output's gamma of `cond`) - AdaptiveOutputScale and the block's residual
void scaleAdd(float* x, const float* y, const float* cond, const std::string& prefix, size_t rows, int dim,
              int condDim = 0);
// rows picked out by index, and put back: `out[r] = in[idx[r]]` and `out[idx[r]] = in[r]`
void gatherRows(const float* in, float* out, const int* idx, size_t rows, int C);
void scatterRows(const float* in, float* out, const int* idx, size_t rows, int C);
// the largest n this port's float32 attention takes (its scores live in threadgroup memory)
constexpr int LF_ATTN_MAX_N = 6144;

// ---------------------------------------------------------------- decoder.mm
// The autoencoder's decoder: latents and alpha carbons in, all-atom coordinates out. Its widths are the bundle's
// `autoencoder_config.nn_ae.decoder` - 12 LayerNorm layers of token dim 768, conditioning width 128 and no pair
// track at all - and `cpu/rfd4p/decoder.js` is the spec.
struct DecDims { int layers = 12, D = 768, C = 128, H = 12, feat = 214, types = 147, latent = 32; };
extern DecDims DD_;
void checkDecoderDims();
// the decoder's input features [n][214] (ca_coors_nm, z_latent_seq, the token-type embedding, the C_SEQ
// embedding), built on the host as every other port builds them
void decoderFeatures(const float* caNm, const float* zLatent, const int* tokenType, const int* seq, const float* pad,
                     float* feats);
// the three heads over the ATTENDED rows: logits [na][147], the structure's 37 relative offsets [na][111], and
// each slot's presence [na][37]. `s` is the finished stack; `na` rows of it.
// `sInitOut` and `perLayer` ([layers][na][D]) are the gate's taps and may be null; DEC_TAP_ADA and DEC_TAP_ATTN
// take layer 0's adaLN and attention outputs the same way (the oracle records both).
void decode(const float* feats, const int* attnIdx, int na, float* sOut, float* logits, float* coors, float* presence,
            float* sInitOut = nullptr, float* perLayer = nullptr);
extern float* DEC_TAP_ADA;
extern float* DEC_TAP_ATTN;
// one decoder layer in place on `rows` gathered rows - the gate's per-stage arm drives it directly
void decoderLayerOne(int layer, float* s, const float* c, size_t rows);
// one layer of either half of the autoencoder ("autoencoder.decoder." / "autoencoder.encoder.")
void aeLayer(const std::string& prefix, int layer, float* s, const float* c, size_t rows);

// ---------------------------------------------------------------- encoder.mm
// The autoencoder's encoder: all-atom coordinates in, a token's 32 local latents out - what a partial-diffusion
// or inverse-folding run starts from. `cpu/rfd4p/encoder.js` is the spec; the layers are the decoder's.
struct EncDims { int layers = 12, D = 768, C = 128, H = 12, feat = 1587, latent = 32; };
extern EncDims ED_;
void checkEncoderDims();
// the encoder's [n][1587] features, built on the host from the structure (its offsets read off the input's
// metadata, which the featuriser derives from `feats_seq` rather than anybody typing them in)
void encoderFeatures(int n, const float* coorsNm, const float* atomMask, const float* chainIid,
                     const float* bondCounts, const int* element, const int* charge, float* feats);
// the latents: `mean` and `logScale` over the attended rows ([na][32] each). The sampled z is the mean plus
// eps * exp(logScale); the gate's arm is the deterministic one (z = mean), as the reference's is.
void encode(const float* feats, const int* attnIdx, int na, float* mean, float* logScale);

// ---------------------------------------------------------------- trunk.mm
// c = transition_c_2(transition_c_1(cond_factory(features))) over `rows` conditioning rows
void condC(const float* condFeats, float* c, size_t rows);
// s_init = init_repr_factory(seq features)
void initSingle(const float* seqFeats, float* sInit);
// z_init = AdaLN(RMSNorm(pair_repr_builder(pair features)), RMSNorm(pair cond)) - the pair conditioning is one row
void initPair(const float* pairFeats, const float* pairCondFeats, float* zInit);
// one `MultiheadAttnAndTransition` block, in place on `s`: the pair-biased gated attention under its adaLN and
// output scale, then the SwiGLU transition under its own, each with a residual
void tokenLayer(int layer, float* s, const float* z, const float* c);
// one `PairReprUpdate`, in place on `z`: the hadamard pair write from the single track, the outgoing and incoming
// triangle multiplications, and the pair transition - each with a residual
void pairUpdate(int layer, const float* s, float* z);
// the finished trunk's outputs: the alpha carbons (3), the local latents (32), the sequence logits (147), and the
// insertion logits (n x n, or null for none)
void heads(const float* s, float* bbCa, float* localLatents, float* logits, float* insertion);
// one cycle's input: s = s_init + Lin(RMSNorm(s)), z = z_init + Lin(RMSNorm(z))
void recycleIn(const float* sInit, const float* zInit, float* s, float* z);
// the whole trunk over one step's features; `s` and `z` must start at zero, `insertion` may be null
void trunkForward(const float* seqF, const float* cond0, const float* cond1, const float* pairF,
                  const float* pairCondF, int nRecycle, float* s, float* z, float* bbCa, float* latents,
                  float* logits, float* insertion);

// ---------------------------------------------------------------- sampler.mm
// The flow sampler: `shared/rfd4p/flow.js`'s schedules and `simulation_step`, over a generation exported by
// `cuda/rfd4p/export_input.mjs --generate` (the features once, the noise replayed, the per-step conditioning
// precomputed). What moves per step is 35 of the 897 sequence columns and the pair track's 30 distance bins.
struct Modality {
  std::vector<double> ts, gt;      // nsteps + 1 times, nsteps noise levels
  int mode = 0;                    // 0 vf_tsr, 1 vf, 2 sc
  double tsrK = 1, tsrSigma = 1, scn = 1, odeLim = 1, odeBelow = 0;
  bool center = false;
};
// one modality's schedule, built from the exported sampling metadata (`samplingMeta`'s fields)
Modality modalitySchedule(const char* tag, int nsteps);
bool stochasticStep(double t, const Modality& m);
// a whole generation from an exported `-gen` directory. When `GATE_CHECK` is set, every step's noised state and
// clean estimate are held to the dump's own trajectory as they are produced.
int generate(const std::string& genDir, const std::string& outPath, int designs = 1, uint64_t seed = 0);
// the gate's comparison, set by main where there is an oracle to compare against: ours (on the device) against
// the dump's tensor `key`, at `bound`, returning the relative RMS. Null when nothing is being held.
extern double (*GATE_CHECK)(const char* label, const std::string& key, const float* ours, size_t n, double bound);
extern double GATE_TRAJ_BOUND;
// the noised state's own bound: a step's `score` divides by (1 - t), so at the last step the velocity's error
// arrives amplified where the clean estimate x1 - which does not divide by anything - does not
extern double GATE_TRAJ_XT_BOUND;

// ---------------------------------------------------------------- chirality.mm
// `align_chirality`, recomputed every step: the clean packed vectors rotated onto the directions their frame's
// neighbours have in the current noisy state. `out` is [n][18] and starts as a copy of `chirClean`.
void alignChirality(int n, const float* caNm, const float* chirClean, const int* frames, const int* owners,
                    const float* pad, int frameWidth, float* out);

// ---------------------------------------------------------------- output.mm
// the decoder's C_SEQ: the job's conditioned residues kept, an emitted token with none taking the argmax of the
// trunk's logits over its TOKEN TYPE's allowed residues. Returns how many it filled.
int sequenceForDecoder(int n, const float* logits, const int* condSeq, const int* tokenType, const float* emit,
                       int* seq);
// the decoder's heads turned into a structure: the residue type, which of the 37 slots exist, and their
// coordinates in angstrom (`decoderPostprocess` + `formatSampleLocalLatents`)
void decoderPostprocess(int n, const float* caNm, const int* seq, const float* logits, const float* coorsFlat,
                        const float* presence, const float* emit, const int* coordGroup, const int* attnIdx, int na,
                        float* coors, uint8_t* atomMask, int* residueType);
// the input's own per-token naming, where a job named a structure (the featuriser's `outputTables`): absent for
// a de-novo design, and `present` says which
struct OutTables {
  bool present = false;
  const int* atomize = nullptr; const int* hetero = nullptr; const int* resName = nullptr;
  const int* inName = nullptr; const int* inElem = nullptr; const int* inCharge = nullptr;
  const int* insCode = nullptr; const int* entity = nullptr;
  // b-factor and occupancy are DOUBLES packed as two ints each (the exporter's `packDouble`)
  const int* inB = nullptr; const int* inOcc = nullptr;
};
// a double the exporter packed into two ints, low word first
double unpackDouble(const int* w);
// the structure written - mmCIF where the path ends `.cif` (RFD4-Proteina's own format), PDB otherwise - and the
// consecutive alpha-carbon spacing within a chain (median, min, max) as the geometry check
int writeStructure(const std::string& path, int n, const float* coors, const uint8_t* atomMask,
                   const int* residueType, const int* outResId, const int* outChain, const int* chainIid,
                   double* caSpacing, const OutTables& in);
