/**
 * AF3's confidence head in WebGL2: pLDDT, PAE and PDE.
 *
 * This is what tells a reader whether to believe the structure, and it is the
 * only part of AF3 that reads the SAMPLED COORDINATES back in - everything else
 * runs before there are any.
 *
 *     pair  += target_feat (both ways) + a distogram of the predicted structure
 *     4 x pairformer block, the same module the trunk runs 48 of
 *     pair  -> distance-error logits (symmetrised) -> PDE
 *           -> aligned-error logits                -> PAE
 *     single -> per-atom pLDDT logits              -> pLDDT
 *            -> per-atom resolved logits
 *
 * 🔴 pLDDT IS PER ATOM SLOT, NOT PER TOKEN. Its projection is (384, 24, 50):
 * one 50-bin distribution for each of a token's 24 dense slots. "Reading it as
 * (384, 50) and broadcasting would run, produce plausible per-residue numbers,
 * and throw away the side-chain resolution that is the point of an atom-level
 * model."
 *
 * 🔴 "left" AND "right" ARE THE OTHER WAY ROUND. AF3's `_embed_features` writes
 * `left_target_feat_project(tf)` with no axis expansion and
 * `right_target_feat_project(tf)[:, None]` with one, so the LEFT projection is
 * indexed by **j** and the right by **i**. "Swapping them transposes a term
 * nothing downstream will complain about."
 *
 * 🔴 THE PDE HEAD IS SYMMETRISED AND THE PAE HEAD IS NOT, because PAE is
 * directional - "how wrong is j when aligned on i". One projection used twice
 * for the first (AF3 sets `right = left` explicitly) and the plain one for the
 * second; swapping which gets the transpose is two plausible tensors.
 *
 * 🔴 AND A HEAD LayerNorm THE BUNDLE DOES NOT CARRY IS NO LayerNorm, NOT ONE AT
 * SCALE 1: normalising still re-centres and rescales. boltz2 calls every logit
 * head directly on z and s, so each norm here is present only where its scale
 * is - which `kernels/layer-norm.js` supports by generating a source without
 * the samplers rather than binding ones and zeros.
 *
 * 🔴 AND THE ONE-HOT DISTOGRAM IS FUSED WITH ITS PROJECTION. The reference
 * materialises `pairs x 39` one-hots and runs a Linear over them; exactly one
 * bin a pair is non-zero, so that Linear is a GATHER of one weight row times
 * the pair mask. One draw instead of two and no `pairs x 39` tensor - which at
 * 300 tokens is 14 MB that only ever holds a single 1. The risk the fusion
 * carries is the bin edge, which is what the gate's arms are on.
 *
 * 🔴 AND rf3 BINS THE SAME RANGE ONE MORE TIME AND COUNTS DIFFERENTLY: 39
 * BOUNDARIES and 40 bins, "the index being how many of them the distance
 * exceeds - so it has a bin BELOW 3.25 that AF3 has no equivalent of". The two
 * are not a reparameterisation of each other, the bundle says which
 * (`distogram_feat_project` is [40, 128] against [39, 128]), and this port
 * REFUSES where the weight and the dialect disagree.
 *
 * 🔴 AND rf3's GLOBAL NORM IS MASKED, WHICH IS WHY IT IS NOT A ONE-LINER. "A
 * statistic that reduces over more than the feature axis is padding-sensitive in
 * a way a per-feature one is not: upstream measured a 76-residue chain padded
 * into a 128-token bucket at PAE ~28 A everywhere and pTM 0.04 against 0.89 for
 * the same fold, purely from counting the padding into the mean." It is also
 * TWO PASSES over the tensor here - partial sums a row, then one fragment over
 * the rows - because the reference centres before squaring and the trunk's own
 * magnitudes are exactly where `E[x^2] - E[x]^2` gives up.
 *
 * pTM and ipTM are NOT here: `cpu/af3/confidence/ptm.js` is pure host arithmetic
 * over the PAE logits this returns, shared with AF2 through
 * `shared/heads/tm-score.js`, and a second copy would be a second chance to
 * disagree about where the bins sit.
 */
import { ONE_TARGET, PRELUDE, sized, memoSource } from "../../runtime/execution.js";
import { drain } from "../../runtime/steps.js";
import { elementwiseOpSource, multiply, scale } from "../../runtime/elementwise.js";
import { layerNorm } from "../../kernels/layer-norm.js";
import { linear } from "../../kernels/linear.js";
import { pairformerBlock } from "../trunk/pairformer-block.js";
import { softmaxExpectationSource } from "../../heads/confidence.js";
import { symmetriseSource } from "../trunk/trunk.js";
import { RELATIVE_WIDTH, relativeEncodingSource } from "../trunk/embedder.js";
import { add } from "../../runtime/elementwise.js";

/** @typedef {import("../../runtime/storage.js").Tensor} Tensor */

export const NUM_BINS = 64;
export const MAX_ERROR_BIN = 31.0;
export const PLDDT_BINS = 50;
export const DGRAM_BINS = 39;
export const DGRAM_MIN = 3.25;
export const DGRAM_MAX = 50.75;
/** rf3's forty - see the note at the top. */
export const CA_DGRAM_BINS = 40;
/** rf3's s_inputs width, where this port's `target_feat` is 447. */
export const RF3_S_INPUTS_WIDTH = 449;

/**
 * Bin centres for an error head: `bins - 1` edges, plus a catch-all.
 *
 * The same centres pTM reads from the same logits, which is why the reference
 * exports them rather than letting two places derive them.
 */
export function errorBinCentres(bins, maxErrorBin) {
  const step = maxErrorBin / (bins - 2);
  const centres = new Float32Array(bins);
  for (let bin = 0; bin < bins - 1; bin += 1) centres[bin] = bin * step + step / 2;
  centres[bins - 1] = centres[bins - 2] + step;
  return centres;
}

/** pLDDT's centres, on 0-1; the head multiplies by 100 afterwards. */
export function plddtBinCentres(bins = PLDDT_BINS) {
  return Float32Array.from({ length: bins }, (_, bin) => (bin + 0.5) / bins);
}

/** `pairMask[i][j] = seqMask[i] * seqMask[j]`. */
const pairMaskSourceUncached = (tokens) => sized({ L: tokens, ELEMENTS: tokens * tokens }, `${PRELUDE}${ONE_TARGET}
uniform sampler2D seqMaskTex;

float at(int pair) {
  int i = pair / L;
  return fetch1(seqMaskTex, i) * fetch1(seqMaskTex, pair - i * L);
}

void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  result = vec4(
    at(base),
    base + 1 < ELEMENTS ? at(base + 1) : 0.0,
    base + 2 < ELEMENTS ? at(base + 2) : 0.0,
    base + 3 < ELEMENTS ? at(base + 3) : 0.0);
}`);

/** Per-row partial sums of `x`, over the rows the mask keeps. */
const rowSumSourceUncached = (rows, channels) => sized({ ROWS: rows }, `${PRELUDE}${ONE_TARGET}
const int CHANNELS = ${channels};
uniform sampler2D xTex;
uniform sampler2D maskTex;

float at(int row) {
  if (!(fetch1(maskTex, row) > 0.0)) return 0.0;
  float total = 0.0;
  int base = row * CHANNELS;
  for (int c = 0; c < CHANNELS; ++c) total += fetch1(xTex, base + c);
  return total;
}

void main() {
  int base = linearTexel() * 4;
  if (base >= ROWS) { result = vec4(0.0); return; }
  result = vec4(
    at(base),
    base + 1 < ROWS ? at(base + 1) : 0.0,
    base + 2 < ROWS ? at(base + 2) : 0.0,
    base + 3 < ROWS ? at(base + 3) : 0.0);
}`);

/** Per-row partial sums of `(x - mean)^2`, over the rows the mask keeps. */
const rowSquaredSourceUncached = (rows, channels) => sized({ ROWS: rows }, `${PRELUDE}${ONE_TARGET}
const int CHANNELS = ${channels};
uniform sampler2D xTex;
uniform sampler2D maskTex;
uniform float mean;

float at(int row) {
  if (!(fetch1(maskTex, row) > 0.0)) return 0.0;
  float total = 0.0;
  int base = row * CHANNELS;
  for (int c = 0; c < CHANNELS; ++c) {
    float d = fetch1(xTex, base + c) - mean;
    total += d * d;
  }
  return total;
}

void main() {
  int base = linearTexel() * 4;
  if (base >= ROWS) { result = vec4(0.0); return; }
  result = vec4(
    at(base),
    base + 1 < ROWS ? at(base + 1) : 0.0,
    base + 2 < ROWS ? at(base + 2) : 0.0,
    base + 3 < ROWS ? at(base + 3) : 0.0);
}`);

/** The live row count and the sum of the per-row partials, in one texel. */
const combineRowsSourceUncached = (rows) => sized({ ROWS: rows }, `${PRELUDE}${ONE_TARGET}
uniform sampler2D partialTex;
uniform sampler2D maskTex;

void main() {
  float total = 0.0;
  float live = 0.0;
  for (int row = 0; row < ROWS; ++row) {
    total += fetch1(partialTex, row);
    if (fetch1(maskTex, row) > 0.0) live += 1.0;
  }
  result = vec4(total, live, 0.0, 0.0);
}`);

/** `out = (x - mean) * inverse`, everywhere - padded rows included, as the reference does. */
const affineAllSourceUncached = (elements) => sized({ ELEMENTS: elements }, `${PRELUDE}${ONE_TARGET}
uniform sampler2D xTex;
uniform float mean;
uniform float inverse;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  result = (fetch4(xTex, texel) - vec4(mean)) * vec4(inverse);
}`);

/**
 * rf3's parameter-free LayerNorm over a WHOLE tensor, real rows only.
 *
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{x: Tensor, mask: Tensor, rows: number, channels: number,
 *   vendorWidth?: number}} call `vendorWidth` is the width to normalise OVER
 *   where it is wider than `channels`: the extra columns are assumed zero and so
 *   contribute `mean^2` apiece.
 * @returns {{values: Tensor, mean: number, inverse: number}} the caller owns
 *   `values`; the two scalars are what `absentTargetFeatTerms` needs, since a
 *   dropped (zero) column normalises to `-mean * inverse`.
 */
export function maskedGlobalNorm(execution, { x, mask, rows, channels, vendorWidth }) {
  const width = vendorWidth ?? channels;
  const partial = execution.allocate(rows, "af3.conf.row-sum");
  execution.dispatch({
    source: rowSumSource(rows, channels),
    output: partial,
    inputs: { xTex: x, maskTex: mask },
    label: `af3.conf.row-sum:${rows}x${channels}`,
  });
  const combined = execution.allocate(4, "af3.conf.combined");
  execution.dispatch({
    source: combineRowsSource(rows),
    output: combined,
    inputs: { partialTex: partial, maskTex: mask },
    label: `af3.conf.combine:${rows}`,
  });
  const [total, live] = Array.from(execution.read(combined));
  const count = Math.max(live * width, 1);
  const mean = total / count;
  execution.dispatch({
    source: rowSquaredSource(rows, channels),
    output: partial,
    inputs: { xTex: x, maskTex: mask },
    floats: { mean },
    label: `af3.conf.row-squared:${rows}x${channels}`,
  });
  execution.dispatch({
    source: combineRowsSource(rows),
    output: combined,
    inputs: { partialTex: partial, maskTex: mask },
    label: `af3.conf.combine:${rows}`,
  });
  // ...each column the vendor has and this port does not is zero, so it
  // contributes `mean^2`, once per real row.
  const variance = execution.read(combined)[0] + (width - channels) * live * mean * mean;
  const inverse = 1 / Math.sqrt(variance / count + 1e-5);
  execution.release(partial);
  execution.release(combined);
  const values = execution.allocate(rows * channels, "af3.conf.global-norm");
  execution.dispatch({
    source: affineAllSource(rows * channels),
    output: values,
    inputs: { xTex: x },
    floats: { mean, inverse },
    label: `af3.conf.global-norm:${rows}x${channels}`,
  });
  return { values, mean, inverse };
}

/**
 * rf3's two s_inputs columns this port's target_feat does not carry, as one
 * constant vector per side.
 *
 * Each column is zero before the global norm, so after it each is the constant
 * `-mean * inverse` on every token, and its contribution is that constant times
 * the sum of its two rows. Null where the bundle has no rows.
 */
export function absentTargetFeatTerms(weights, stats, pairChannels) {
  if (weights.leftTargetFeatProjectAbsent === undefined
    || weights.leftTargetFeatProjectAbsent === null) return null;
  const value = -stats.mean * stats.inverse;
  const term = (rows) => {
    if (rows.length !== 2 * pairChannels) {
      throw new RangeError(
        `af3 confidence: rf3's absent target_feat rows are ${rows.length} values,`
        + ` not 2 x ${pairChannels}`);
    }
    const out = new Float32Array(pairChannels);
    for (let c = 0; c < pairChannels; c += 1) {
      out[c] = value * rows[c] + value * rows[pairChannels + c];
    }
    return out;
  };
  return {
    left: term(weights.leftTargetFeatProjectAbsent),
    right: term(weights.rightTargetFeatProjectAbsent),
  };
}

/**
 * The whole pair assembly in one draw:
 *
 *     out[i][j][c] = pair[i][j][c] + left[j][c] + right[i][c]
 *                  + mask[i][j] * W_dgram[bin(i,j)][c]
 *                  + distance(i,j) * W_distance[c]      (protenix2 only)
 *                  + absent.left[c] + absent.right[c]   (rf3 only)
 *
 * 🔴 THE DISTOGRAM TERM IS A GATHER, NOT A PROJECTION - see the note at the top.
 * 🔴 AND AF3 COMPARES SQUARED DISTANCES AGAINST SQUARED EDGES where rf3 takes a
 * real square root with 1e-10 under it: "the boundary arithmetic is what decides
 * a bin and squaring the bounds moves the comparison by an ulp at the edges", so
 * each dialect's own spelling is kept rather than one being rewritten as the
 * other.
 */
const assemblePairSourceUncached = (
  tokens, channels, bins, caStyle, hasDistance, hasAbsent,
) => sized({ L: tokens, ELEMENTS: tokens * tokens * channels }, `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
const int BINS = ${bins};
const float DGRAM_MIN = ${DGRAM_MIN.toFixed(6)};
const float DGRAM_MAX = ${DGRAM_MAX.toFixed(6)};
uniform sampler2D pairTex;
uniform sampler2D leftTex;
uniform sampler2D rightTex;
uniform sampler2D maskTex;
uniform sampler2D positionsTex;
uniform sampler2D dgramWeightTex;
${hasDistance ? "uniform sampler2D distanceWeightTex;" : ""}
${hasAbsent ? "uniform sampler2D absentTex;" : ""}

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int pair = texel / C4;
  int i = pair / L;
  int j = pair - i * L;

  float squared = ${caStyle ? "1e-10" : "0.0"};
  for (int axis = 0; axis < 3; ++axis) {
    float d = fetch1(positionsTex, i * 3 + axis) - fetch1(positionsTex, j * 3 + axis);
    squared += d * d;
  }
${caStyle
    ? `  // rf3: the bin is how many of 39 BOUNDARIES the distance is past, so it has
  // a bin below DGRAM_MIN that AF3 has no equivalent of.
  float distance = sqrt(squared);
  float stride = (DGRAM_MAX - DGRAM_MIN) / float(BINS - 1);
  int bin = 0;
  for (int at = 0; at < BINS - 1; ++at) {
    if (distance > DGRAM_MIN + float(at) * stride) bin += 1;
  }`
    : `  // AF3: 39 EDGES and a bin an edge, the last catching everything past
  // DGRAM_MAX, compared on SQUARES against squared edges.
  int bin = -1;
  for (int at = 0; at < BINS; ++at) {
    float lower = DGRAM_MIN + (DGRAM_MAX - DGRAM_MIN) * float(at) / float(BINS - 1);
    float upper = at + 1 < BINS
      ? DGRAM_MIN + (DGRAM_MAX - DGRAM_MIN) * float(at + 1) / float(BINS - 1)
      : 0.0;
    float lowerSquared = lower * lower;
    float upperSquared = at + 1 < BINS ? upper * upper : 1e8;
    if (squared > lowerSquared && squared < upperSquared) bin = at;
  }`}

  vec4 value = fetch4(pairTex, texel)
    + fetch4(leftTex, j * C4 + group)
    + fetch4(rightTex, i * C4 + group);
  float mask = fetch1(maskTex, pair);
  if (bin >= 0) value += mask * fetch4(dgramWeightTex, bin * C4 + group);
${hasDistance ? `  value += sqrt(squared + 1e-10) * fetch4(distanceWeightTex, group);` : ""}
${hasAbsent ? `  value += fetch4(absentTex, group) + fetch4(absentTex, C4 + group);` : ""}
  result = value;
}`);

/** `out = x`, for the branch that takes the trunk's single unchanged. */
const copyRowsSourceUncached = (elements) => elementwiseOpSource("copy", elements);

/** `out = clamp(x, -512, 512)`, which protenix2 applies before its LayerNorm. */
const clampSourceUncached = (elements, limit) => sized({ ELEMENTS: elements }, `${PRELUDE}${ONE_TARGET}
uniform sampler2D xTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  result = clamp(fetch4(xTex, texel), vec4(${(-limit).toFixed(1)}), vec4(${limit.toFixed(1)}));
}`);

/** `out[i][j] = pair[i][j] + pair[j][i]`, boltz2's pre-symmetrised PDE source. */
const symmetrisePairSourceUncached = (tokens, channels) => sized({ L: tokens, ELEMENTS: tokens * tokens * channels }, `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
uniform sampler2D pairTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int pair = texel / C4;
  int i = pair / L;
  int j = pair - i * L;
  result = fetch4(pairTex, (i * L + j) * C4 + group)
    + fetch4(pairTex, (j * L + i) * C4 + group);
}`);

/**
 * boltz2's split pair heads: intra-chain logits where the two tokens share an
 * asym_id and inter-chain logits where they do not, masked HARD rather than
 * blended - so on a MONOMER the inter head never fires and a single-chain gate
 * cannot see whether it is there at all.
 */
const selectByChainSourceUncached = (tokens, bins) => sized({ L: tokens, ELEMENTS: tokens * tokens * bins }, `${PRELUDE}${ONE_TARGET}
const int B4 = ${bins / 4};
uniform sampler2D intraTex;
uniform sampler2D interTex;
uniform sampler2D asymTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int pair = texel / B4;
  int i = pair / L;
  int j = pair - i * L;
  bool same = fetch1(asymTex, i) == fetch1(asymTex, j);
  result = same ? fetch4(intraTex, texel) : fetch4(interTex, texel);
}`);

/**
 * The per-atom heads read by the atom's RANK among its token's real atoms, not
 * by its dense slot (`perAtomHeadsByRank`: rf3, intellifold2, openbind0).
 *
 * "The two agree until a slot is empty: a nucleotide's dropped OP3 is dense slot
 * 0, so every atom read its neighbour's logit (rna pLDDT corr 0.43 against stock
 * RF3, 0.9996 by rank)." A slot is real when its name is not empty.
 */
const byRankSourceUncached = (tokens, dense) => sized({ ELEMENTS: tokens * dense }, `${PRELUDE}${ONE_TARGET}
const int DENSE = ${dense};
uniform sampler2D valuesTex;
uniform sampler2D nameCharsTex;

// flat would be the obvious name for this index and is a GLSL interpolation
// QUALIFIER: the compiler says "'flat' : syntax error", which names the token
// and not the reason. (And no backticks in a GLSL comment - it lives inside a
// JS template literal and one would end it. Fourth time in this tree.)
float at(int element) {
  int token = element / DENSE;
  int slot = element - token * DENSE;
  int rank = -1;
  for (int s = 0; s <= slot; ++s) {
    bool real = false;
    for (int c = 0; c < 4; ++c) {
      if (fetch1(nameCharsTex, (token * DENSE + s) * 4 + c) != 0.0) real = true;
    }
    if (real) rank += 1;
  }
  int picked = clamp(rank, 0, DENSE - 1);
  return fetch1(valuesTex, token * DENSE + picked);
}

void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  result = vec4(
    at(base),
    base + 1 < ELEMENTS ? at(base + 1) : 0.0,
    base + 2 < ELEMENTS ? at(base + 2) : 0.0,
    base + 3 < ELEMENTS ? at(base + 3) : 0.0);
}`);

const SCALARS = new Set(["targetFeatWidth", "pairChannels", "singleChannels", "dense"]);

/**
 * The head's own tensors. The four pairformer blocks go through
 * `uploadPairformerBlockWeights` and arrive as `blocks` on the call, because
 * they are the trunk's module and this file does not restate its weight list.
 *
 * @param {import("../../runtime/execution.js").Execution} execution
 */
export function uploadConfidenceHeadWeights(execution, weights) {
  /** @type {any} */
  const out = {};
  for (const [name, value] of Object.entries(weights)) {
    if (value === undefined || value === null) continue;
    // 🔴 THE ABSENT ROWS ARE NOT UPLOADED - they are read as numbers below. The
    // first version let the Float32Array branch catch them, uploaded two
    // tensors, then overwrote the entries with the host arrays: two textures
    // nothing could release, which `Storage.stats()` reported as a leak.
    if (name.endsWith("Absent")) continue;
    if (SCALARS.has(name) || typeof value === "number") {
      out[name] = value;
    } else if (value instanceof Float32Array) {
      out[name] = execution.uploadWeight(value, `w.af3.conf.${name}`);
    }
  }
  // boltz2's re-embedding weights, a subtree of their own
  if (weights.reembed !== undefined && weights.reembed !== null) {
    out.reembed = {};
    for (const [name, value] of Object.entries(weights.reembed)) {
      if (value instanceof Float32Array) out.reembed[name] = execution.uploadWeight(value, `w.af3.conf.reembed.${name}`);
    }
  }
  // The absent rows stay on the HOST: they are read as numbers to build one
  // constant vector per side, after the global norm's scalars are known.
  for (const name of ["leftTargetFeatProjectAbsent", "rightTargetFeatProjectAbsent"]) {
    if (weights[name] !== undefined && weights[name] !== null) out[name] = weights[name];
  }
  return out;
}

export function releaseConfidenceHeadWeights(execution, weights) {
  for (const value of Object.values(weights.reembed ?? {})) execution.release(value);
  for (const [name, value] of Object.entries(weights)) {
    if (name.endsWith("Absent")) continue;
    if (value !== undefined && value !== null && typeof value === "object"
      && "elements" in /** @type {any} */ (value)) {
      execution.release(/** @type {any} */ (value));
    }
  }
}

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{pair: Tensor, single: Tensor, targetFeat: Tensor, pseudoBeta: Tensor,
 *   seqMask: Tensor, asymId?: Tensor, atomNameChars?: Tensor,
 *   tokens: number, dense: number, weights: any, blocks: any[],
 *   confidenceGlobalNorm?: boolean, confidenceCaDgram?: boolean,
 *   preSymmetrisedPde?: boolean, perAtomHeadsByRank?: boolean,
 *   reembedConfidencePair?: boolean,
 *   swapTransposedBias?: boolean, divideByLength?: boolean, epsilon?: number,
 *   onStage?: (stage: string, tensor: Tensor) => void}} call
 * @returns {{plddt: Tensor, pae: Tensor, pde: Tensor, distanceLogits: Tensor,
 *   paeLogits: Tensor, resolvedLogits: Tensor, pairMask: Tensor}} the caller
 *   owns all seven
 */
export function confidenceHead(execution, call) {
  return drain(confidenceHeadSteps(execution, call));
}

/**
 * boltz2's confidence re-embedding (cpu/af3/confidence/confidence.js `boltz2Reembed`): z and s REBUILT from the trunk's
 * outputs rather than added to - nine terms where AF3 has two.
 *
 *     sInputs = LN(target_feat);  s = LN(s) + sInputToS(sInputs)
 *     z = LN(z) + relpos + dgram64(pseudoBeta) + bond * tokenBondsProject + typeEmbed[order] + contact
 *         + right(sInputs)[i] + left(sInputs)[j] + prodOut(prodIn1(sInputs)[i] * prodIn2(sInputs)[j])
 *
 * "left" and "right" are the other way round from `_embed_features`: boltz adds `s_to_z(s)[:, :, None]`, indexed by i,
 * which is the RIGHT projection.
 */
function boltz2Reembed(execution, call, w, pairMask, epsilon) {
  const { pair: pairIn, single: singleIn, targetFeat, pseudoBeta, tokens } = call;
  const r = w.reembed;
  const pairChannels = w.pairChannels;
  const singleChannels = w.singleChannels;
  const width = w.targetFeatWidth;
  const pairs = tokens * tokens;
  if (r.relPosProject.elements !== RELATIVE_WIDTH * pairChannels) {
    throw new RangeError(`af3 confidence: rel_pos_project holds ${r.relPosProject.elements} floats, not`
      + ` ${RELATIVE_WIDTH} x ${pairChannels}`);
  }
  const sInputs = layerNorm(execution, {
    x: targetFeat, weight: r.sInputsNormScale, bias: r.sInputsNormOffset, rows: tokens, channels: width, epsilon,
  });
  const project = (weight, outChannels) => linear(execution, {
    x: sInputs, weight, rows: tokens, inChannels: width, outChannels, weightLayout: "in-major",
  });
  const normalisedSingle = layerNorm(execution, {
    x: singleIn, weight: r.sNormScale, bias: r.sNormOffset, rows: tokens, channels: singleChannels, epsilon,
  });
  const fromInputs = project(r.sInputToS, singleChannels);
  const single = add(execution, normalisedSingle, fromInputs);
  execution.release(normalisedSingle);
  execution.release(fromInputs);
  const left = project(r.leftTargetFeatProject, pairChannels);
  const right = project(r.rightTargetFeatProject, pairChannels);
  const in1 = project(r.sToZProdIn1, pairChannels);
  const in2 = project(r.sToZProdIn2, pairChannels);
  execution.release(sInputs);
  // the outer product, then its projection
  const product = execution.allocate(pairs * pairChannels, "af3.conf.reembed.product");
  execution.dispatch({
    source: outerProductRowsSource(tokens, pairChannels), output: product,
    inputs: { aTex: in1, bTex: in2 }, label: `af3.conf.reembed.product:${tokens}x${pairChannels}`,
  });
  execution.release(in1);
  execution.release(in2);
  const productOut = linear(execution, {
    x: product, weight: r.sToZProdOut, rows: pairs, inChannels: pairChannels, outChannels: pairChannels,
    weightLayout: "in-major",
  });
  execution.release(product);
  const normalisedPair = layerNorm(execution, {
    x: pairIn, weight: r.zNormScale, bias: r.zNormOffset, rows: pairs, channels: pairChannels, epsilon,
  });
  const positioned = execution.allocate(pairs * pairChannels, "af3.conf.reembed.relpos");
  execution.dispatch({
    source: relativeEncodingSource(tokens, pairChannels), output: positioned,
    inputs: { identityTex: call.identity, weightTex: r.relPosProject },
    label: `af3.conf.reembed.relpos:${tokens}x${pairChannels}`,
  });
  const pair = execution.allocate(pairs * pairChannels, "af3.conf.pair");
  execution.dispatch({
    source: reembedPairSource(tokens, pairChannels), output: pair,
    inputs: {
      pairTex: normalisedPair, relposTex: positioned, productTex: productOut, leftTex: left, rightTex: right,
      positionsTex: pseudoBeta, maskTex: pairMask, dgramWeightTex: r.distogramFeatProject,
      bondTex: call.bondMatrix, bondWeightTex: r.tokenBondsProject,
      orderTex: call.bondOrderMatrix, typeWeightTex: r.tokenBondsTypeEmbed,
      contactIndexTex: call.contactTerms.index, contactTableTex: call.contactTerms.table,
    },
    label: `af3.conf.reembed:${tokens}x${pairChannels}`,
  });
  for (const tensor of [normalisedPair, positioned, productOut, left, right]) execution.release(tensor);
  return { pair, single };
}

/** `out[i][j][c] = a[i][c] * b[j][c]`. */
const outerProductRowsSourceUncached = (tokens, channels) => sized({ L: tokens, ELEMENTS: tokens * tokens * channels }, `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
uniform sampler2D aTex;
uniform sampler2D bTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int pair = texel / C4;
  int i = pair / L;
  int j = pair - i * L;
  result = fetch4(aTex, i * C4 + group) * fetch4(bTex, j * C4 + group);
}`);

/** boltz2's re-embedded z: the sum above, with its 64-bin distogram of the pseudo-beta positions taken in place. */
const BOLTZ2_DGRAM_BINS = 64;
const reembedPairSourceUncached = (tokens, channels) => sized({ L: tokens, ELEMENTS: tokens * tokens * channels }, `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
const int BINS = ${BOLTZ2_DGRAM_BINS};
uniform sampler2D pairTex;
uniform sampler2D relposTex;
uniform sampler2D productTex;
uniform sampler2D leftTex;
uniform sampler2D rightTex;
uniform sampler2D positionsTex;
uniform sampler2D maskTex;
uniform sampler2D dgramWeightTex;
uniform sampler2D bondTex;
uniform sampler2D bondWeightTex;
uniform sampler2D orderTex;
uniform sampler2D typeWeightTex;
uniform sampler2D contactIndexTex;
uniform sampler2D contactTableTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int pair = texel / C4;
  int i = pair / L;
  int j = pair - i * L;
  vec3 a = vec3(fetch1(positionsTex, i * 3), fetch1(positionsTex, i * 3 + 1), fetch1(positionsTex, i * 3 + 2));
  vec3 b = vec3(fetch1(positionsTex, j * 3), fetch1(positionsTex, j * 3 + 1), fetch1(positionsTex, j * 3 + 2));
  vec3 d = a - b;
  float distance = sqrt(dot(d, d) + 1e-10);
  int bin = 0;
  for (int edge = 0; edge < BINS - 1; ++edge) {
    if (distance > 2.0 + (20.0 * float(edge)) / float(BINS - 2)) bin += 1;
  }
  result = fetch4(pairTex, texel) + fetch4(relposTex, texel) + fetch4(productTex, texel)
    + fetch4(rightTex, i * C4 + group) + fetch4(leftTex, j * C4 + group)
    + fetch1(maskTex, pair) * fetch4(dgramWeightTex, bin * C4 + group)
    + fetch1(bondTex, pair) * fetch4(bondWeightTex, group)
    + fetch4(typeWeightTex, clamp(int(fetch1(orderTex, pair)), 0, 6) * C4 + group)
    + fetch4(contactTableTex, int(fetch1(contactIndexTex, pair)) * C4 + group);
}`);

/** `confidenceHead` as a generator: yields after the embedding and each of its pairformer blocks. */
export function* confidenceHeadSteps(execution, call) {
  const {
    pair: pairIn, single: singleIn, targetFeat: targetFeatIn, pseudoBeta, seqMask,
    tokens, dense, weights: w, blocks, epsilon = 1e-5,
  } = call;
  const pairChannels = w.pairChannels;
  const singleChannels = w.singleChannels;
  const pairs = tokens * tokens;
  const reembedding = call.reembedConfidencePair === true;
  if (reembedding && (w.reembed === undefined || call.identity === undefined || call.bondMatrix === undefined
    || call.bondOrderMatrix === undefined || call.contactTerms === undefined)) {
    throw new Error("af3 confidence: reembedConfidencePair (boltz2's nine-term rebuild of z) needs the reembed weights,"
      + " the token identity, bondMatrix, bondOrderMatrix and contactTerms");
  }
  // 🔴 EVERY PRECONDITION BEFORE THE FIRST ALLOCATION. These two were checked
  // where they are USED - the split heads just before the PDE projection, the
  // rank heads just after pLDDT - and each refusal then leaked everything the
  // head had allocated by then: `Storage.stats()` read 19 tensors live and
  // turned the gate red. Same lesson as `diffusionConditioning`'s width check,
  // which had to move above its allocation for the same reason.
  if (w.interHalfDistanceLogits !== undefined && call.asymId === undefined) {
    throw new Error(
      "af3 confidence: this bundle has split intra/inter pair heads, which need"
      + " `asymId` - the mask is hard, not blended");
  }
  if (call.perAtomHeadsByRank === true && call.atomNameChars === undefined) {
    throw new Error(
      "af3 confidence: perAtomHeadsByRank reads the atom's RANK among its"
      + " token's real atoms, which needs `atomNameChars` - a slot is real when"
      + " its name is not empty");
  }
  const temporary = [];
  const keep = (tensor) => {
    temporary.push(tensor);
    return tensor;
  };

  const pairMask = execution.allocate(pairs, "af3.conf.pair-mask");
  execution.dispatch({
    source: pairMaskSource(tokens),
    output: pairMask,
    inputs: { seqMaskTex: seqMask },
    label: `af3.conf.pair-mask:${tokens}`,
  });

  let pair;
  let single;
  if (reembedding) {
    ({ pair, single } = boltz2Reembed(execution, call, w, pairMask, epsilon));
  } else {
    // 🔴 rf3 NORMALISES EVERY DETACHED TRUNK INPUT OVER THE WHOLE TENSOR FIRST,
    // parameter-free and over REAL TOKENS ONLY, before anything else reads them.
    let pairSource = pairIn;
    let singleSource = singleIn;
    let targetFeat = targetFeatIn;
    let absent = null;
    if (call.confidenceGlobalNorm === true) {
      pairSource = keep(maskedGlobalNorm(execution, {
        x: pairIn, mask: pairMask, rows: pairs, channels: pairChannels,
      }).values);
      singleSource = keep(maskedGlobalNorm(execution, {
        x: singleIn, mask: seqMask, rows: tokens, channels: singleChannels,
      }).values);
      // ...and target_feat over the VENDOR's width, two columns wider than ours.
      const normalised = maskedGlobalNorm(execution, {
        x: targetFeatIn, mask: seqMask, rows: tokens, channels: w.targetFeatWidth,
        vendorWidth: RF3_S_INPUTS_WIDTH,
      });
      targetFeat = keep(normalised.values);
      absent = absentTargetFeatTerms(w, normalised, pairChannels);
    }
    call.onStage?.("globalNorm.pair", pairSource);
    call.onStage?.("globalNorm.single", singleSource);
    call.onStage?.("globalNorm.targetFeat", targetFeat);

    // The target features, once along each axis - LEFT indexed by j, RIGHT by i.
    const left = keep(linear(execution, {
      x: targetFeat, weight: w.leftTargetFeatProject, rows: tokens,
      inChannels: w.targetFeatWidth, outChannels: pairChannels, weightLayout: "in-major",
    }));
    const right = keep(linear(execution, {
      x: targetFeat, weight: w.rightTargetFeatProject, rows: tokens,
      inChannels: w.targetFeatWidth, outChannels: pairChannels, weightLayout: "in-major",
    }));
    call.onStage?.("embed.left", left);
    call.onStage?.("embed.right", right);

    // 🔴 THE BIN COUNT IS THE WEIGHT'S, so a bundle that disagrees with the
    // dialect is a shape error rather than a silent prefix.
    const caDgram = call.confidenceCaDgram === true;
    const dgramBins = w.distogramFeatProject.elements / pairChannels;
    if (dgramBins !== (caDgram ? CA_DGRAM_BINS : DGRAM_BINS)) {
      for (const tensor of temporary) execution.release(tensor);
      execution.release(pairMask);
      throw new RangeError(
        `af3 confidence: this bundle's distogram_feat_project has ${dgramBins} bins`
        + ` and its dialect asks for ${caDgram ? CA_DGRAM_BINS : DGRAM_BINS}`);
    }
    const absentTensor = absent === null ? undefined : keep(execution.upload(
      Float32Array.from([...absent.left, ...absent.right]), "af3.conf.absent"));
    pair = execution.allocate(pairs * pairChannels, "af3.conf.pair");
    execution.dispatch({
      source: assemblePairSource(
        tokens, pairChannels, dgramBins, caDgram,
        w.distanceFeatProject !== undefined, absentTensor !== undefined),
      output: pair,
      inputs: {
        pairTex: pairSource, leftTex: left, rightTex: right, maskTex: pairMask,
        positionsTex: pseudoBeta, dgramWeightTex: w.distogramFeatProject,
        ...(w.distanceFeatProject === undefined
          ? {} : { distanceWeightTex: w.distanceFeatProject }),
        ...(absentTensor === undefined ? {} : { absentTex: absentTensor }),
      },
      label: `af3.conf.embed:${tokens}x${pairChannels}:${dgramBins}`
        + `${caDgram ? ":ca" : ""}`,
    });
    call.onStage?.("embed.pair", pair);

    // 🔴 AND THE TRUNK SINGLE IS CLAMPED AND NORMALISED BEFORE ANY USE. Both the
    // pairformer and every head read the normalised one.
    if (w.inputSingleNormScale === undefined) {
      // 🔴 NO NORM MEANS NO CLAMP EITHER, AND THE CLAMP ALONE IS A WRECKING BALL.
      // The clamp exists because protenix2 normalises here; AF3 uses the trunk's
      // single RAW, and that single runs to a standard deviation of 211 with peaks
      // far past 512. The first version of this branch clamped and copied - which
      // is a no-op at the noise scale every gate drove it with and, on a real
      // fold, took pLDDT from 75 to 6 while PAE and the coordinates stayed in
      // band, because the pair half of a pairformer block never reads the single.
      single = execution.allocate(tokens * singleChannels, "af3.conf.single");
      execution.dispatch({
        source: copyRowsSource(tokens * singleChannels),
        output: single,
        inputs: { aTex: singleSource, bTex: singleSource },
        label: `af3.conf.single-copy:${tokens}x${singleChannels}`,
      });
    } else {
      const clamped = execution.allocate(tokens * singleChannels, "af3.conf.clamped");
      execution.dispatch({
        source: clampSource(tokens * singleChannels, 512),
        output: clamped,
        inputs: { xTex: singleSource },
        label: `af3.conf.clamp:${tokens}x${singleChannels}`,
      });
      single = layerNorm(execution, {
        x: clamped, weight: w.inputSingleNormScale, bias: w.inputSingleNormOffset,
        rows: tokens, channels: singleChannels, epsilon,
      });
      execution.release(clamped);
    }
  }
  call.onStage?.("embed.single", single);
  yield { stage: "confidence.embed" };

  // Four pairformer blocks, the same module the trunk runs 48 of.
  for (const [index, block] of blocks.entries()) {
    const next = pairformerBlock(execution, {
      pair, single, pairMask, seqMask, tokens, weights: block,
      swapTransposedBias: call.swapTransposedBias,
      divideByLength: call.divideByLength, epsilon,
    });
    execution.release(pair);
    execution.release(single);
    pair = next.pair;
    single = next.single;
    call.onStage?.(`block${index}.pair`, pair);
    call.onStage?.(`block${index}.single`, single);
    yield { stage: "confidence.block", index };
  }

  const headNorm = (x, rows, channels, scaleWeight, offset) => (scaleWeight === undefined
    ? x
    : layerNorm(execution, {
      x, weight: scaleWeight, bias: offset, rows, channels, epsilon,
    }));
  const splitHeads = w.interHalfDistanceLogits !== undefined;
  /** intra where the asym ids agree, inter where they do not. */
  const perPair = (intraLogits, interLogits) => {
    if (!splitHeads) return intraLogits;
    const out = execution.allocate(pairs * NUM_BINS, "af3.conf.split-logits");
    execution.dispatch({
      source: selectByChainSource(tokens, NUM_BINS),
      output: out,
      inputs: {
        intraTex: intraLogits, interTex: interLogits,
        asymTex: /** @type {Tensor} */ (call.asymId),
      },
      label: `af3.conf.split:${tokens}x${NUM_BINS}`,
    });
    execution.release(intraLogits);
    if (interLogits !== intraLogits) execution.release(interLogits);
    return out;
  };

  // 🔴 boltz2 SYMMETRISES BEFORE THE PROJECTION, not after it: "AF3 projects
  // once and adds the transpose of the result; the two agree only when the
  // projection is the same on both halves, which under split heads it is not."
  const preSymmetrised = call.preSymmetrisedPde === true;
  let pdeSource = pair;
  if (preSymmetrised) {
    pdeSource = execution.allocate(pairs * pairChannels, "af3.conf.pde-source");
    execution.dispatch({
      source: symmetrisePairSource(tokens, pairChannels),
      output: pdeSource,
      inputs: { pairTex: pair },
      label: `af3.conf.pre-symmetrise:${tokens}x${pairChannels}`,
    });
  }
  const normalisedPde = headNorm(
    pdeSource, pairs, pairChannels, w.logitsLnScale, w.logitsLnOffset);
  const half = linear(execution, {
    x: normalisedPde, weight: w.leftHalfDistanceLogits, rows: pairs,
    inChannels: pairChannels, outChannels: NUM_BINS, weightLayout: "in-major",
  });
  let distanceLogits;
  if (preSymmetrised) {
    distanceLogits = perPair(half, splitHeads
      ? linear(execution, {
        x: normalisedPde, weight: w.interHalfDistanceLogits, rows: pairs,
        inChannels: pairChannels, outChannels: NUM_BINS, weightLayout: "in-major",
      })
      : half);
  } else {
    // One projection, used twice: AF3 sets `right = left` explicitly, and
    // `trunk.js`'s own symmetrise is this draw at the same 64 bins.
    distanceLogits = execution.allocate(pairs * NUM_BINS, "af3.conf.distance-logits");
    execution.dispatch({
      source: symmetriseSource(tokens),
      output: distanceLogits,
      inputs: { halfTex: half },
      label: `af3.conf.symmetrise:${tokens}`,
    });
    execution.release(half);
  }
  if (normalisedPde !== pdeSource) execution.release(normalisedPde);
  if (pdeSource !== pair) execution.release(pdeSource);
  call.onStage?.("distanceLogits", distanceLogits);

  // ...and the aligned-error head, which is NOT symmetrised.
  const normalisedPae = headNorm(
    pair, pairs, pairChannels, w.paeLogitsLnScale, w.paeLogitsLnOffset);
  const paeLogits = perPair(
    linear(execution, {
      x: normalisedPae, weight: w.paeLogits, rows: pairs,
      inChannels: pairChannels, outChannels: NUM_BINS, weightLayout: "in-major",
    }),
    w.paeInterLogits === undefined ? undefined : linear(execution, {
      x: normalisedPae, weight: w.paeInterLogits, rows: pairs,
      inChannels: pairChannels, outChannels: NUM_BINS, weightLayout: "in-major",
    }));
  if (normalisedPae !== pair) execution.release(normalisedPae);
  call.onStage?.("paeLogits", paeLogits);

  const centres = keep(execution.upload(
    errorBinCentres(NUM_BINS, MAX_ERROR_BIN), "af3.conf.error-centres"));
  const expectation = (logits, rows, bins, centresTensor, label) => {
    const out = execution.allocate(rows, `af3.conf.${label}`);
    execution.dispatch({
      source: softmaxExpectationSource(rows, bins),
      output: out,
      inputs: { logitsTex: logits, centresTex: centresTensor },
      label: `af3.conf.${label}:${rows}x${bins}`,
    });
    return out;
  };
  const rawPde = expectation(distanceLogits, pairs, NUM_BINS, centres, "pde");
  const pde = multiply(execution, rawPde, pairMask);
  execution.release(rawPde);
  const rawPae = expectation(paeLogits, pairs, NUM_BINS, centres, "pae");
  const pae = multiply(execution, rawPae, pairMask);
  execution.release(rawPae);

  // pLDDT, per atom slot: the projection's output is (dense, bins) flattened.
  const plddtSource = headNorm(
    single, tokens, singleChannels, w.plddtLnScale, w.plddtLnOffset);
  const plddtLogits = linear(execution, {
    x: plddtSource, weight: w.plddtLogits, rows: tokens,
    inChannels: singleChannels, outChannels: dense * PLDDT_BINS,
    weightLayout: "in-major",
  });
  if (plddtSource !== single) execution.release(plddtSource);
  const plddtCentres = keep(execution.upload(plddtBinCentres(), "af3.conf.plddt-centres"));
  const fraction = expectation(
    plddtLogits, tokens * dense, PLDDT_BINS, plddtCentres, "plddt");
  execution.release(plddtLogits);
  let plddt = scale(execution, fraction, 100);
  execution.release(fraction);
  if (call.perAtomHeadsByRank === true) {
    const ranked = execution.allocate(tokens * dense, "af3.conf.plddt-ranked");
    execution.dispatch({
      source: byRankSource(tokens, dense),
      output: ranked,
      inputs: { valuesTex: plddt, nameCharsTex: call.atomNameChars },
      label: `af3.conf.by-rank:${tokens}x${dense}`,
    });
    execution.release(plddt);
    plddt = ranked;
  }

  const resolvedSource = headNorm(
    single, tokens, singleChannels, w.resolvedLnScale, w.resolvedLnOffset);
  const resolvedLogits = linear(execution, {
    x: resolvedSource, weight: w.experimentallyResolvedLogits, rows: tokens,
    inChannels: singleChannels, outChannels: dense * 2, weightLayout: "in-major",
  });
  if (resolvedSource !== single) execution.release(resolvedSource);

  for (const tensor of temporary) execution.release(tensor);
  execution.release(pair);
  execution.release(single);
  plddt.label = "af3.conf.plddt";
  pae.label = "af3.conf.pae";
  pde.label = "af3.conf.pde";
  return { plddt, pae, pde, distanceLogits, paeLogits, resolvedLogits, pairMask };
}

// Every builder above, memoised on its arguments (`memoSource` in webgl/runtime/execution.js).
export const pairMaskSource = memoSource(pairMaskSourceUncached);
export const rowSumSource = memoSource(rowSumSourceUncached);
export const rowSquaredSource = memoSource(rowSquaredSourceUncached);
export const combineRowsSource = memoSource(combineRowsSourceUncached);
export const affineAllSource = memoSource(affineAllSourceUncached);
export const assemblePairSource = memoSource(assemblePairSourceUncached);
export const outerProductRowsSource = memoSource(outerProductRowsSourceUncached);
export const reembedPairSource = memoSource(reembedPairSourceUncached);
export const copyRowsSource = memoSource(copyRowsSourceUncached);
export const clampSource = memoSource(clampSourceUncached);
export const symmetrisePairSource = memoSource(symmetrisePairSourceUncached);
export const selectByChainSource = memoSource(selectByChainSourceUncached);
export const byRankSource = memoSource(byRankSourceUncached);
