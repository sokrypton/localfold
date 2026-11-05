/**
 * Synthyra's ESMFold2 confidence head in WebGL2: pLDDT, PAE, pTM and ipTM.
 *
 * 🔴 THIS CHECKPOINT DID NOT HAVE ONE. "biohub ships ESMFold2 with
 * `confidence_head.enabled: false` and zero confidence tensors... Synthyra froze
 * that trunk and trained a head on it; the trunk underneath is ours byte for
 * byte, so this module is the whole of the difference."
 *
 * 🔴 AND ITS FOUR BLOCKS ARE THE TRUNK'S BLOCK, PAIR-ONLY: no single track
 * crosses them - `singleFromPair` makes one at the END out of the finished pair
 * - so this calls `foldingTrunk` rather than a pairformer block, for the same
 * reason the trunk does.
 *
 * 🔴 AND `pair + foldingTrunk(pair)` ADDS THE INPUT TWICE, WHICH IS THEIR LINE
 * AND NOT A SLIP. Their `FoldingTrunk.forward` returns the UPDATED pair - each
 * block's residuals are already inside it - and the head then writes
 * `pair = pair + self.folding_trunk(pair, ...)`. "A port that read that line as
 * an ordinary residual is short exactly one copy of the input, which is a PAE
 * that looks entirely reasonable and is wrong."
 *
 * 🔴 AND THE RELATIVE-POSITION ENCODING GOES IN AFTER THE NORM, NOT BEFORE: "it
 * is a keyword defaulting to None, which is why leaving it out was silent". This
 * port REFUSES a call without it rather than defaulting to zeros.
 *
 * 🔴 AND THE DISTANCE TERM IS A BUCKET COUNT, NOT A ONE-HOT MATMUL.
 * `dist_bin_pairwise_embed` is an `nn.Embedding(128, 256)` looked up by how many
 * of the 127 boundaries the distance exceeds - "the table's rows are read
 * directly and the tensor is the one matrix in this head that the exporter does
 * NOT transpose".
 *
 * 🔴 AND THE POOLING MASK IS A BIAS OF -1e9 AND NOT A ZEROED WEIGHT: "an
 * all-masked row would divide by zero the other way".
 *
 * 🔴 AND THE pLDDT TABLE IS INDEXED BY THE ATOM'S SLOT WITHIN ITS TOKEN, a
 * running count that RESETS at each token boundary and is clamped to the table's
 * 23 rows. "Indexing it by the atom's global position instead reads a different
 * residue's weights and still returns a number."
 *
 * pTM and ipTM stay on the host: `cpu/esmfold2/confidence.js`'s `tmScores` reads
 * the PAE logits this returns, `d0` is computed from the LIVE token count and
 * clamped at 19, and the row expectation is MAXIMISED over i rather than
 * averaged - every step of which is a place to be subtly wrong twice.
 */
import { ONE_TARGET, PRELUDE } from "../runtime/execution.js";
import { layerNorm } from "../kernels/layer-norm.js";
import { linear } from "../kernels/linear.js";
import { foldingTrunk } from "./trunk.js";
import { sumSource } from "./atom-transformer.js";

/** @typedef {import("../runtime/storage.js").Tensor} Tensor */

/**
 * The whole pair assembly in one draw, bar the projections:
 *
 *     out[i][j][c] = normed[i][j][c] + bias[i][j][c]
 *                  + rows[i][c] + columns[j][c] + product[i][j][c]
 *                  + table[bucket(d(i, j))][c]
 */
export const assembleSource = (tokens, channels, edges) => `${PRELUDE}${ONE_TARGET}
const int L = ${tokens};
const int C4 = ${channels / 4};
const int EDGES = ${edges};
const int ELEMENTS = ${tokens * tokens * channels};
uniform sampler2D normedTex;
uniform sampler2D biasTex;
uniform sampler2D rowsTex;
uniform sampler2D columnsTex;
uniform sampler2D productTex;
uniform sampler2D positionsTex;
uniform sampler2D repAtomTex;
uniform sampler2D edgesTex;
uniform sampler2D tableTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int pair = texel / C4;
  int i = pair / L;
  int j = pair - i * L;

  int a = int(fetch1(repAtomTex, i)) * 3;
  int b = int(fetch1(repAtomTex, j)) * 3;
  float squared = 0.0;
  for (int axis = 0; axis < 3; ++axis) {
    float d = fetch1(positionsTex, a + axis) - fetch1(positionsTex, b + axis);
    squared += d * d;
  }
  float distance = sqrt(squared);
  int bucket = 0;
  for (int edge = 0; edge < EDGES; ++edge) {
    if (distance > fetch1(edgesTex, edge)) bucket += 1;
  }

  result = fetch4(normedTex, texel) + fetch4(biasTex, texel)
    + fetch4(rowsTex, i * C4 + group) + fetch4(columnsTex, j * C4 + group)
    + fetch4(productTex, texel) + fetch4(tableTex, bucket * C4 + group);
}`;

/** `out[i][j][c] = left[i][c] * right[j][c]`, before its own projection. */
export const outerProductSource = (tokens, channels) => `${PRELUDE}${ONE_TARGET}
const int L = ${tokens};
const int C4 = ${channels / 4};
const int ELEMENTS = ${tokens * tokens * channels};
uniform sampler2D leftTex;
uniform sampler2D rightTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int pair = texel / C4;
  int i = pair / L;
  int j = pair - i * L;
  result = fetch4(leftTex, i * C4 + group) * fetch4(rightTex, j * C4 + group);
}`;

/** `pairMask[i][j] = tokenMask[i] * tokenMask[j]`. */
export const pairMaskSource = (tokens) => `${PRELUDE}${ONE_TARGET}
const int L = ${tokens};
const int ELEMENTS = ${tokens * tokens};
uniform sampler2D maskTex;

float at(int pair) {
  int i = pair / L;
  return fetch1(maskTex, i) * fetch1(maskTex, pair - i * L);
}

void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  result = vec4(
    at(base),
    base + 1 < ELEMENTS ? at(base + 1) : 0.0,
    base + 2 < ELEMENTS ? at(base + 2) : 0.0,
    base + 3 < ELEMENTS ? at(base + 3) : 0.0);
}`;

/**
 * Row-attention pooling: a masked softmax along j over one scalar a pair, then
 * the weighted sum of the pair's channels.
 */
export const poolSource = (tokens, channels) => `${PRELUDE}${ONE_TARGET}
const int L = ${tokens};
const int C4 = ${channels / 4};
const int ELEMENTS = ${tokens * channels};
uniform sampler2D scoresTex;
uniform sampler2D pairTex;
uniform sampler2D maskTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int i = texel / C4;
  // 🔴 A BIAS OF -1e9, NOT A ZEROED WEIGHT: see the note at the top.
  float top = -1e30;
  for (int j = 0; j < L; ++j) {
    float value = fetch1(maskTex, j) > 0.5 ? fetch1(scoresTex, i * L + j) : -1e9;
    top = max(top, value);
  }
  float total = 0.0;
  vec4 pooled = vec4(0.0);
  for (int j = 0; j < L; ++j) {
    float value = fetch1(maskTex, j) > 0.5 ? fetch1(scoresTex, i * L + j) : -1e9;
    float weight = exp(value - top);
    total += weight;
    pooled += weight * fetch4(pairTex, (i * L + j) * C4 + group);
  }
  result = pooled / total;
}`;

/** `out[atom][c] = tokenValues[atomToToken[atom]][c]`. */
export const gatherToAtomsSource = (atoms, channels) => `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
const int ELEMENTS = ${atoms * channels};
uniform sampler2D tokenTex;
uniform sampler2D indexTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int atom = texel / C4;
  int token = int(fetch1(indexTex, atom));
  result = fetch4(tokenTex, token * C4 + group);
}`;

/**
 * The pLDDT logits: a per-SLOT weight table, the slot being the atom's index
 * within its token.
 *
 * The slot is a host-side running count (it resets at each token boundary and
 * clamps to the table's rows), because a fragment that recomputed it would walk
 * the whole index array per output element.
 */
export const plddtLogitsSource = (atoms, channels, bins) => `${PRELUDE}${ONE_TARGET}
const int CHANNELS = ${channels};
const int BINS = ${bins};
const int ELEMENTS = ${atoms * bins};
uniform sampler2D normedTex;
uniform sampler2D slotTex;
uniform sampler2D tableTex;

float at(int element) {
  int atom = element / BINS;
  int bin = element - atom * BINS;
  int table = int(fetch1(slotTex, atom)) * CHANNELS * BINS;
  float sum = 0.0;
  for (int c = 0; c < CHANNELS; ++c) {
    sum += fetch1(normedTex, atom * CHANNELS + c) * fetch1(tableTex, table + c * BINS + bin);
  }
  return sum;
}

void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  result = vec4(
    at(base),
    base + 1 < ELEMENTS ? at(base + 1) : 0.0,
    base + 2 < ELEMENTS ? at(base + 2) : 0.0,
    base + 3 < ELEMENTS ? at(base + 3) : 0.0);
}`;

/**
 * The expectation of a categorical distribution over evenly spaced bins.
 *
 * Their `_categorical_mean`: the MIDPOINTS of `linspace(start, end, bins + 1)`,
 * so 64 bins over [0, 32] are centred 0.25 to 31.75 and 50 over [0, 1] are
 * centred 0.01 to 0.99.
 */
export const categoricalMeanSource = (rows, bins, start, end) => `${PRELUDE}${ONE_TARGET}
const int BINS = ${bins};
const int ROWS = ${rows};
const float START = ${start.toFixed(6)};
const float WIDTH = ${((end - start) / bins).toFixed(10)};
uniform sampler2D logitsTex;

float at(int row) {
  int base = row * BINS;
  float top = -1e30;
  for (int bin = 0; bin < BINS; ++bin) top = max(top, fetch1(logitsTex, base + bin));
  float total = 0.0;
  float sum = 0.0;
  for (int bin = 0; bin < BINS; ++bin) {
    float weight = exp(fetch1(logitsTex, base + bin) - top);
    total += weight;
    sum += weight * (START + WIDTH * (float(bin) + 0.5));
  }
  return sum / total;
}

void main() {
  int base = linearTexel() * 4;
  if (base >= ROWS) { result = vec4(0.0); return; }
  result = vec4(
    at(base),
    base + 1 < ROWS ? at(base + 1) : 0.0,
    base + 2 < ROWS ? at(base + 2) : 0.0,
    base + 3 < ROWS ? at(base + 3) : 0.0);
}`;

/** A token's masked mean over its atoms, and the representative atom's value. */
export const perTokenSource = (atoms, tokens) => `${PRELUDE}${ONE_TARGET}
const int ATOMS = ${atoms};
const int ELEMENTS = ${tokens};
uniform sampler2D valuesTex;
uniform sampler2D indexTex;
uniform sampler2D maskTex;

float at(int token) {
  float sum = 0.0;
  float count = 0.0;
  for (int atom = 0; atom < ATOMS; ++atom) {
    if (int(fetch1(indexTex, atom)) != token) continue;
    float w = fetch1(maskTex, atom);
    sum += fetch1(valuesTex, atom) * w;
    count += w;
  }
  return sum / max(count, 1e-6);
}

void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  result = vec4(
    at(base),
    base + 1 < ELEMENTS ? at(base + 1) : 0.0,
    base + 2 < ELEMENTS ? at(base + 2) : 0.0,
    base + 3 < ELEMENTS ? at(base + 3) : 0.0);
}`;

/** The atom's slot within its token: a running count that resets per token. */
export function intraTokenSlots(atomToToken, maxAtomsPerToken) {
  const out = new Float32Array(atomToToken.length);
  let slot = 0;
  for (let atom = 0; atom < atomToToken.length; atom += 1) {
    if (atom > 0 && atomToToken[atom] !== atomToToken[atom - 1]) slot = 0;
    out[atom] = Math.min(slot, maxAtomsPerToken - 1);
    slot += 1;
  }
  return out;
}

/**
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {{tokens: number, atoms: number, sInputs: Tensor, pair: Tensor,
 *   pairBias: Tensor, coordinates: Tensor, repAtom: Tensor, atomToToken: Tensor,
 *   tokenMask: Tensor, atomMask: Tensor, slots: Tensor,
 *   weights: any, blocks: any[], epsilon?: number,
 *   onStage?: (stage: string, tensor: Tensor) => void}} call
 * @returns {{pair: Tensor, single: Tensor, plddtLogits: Tensor,
 *   plddtPerAtom: Tensor, plddt: Tensor, paeLogits: Tensor, pae: Tensor}}
 *   the caller owns all seven
 */
export function esmfold2Confidence(execution, call) {
  const { tokens, atoms, weights: w, blocks, epsilon = 1e-5 } = call;
  const dPair = w.pairChannels;
  const dSingle = w.singleChannels;
  const dInputs = w.singleInputs;
  const pairs = tokens * tokens;
  if (call.pairBias === undefined) {
    throw new Error(
      "esmfold2 confidence: pairBias is required - the relative-position encoding"
      + " goes in AFTER the norm and is a keyword defaulting to None upstream,"
      + " which is why leaving it out was silent (docs/EF2FAST.md)");
  }

  const single = layerNorm(execution, {
    x: call.sInputs, weight: w.sInputsNormScale, bias: w.sInputsNormOffset,
    rows: tokens, channels: dInputs, epsilon,
  });
  const normedPair = layerNorm(execution, {
    x: call.pair, weight: w.zNormScale, bias: w.zNormOffset,
    rows: pairs, channels: dPair, epsilon,
  });
  const project = (weight) => linear(execution, {
    x: single, weight, rows: tokens,
    inChannels: dInputs, outChannels: dPair, weightLayout: "in-major",
  });
  const rows = project(w.sToZ);
  const columns = project(w.sToZTranspose);
  const left = project(w.sToZProdIn1);
  const right = project(w.sToZProdIn2);
  const product = execution.allocate(pairs * dPair, "ef2.conf.product");
  execution.dispatch({
    source: outerProductSource(tokens, dPair),
    output: product,
    inputs: { leftTex: left, rightTex: right },
    label: `ef2.conf.product:${tokens}x${dPair}`,
  });
  execution.release(left);
  execution.release(right);
  const projectedProduct = linear(execution, {
    x: product, weight: w.sToZProdOut, rows: pairs,
    inChannels: dPair, outChannels: dPair, weightLayout: "in-major",
  });
  execution.release(product);

  const initial = execution.allocate(pairs * dPair, "ef2.conf.pair");
  execution.dispatch({
    source: assembleSource(tokens, dPair, w.boundaries.elements),
    output: initial,
    inputs: {
      normedTex: normedPair, biasTex: call.pairBias,
      rowsTex: rows, columnsTex: columns, productTex: projectedProduct,
      positionsTex: call.coordinates, repAtomTex: call.repAtom,
      edgesTex: w.boundaries, tableTex: w.distanceEmbedding,
    },
    label: `ef2.conf.assemble:${tokens}x${dPair}`,
  });
  execution.release(normedPair);
  execution.release(rows);
  execution.release(columns);
  execution.release(projectedProduct);
  call.onStage?.("confidence.initial", initial);

  const pairMask = execution.allocate(pairs, "ef2.conf.pair-mask");
  execution.dispatch({
    source: pairMaskSource(tokens),
    output: pairMask,
    inputs: { maskTex: call.tokenMask },
    label: `ef2.conf.pair-mask:${tokens}`,
  });
  // 🔴 THE STACK'S ANSWER IS ADDED TO ITS OWN INPUT - see the note at the top.
  const stacked = foldingTrunk(execution, {
    pair: initial, mask: pairMask, tokens, channels: dPair, blocks, epsilon,
  });
  const pair = execution.allocate(pairs * dPair, "ef2.conf.pair");
  execution.dispatch({
    source: sumSource(pairs * dPair),
    output: pair,
    inputs: { aTex: initial, bTex: stacked },
    label: `ef2.conf.double-add:${pairs}x${dPair}`,
  });
  execution.release(stacked);
  execution.release(initial);
  call.onStage?.("confidence.pair", pair);

  // The single, pooled out of the finished pair.
  const scores = linear(execution, {
    x: pair, weight: w.poolingAttention, rows: pairs,
    inChannels: dPair, outChannels: 1, weightLayout: "in-major",
  });
  const pooled = execution.allocate(tokens * dPair, "ef2.conf.pooled");
  execution.dispatch({
    source: poolSource(tokens, dPair),
    output: pooled,
    inputs: { scoresTex: scores, pairTex: pair, maskTex: call.tokenMask },
    label: `ef2.conf.pool:${tokens}x${dPair}`,
  });
  execution.release(scores);
  const pooledSingle = linear(execution, {
    x: pooled, weight: w.poolingOutput, rows: tokens,
    inChannels: dPair, outChannels: dSingle, weightLayout: "in-major",
  });
  execution.release(pooled);
  call.onStage?.("confidence.single", pooledSingle);

  // pLDDT, per atom and then per token.
  const gathered = execution.allocate(atoms * dSingle, "ef2.conf.per-atom");
  execution.dispatch({
    source: gatherToAtomsSource(atoms, dSingle),
    output: gathered,
    inputs: { tokenTex: pooledSingle, indexTex: call.atomToToken },
    label: `ef2.conf.gather:${atoms}x${dSingle}`,
  });
  const normed = layerNorm(execution, {
    x: gathered, weight: w.plddtNormScale, bias: w.plddtNormOffset,
    rows: atoms, channels: dSingle, epsilon,
  });
  execution.release(gathered);
  const plddtLogits = execution.allocate(atoms * w.plddtBins, "ef2.conf.plddt-logits");
  execution.dispatch({
    source: plddtLogitsSource(atoms, dSingle, w.plddtBins),
    output: plddtLogits,
    inputs: { normedTex: normed, slotTex: call.slots, tableTex: w.plddtWeight },
    label: `ef2.conf.plddt-logits:${atoms}x${w.plddtBins}`,
  });
  execution.release(normed);
  const plddtPerAtom = execution.allocate(atoms, "ef2.conf.plddt-per-atom");
  execution.dispatch({
    source: categoricalMeanSource(atoms, w.plddtBins, 0, 1),
    output: plddtPerAtom,
    inputs: { logitsTex: plddtLogits },
    label: `ef2.conf.plddt-mean:${atoms}x${w.plddtBins}`,
  });
  const plddt = execution.allocate(tokens, "ef2.conf.plddt");
  execution.dispatch({
    source: perTokenSource(atoms, tokens),
    output: plddt,
    inputs: {
      valuesTex: plddtPerAtom, indexTex: call.atomToToken, maskTex: call.atomMask,
    },
    label: `ef2.conf.plddt-per-token:${atoms}->${tokens}`,
  });

  const paeLogits = linear(execution, {
    x: pair, weight: w.pae, rows: pairs,
    inChannels: dPair, outChannels: w.paeBins, weightLayout: "in-major",
  });
  const pae = execution.allocate(pairs, "ef2.conf.pae");
  execution.dispatch({
    source: categoricalMeanSource(pairs, w.paeBins, 0, 32),
    output: pae,
    inputs: { logitsTex: paeLogits },
    label: `ef2.conf.pae-mean:${pairs}x${w.paeBins}`,
  });
  execution.release(pairMask);
  execution.release(single);
  return {
    pair, single: pooledSingle, plddtLogits, plddtPerAtom, plddt, paeLogits, pae,
  };
}
