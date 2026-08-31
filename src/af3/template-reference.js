/**
 * AF3's template embedder, which contributes whether or not there are templates.
 *
 * 🔴 THIS IS THE MODULE EVERYONE SKIPS AND NOBODY SHOULD. On a de novo protein
 * with FOUR EMPTY TEMPLATE SLOTS its output measures std 13.1 against a pair
 * whose own std is 55 - about a quarter of the representation entering the MSA
 * stack. It is not a residual correction that a missing template makes zero.
 *
 * The reason is visible in the feature list below. Nine features are summed
 * into the embedding and only six of them are template geometry; the other
 * three are the QUERY's own aatype (twice, once per axis) and the query pair
 * representation itself. With no template the geometry vanishes and those three
 * do not, so the module becomes a learned transform of the query - and then
 * runs it through two pairformer blocks, a LayerNorm, a relu and a projection.
 *
 * AF2-multimer had the identical trap. It cost this project a week there.
 *
 * 🔴 ONLY THE EMPTY-TEMPLATE PATH IS IMPLEMENTED, AND REAL TEMPLATES RAISE.
 * The six geometry features - a 39-bin pseudo-beta distogram, its mask, three
 * components of a unit vector in each residue's backbone frame, and the
 * backbone mask - are all identically zero when no template is present, so
 * nothing in this repository can currently tell a correct implementation of
 * them from a wrong one. Writing them anyway would add a hundred lines that no
 * measurement covers, which is how a silent error gets in. They raise instead.
 */
import {
  gridSelfAttention, layerNorm, linear, transition, triangleMultiplication,
} from "./pairformer-reference.js";

const RESTYPES = 31;
const CHANNELS = 64;

/** One block of the template stack: the pair half of a pairformer block. */
function templateBlock(pair, pairMask, tokens, weights, dialect) {
  let act = Float32Array.from(pair);
  const add = (delta) => {
    for (let index = 0; index < act.length; index += 1) act[index] += delta[index];
  };
  add(triangleMultiplication(act, pairMask, tokens, CHANNELS, "outgoing",
                             weights.triangleMultiplicationOutgoing));
  add(triangleMultiplication(act, pairMask, tokens, CHANNELS, "incoming",
                             weights.triangleMultiplicationIncoming));
  add(gridSelfAttention(act, pairMask, tokens, CHANNELS, false,
                        weights.pairAttention1, dialect));
  add(gridSelfAttention(act, pairMask, tokens, CHANNELS, true,
                        weights.pairAttention2, dialect));
  // ...factor 2 here, against the trunk's 4. See transition() in
  // pairformer-reference.js.
  add(transition(act, tokens * tokens, CHANNELS, weights.pairTransition, 2));
  return act;
}

/**
 * The template embedding added to the pair representation.
 *
 * @param {{pair: Float32Array, tokens: number, pairMask: Float32Array,
 *          aatype: ArrayLike<number>, templates: number,
 *          templateOccupied: boolean}} input
 * @param {object} weights
 * @param {{swapTransposedBias: boolean}} dialect
 * @returns {Float32Array} tokens * tokens * pairChannels
 */
export function templateEmbedding(input, weights, dialect) {
  const { tokens, pair, pairMask, templates } = input;
  if (input.templateOccupied) {
    throw new Error("this template embedder only implements the empty-template"
      + " path: the six geometry features are identically zero without a"
      + " template, so nothing here can verify an implementation of them."
      + " See the note at the top of src/af3/template-reference.js.");
  }
  const pairs = tokens * tokens;

  // Feature 8: the query pair representation, normalised. With no template this
  // is the only per-pair signal, and it is what makes the module's output large.
  const normalised = layerNorm(pair, pairs, weights.queryChannels,
                               weights.queryEmbeddingNormScale,
                               weights.queryEmbeddingNormOffset);
  const act = linear(normalised, pairs, weights.queryChannels, CHANNELS,
                     weights.templatePairEmbedding8);

  // Features 2 and 3: the query aatype, once along each axis. The template's
  // OWN aatype is what AF3 embeds here, and an empty slot carries type 0 - so
  // these contribute row 0 of each weight rather than nothing.
  const oneHot = new Float32Array(tokens * RESTYPES);
  for (let token = 0; token < tokens; token += 1) {
    const code = input.templateAatype ? input.templateAatype[token] : 0;
    if (code >= 0 && code < RESTYPES) oneHot[token * RESTYPES + code] = 1;
  }
  const row = linear(oneHot, tokens, RESTYPES, CHANNELS,
                     weights.templatePairEmbedding2);
  const column = linear(oneHot, tokens, RESTYPES, CHANNELS,
                        weights.templatePairEmbedding3);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) {
      const base = (i * tokens + j) * CHANNELS;
      for (let c = 0; c < CHANNELS; c += 1) {
        // ...feature 2 is aatype[None, :, :], so it varies along j; feature 3
        // is aatype[:, None, :] and varies along i.
        act[base + c] += row[j * CHANNELS + c] + column[i * CHANNELS + c];
      }
    }
  }

  // Features 0, 1, 4, 5, 6 and 7 are the template geometry and are all exactly
  // zero here: the distogram is multiplied by a pseudo-beta mask that is zero,
  // the unit vectors by a backbone mask that is zero, and the two masks are
  // themselves the remaining features.

  let embedded = act;
  for (let index = 0; index < weights.blocks.length; index += 1) {
    embedded = templateBlock(embedded, pairMask, tokens, weights.blocks[index], dialect);
  }
  embedded = layerNorm(embedded, pairs, CHANNELS, weights.outputLayerNormScale,
                       weights.outputLayerNormOffset);

  // 🔴 THE SUM IS DIVIDED BY THE TEMPLATE COUNT, NOT BY HOW MANY ARE REAL.
  // Four empty slots each produce the SAME embedding, so the sum is four times
  // one of them and the division puts it back - the module behaves as though
  // there were exactly one template, whatever the slot count. Dividing by the
  // number of real templates instead would be a division by zero here.
  const summed = new Float32Array(embedded.length);
  for (let index = 0; index < embedded.length; index += 1) {
    summed[index] = embedded[index] * templates / (1e-7 + templates);
  }

  // ...relu before the projection, so the module can only add along the
  // directions its output_linear selects from a non-negative combination.
  for (let index = 0; index < summed.length; index += 1) {
    if (summed[index] < 0) summed[index] = 0;
  }
  return linear(summed, pairs, CHANNELS, weights.queryChannels, weights.outputLinear);
}
