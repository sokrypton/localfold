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
 * 🔴 REAL TEMPLATES USED TO RAISE, AND THE REASON WAS GOOD. The six geometry
 * features - a 39-bin pseudo-beta distogram, its mask, three components of a
 * unit vector in each residue's backbone frame, and the backbone mask - are
 * all identically zero when no template is present, so nothing here could tell
 * a correct implementation of them from a wrong one, and writing them anyway
 * would have been a hundred lines no measurement covers.
 *
 * `tools/oracle/dump_af3_trunk.py --template <pdb>` produces the measurement.
 * The features themselves are in src/af3/template-features.js, kept separate
 * because they are arithmetic over coordinates and can be checked before any
 * embedding is involved; `tools/oracle/check_af3_template.js` does both.
 */
import {
  gridSelfAttention, layerNorm, linear, transition, triangleMultiplication,
} from "./pairformer-reference.js";
import {
  DGRAM_BINS, coverageOf, multichainMaskFor, templateGeometry,
} from "./template-features.js";

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
 *          templates: number, slots?: (object|undefined)[],
 *          multichainMask2d?: ArrayLike<number>,
 *          onSlot?: (slot: number, embedded: Float32Array) => void}} input
 *   `slots` holds one entry per OCCUPIED slot - `{aatype, atomPositions,
 *   atomMask}` in AF3's dense-24 layout - with `undefined` for an empty one.
 * @param {object} weights
 * @param {{swapTransposedBias: boolean}} dialect
 * @returns {Float32Array} tokens * tokens * pairChannels
 */
export function templateEmbedding(input, weights, dialect) {
  const { tokens, pair, pairMask, templates } = input;
  const pairs = tokens * tokens;
  const slots = input.slots ?? [];
  if (slots.length > templates) {
    throw new RangeError(`${slots.length} templates for ${templates} slots`);
  }
  // 🔴 THE OLD FLAG STILL REFUSES, RATHER THAN BEING IGNORED. Callers wrote
  // `templateOccupied: <does the dump have a template>` to fail loudly when
  // one appeared, back when this path could not handle it. Now that it can,
  // dropping the flag would turn that deliberate noise into silence: a dump
  // WITH a template would be folded WITHOUT one and simply score worse.
  if (input.templateOccupied === true && slots.filter(Boolean).length === 0) {
    throw new Error("templateOccupied is true but no slots were given:"
      + " pass `slots` with {aatype, atomPositions, atomMask} per template");
  }
  const EMPTY_MASK = new Float32Array(pairs);
  // 🔴 THE MASK IS PER SLOT AND IS NOT ALLOWED TO DEFAULT TO "EVERYTHING". It
  // did, and a two-chain query with a template on each chain then scored
  // relRMS 1.09 against AF3 - the cross-chain geometry is most of the module's
  // answer, so a permissive default is not a small error. It went unnoticed
  // because every check had a ONE-CHAIN query, where all-ones and per-chain
  // are the same array.
  const chainMaskFor = (template) => {
    if (input.multichainMask2d !== undefined) return input.multichainMask2d;
    if (input.asymId === undefined) {
      if (template === undefined || template === null) {
        // An empty slot has no geometry to mask, so the mask is unread.
        return EMPTY_MASK;
      }
      throw new Error("a template needs `asymId` (or `multichainMask2d`):"
        + " AF3 masks the geometry features across chains, and assuming one"
        + " chain silently lets a template speak about pairs it has never"
        + " seen in one coordinate frame");
    }
    return multichainMaskFor(input.asymId, tokens, {
      coverage: coverageOf(template, tokens),
      // ...opt in, and only where one structure covered both chains. See
      // multichainMaskFor.
      spanChains: template.spanChains === true,
    });
  };

  // Feature 8: the query pair representation, normalised. It does not depend on
  // the slot, so it and its projection are computed ONCE - which is most of the
  // module's arithmetic when the slots are empty and all of it when there are
  // none.
  const normalised = layerNorm(pair, pairs, weights.queryChannels,
                               weights.queryEmbeddingNormScale,
                               weights.queryEmbeddingNormOffset);
  const queryTerm = linear(normalised, pairs, weights.queryChannels, CHANNELS,
                           weights.templatePairEmbedding8);

  const summed = new Float32Array(pairs * CHANNELS);
  for (let slot = 0; slot < templates; slot += 1) {
    const template = slots[slot];
    const act = Float32Array.from(queryTerm);

    // Features 2 and 3: the TEMPLATE's aatype, once along each axis. An empty
    // slot carries type 0 - ALA - so these contribute row 0 of each weight
    // rather than nothing, which is half of why an empty slot is not a no-op.
    const oneHot = new Float32Array(tokens * RESTYPES);
    for (let token = 0; token < tokens; token += 1) {
      const code = template ? template.aatype[token] : 0;
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

    // Features 0, 1, 4, 5, 6 and 7: the geometry. All exactly zero for an empty
    // slot - the distogram is multiplied by a pseudo-beta mask that is zero,
    // the unit vectors by a backbone mask that is zero, and the two masks are
    // themselves two of the features - so an empty slot skips the work rather
    // than computing zeros.
    if (template !== undefined && template !== null) {
      const geometry = templateGeometry(template, chainMaskFor(template), tokens);
      for (let index = 0; index < pairs; index += 1) {
        const base = index * CHANNELS;
        for (let bin = 0; bin < DGRAM_BINS; bin += 1) {
          const value = geometry.distogram[index * DGRAM_BINS + bin];
          if (value === 0) continue;
          for (let c = 0; c < CHANNELS; c += 1) {
            act[base + c] += value * weights.templatePairEmbedding0[bin * CHANNELS + c];
          }
        }
        // 🔴 FEATURES 1, 4, 5, 6 AND 7 ARE SCALARS TIMES A [64] VECTOR, not
        // matrix products. AF3 builds them with `num_input_dims=0`, which makes
        // the weight a per-channel scale rather than a projection - so reading
        // any of these four as a [1, 64] matmul is right by accident and
        // reading them as [39, 64] or [31, 64] is a shape error that only
        // shows up as a wrong answer.
        const scalars = [
          [geometry.pseudoBetaMask2d[index], weights.templatePairEmbedding1],
          [geometry.unitVector[index * 3], weights.templatePairEmbedding4],
          [geometry.unitVector[index * 3 + 1], weights.templatePairEmbedding5],
          [geometry.unitVector[index * 3 + 2], weights.templatePairEmbedding6],
          [geometry.backboneMask2d[index], weights.templatePairEmbedding7],
        ];
        for (const [value, weight] of scalars) {
          if (value === 0) continue;
          for (let c = 0; c < CHANNELS; c += 1) act[base + c] += value * weight[c];
        }
      }
    }

    let embedded = act;
    for (let index = 0; index < weights.blocks.length; index += 1) {
      embedded = templateBlock(embedded, pairMask, tokens, weights.blocks[index], dialect);
    }
    embedded = layerNorm(embedded, pairs, CHANNELS, weights.outputLayerNormScale,
                         weights.outputLayerNormOffset);
    // 🔴 REPORTED PER SLOT, BECAUSE THE SUM HIDES WHICH SLOT WAS WRONG. AF3
    // captures `single_template_embedding/__call__#k` at exactly this point -
    // after the two blocks and the LayerNorm, before the summation and the
    // output projection - so a checker can hold ONE slot's 64 channels to it
    // and see the geometry on its own. Summed and projected, a wrong unit
    // vector and a wrong distogram bin are the same number.
    input.onSlot?.(slot, embedded);
    for (let index = 0; index < summed.length; index += 1) summed[index] += embedded[index];
  }

  // 🔴 DIVIDED BY THE SLOT COUNT, NOT BY HOW MANY SLOTS ARE REAL. Four empty
  // slots each produce the SAME embedding, so the sum is four times one of them
  // and the division puts it back - the module behaves as though there were
  // exactly one template, whatever the slot count. With one real template among
  // four slots the real one is therefore worth a QUARTER of what it would be
  // alone, which is AF3's arithmetic and not an oversight to correct.
  const scale = 1 / (1e-7 + templates);
  for (let index = 0; index < summed.length; index += 1) {
    // ...relu before the projection, so the module can only add along the
    // directions its output_linear selects from a non-negative combination.
    summed[index] = Math.max(0, summed[index] * scale);
  }
  return linear(summed, pairs, CHANNELS, weights.queryChannels, weights.outputLinear);
}
