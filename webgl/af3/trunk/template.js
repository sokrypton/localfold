/**
 * AF3's template embedder, which contributes whether or not there are templates.
 *
 * 🔴 THIS IS THE MODULE EVERYONE SKIPS AND NOBODY SHOULD. "On a de novo protein
 * with FOUR EMPTY TEMPLATE SLOTS its output measures std 13.1 against a pair
 * whose own std is 55 - about a quarter of the representation entering the MSA
 * stack. It is not a residual correction that a missing template makes zero."
 * Nine features are summed and only six are template geometry; the other three
 * are the QUERY's own aatype (twice, once per axis) and the query pair itself,
 * so with no template the module becomes a learned transform of the query - and
 * then runs it through two blocks, a LayerNorm, a relu and a projection.
 * AF2-multimer had the identical trap and it cost this project a week there.
 *
 *     query  = Linear(LayerNorm(pair))                computed ONCE
 *     slot:  act = query + W2[aatype_j] + W3[aatype_i]
 *                       + (occupied ? the six geometry features : 0)
 *            act = templateBlock(act) x 2
 *            act = LayerNorm(act)
 *     out    = Linear(relu(sum_slots(act) / (1e-7 + slots)))
 *
 * 🔴 DIVIDED BY THE SLOT COUNT, NOT BY HOW MANY SLOTS ARE REAL. "Four empty
 * slots each produce the SAME embedding, so the sum is four times one of them
 * and the division puts it back - the module behaves as though there were
 * exactly one template, whatever the slot count. With one real template among
 * four slots the real one is therefore worth a QUARTER of what it would be
 * alone, which is AF3's arithmetic and not an oversight to correct."
 *
 * 🔴 AND FIVE OF THE SIX GEOMETRY FEATURES ARE SCALARS TIMES A VECTOR, NOT
 * MATRIX PRODUCTS. "AF3 builds them with `num_input_dims=0`, which makes the
 * weight a per-channel scale rather than a projection - so reading any of these
 * four as a [1, 64] matmul is right by accident and reading them as [39, 64] or
 * [31, 64] is a shape error that only shows up as a wrong answer."
 *
 * 🔴 AND WHICH aatype ROW AN EMPTY SLOT GETS IS THE DIALECT'S, AND ONLY THE
 * FIRST EMPTY SLOT GETS IT. "OpenDDE and protenix2 take protenix's featuriser,
 * which fills its one empty template with the GAP restype and zero-pads the
 * rest... Writing 0 everywhere put row 0 (ALA) where row 21 belongs and was
 * worth 2.83e-1 on the module and 2.07e-2 on the trunk's `z_after_template`
 * seam - a defect that needed no template to appear."
 *
 * 🔴 AND THE CHAIN MASK IS PER SLOT AND MAY NOT DEFAULT TO "EVERYTHING". It did,
 * "and a two-chain query with a template on each chain then scored relRMS 1.09
 * against AF3 - the cross-chain geometry is most of the module's answer... It
 * went unnoticed because every check had a ONE-CHAIN query, where all-ones and
 * per-chain are the same array."
 */
import { ONE_TARGET, PRELUDE, sized } from "../../runtime/execution.js";
import { drain } from "../../runtime/steps.js";
import { add, relu, scale } from "../../runtime/elementwise.js";
import { layerNorm } from "../../kernels/layer-norm.js";
import { linear } from "../../kernels/linear.js";
import { pairHalf } from "./pairformer-block.js";
import {
  releaseGridAttentionWeights, uploadGridAttentionWeights,
} from "./grid-attention.js";
import { releaseTransitionWeights, uploadTransitionWeights } from "./transition.js";
import { releaseTriangleWeights, uploadTriangleWeights } from "./triangle.js";
import {
  DGRAM_BINS, GEOMETRY_STRIDE, coverageOf, multichainMaskFor,
  packTemplateGeometry, templateGeometry,
} from "../../../shared/af3/featurise/template-features.js";

/** @typedef {import("../../runtime/storage.js").Tensor} Tensor */

/** AF3's template restype vocabulary: 31 classes, with 21 the GAP. */
export const RESTYPES = 31;
export const GAP_RESTYPE = 21;

export const TEMPLATE_WEIGHTS = [
  "queryEmbeddingNormScale", "queryEmbeddingNormOffset",
  "templatePairEmbedding8", "templatePairEmbedding2", "templatePairEmbedding3",
  "templatePairEmbedding0", "templatePairEmbedding1",
  "templatePairEmbedding4", "templatePairEmbedding5", "templatePairEmbedding6",
  "templatePairEmbedding7",
  "outputLayerNormScale", "outputLayerNormOffset", "outputLinear",
];

/**
 * One slot's nine features, summed into the stack's input in one draw.
 *
 * The aatype one-hots are gathers rather than matrix products - a 31-wide input
 * is not a multiple of four, so `kernels/linear.js` refuses it, and a one-hot
 * needs one row anyway. The geometry arrives packed six floats to a pair
 * (`packTemplateGeometry`), with the distogram bin stored PLUS ONE so that a
 * zeroed buffer is an empty slot.
 */
export const slotInputSource = (tokens, channels, occupied) => sized({ L: tokens, ELEMENTS: tokens * tokens * channels }, `${PRELUDE}${ONE_TARGET}
const int C4 = ${channels / 4};
const int STRIDE = ${GEOMETRY_STRIDE};
uniform sampler2D queryTex;
uniform sampler2D codeTex;
uniform sampler2D rowWeightTex;
uniform sampler2D columnWeightTex;
${occupied ? `uniform sampler2D geometryTex;
uniform sampler2D dgramWeightTex;
uniform sampler2D maskWeightTex;
uniform sampler2D unitXWeightTex;
uniform sampler2D unitYWeightTex;
uniform sampler2D unitZWeightTex;
uniform sampler2D backboneWeightTex;` : ""}

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int pair = texel / C4;
  int i = pair / L;
  int j = pair - i * L;

  // Feature 8, the query pair, already projected and slot-independent.
  vec4 total = fetch4(queryTex, texel);
  // Features 2 and 3: feature 2 varies along j, feature 3 along i.
  int rowCode = int(fetch1(codeTex, j));
  int columnCode = int(fetch1(codeTex, i));
  if (rowCode >= 0 && rowCode < ${RESTYPES}) {
    total += fetch4(rowWeightTex, rowCode * C4 + group);
  }
  if (columnCode >= 0 && columnCode < ${RESTYPES}) {
    total += fetch4(columnWeightTex, columnCode * C4 + group);
  }
${occupied ? `  // Feature 0, the distogram: the bin is stored plus one, so zero is no bin.
  int g = pair * STRIDE;
  int bin = int(fetch1(geometryTex, g));
  if (bin > 0) total += fetch4(dgramWeightTex, (bin - 1) * C4 + group);
  // Features 1, 4, 5, 6 and 7: a SCALAR times a per-channel vector.
  total += fetch1(geometryTex, g + 1) * fetch4(maskWeightTex, group);
  total += fetch1(geometryTex, g + 2) * fetch4(unitXWeightTex, group);
  total += fetch1(geometryTex, g + 3) * fetch4(unitYWeightTex, group);
  total += fetch1(geometryTex, g + 4) * fetch4(unitZWeightTex, group);
  total += fetch1(geometryTex, g + 5) * fetch4(backboneWeightTex, group);`
    : "  // An empty slot has no geometry: every one of the six is exactly zero."}
  result = total;
}`);

/**
 * The per-token restype code for one slot, on the host.
 *
 * @typedef {{aatype: ArrayLike<number>, atomMask: ArrayLike<number>,
 *   atomPositions: ArrayLike<number>, spanChains?: boolean}} TemplateSlot
 *
 * @param {{template?: TemplateSlot | null, tokens: number, slot: number,
 *   firstEmpty: number, emptyTemplateAatype?: number | null,
 *   templateGapWhereNoAtoms?: boolean}} input
 */
export function slotCodes(input) {
  const { template, tokens, slot, firstEmpty } = input;
  const out = new Float32Array(tokens);
  if (template === undefined || template === null) {
    // 🔴 THE FIRST EMPTY SLOT, NOT SLOT ZERO: "with a real template in slot 0
    // the featuriser's 'one empty template' is the first unoccupied one".
    const gap = input.emptyTemplateAatype ?? null;
    const code = input.templateGapWhereNoAtoms === true
      ? GAP_RESTYPE
      : (gap !== null && slot === firstEmpty ? gap : 0);
    return out.fill(code);
  }
  const dense = template.atomMask.length / tokens;
  for (let token = 0; token < tokens; token += 1) {
    let code = template.aatype[token];
    if (input.templateGapWhereNoAtoms === true) {
      let covered = false;
      for (let atom = 0; atom < dense; atom += 1) {
        if (template.atomMask[token * dense + atom] > 0) { covered = true; break; }
      }
      if (!covered) code = GAP_RESTYPE;
    }
    out[token] = code;
  }
  return out;
}

/**
 * One slot's packed geometry, or undefined for an empty slot.
 *
 * 🔴 AN EMPTY SLOT SKIPS THE WORK RATHER THAN COMPUTING ZEROS, which is what the
 * reference does and what the `occupied` branch of the kernel above expresses:
 * "the distogram is multiplied by a pseudo-beta mask that is zero, the unit
 * vectors by a backbone mask that is zero, and the two masks are themselves two
 * of the features".
 */
export function slotGeometry(input) {
  const { template, tokens } = input;
  if (template === undefined || template === null) return undefined;
  const chainMask = input.multichainMask2d ?? (() => {
    if (input.asymId === undefined) {
      throw new Error(
        "a template needs `asymId` (or `multichainMask2d`): AF3 masks the geometry features"
        + " across chains, and assuming one chain silently lets a template speak about"
        + " pairs it has never seen in one coordinate frame");
    }
    return multichainMaskFor(input.asymId, tokens, {
      coverage: coverageOf(template, tokens),
      spanChains: template.spanChains === true,
    });
  })();
  return packTemplateGeometry(/** @type {any} */ (
    templateGeometry(template, chainMask, tokens)), tokens);
}

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 */
export function uploadTemplateWeights(execution, weights) {
  /** @type {any} */
  const out = {
    channels: weights.channels,
    queryChannels: weights.queryChannels,
  };
  for (const name of TEMPLATE_WEIGHTS) {
    const values = weights[name];
    if (values === undefined) throw new Error(`af3 template weights: ${name} is missing`);
    out[name] = execution.uploadWeight(values, `w.af3.template.${name}`);
  }
  out.blocks = weights.blocks.map((block) => ({
    triangleMultiplicationOutgoing: uploadTriangleWeights(
      execution, block.triangleMultiplicationOutgoing),
    triangleMultiplicationIncoming: uploadTriangleWeights(
      execution, block.triangleMultiplicationIncoming),
    pairAttention1: uploadGridAttentionWeights(execution, block.pairAttention1),
    pairAttention2: uploadGridAttentionWeights(execution, block.pairAttention2),
    pairTransition: uploadTransitionWeights(execution, block.pairTransition),
  }));
  return out;
}

export function releaseTemplateWeights(execution, weights) {
  for (const name of TEMPLATE_WEIGHTS) execution.release(weights[name]);
  for (const block of weights.blocks) {
    releaseTriangleWeights(execution, block.triangleMultiplicationOutgoing);
    releaseTriangleWeights(execution, block.triangleMultiplicationIncoming);
    releaseGridAttentionWeights(execution, block.pairAttention1);
    releaseGridAttentionWeights(execution, block.pairAttention2);
    releaseTransitionWeights(execution, block.pairTransition);
  }
}

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{pair: Tensor, pairMask: Tensor, tokens: number, templates: number,
 *   weights: ReturnType<typeof uploadTemplateWeights>,
 *   codes: Tensor[], geometry: (Tensor | undefined)[],
 *   swapTransposedBias?: boolean, divideByLength?: boolean, epsilon?: number,
 *   onSlot?: (slot: number, values: Float32Array) => void}} call
 * @returns {Tensor} `tokens^2 * queryChannels`, the term the embedder's pair
 *   gets ADDED - the caller adds it, because the stack read the pair before it
 */
export function templateEmbedding(execution, call) {
  return drain(templateEmbeddingSteps(execution, call));
}

/** `templateEmbedding` as a generator: yields after each template slot (its own pairformer blocks). */
export function* templateEmbeddingSteps(execution, call) {
  const {
    pair, pairMask, tokens, templates, weights: w, codes, geometry,
    swapTransposedBias = false, divideByLength = false, epsilon = 1e-5,
  } = call;
  const channels = w.channels;
  const queryChannels = w.queryChannels;
  const pairs = tokens * tokens;
  if (codes.length !== templates || geometry.length !== templates) {
    throw new RangeError(
      `af3 template: ${codes.length} code sets and ${geometry.length} geometry sets`
      + ` for ${templates} slots`);
  }
  if (channels % 4 !== 0) {
    throw new RangeError(`af3 template: channels must be a multiple of 4, got ${channels}`);
  }

  // Feature 8 does not depend on the slot, so it and its projection run ONCE -
  // "which is most of the module's arithmetic when the slots are empty and all
  // of it when there are none".
  const normalised = layerNorm(execution, {
    x: pair, weight: w.queryEmbeddingNormScale, bias: w.queryEmbeddingNormOffset,
    rows: pairs, channels: queryChannels, epsilon, variance: "fast",
  });
  const queryTerm = linear(execution, {
    x: normalised, weight: w.templatePairEmbedding8, rows: pairs,
    inChannels: queryChannels, outChannels: channels, weightLayout: "in-major",
  });
  execution.release(normalised);

  let summed;
  for (let slot = 0; slot < templates; slot += 1) {
    const occupied = geometry[slot] !== undefined;
    let act = execution.allocate(pairs * channels, `af3.template.slot${slot}`);
    execution.dispatch({
      source: slotInputSource(tokens, channels, occupied),
      output: act,
      inputs: {
        queryTex: queryTerm, codeTex: codes[slot],
        rowWeightTex: w.templatePairEmbedding2,
        columnWeightTex: w.templatePairEmbedding3,
        ...(occupied
          ? {
            geometryTex: geometry[slot],
            dgramWeightTex: w.templatePairEmbedding0,
            maskWeightTex: w.templatePairEmbedding1,
            unitXWeightTex: w.templatePairEmbedding4,
            unitYWeightTex: w.templatePairEmbedding5,
            unitZWeightTex: w.templatePairEmbedding6,
            backboneWeightTex: w.templatePairEmbedding7,
          }
          : {}),
      },
      label: `af3.template.slot:${tokens}x${channels}${occupied ? ":occupied" : ":empty"}`,
    });
    for (const block of w.blocks) {
      act = pairHalf(execution, {
        pair: act, pairMask, tokens, channels, weights: block,
        swapTransposedBias, divideByLength, epsilon,
      }, true);
    }
    const embedded = layerNorm(execution, {
      x: act, weight: w.outputLayerNormScale, bias: w.outputLayerNormOffset,
      rows: pairs, channels, epsilon, variance: "fast",
    });
    execution.release(act);
    // 🔴 REPORTED PER SLOT, BECAUSE THE SUM HIDES WHICH SLOT WAS WRONG. AF3
    // captures its own tap at exactly this point - after the blocks and the
    // LayerNorm, before the summation and the output projection - so a checker
    // can hold ONE slot's channels to it and see the geometry on its own.
    // "Summed and projected, a wrong unit vector and a wrong distogram bin are
    // the same number."
    call.onSlot?.(slot, Float32Array.from(execution.read(embedded)));
    if (summed === undefined) {
      summed = embedded;
    } else {
      const next = add(execution, summed, embedded);
      execution.release(summed);
      execution.release(embedded);
      summed = next;
    }
    yield { stage: "template", slot };
  }
  execution.release(queryTerm);
  if (summed === undefined) {
    throw new RangeError("af3 template: at least one slot is required - AF3 pads to four");
  }

  // Divided by the SLOT COUNT, then a relu, then the projection.
  const averaged = scale(execution, summed, 1 / (1e-7 + templates));
  execution.release(summed);
  const activated = relu(execution, averaged);
  execution.release(averaged);
  const out = linear(execution, {
    x: activated, weight: w.outputLinear, rows: pairs,
    inChannels: channels, outChannels: queryChannels, weightLayout: "in-major",
  });
  execution.release(activated);
  out.label = "af3.template";
  return out;
}
