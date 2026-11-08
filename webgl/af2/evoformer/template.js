/**
 * AF2-monomer's template stack: the one place a known structure reaches the
 * pair track.
 *
 *     pair = embedding2d(geometry features)        one Linear over 88 columns
 *     pair = templatePairBlock(pair) x blocks      two attentions, two
 *                                                 multiplications, a transition
 *     update = Linear(Linear(LayerNorm(pair)))     the pointwise attention
 *
 * 🔴 ONE Linear OVER A CONCATENATION, where AF2-multimer and AF3 sum nine
 * separate projections. Same arithmetic, one `[88, 64]` weight, and the columns
 * are laid out in `to_concat`'s order:
 *
 *     0..38   the distogram, one-hot over 39 bins
 *     39      the pseudo-beta mask
 *     40..61  the template aatype along j, one-hot over 22
 *     62..83  ...and along i
 *     84..86  the unit vector
 *     87      the backbone mask
 *
 * 🔴 THE WHOLE 88 IS MULTIPLIED BY THE BACKBONE MASK, not each feature by its
 * own - `act *= template_mask_2d[..., None]`, where that mask is N, CA and C.
 * So the monomer's distogram survives at a pair whose pseudo-beta is missing and
 * whose backbone is not, which is the opposite of what AF3 and the multimer do.
 * The BIAS is outside the mask, so an all-masked template leaves exactly the
 * bias - and that bias then runs the whole stack and the pointwise projections,
 * which is why a masked template does NOT produce a zero update.
 *
 * 🔴 AND `use_template_unit_vector` IS FALSE IN EVERY SHIPPED MONOMER CONFIG, so
 * columns 84..86 are deliberately zero. It is baked into the source here rather
 * than defaulted silently: the flag is per model, and a checkpoint that set it
 * true would otherwise be mis-embedded without a word.
 *
 * 🔴 THE POINTWISE ATTENTION HAS NO QUERY, NO KEY AND NO SOFTMAX HERE. AF2's
 * `TemplatePointwiseAttention` attends over the template axis; the monomer runs
 * ONE template, so the softmax over one logit is exactly 1 and the whole thing
 * collapses to `Linear(Linear(x))` - the value projection, then the output
 * projection with its bias. `query_w` and `key_w` are in the checkpoint and are
 * genuinely unread; `webgpu/af2/evoformer/template.js` reads neither either.
 */
import { ONE_TARGET, PRELUDE, memoSource } from "../../runtime/execution.js";
import { layerNorm } from "../../kernels/layer-norm.js";
import { linear } from "../../kernels/linear.js";
import { templatePairBlock } from "./block.js";
import {
  AF2_ATOM37_MONOMER, GEOMETRY_STRIDE, coverageOf, multichainMaskFor,
  packTemplateGeometry, templateGeometry,
} from "../../../shared/af3/featurise/template-features.js";
import { GAP_AATYPE } from "../../../shared/af3/featurise/template-input.js";

/** @typedef {import("../../runtime/storage.js").Tensor} Tensor */

/** Channels in AF2-monomer's `to_concat`: 39 + 1 + 22 + 22 + 3 + 1. */
export const FEATURE_COLUMNS = 88;

export const TEMPLATE_WEIGHTS = [
  "embeddingWeight", "embeddingBias",
  "outputNormScale", "outputNormOffset",
  "valueWeight", "outputWeight", "outputBias",
];

/**
 * The 88-column embedding, one fragment per four output channels of one pair.
 *
 * The geometry arrives packed six floats to a pair (`packTemplateGeometry`):
 * the distogram bin PLUS ONE, the pseudo-beta mask, three unit-vector
 * components and the backbone mask. The offset on the bin is what lets a zeroed
 * buffer mean "no template here" rather than "bin 0", which is the
 * shortest-distance bin and the strongest signal there is.
 */
const templateEmbedSourceUncached = (shape) => {
  const { length, templateChannels, unitVectors } = shape;
  const unit = unitVectors
    ? `    inner += fetch1(geometryTex, g + 2) * fetch4(weightTex, 84 * CT4 + group);
    inner += fetch1(geometryTex, g + 3) * fetch4(weightTex, 85 * CT4 + group);
    inner += fetch1(geometryTex, g + 4) * fetch4(weightTex, 86 * CT4 + group);`
    : "    // use_template_unit_vector is false: columns 84..86 contribute nothing.";
  return `${PRELUDE}${ONE_TARGET}
const int L = ${length};
const int CT4 = ${templateChannels / 4};
const int STRIDE = ${GEOMETRY_STRIDE};
const int ELEMENTS = ${length * length * templateChannels};
uniform sampler2D biasTex;
uniform sampler2D weightTex;
uniform sampler2D geometryTex;
uniform sampler2D aatypeTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % CT4;
  int entry = texel / CT4;
  int i = entry / L;
  int j = entry - i * L;

  int g = entry * STRIDE;
  float backbone = fetch1(geometryTex, g + 5);
  vec4 value = fetch4(biasTex, group);
  if (backbone != 0.0) {
    vec4 inner = vec4(0.0);
    // The bin is stored plus one, so zero is "no bin at all" - a pair closer
    // than 3.25 A contributes nothing rather than falling into bin 0.
    int bin = int(fetch1(geometryTex, g));
    if (bin > 0) inner += fetch4(weightTex, (bin - 1) * CT4 + group);
    inner += fetch1(geometryTex, g + 1) * fetch4(weightTex, 39 * CT4 + group);
    int rowType = int(clamp(fetch1(aatypeTex, j), 0.0, 21.0));
    int colType = int(clamp(fetch1(aatypeTex, i), 0.0, 21.0));
    inner += fetch4(weightTex, (40 + rowType) * CT4 + group);
    inner += fetch4(weightTex, (62 + colType) * CT4 + group);
${unit}
    inner += fetch4(weightTex, 87 * CT4 + group);
    value += backbone * inner;
  }
  result = value;
}`;
};

/**
 * The host-side half: the packed geometry and the per-residue template aatype.
 *
 * 🔴 AN UNCOVERED POSITION IS THE GAP TYPE, NOT ALANINE. AF2's own pipeline maps
 * an alignment gap to `restypes_with_x_and_gap`'s 21, so an absent template
 * fills the whole vector with 21 rather than leaving it zeroed - zero is
 * alanine, and a template of alanines is a plausible wrong answer.
 *
 * 🔴 AND A MONOMER IS ONE CHAIN, WHICH IS STILL NOT A LICENCE TO ASSUME IT.
 * `asymId` is honoured when given, because AF3 and AF2-multimer both shipped a
 * permissive default that was wrong for a complex.
 *
 * @typedef {{aatype: ArrayLike<number>, atomMask: ArrayLike<number>,
 *   atomPositions: ArrayLike<number>, spanChains?: boolean}} TemplateSlot
 *
 * @param {{template?: TemplateSlot, length: number, asymId?: ArrayLike<number>,
 *   multichainMask2d?: Float32Array}} input
 * @returns {{geometry: Float32Array, aatype: Float32Array}}
 */
export function templateFeatures(input) {
  const { length, template } = input;
  const chainMask = input.multichainMask2d
    ?? (input.asymId !== undefined
      ? multichainMaskFor(input.asymId, length, {
        coverage: coverageOf(template, length, AF2_ATOM37_MONOMER),
        spanChains: template?.spanChains === true,
      })
      : new Float32Array(length * length).fill(1));
  // A masked template is expressed as DATA, not as a second path: zeroed
  // geometry has a zero backbone mask, and the shader then leaves exactly the
  // embedding bias.
  const geometry = template === undefined
    ? new Float32Array(length * length * GEOMETRY_STRIDE)
    // The cast is the one shared-side wart here: `templateGeometry` returns a
    // Uint8Array distogram on one branch and a Float32Array on the other, and
    // `packTemplateGeometry` declares the second. `webgpu/af2/evoformer/
    // template.js` carries the identical error on the identical call.
    : packTemplateGeometry(/** @type {any} */ (
      templateGeometry(template, chainMask, length, AF2_ATOM37_MONOMER)), length);
  const aatype = new Float32Array(length).fill(GAP_AATYPE);
  if (template !== undefined) {
    for (let token = 0; token < length; token += 1) aatype[token] = template.aatype[token];
  }
  return { geometry, aatype };
}

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 */
export function uploadTemplateWeights(execution, weights) {
  /** @type {Record<string, Tensor>} */
  const out = {};
  for (const name of TEMPLATE_WEIGHTS) {
    const values = weights[name];
    if (values === undefined) throw new Error(`template weights: ${name} is missing`);
    out[name] = execution.uploadWeight(values, `w.template.${name}`);
  }
  return out;
}

export function releaseTemplateWeights(execution, weights) {
  for (const name of TEMPLATE_WEIGHTS) execution.release(weights[name]);
}

/**
 * The whole stage: embed, run the blocks, normalise, project twice.
 *
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{geometry: Tensor, aatype: Tensor, pairMask: Tensor,
 *   weights: Record<string, Tensor>,
 *   blockWeights: ReturnType<typeof import("./block.js").uploadTemplatePairBlockWeights>[],
 *   length: number, templateChannels: number, pairChannels: number,
 *   triangleHidden: number, pairHidden: number,
 *   useTemplateUnitVector?: boolean, epsilon?: number,
 *   triangleAttentionPrecision?: "f16" | "f32",
 *   trianglePrecision?: "f16" | "f32"}} call
 * @returns {Tensor} the pair UPDATE, `length * length * pairChannels` floats.
 *   The caller adds it; AF2's own `pair += template_pair_representation`.
 */
export function templateEmbedding(execution, call) {
  const {
    geometry, aatype, pairMask, weights: w, blockWeights,
    length, templateChannels, pairChannels, triangleHidden, pairHidden, epsilon = 1e-5,
  } = call;
  if (templateChannels % 4 !== 0) {
    throw new RangeError(
      `templateEmbedding: templateChannels must be a multiple of 4, got ${templateChannels}`);
  }
  if (w.embeddingWeight.elements !== FEATURE_COLUMNS * templateChannels) {
    throw new RangeError(
      `templateEmbedding: embeddingWeight holds ${w.embeddingWeight.elements} floats,`
      + ` not ${FEATURE_COLUMNS} x ${templateChannels}`);
  }
  const pairs = length * length;
  if (geometry.elements !== pairs * GEOMETRY_STRIDE) {
    throw new RangeError(
      `templateEmbedding: geometry holds ${geometry.elements} floats,`
      + ` not ${pairs} x ${GEOMETRY_STRIDE}`);
  }

  let pair = execution.allocate(pairs * templateChannels, "template.pair");
  execution.dispatch({
    source: templateEmbedSource({
      length, templateChannels, unitVectors: call.useTemplateUnitVector === true,
    }),
    output: pair,
    inputs: {
      biasTex: w.embeddingBias, weightTex: w.embeddingWeight,
      geometryTex: geometry, aatypeTex: aatype,
    },
    label: `template.embed:${length}x${templateChannels}`,
  });

  for (const block of blockWeights) {
    const next = templatePairBlock(execution, {
      pair, pairMask, length, cZ: templateChannels, triangleHidden, pairHidden,
      weights: block, epsilon,
      triangleAttentionPrecision: call.triangleAttentionPrecision,
      trianglePrecision: call.trianglePrecision,
    });
    execution.release(pair);
    pair = next;
  }

  const normalised = layerNorm(execution, {
    x: pair, weight: w.outputNormScale, bias: w.outputNormOffset,
    rows: pairs, channels: templateChannels, epsilon,
  });
  execution.release(pair);
  const projected = w.valueWeight.elements / templateChannels;
  if (!Number.isInteger(projected) || projected % 4 !== 0) {
    throw new RangeError(
      `templateEmbedding: valueWeight holds ${w.valueWeight.elements} floats, which is not`
      + ` ${templateChannels} x a multiple of four`);
  }
  const value = linear(execution, {
    x: normalised, weight: w.valueWeight, rows: pairs,
    inChannels: templateChannels, outChannels: projected, weightLayout: "in-major",
  });
  execution.release(normalised);
  const update = linear(execution, {
    x: value, weight: w.outputWeight, bias: w.outputBias, rows: pairs,
    inChannels: projected, outChannels: pairChannels, weightLayout: "in-major",
  });
  execution.release(value);
  update.label = "template.update";
  return update;
}

// Every builder above, memoised on its arguments (`memoSource` in webgl/runtime/execution.js).
export const templateEmbedSource = memoSource(templateEmbedSourceUncached);
