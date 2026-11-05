/**
 * AF2-MULTIMER's template embedder: nine projections summed, where the monomer
 * has one Linear over an 88-column concatenation.
 *
 *     act    = sum(biases) + W_query . LayerNorm(pair)
 *            + W_aatype_row[aatype[j]] + W_aatype_col[aatype[i]]
 *            + W_dgram[bin] + sum_f geometry[f] * W_f
 *     act    = templatePairBlock(act) x 2            multimer's order
 *     update = relu(LayerNorm(act) / templates) . W_out + b_out
 *     pair  += update
 *
 * 🔴 IT RUNS WHETHER OR NOT THERE ARE TEMPLATES, AND THAT IS NOT A DETAIL.
 * Multimer's config has `template.enabled` true and the embedding wrapper adds
 * this to the pair unconditionally. Masking every template off does NOT zero
 * the term: the input Linears still contribute their biases, a one-hot of
 * aatype zero is still a one-hot, and the query embedding is still read. The
 * reference records what omitting it cost - "the pair track 30% out from the
 * first evoformer block, which is what shattered backbones at six copies and
 * made recycling wander instead of converge".
 *
 * 🔴 AND THE NINE INDICES ARE NOT IN THE ORDER THE FEATURES ARE PACKED IN.
 * `template_pair_embedding_0..8` are the distogram, its mask, the aatype along
 * each axis, three unit-vector components, the backbone mask and the query
 * pair; the packed geometry is bin, mask, three unit-vector components,
 * backbone mask. So the six geometry embeddings are 0, 1, 4, 5, 6, 7 and the
 * two aatype ones are 2 and 3 - a plausible-looking wrong answer is one
 * mis-mapped index away.
 *
 * 🔴 AND THE TWO AATYPE TERMS ARE PER RESIDUE RATHER THAN CONSTANT, which the
 * reference had to learn: "That was right while a masked template was the only
 * case - one residue type repeated is the same weight row everywhere - and is
 * wrong the moment a template has real residues in it."
 *
 * The LayerNorms and the three projections here are the gated kernels rather
 * than one fused pass: the reference fuses its query LayerNorm into the input
 * shader and its output LayerNorm into the output one, which is a speed choice
 * this port has not earned yet. Four draws and two extra tensors.
 */
import { ONE_TARGET, PRELUDE } from "../../runtime/execution.js";
import { layerNorm } from "../../kernels/layer-norm.js";
import { linear } from "../../kernels/linear.js";
import { relu, scale } from "../../runtime/elementwise.js";
import { templatePairBlock } from "../evoformer/block.js";
import {
  AF2_ATOM37, GEOMETRY_STRIDE, coverageOf, multichainMaskFor,
  packTemplateGeometry, templateGeometry,
} from "../../../shared/af3/featurise/template-features.js";

/** @typedef {import("../../runtime/storage.js").Tensor} Tensor */

export const TEMPLATE_CHANNELS = 64;
export const AATYPE_CLASSES = 22;
/** The distogram bins AF3's own featuriser produces, which this shares. */
export const DGRAM_BINS = 39;
/** Which of the nine embeddings are geometry, in the packed features' order. */
export const GEOMETRY_FEATURES = [0, 1, 4, 5, 6, 7];
/** ...and which are the two aatype axes, along j then along i. */
export const AATYPE_FEATURES = [2, 3];
/** ...and the query pair embedding. */
export const QUERY_FEATURE = 8;

export const TEMPLATE_WEIGHTS = [
  "queryNormScale", "queryNormOffset", "outputNormScale", "outputNormOffset",
  "outputWeight", "outputBias",
];

/**
 * The constant part of `construct_input`: every embedding's bias but the
 * query's, which the input kernel adds separately because it sits beside that
 * embedding's weight.
 */
export function templateConstantTerm(weights) {
  const constant = new Float32Array(TEMPLATE_CHANNELS);
  for (const [index, { bias }] of weights.pairEmbeddings.entries()) {
    if (index === QUERY_FEATURE) continue;
    for (let c = 0; c < TEMPLATE_CHANNELS; c += 1) constant[c] += bias[c];
  }
  return constant;
}

/**
 * The two aatype terms, gathered on the host: `[length, 64]` each, the row one
 * read at j and the column one at i.
 */
export function templateAatypeTerms(weights, aatype, length) {
  const row = new Float32Array(length * TEMPLATE_CHANNELS);
  const column = new Float32Array(length * TEMPLATE_CHANNELS);
  /** @type {[number, Float32Array][]} */
  const axes = [[AATYPE_FEATURES[0], row], [AATYPE_FEATURES[1], column]];
  for (const [index, into] of axes) {
    const { weight } = weights.pairEmbeddings[index];
    for (let residue = 0; residue < length; residue += 1) {
      const code = Math.min(Math.max(aatype[residue] ?? 0, 0), AATYPE_CLASSES - 1);
      for (let c = 0; c < TEMPLATE_CHANNELS; c += 1) {
        into[residue * TEMPLATE_CHANNELS + c] = weight[code * TEMPLATE_CHANNELS + c];
      }
    }
  }
  return { row, column };
}

/**
 * The five scalar geometry weights as one `[5, 64]` tensor, in the order the
 * packed geometry stores them: the pseudo-beta mask, three unit-vector
 * components, the backbone mask. The distogram's is `[39, 64]` and stays its
 * own tensor because it is indexed by a bin rather than scaled by a value.
 */
export function scalarGeometryWeights(weights) {
  const scalars = GEOMETRY_FEATURES.slice(1);
  const out = new Float32Array(scalars.length * TEMPLATE_CHANNELS);
  for (const [slot, index] of scalars.entries()) {
    out.set(weights.pairEmbeddings[index].weight.subarray(0, TEMPLATE_CHANNELS),
      slot * TEMPLATE_CHANNELS);
  }
  return out;
}

/**
 * The geometry and the template aatype, on the host.
 *
 * 🔴 AF2-MULTIMER IS THE SECOND OF THREE MASKING DIALECTS. `AF2_ATOM37` leaves
 * `maskDistogram` at its default, so the multimer multiplies the distogram by
 * the pseudo-beta mask where `AF2_ATOM37_MONOMER` sets it false and masks the
 * whole concatenation by the BACKBONE mask instead. Same six features, same
 * atom layout, opposite masking - and `shared/af3/featurise/template-features.js`
 * is where that is stated once for all three.
 */
export function multimerTemplateFeatures(input) {
  const { length, template } = input;
  // 🔴 A TEMPLATE WITHOUT `asymId` IS REFUSED, AND THAT REFUSAL IS THE
  // REFERENCE'S. `webgpu/af2/multimer/template.js` raises "a template needs
  // `asymId` (or `multichainMask2d`): AF2-multimer masks the template's geometry
  // across chains, and assuming one chain lets it speak about pairs it has never
  // seen in one coordinate frame" - and this port defaulted to all-ones instead,
  // copied from the monomer path where one chain IS the model. A permissive
  // default here is the exact mistake the monomer template's own comment warns
  // about ("AF3 and AF2-multimer both shipped a permissive default that was
  // wrong for a complex"), and it was caught by the reference refusing where
  // this port would have folded.
  if (template !== undefined
      && input.multichainMask2d === undefined && input.asymId === undefined) {
    throw new Error(
      "a template needs `asymId` (or `multichainMask2d`): AF2-multimer masks the template's"
      + " geometry across chains, and assuming one chain lets it speak about pairs it has"
      + " never seen in one coordinate frame");
  }
  // ...and the mask is built only where there is a template to mask, which is
  // also why the refusal above cannot fire on the masked path. A masked slot has
  // no geometry at all, so there is nothing for a chain mask to say about it.
  const chainMaskFor = () => input.multichainMask2d
    ?? multichainMaskFor(input.asymId, length, {
      coverage: coverageOf(template, length, AF2_ATOM37),
      spanChains: template?.spanChains === true,
    });
  const geometry = template === undefined
    ? new Float32Array(length * length * GEOMETRY_STRIDE)
    : packTemplateGeometry(/** @type {any} */ (
      templateGeometry(template, chainMaskFor(), length, AF2_ATOM37)), length);
  // A masked template's residue type is zero - and a one-hot of zero is still a
  // one-hot, which is why the aatype terms survive the mask.
  const aatype = template === undefined
    ? new Float32Array(length)
    : Float32Array.from(template.aatype);
  return { geometry, aatype };
}

export const templateInputSource = (shape) => {
  const { length, pairChannels, templateChannels } = shape;
  return `${PRELUDE}${ONE_TARGET}
const int L = ${length};
const int PAIR_C = ${pairChannels};
const int CT4 = ${templateChannels / 4};
const int STRIDE = ${GEOMETRY_STRIDE};
const int SCALARS = ${GEOMETRY_FEATURES.length - 1};
const int ELEMENTS = ${length * length * templateChannels};
uniform sampler2D normalisedTex;
uniform sampler2D constantTex;
uniform sampler2D queryWeightTex;
uniform sampler2D queryBiasTex;
uniform sampler2D aatypeRowTex;
uniform sampler2D aatypeColumnTex;
uniform sampler2D geometryTex;
uniform sampler2D dgramWeightTex;
uniform sampler2D scalarWeightTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % CT4;
  int entry = texel / CT4;
  int i = entry / L;
  int j = entry - i * L;

  // Embedding 8, the query pair, over the normalised pair track.
  vec4 total = fetch4(constantTex, group) + fetch4(queryBiasTex, group);
  for (int c = 0; c < PAIR_C; ++c) {
    total += fetch1(normalisedTex, entry * PAIR_C + c)
      * fetch4(queryWeightTex, c * CT4 + group);
  }

  // Embeddings 2 and 3: the template's aatype along j, then along i.
  total += fetch4(aatypeRowTex, j * CT4 + group);
  total += fetch4(aatypeColumnTex, i * CT4 + group);

  // Embedding 0, the distogram: the bin is stored PLUS ONE, so zero means no
  // bin at all - which is what a masked template and a pair closer than the
  // first edge both have.
  int g = entry * STRIDE;
  int bin = int(fetch1(geometryTex, g));
  if (bin > 0) total += fetch4(dgramWeightTex, (bin - 1) * CT4 + group);

  // Embeddings 1, 4, 5, 6 and 7: the five scalar features, each scaled by its
  // own value rather than indexed.
  for (int f = 0; f < SCALARS; ++f) {
    total += fetch1(geometryTex, g + 1 + f) * fetch4(scalarWeightTex, f * CT4 + group);
  }
  result = total;
}`;
};

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 */
export function uploadMultimerTemplateWeights(execution, weights) {
  /** @type {any} */
  const out = {};
  for (const name of TEMPLATE_WEIGHTS) {
    const values = weights[name];
    if (values === undefined) throw new Error(`multimer template weights: ${name} is missing`);
    out[name] = execution.upload(values, `w.mt.${name}`);
  }
  const query = weights.pairEmbeddings[QUERY_FEATURE];
  out.queryWeight = execution.upload(query.weight, "w.mt.queryWeight");
  out.queryBias = execution.upload(query.bias, "w.mt.queryBias");
  out.dgramWeight = execution.upload(
    weights.pairEmbeddings[GEOMETRY_FEATURES[0]].weight, "w.mt.dgramWeight");
  out.scalarWeight = execution.upload(scalarGeometryWeights(weights), "w.mt.scalarWeight");
  out.constant = execution.upload(templateConstantTerm(weights), "w.mt.constant");
  return out;
}

export function releaseMultimerTemplateWeights(execution, weights) {
  for (const name of [...TEMPLATE_WEIGHTS, "queryWeight", "queryBias", "dgramWeight",
    "scalarWeight", "constant"]) {
    execution.release(weights[name]);
  }
}

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{pair: Tensor, geometry: Tensor, aatypeRow: Tensor, aatypeColumn: Tensor,
 *   pairMask: Tensor, weights: any, blockWeights: any[],
 *   length: number, pairChannels: number, templateChannels: number,
 *   triangleHidden: number, pairHidden: number, templates?: number,
 *   epsilon?: number, templateOrder?: "monomer" | "multimer",
 *   transitionPrecision?: "f16" | "f32", trianglePrecision?: "f16" | "f32",
 *   triangleAttentionPrecision?: "f16" | "f32"}} call
 * @returns {Tensor} the pair UPDATE, `length^2 * pairChannels` floats; the
 *   caller adds it, as AF2's own embedding wrapper does
 */
export function multimerTemplateEmbedding(execution, call) {
  const {
    pair, geometry, aatypeRow, aatypeColumn, pairMask, weights: w, blockWeights,
    length, pairChannels, templateChannels, triangleHidden, pairHidden,
    templates = 1, epsilon = 1e-5,
  } = call;
  const pairs = length * length;
  if (geometry.elements !== pairs * GEOMETRY_STRIDE) {
    throw new RangeError(
      `multimerTemplateEmbedding: geometry holds ${geometry.elements} floats,`
      + ` not ${pairs} x ${GEOMETRY_STRIDE}`);
  }
  if (!(templates >= 1)) {
    throw new RangeError(
      `multimerTemplateEmbedding: templates is the DIVISOR of the mean and must be at least`
      + ` one, got ${templates}`);
  }

  const normalised = layerNorm(execution, {
    x: pair, weight: w.queryNormScale, bias: w.queryNormOffset,
    rows: pairs, channels: pairChannels, epsilon,
  });
  let act = execution.allocate(pairs * templateChannels, "mt.act");
  execution.dispatch({
    source: templateInputSource({ length, pairChannels, templateChannels }),
    output: act,
    inputs: {
      normalisedTex: normalised, constantTex: w.constant,
      queryWeightTex: w.queryWeight, queryBiasTex: w.queryBias,
      aatypeRowTex: aatypeRow, aatypeColumnTex: aatypeColumn,
      geometryTex: geometry, dgramWeightTex: w.dgramWeight,
      scalarWeightTex: w.scalarWeight,
    },
    label: `mt.input:${length}x${templateChannels}`,
  });
  execution.release(normalised);

  for (const block of blockWeights) {
    const next = templatePairBlock(execution, {
      pair: act, pairMask, length, cZ: templateChannels, triangleHidden, pairHidden,
      weights: block, epsilon,
      // ...the multiplications first, which is this stack's order and not the
      // monomer template stack's.
      templateOrder: call.templateOrder ?? "multimer",
      // 🔴 AND THE MULTIMER GRAPH'S PRECISIONS, WHICH ARE NOT THE MONOMER'S.
      // These blocks run through `webgpu/af2/multimer/block.js`, whose attention
      // targets are all f32 and whose transitions keep their hidden row in f32 -
      // where the monomer's are f16 at both seams with no switch. Defaulted here
      // rather than left to the caller, because which graph this stack belongs
      // to is not the caller's question.
      transitionPrecision: call.transitionPrecision ?? "f32",
      trianglePrecision: call.trianglePrecision ?? "f32",
      triangleAttentionPrecision: call.triangleAttentionPrecision ?? "f32",
    });
    execution.release(act);
    act = next;
  }

  const outputNorm = layerNorm(execution, {
    x: act, weight: w.outputNormScale, bias: w.outputNormOffset,
    rows: pairs, channels: templateChannels, epsilon,
  });
  execution.release(act);
  // 🔴 THE MEAN OVER TEMPLATES IS A DIVISION BY THE SLOT COUNT, AND IT IS
  // INSIDE THE RELU. `max(normalized / templates, 0)` - so the divisor is the
  // padded slot count rather than the number of real templates, which is the
  // same trap `templateFeatureMeanOnePass` records for rosettafold3: a stage
  // exact at one slot and a quarter strength at four.
  const averaged = scale(execution, outputNorm, 1 / templates);
  execution.release(outputNorm);
  const activated = relu(execution, averaged);
  execution.release(averaged);
  const update = linear(execution, {
    x: activated, weight: w.outputWeight, bias: w.outputBias, rows: pairs,
    inChannels: templateChannels, outChannels: pairChannels, weightLayout: "in-major",
  });
  execution.release(activated);
  update.label = "mt.update";
  return update;
}
