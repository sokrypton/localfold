/**
 * The pair bias: `bias[h][i][j] = sum_c LayerNorm(pair[i][j])[c] * W[c][h]`.
 *
 * This is how the pair track reaches an attention that runs over something
 * else - AF2's MSA row attention attends over the MSA and takes its logit bias
 * from the pair, which is the only place the two tracks meet before the outer
 * product mean sends information the other way. The triangle attentions take
 * their bias the same way from their own normalised pair.
 *
 * 🔴 THE OUTPUT IS HEAD-MAJOR, WHICH IS NOT THE PROJECTION'S NATURAL ORDER.
 * The WGSL kernel stores `output[h * pairs + pair_index]` - every head's plane
 * contiguous - and that is exactly what the flash core indexes as
 * `(head * Q + q) * Q + k`. A projection, though, naturally writes
 * `[pair][head]`, because that is the order in which one pair row's dot
 * products come out. So this is three draws: normalise, project pair-major,
 * transpose to head-major. Writing head-major directly would make a fragment's
 * four outputs four UNRELATED pairs, each needing its own pass over
 * `pairChannels` - four row reads where the pair-major form shares one.
 *
 * 🔴 AND THE WEIGHT IS `[c][heads]` WITH NO BIAS TERM: `weights[p.pair_weight +
 * c * HEADS + h]`, read off the WGSL shader. Three of the four operations
 * ported here take `[in][out]`; the triangle takes `[out][in]`; none of them
 * can be inferred from a shape.
 *
 * 🔴 HEADS MUST BE A MULTIPLE OF FOUR, AND EVERY SHAPE THIS PORT RUNS IS. A
 * fragment owns four consecutive output channels, so a head count that is not a
 * multiple of four would put one fragment's four outputs across two pairs with
 * two different rows. AF2 runs 8 heads on its MSA row attention and 4 on its
 * triangle attentions; AF3's grid attention runs 4, 8 and 16. Refused by name
 * rather than padded, because a padded head is a head whose bias is garbage and
 * which the attention would then read.
 */
import { ONE_TARGET, PRELUDE, sized } from "../runtime/execution.js";
import { layerNorm } from "./layer-norm.js";
import { linear } from "./linear.js";

/** @typedef {import("../runtime/storage.js").Tensor} Tensor */

export const PAIR_BIAS_WEIGHTS = ["layerNormScale", "layerNormOffset", "projectionWeight"];

/**
 * `out[h * pairs + pair] = source[pair * heads + h]`
 *
 * Per component rather than per texel: four consecutive OUTPUT elements are
 * four consecutive pairs of one head, and `pairs` is `length^2`, which is odd
 * whenever the chain length is - so a fragment's four can straddle two heads
 * and each one has to resolve its own (head, pair).
 */
export const toHeadMajorSource = (pairs, heads) => sized({ PAIRS: pairs, ELEMENTS: pairs * heads }, `${PRELUDE}${ONE_TARGET}
const int HEADS = ${heads};
uniform sampler2D sourceTex;

float at(int element) {
  int head = element / PAIRS;
  int pair = element - head * PAIRS;
  return fetch1(sourceTex, pair * HEADS + head);
}

void main() {
  int texel = linearTexel();
  int base = texel * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  result = vec4(
    at(base),
    base + 1 < ELEMENTS ? at(base + 1) : 0.0,
    base + 2 < ELEMENTS ? at(base + 2) : 0.0,
    base + 3 < ELEMENTS ? at(base + 3) : 0.0);
}`);

/**
 * @param {import("../runtime/execution.js").Execution} execution
 */
export function uploadPairBiasWeights(execution, weights) {
  /** @type {Record<string, Tensor>} */
  const out = {};
  for (const name of PAIR_BIAS_WEIGHTS) {
    const values = weights[name];
    if (values === undefined) throw new Error(`pair bias weights: ${name} is missing`);
    out[name] = execution.uploadWeight(values, `w.pairBias.${name}`);
  }
  return out;
}

export function releasePairBiasWeights(execution, weights) {
  for (const name of PAIR_BIAS_WEIGHTS) execution.release(weights[name]);
}

/**
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {{pair: Tensor, weights: Record<string, Tensor>, length: number,
 *   pairChannels: number, heads: number, epsilon?: number}} call
 * @returns {Tensor} `heads x length x length`, head-major; the caller owns it
 */
export function pairBias(execution, call) {
  const { pair, weights: w, length, pairChannels, heads, epsilon = 1e-5 } = call;
  const pairs = length * length;
  if (pair.elements !== pairs * pairChannels) {
    throw new RangeError(
      `pairBias: pair holds ${pair.elements} floats, not ${length}^2 x ${pairChannels}`);
  }
  if (heads % 4 !== 0) {
    throw new RangeError(
      `pairBias: heads must be a multiple of 4, got ${heads}; every shape this port `
      + "runs is (AF2 8 and 4, AF3 4, 8 and 16)");
  }

  const normalised = layerNorm(execution, {
    x: pair, weight: w.layerNormScale, bias: w.layerNormOffset,
    rows: pairs, channels: pairChannels, epsilon,
  });
  const pairMajor = linear(execution, {
    x: normalised, weight: w.projectionWeight,
    rows: pairs, inChannels: pairChannels, outChannels: heads,
    weightLayout: "in-major",
  });
  execution.release(normalised);

  const output = execution.allocate(pairs * heads, "pairBias");
  execution.dispatch({
    source: toHeadMajorSource(pairs, heads),
    output,
    inputs: { sourceTex: pairMajor },
    label: `pair-bias.head-major:${heads}x${pairs}`,
  });
  execution.release(pairMajor);
  return output;
}
