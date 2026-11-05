/**
 * AF3's 24-block token transformer: the 198 M parameters that are 98% of the
 * diffusion head.
 *
 *     per block:  act += conditionedSelfAttention(act, cond, pairLogits)
 *                 act += conditionedTransition(act, cond)
 *
 * 🔴 THE BLOCKS ARE NESTED SIX BY FOUR, AND THE PAIR LOGITS FOLLOW THAT
 * NESTING. "The LayerNorm over the pair conditioning is computed ONCE and
 * shared, but each of the six SUPER-BLOCKS then projects it to its own four
 * blocks' worth of head biases. So there are six projections, not one and not
 * twenty-four, and a flat reading of the stack indexes the wrong weights for
 * every block after the fourth."
 *
 * 🔴 THE SCALE IS THE PER-HEAD DIMENSION, TAKEN AFTER THE DIVISION BY THE HEAD
 * COUNT. "AF3 writes `key_dim = key_dim // num_head` and only then
 * `key_dim ** -0.5`. Using the full 768 instead of 48 is a factor of four on
 * every logit, which softmax turns into a much flatter attention."
 *
 * 🔴 AND A MISSING HEAD COUNT IS NOT AN ERROR IN JAVASCRIPT, IT IS A ZERO. The
 * reference records what that cost: it "read `weights.heads` while the head
 * count lives on the transformer config rather than the per-block weights, so
 * the loop below ran zero times and the whole attention branch returned an array
 * of zeros - no exception, no NaN, just a residual stream missing half its
 * updates. It cost a bisection down to the block to find, past four stages that
 * were all exact." So the shape is validated here rather than trusted.
 *
 * 🔴 AND THE CONDITIONING IS WIDER THAN THE ACTIVATION HERE - 384 against 768 -
 * which is the case `crossAttentionBlock` cannot be asked about and this one is
 * built for.
 *
 * The attention itself is `kernels/attention-flash.js` at `batch = 1`, which is
 * the same reuse AF3's single-track attention gets: that kernel's mask index
 * collapses to the key position and its bias layout is already `[head][i][j]`.
 */
import { ONE_TARGET, PRELUDE } from "../../runtime/execution.js";
import { adaptiveLayerNorm, adaptiveZeroInit, elementwiseProductSource, sumSource }
  from "./atom-attention.js";
import { attentionFlash } from "../../kernels/attention-flash.js";
import { layerNorm } from "../../kernels/layer-norm.js";
import { linear } from "../../kernels/linear.js";

/** @typedef {import("../../runtime/storage.js").Tensor} Tensor */

/** `swish(wide[i]) * wide[intermediate + i]` over a blocked row. */
export const swiGluSource = (rows, intermediate) => `${PRELUDE}${ONE_TARGET}
const int INTER4 = ${intermediate / 4};
const int ELEMENTS = ${rows * intermediate};
uniform sampler2D wideTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % INTER4;
  int row = texel / INTER4;
  int base = row * 2 * INTER4;
  vec4 gate = fetch4(wideTex, base + group);
  vec4 value = fetch4(wideTex, base + INTER4 + group);
  result = (gate / (vec4(1.0) + exp(-gate))) * value;
}`;

/**
 * `logits[head][i][j] = flat[(i * tokens + j) * perSuper * heads + inner * heads + head]`
 *
 * The nesting made explicit: a super-block's projection carries `perSuper`
 * blocks' biases interleaved by head, and this pulls out one block's.
 */
export const superBlockLogitsSource = (tokens, heads, perSuper, inner) => `${PRELUDE}${ONE_TARGET}
const int L = ${tokens};
const int HEADS = ${heads};
const int PER_SUPER = ${perSuper};
const int INNER = ${inner};
const int ELEMENTS = ${heads * tokens * tokens};
uniform sampler2D flatTex;

float at(int element) {
  int j = element % L;
  int rest = element / L;
  int i = rest % L;
  int head = rest / L;
  int source = (i * L + j) * PER_SUPER * HEADS + INNER * HEADS + head;
  return fetch1(flatTex, source);
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
 * A transition block with AdaLN conditioning, or none.
 *
 * 🔴 WITH `cond` ABSENT IT DEGRADES TO THE PLAIN FORM, which is what the
 * diffusion conditioning's own transitions use - and boltz2's extra up-gate is
 * gated on `cond` AS WELL AS on the weight, because "boltz2's plain Transition -
 * the one the diffusion conditioning's four transitions use - has NO up-gate,
 * and only its adaLN-conditioned ConditionedTransitionBlock does".
 *
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{x: Tensor, cond?: Tensor, rows: number, channels: number, factor: number,
 *   weights: any, prefix: string, condChannels?: number, epsilon?: number}} call
 * @returns {Tensor}
 */
export function conditionedTransition(execution, call) {
  const { x, cond, rows, channels, factor, weights: w, prefix, epsilon = 1e-5 } = call;
  const condChannels = call.condChannels ?? channels;
  const intermediate = channels * factor;
  // 🔴 THE KEY NAMES FOLLOW `adaptiveLayerNorm`'S, "which prefixes with 'ffw'
  // and then capitalises the leaf - so a conditioned transition reads
  // `ffwTransition1`, not `FfwTransition1`. Getting it wrong hands `linear` an
  // undefined weight, which at least throws; the danger is 'fixing' it by
  // renaming here and silently splitting the convention in two."
  const normalised = cond === undefined
    ? layerNorm(execution, {
      x, weight: w[`${prefix}ffwLayerNormScale`], bias: w[`${prefix}ffwLayerNormOffset`],
      rows, channels, epsilon,
    })
    : adaptiveLayerNorm(execution, {
      x, cond, rows, channels, condChannels, weights: w, prefix: `${prefix}ffw`, epsilon,
    });
  const wide = linear(execution, {
    x: normalised, weight: w[`${prefix}ffwTransition1`], rows,
    inChannels: channels, outChannels: intermediate * 2, weightLayout: "in-major",
  });
  let gated = execution.allocate(rows * intermediate, "af3.difftx.swiglu");
  execution.dispatch({
    source: swiGluSource(rows, intermediate),
    output: gated,
    inputs: { wideTex: wide },
    label: `af3.difftx.swiglu:${rows}x${intermediate}`,
  });
  execution.release(wide);
  if (cond !== undefined && w[`${prefix}ffwAToB`] !== undefined
      && w[`${prefix}ffwAToB`] !== null) {
    const upGate = linear(execution, {
      x: normalised, weight: w[`${prefix}ffwAToB`], rows,
      inChannels: channels, outChannels: intermediate, weightLayout: "in-major",
    });
    const multiplied = execution.allocate(rows * intermediate, "af3.difftx.up-gated");
    execution.dispatch({
      source: elementwiseProductSource(rows * intermediate),
      output: multiplied,
      inputs: { aTex: gated, bTex: upGate },
      label: `af3.difftx.up-gate:${rows}x${intermediate}`,
    });
    execution.release(gated);
    execution.release(upGate);
    gated = multiplied;
  }
  if (cond === undefined) {
    execution.release(normalised);
    const out = linear(execution, {
      x: gated, weight: w[`${prefix}ffwTransition2`], rows,
      inChannels: intermediate, outChannels: channels, weightLayout: "in-major",
    });
    execution.release(gated);
    return out;
  }
  execution.release(normalised);
  const out = adaptiveZeroInit(execution, {
    x: gated, cond, rows, channels, condChannels, inChannels: intermediate,
    weights: w, prefix: `${prefix}ffw`,
  });
  execution.release(gated);
  return out;
}

/**
 * Self-attention over tokens, conditioned by AdaLN and biased by the pair.
 *
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{act: Tensor, cond: Tensor, pairLogits: Tensor, mask: Tensor,
 *   tokens: number, channels: number, condChannels: number, heads: number,
 *   dimension: number, weights: any, epsilon?: number}} call
 * @returns {Tensor}
 */
export function conditionedSelfAttention(execution, call) {
  const {
    act, cond, pairLogits, mask, tokens, channels, condChannels, heads, dimension,
    weights: w, epsilon = 1e-5,
  } = call;
  if (!Number.isInteger(heads) || !Number.isInteger(dimension)) {
    throw new Error(
      `af3 diffusion attention needs heads and dimension, got ${heads} and ${dimension}`
      + " - a missing head count is not an error in JavaScript, it is a zero, and the"
      + " whole attention branch then returns zeros with no exception and no NaN");
  }
  const width = heads * dimension;
  const x = adaptiveLayerNorm(execution, {
    x: act, cond, rows: tokens, channels, condChannels, weights: w, prefix: "", epsilon,
  });
  const project = (weight, bias, activation) => linear(execution, {
    x, weight, rows: tokens, inChannels: channels, outChannels: width,
    weightLayout: "in-major",
    ...(bias === undefined ? {} : { bias }),
    ...(activation === undefined ? {} : { activation }),
  });
  const query = project(w.qProjection, w.qBias);
  const key = project(w.kProjection);
  const value = project(w.vProjection);
  const gate = project(w.gatingQuery, undefined, "sigmoid");
  // ...the scale is the PER-HEAD dimension, which `attentionFlash` defaults to.
  const { output: gathered } = attentionFlash(execution, {
    query, key, value, gate, mask, pairBias: pairLogits,
    batch: 1, queries: tokens, heads, headDim: dimension,
  });
  for (const tensor of [query, key, value, gate]) execution.release(tensor);
  execution.release(x);
  const out = adaptiveZeroInit(execution, {
    x: gathered, cond, rows: tokens, channels, condChannels, inChannels: width,
    weights: w, prefix: "",
  });
  execution.release(gathered);
  return out;
}

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{act: Tensor, cond: Tensor, pairCond: Tensor, mask: Tensor, tokens: number,
 *   weights: any, epsilon?: number,
 *   onBlock?: (superBlock: number, inner: number) => void}} call
 * @returns {Tensor} a new activation; the caller's `act` is untouched
 */
export function diffusionTransformer(execution, call) {
  const { act, cond, pairCond, mask, tokens, weights: w, epsilon = 1e-5 } = call;
  const channels = w.channels;
  const condChannels = w.condChannels;
  const heads = w.heads;
  const perSuper = w.blocksPerSuperBlock;
  const pairs = tokens * tokens;

  // One LayerNorm over the pair conditioning, shared by every super-block.
  const normalisedPair = layerNorm(execution, {
    x: pairCond, weight: w.pairInputLayerNormScale,
    rows: pairs, channels: w.pairChannels, epsilon,
  });

  let current = act;
  let own = false;
  for (const [superBlock, group] of w.superBlocks.entries()) {
    // ...and one projection a SUPER-BLOCK, carrying its four blocks' biases.
    const flat = linear(execution, {
      x: normalisedPair, weight: group.pairLogitsProjection, rows: pairs,
      inChannels: w.pairChannels, outChannels: perSuper * heads, weightLayout: "in-major",
    });
    for (let inner = 0; inner < perSuper; inner += 1) {
      const pairLogits = execution.allocate(heads * pairs, "af3.difftx.logits");
      execution.dispatch({
        source: superBlockLogitsSource(tokens, heads, perSuper, inner),
        output: pairLogits,
        inputs: { flatTex: flat },
        label: `af3.difftx.logits:${tokens}x${heads}:${inner}/${perSuper}`,
      });
      const block = group.blocks[inner];
      const attention = conditionedSelfAttention(execution, {
        act: current, cond, pairLogits, mask, tokens, channels, condChannels,
        heads, dimension: w.dimension, weights: block, epsilon,
      });
      execution.release(pairLogits);
      const afterAttention = execution.allocate(
        tokens * channels, "af3.difftx.after-attention");
      execution.dispatch({
        source: sumSource(tokens * channels),
        output: afterAttention,
        inputs: { aTex: current, bTex: attention },
        label: `af3.difftx.residual:${tokens}x${channels}`,
      });
      execution.release(attention);
      if (own) execution.release(current);
      const transitioned = conditionedTransition(execution, {
        x: afterAttention, cond, rows: tokens, channels,
        factor: w.transitionFactor, weights: block, prefix: "", condChannels, epsilon,
      });
      const next = execution.allocate(tokens * channels, "af3.difftx.block");
      execution.dispatch({
        source: sumSource(tokens * channels),
        output: next,
        inputs: { aTex: afterAttention, bTex: transitioned },
        label: `af3.difftx.residual2:${tokens}x${channels}`,
      });
      execution.release(afterAttention);
      execution.release(transitioned);
      current = next;
      own = true;
      call.onBlock?.(superBlock, inner);
    }
    execution.release(flat);
  }
  execution.release(normalisedPair);
  current.label = "af3.difftx";
  return current;
}
