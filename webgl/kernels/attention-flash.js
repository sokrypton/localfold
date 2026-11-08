/**
 * Gated attention with an optional pair bias, for any head width - the core the
 * MSA and pair stacks of every model here are built on.
 *
 * 🔴 THE RENDER-TARGET BUDGET IS WHAT MAKES THIS AFFORDABLE, AND GENERALISING
 * OFF 32 CHANNELS IS WHY IT HAD TO BE REWRITTEN. A fragment writes one `vec4`,
 * so a head of `D` channels needs `D/4` of them; the obvious port gives each
 * fragment one group and recomputes the SAME q.k dot product `D/4` times over -
 * 4.5x the arithmetic at D = 32. With `MAX_DRAW_BUFFERS` attachments (eight
 * here, four guaranteed) one invocation owns up to that many groups and
 * computes each dot product exactly once. The first version of this kernel hard
 * coded `D = 32` and `8` targets, which is this machine's budget and AF2's
 * commonest head, and neither is a constant: AF3's grid attention runs heads of
 * 16 and of 8, and the specification floor is four targets. So the geometry is
 * derived - `targets = min(D/4, MAX_DRAW_BUFFERS)` - and a head too wide for one
 * draw is drawn in several, each recomputing the dot products for its own
 * groups. That redundancy is reported by `attentionFlashPlan` rather than
 * hidden, because it is the one case where this kernel is not the 1.55x it is
 * quoted at.
 *
 * 🔴 THE OUTPUT IS PLANES, AND THE INTERLEAVE IS A SEPARATE DRAW. Attachment
 * `t` holds channels `4t..4t+3` for every (batch, query, head), because separate
 * attachments are separate TEXTURES and no MRT arrangement can write one
 * interleaved tensor. A pipeline would have the pass downstream read the planes
 * directly; `interleavePlanes` exists so that a consumer which wants the
 * ordinary `[batch][query][head][channel]` tensor - a differential, or the
 * output projection - can have one for the price of one fetch per texel.
 *
 * 🔴 AND THE SCALE IS EXPLICIT, BECAUSE THE FIRST VERSION LEFT IT TO THE CALLER
 * AND SAID SO NOWHERE. AlphaFold scales the logits by `inverseSqrt(head_dim)`
 * (`let scale = inverseSqrt(f32(p.head_dim))` on the WGSL side); the original
 * kernel applied none, so it was correct only for a caller that had pre-scaled
 * its queries, and nothing stated that. `scale` defaults to the model's value
 * and a benchmark reproducing the old numbers passes 1.
 *
 * The masking convention is the WGSL kernel's: `logit += 1e9 * (mask - 1)`, so
 * a mask of 1 keeps a key and 0 withholds it, and `transpose` switches the mask
 * between `[batch][key]` and `[key][batch]`.
 */
import { PRELUDE, targetDeclarations, sized, memoSource } from "../runtime/execution.js";
/** @typedef {import("../runtime/storage.js").Tensor} Tensor */


/**
 * How this head width will be drawn on this context.
 *
 * `redundancy` is how many times each q.k dot product is computed: 1 when the
 * whole head fits one draw, and `draws` when it does not.
 */
export function attentionFlashPlan(headDim, maxDrawBuffers) {
  if (headDim % 4 !== 0) {
    throw new RangeError(`attention: head width must be a multiple of 4, got ${headDim}`);
  }
  const groups = headDim / 4;
  const targets = Math.min(groups, maxDrawBuffers);
  const draws = Math.ceil(groups / targets);
  return { groups, targets, draws, redundancy: draws };
}

const attentionFlashSourceUncached = (
  { batch, queries, heads, headDim, targets, hasPairBias, roundHalf = false },
) => {
  const groups = headDim / 4;
  const register = (name, body) => Array.from({ length: groups },
    (_, g) => `  vec4 ${name}${g} = ${body(g)};`).join("\n");
  const perTarget = (body) => Array.from({ length: targets },
    (_, t) => `    ${body(t)}`).join("\n");
  const dot = Array.from({ length: groups },
    (_, g) => `    score += dot(q${g}, fetch4(keyTex, keyBase + ${g}));`).join("\n");

  return sized({ BATCH: batch, Q: queries }, `${PRELUDE}
${targetDeclarations(targets)}
const int HEADS = ${heads};
const int HD4 = ${groups};
uniform sampler2D queryTex;
uniform sampler2D keyTex;
uniform sampler2D valueTex;
uniform sampler2D gateTex;
uniform sampler2D maskTex;
${hasPairBias ? "uniform sampler2D pairTex;" : ""}
uniform int transposeMask;
uniform int groupBase;
uniform float scale;

void main() {
  // One fragment per (batch, query, head) - which is exactly one texel of an
  // output plane, since a plane holds four channels of each.
  int bqh = linearTexel();
  if (bqh >= BATCH * Q * HEADS) {
${perTarget((t) => `result${t} = vec4(0.0);`)}
    return;
  }
  int head = bqh % HEADS;
  int bq = bqh / HEADS;
  int q = bq % Q;
  int b = bq / Q;
  int queryBase = bqh * HD4;

${register("q", (g) => `fetch4(queryTex, queryBase + ${g})`)}
${perTarget((t) => `vec4 acc${t} = vec4(0.0);`)}

  float runningMax = -1e30;
  float runningSum = 0.0;

  for (int k = 0; k < Q; ++k) {
    int keyBase = ((b * Q + k) * HEADS + head) * HD4;
    float score = 0.0;
${dot}
    score *= scale;

    int maskIndex = transposeMask == 0 ? b * Q + k : k * BATCH + b;
    float logit = score + 1e9 * (fetch1(maskTex, maskIndex) - 1.0);
${hasPairBias ? "    logit += fetch1(pairTex, (head * Q + q) * Q + k);" : ""}
    logit = clamp(logit, -1e8, 1e8);

    // ...the online softmax, kept in registers. The WGSL kernel spends a
    // 32-lane tree reduction and three barriers per key to publish this state;
    // a fragment has no one to synchronise with, which is the whole finding the
    // original spike produced - the 11.4x.
    float newMax = max(runningMax, logit);
    float previousScale = exp(runningMax - newMax);
    float weight = exp(logit - newMax);
    runningSum = runningSum * previousScale + weight;
    runningMax = newMax;
${perTarget((t) => `acc${t} = acc${t} * previousScale + weight * fetch4(valueTex, keyBase + groupBase + ${t});`)}
  }

${perTarget((t) => {
    const value = `(acc${t} / runningSum) * fetch4(gateTex, queryBase + groupBase + ${t})`;
    // The shipped AF2 attention asks its flash kernel for an f16 output; see
    // half4 in webgl/runtime/execution.js.
    return `result${t} = ${roundHalf ? `half4(${value})` : value};`;
  })}
}`);
};

/**
 * Gather `headDim / 4` planes into one `[batch][query][head][channel]` tensor.
 *
 * 🔴 THE SAMPLER IS CHOSEN BY AN UNROLLED SELECT, NOT BY INDEXING AN ARRAY.
 * GLSL ES 3.00 forbids indexing a sampler array by anything but a constant
 * expression, so `planes[t]` with a computed `t` does not compile - and a
 * driver that accepted it would be compiling something else.
 */
const interleavePlanesSourceUncached = (elements, headDim) => {
  const groups = headDim / 4;
  const declarations = Array.from({ length: groups },
    (_, g) => `uniform sampler2D plane${g};`).join("\n");
  const select = Array.from({ length: groups - 1 },
    (_, g) => `  if (group == ${g}) { result = fetch4(plane${g}, bqh); return; }`).join("\n");
  return sized({ ELEMENTS: elements }, `${PRELUDE}
out vec4 result;
const int HD4 = ${groups};
${declarations}

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % HD4;
  int bqh = texel / HD4;
${select}
  result = fetch4(plane${groups - 1}, bqh);
}`);
};

/**
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {{query: Tensor, key: Tensor, value: Tensor, gate: Tensor, mask: Tensor,
 *   pairBias?: Tensor, batch: number, queries: number, heads: number, headDim: number,
 *   transpose?: boolean, scale?: number, roundHalf?: boolean}} call
 * @returns {{planes: Tensor[], plan: {groups: number, targets: number, draws: number,
 *   redundancy: number}}} the caller owns the planes
 */
export function attentionFlashPlanes(execution, call) {
  const {
    query, key, value, gate, mask, pairBias,
    batch, queries, heads, headDim, transpose = false,
  } = call;
  const scale = call.scale ?? 1 / Math.sqrt(headDim);
  const roundHalf = call.roundHalf === true;
  const plan = attentionFlashPlan(headDim, execution.maxDrawBuffers);
  const elements = batch * queries * heads * headDim;
  /** @type {[string, Tensor][]} */
  const operands = [["query", query], ["key", key], ["value", value], ["gate", gate]];
  for (const [name, tensor] of operands) {
    if (tensor.elements !== elements) {
      throw new RangeError(
        `attention: ${name} holds ${tensor.elements} floats, not `
        + `${batch} x ${queries} x ${heads} x ${headDim}`);
    }
  }
  if (mask.elements < batch * queries) {
    throw new RangeError(`attention: mask holds ${mask.elements} floats, not ${batch * queries}`);
  }
  if (pairBias !== undefined && pairBias.elements !== heads * queries * queries) {
    throw new RangeError(
      `attention: pairBias holds ${pairBias.elements} floats, not `
      + `${heads} x ${queries} x ${queries}`);
  }

  const planeElements = batch * queries * heads * 4;
  const planes = Array.from({ length: plan.groups },
    (_, g) => execution.allocate(planeElements, `attention.plane${g}`));

  const inputs = { queryTex: query, keyTex: key, valueTex: value, gateTex: gate, maskTex: mask };
  if (pairBias !== undefined) inputs.pairTex = pairBias;

  for (let base = 0; base < plan.groups; base += plan.targets) {
    const span = Math.min(plan.targets, plan.groups - base);
    execution.dispatch({
      source: attentionFlashSource({
        batch, queries, heads, headDim, targets: span,
        hasPairBias: pairBias !== undefined, roundHalf,
      }),
      outputs: planes.slice(base, base + span),
      inputs,
      ints: { transposeMask: transpose ? 1 : 0, groupBase: base },
      floats: { scale },
      label: `attention.flash:${batch}x${queries}x${heads}x${headDim}`,
    });
  }
  return { planes, plan };
}

/**
 * The same, gathered into one `[batch][query][head][channel]` tensor.
 *
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {Parameters<typeof attentionFlashPlanes>[1]} call
 * @param {Tensor} [into]
 */
export function attentionFlash(execution, call, into) {
  const { batch, queries, heads, headDim } = call;
  const { planes, plan } = attentionFlashPlanes(execution, call);
  const elements = batch * queries * heads * headDim;
  const output = into ?? execution.allocate(elements, "attention.output");
  const inputs = {};
  planes.forEach((plane, g) => { inputs[`plane${g}`] = plane; });
  execution.dispatch({
    source: interleavePlanesSource(elements, headDim),
    output,
    inputs,
    label: `attention.interleave:${headDim}`,
  });
  for (const plane of planes) execution.release(plane);
  return { output, plan };
}

// Every builder above, memoised on its arguments (`memoSource` in webgl/runtime/execution.js).
export const attentionFlashSource = memoSource(attentionFlashSourceUncached);
export const interleavePlanesSource = memoSource(interleavePlanesSourceUncached);
