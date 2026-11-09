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
  { batch, queries, heads, headDim, targets, hasPairBias, roundHalf = false, storedLogits = false },
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
${storedLogits ? "" : "uniform sampler2D queryTex;\nuniform sampler2D keyTex;\nuniform sampler2D maskTex;"}
uniform sampler2D valueTex;
uniform sampler2D gateTex;
${hasPairBias && !storedLogits ? "uniform sampler2D pairTex;" : ""}
${storedLogits ? "uniform sampler2D logitTex;" : "uniform int transposeMask;\nuniform float scale;"}
uniform int groupBase;

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

${storedLogits ? "" : register("q", (g) => `fetch4(queryTex, queryBase + ${g})`)}
${perTarget((t) => `vec4 acc${t} = vec4(0.0);`)}

  float runningMax = -1e30;
  float runningSum = 0.0;

  for (int k = 0; k < Q; ++k) {
    int keyBase = ((b * Q + k) * HEADS + head) * HD4;
${storedLogits ? "    float logit = fetch1(logitTex, bqh * Q + k);" : `    float score = 0.0;
${dot}
    score *= scale;

    int maskIndex = transposeMask == 0 ? b * Q + k : k * BATCH + b;
    float logit = score + 1e9 * (fetch1(maskTex, maskIndex) - 1.0);
${hasPairBias ? "    logit += fetch1(pairTex, (head * Q + q) * Q + k);" : ""}
    logit = clamp(logit, -1e8, 1e8);`}

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
 * Every logit the flash kernel's online softmax reads - `[batch][query][head][key]`, four keys a texel - computed ONCE,
 * in the same arithmetic in the same order (dot by group, the scale, the mask, the pair bias, the clamp), so that a head
 * too wide for one draw (`attentionFlashPlan`'s `draws > 1`) does not recompute them in every draw: its draws then read
 * the logit (`storedLogits`) and keep only the value accumulation, which is what each one is for.
 */
const attentionLogitsSourceUncached = ({ batch, queries, heads, headDim, hasPairBias }) => {
  const groups = headDim / 4;
  const dot = Array.from({ length: groups },
    (_, g) => `      score += dot(fetch4(queryTex, queryBase + ${g}), fetch4(keyTex, keyBase + ${g}));`).join("\n");
  return sized({ BATCH: batch, Q: queries, ELEMENTS: batch * queries * heads * queries }, `${PRELUDE}
out vec4 result;
const int HEADS = ${heads};
const int HD4 = ${groups};
uniform sampler2D queryTex;
uniform sampler2D keyTex;
uniform sampler2D maskTex;
${hasPairBias ? "uniform sampler2D pairTex;" : ""}
uniform int transposeMask;
uniform float scale;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  vec4 out4 = vec4(0.0);
  for (int lane = 0; lane < 4; ++lane) {
    int element = texel * 4 + lane;
    if (element >= ELEMENTS) break;
    int bqh = element / Q;
    int k = element - bqh * Q;
    int head = bqh % HEADS;
    int bq = bqh / HEADS;
    int q = bq % Q;
    int b = bq / Q;
    int queryBase = bqh * HD4;
    int keyBase = ((b * Q + k) * HEADS + head) * HD4;
    float score = 0.0;
${dot}
    score *= scale;
    int maskIndex = transposeMask == 0 ? b * Q + k : k * BATCH + b;
    float logit = score + 1e9 * (fetch1(maskTex, maskIndex) - 1.0);
${hasPairBias ? "    logit += fetch1(pairTex, (head * Q + q) * Q + k);" : ""}
    logit = clamp(logit, -1e8, 1e8);
    if (lane == 0) out4.x = logit;
    else if (lane == 1) out4.y = logit;
    else if (lane == 2) out4.z = logit;
    else out4.w = logit;
  }
  result = out4;
}`);
};

/**
 * The value half straight from stored logits (`attentionLogitsSource`), one fragment per OUTPUT texel - (batch, query,
 * head, channel group) - written in the interleaved `[batch][query][head][channel]` layout. The flash draws give one
 * fragment a whole head and spread its groups over render targets, which at the denoiser's 59 queries x 16 heads is
 * 944 fragments each walking every key: under-occupied, and a 48-wide head needs two draws and an interleave. Here
 * each fragment takes the row's max, then the exponentials' sum and the weighted values, for its own four channels.
 * The same softmax, taken in two sweeps rather than online.
 */
const attentionValuesSourceUncached = ({ batch, queries, heads, headDim, roundHalf = false }) => sized({ BATCH: batch, Q: queries, ELEMENTS: batch * queries * heads * headDim }, `${PRELUDE}
out vec4 result;
const int HEADS = ${heads};
const int HD4 = ${headDim / 4};
uniform sampler2D logitTex;
uniform sampler2D valueTex;
uniform sampler2D gateTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % HD4;
  int bqh = texel / HD4;
  int head = bqh % HEADS;
  int b = (bqh / HEADS) / Q;
  float rowMax = -1e30;
  for (int k = 0; k < Q; ++k) rowMax = max(rowMax, fetch1(logitTex, bqh * Q + k));
  float total = 0.0;
  vec4 acc = vec4(0.0);
  for (int k = 0; k < Q; ++k) {
    float weight = exp(fetch1(logitTex, bqh * Q + k) - rowMax);
    total += weight;
    acc += weight * fetch4(valueTex, ((b * Q + k) * HEADS + head) * HD4 + group);
  }
  vec4 value = (acc / total) * fetch4(gateTex, texel);
  result = ${roundHalf ? "half4(value)" : "value"};
}`);

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

  // ...a head that takes more than one draw has its logits computed once, before them (`attentionLogitsSource`)
  const storedLogits = plan.draws > 1 && !execution.noStoredLogits;
  let logits;
  if (storedLogits) {
    logits = execution.allocate(batch * queries * heads * queries, "attention.logits");
    execution.dispatch({
      source: attentionLogitsSource({ batch, queries, heads, headDim, hasPairBias: pairBias !== undefined }),
      output: logits,
      inputs: {
        queryTex: query, keyTex: key, maskTex: mask, ...(pairBias === undefined ? {} : { pairTex: pairBias }),
      },
      ints: { transposeMask: transpose ? 1 : 0 },
      floats: { scale },
      label: `attention.logits:${batch}x${queries}x${heads}x${headDim}`,
    });
  }
  // ...and then the value draws bind only what they still read
  const drawInputs = storedLogits ? { valueTex: value, gateTex: gate, logitTex: logits } : inputs;
  for (let base = 0; base < plan.groups; base += plan.targets) {
    const span = Math.min(plan.targets, plan.groups - base);
    execution.dispatch({
      source: attentionFlashSource({
        batch, queries, heads, headDim, targets: span,
        hasPairBias: pairBias !== undefined, roundHalf, storedLogits,
      }),
      outputs: planes.slice(base, base + span),
      inputs: drawInputs,
      ints: storedLogits ? { groupBase: base } : { transposeMask: transpose ? 1 : 0, groupBase: base },
      floats: storedLogits ? {} : { scale },
      label: `attention.flash:${batch}x${queries}x${heads}x${headDim}`,
    });
  }
  if (logits !== undefined) execution.release(logits);
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
  // ...a head too wide for one draw: its logits once, then its values one fragment an output texel (no planes)
  const plan0 = attentionFlashPlan(headDim, execution.maxDrawBuffers);
  if (plan0.draws > 1 && !execution.noStoredLogits && !execution.noDirectValues) {
    const { query, key, value, gate, mask, pairBias, transpose = false } = call;
    const scale = call.scale ?? 1 / Math.sqrt(headDim);
    const elements = batch * queries * heads * headDim;
    const logits = execution.allocate(batch * queries * heads * queries, "attention.logits");
    execution.dispatch({
      source: attentionLogitsSource({ batch, queries, heads, headDim, hasPairBias: pairBias !== undefined }),
      output: logits,
      inputs: {
        queryTex: query, keyTex: key, maskTex: mask, ...(pairBias === undefined ? {} : { pairTex: pairBias }),
      },
      ints: { transposeMask: transpose ? 1 : 0 },
      floats: { scale },
      label: `attention.logits:${batch}x${queries}x${heads}x${headDim}`,
    });
    const output = into ?? execution.allocate(elements, "attention.output");
    execution.dispatch({
      source: attentionValuesSource({ batch, queries, heads, headDim, roundHalf: call.roundHalf === true }),
      output,
      inputs: { logitTex: logits, valueTex: value, gateTex: gate },
      label: `attention.values:${batch}x${queries}x${heads}x${headDim}`,
    });
    execution.release(logits);
    return { output, plan: { ...plan0, draws: 1, redundancy: 1 } };
  }
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
export const attentionLogitsSource = memoSource(attentionLogitsSourceUncached);
export const attentionValuesSource = memoSource(attentionValuesSourceUncached);
export const interleavePlanesSource = memoSource(interleavePlanesSourceUncached);
