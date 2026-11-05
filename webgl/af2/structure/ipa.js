/**
 * Invariant point attention - the heart of AF2's structure module, and the one
 * attention in the model whose logits are a DISTANCE.
 *
 * Eight draws after the projections:
 *
 *     points      local (x, y, z) rotated into each residue's frame
 *     pair bias   bias[head][q][k] from the normalised pair
 *     logits      scalar dot + point distance + pair bias + mask
 *     attention   softmax over the key axis
 *     scalar      attention . kv scalar values
 *     point       attention . kv points, rotated BACK into the query's frame
 *     pair        attention . pair
 *     features    the three gathered into one row, then the output projection
 *
 * 🔴 THE POINT TERM IS SUBTRACTED, NOT ADDED, AND IS A SQUARED DISTANCE.
 * `result -= 0.5 * point_weight * distance` with
 * `point_weight = point_factor * log(1 + exp(trainable[head]))` - a softplus, so
 * the weight is positive however the parameter is trained, and the logit falls
 * as the points separate. An added term, or a distance rather than its square,
 * is a plausible attention that is not invariant.
 *
 * 🔴 AND THE THREE SCALE FACTORS ARE NOT INTERCHANGEABLE:
 * `sqrt(1 / (3 * scalarQk))` on the scalar dot, `sqrt(1 / (3 * pointQk * 4.5))`
 * on the point term and `sqrt(1 / 3)` on the pair bias. The 4.5 is the one that
 * cannot be guessed.
 *
 * 🔴 AND THE POINT FEATURE COMES BACK INTO THE QUERY'S FRAME, NOT THE KEY'S.
 * The values are summed in global space and then `inverse_rotate`d by the QUERY
 * residue's quaternion with its translation removed - which is what makes the
 * output invariant - and a FOURTH plane is appended holding the vector's norm,
 * guarded at 1e-8 so a zero vector gives 0 rather than NaN.
 *
 * 🔴 AND THE THREE FEATURE BLOCKS GET THEIR OWN TENSORS. They occupy disjoint
 * ranges of one row - scalar, then four point planes, then pair - and those
 * boundaries are not texel-aligned, so three kernels writing one tensor would
 * each have to own partial texels. They write separately and a gather assembles
 * the row, which costs one pass and no correctness.
 */
import { ONE_TARGET, PRELUDE } from "../../runtime/execution.js";
import { layerNorm } from "../../kernels/layer-norm.js";
import { linear } from "../../kernels/linear.js";

/** @typedef {import("../../runtime/storage.js").Tensor} Tensor */

export const IPA_WEIGHTS = [
  "pairNormScale", "pairNormOffset",
  "queryScalarWeight", "queryScalarBias", "keyValueScalarWeight", "keyValueScalarBias",
  "queryPointWeight", "queryPointBias", "keyValuePointWeight", "keyValuePointBias",
  "trainablePointWeights", "attention2dWeight", "attention2dBias",
  "outputWeight", "outputBias",
];

const QUATERNION = `
// The reference's own rotation, from an unnormalised quaternion (w, x, y, z).
vec3 rotateBy(vec4 q, vec3 v) {
  vec3 u = q.yzw;
  return 2.0 * dot(u, v) * u + (q.x * q.x - dot(u, u)) * v + 2.0 * q.x * cross(u, v);
}
vec3 inverseRotateBy(vec4 q, vec3 v) {
  vec3 u = q.yzw;
  return 2.0 * dot(u, v) * u + (q.x * q.x - dot(u, u)) * v - 2.0 * q.x * cross(u, v);
}
vec4 quaternionOf(sampler2D affineTex, int residue) {
  int base = residue * 7;
  return vec4(fetch1(affineTex, base), fetch1(affineTex, base + 1),
    fetch1(affineTex, base + 2), fetch1(affineTex, base + 3));
}
vec3 translationOf(sampler2D affineTex, int residue) {
  int base = residue * 7;
  return vec3(fetch1(affineTex, base + 4), fetch1(affineTex, base + 5),
    fetch1(affineTex, base + 6));
}
`;

/**
 * Local points into global space.
 *
 * The LOCAL layout is `[residue][component][head][point]` - the three
 * components are a plane apart, not adjacent - and the GLOBAL one it writes is
 * `[residue][head][point][xyz]`. Both are the reference's, and they differ.
 */
export const transformPointsSource = (length, heads, points) => `${PRELUDE}${ONE_TARGET}
const int L = ${length};
const int HEADS = ${heads};
const int POINTS = ${points};
const int PLANE = ${heads * points};
const int ELEMENTS = ${length * heads * points * 3};
uniform sampler2D localTex;
uniform sampler2D affineTex;
${QUATERNION}

float coordinateOf(int element) {
  int axis = element % 3;
  int slot = element / 3;
  int point = slot % POINTS;
  int head = (slot / POINTS) % HEADS;
  int residue = slot / (POINTS * HEADS);
  int base = residue * 3 * PLANE + head * POINTS + point;
  vec3 local = vec3(
    fetch1(localTex, base), fetch1(localTex, base + PLANE), fetch1(localTex, base + 2 * PLANE));
  vec3 global = rotateBy(quaternionOf(affineTex, residue), local)
    + translationOf(affineTex, residue);
  return axis == 0 ? global.x : axis == 1 ? global.y : global.z;
}

void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  result = vec4(
    coordinateOf(base),
    base + 1 < ELEMENTS ? coordinateOf(base + 1) : 0.0,
    base + 2 < ELEMENTS ? coordinateOf(base + 2) : 0.0,
    base + 3 < ELEMENTS ? coordinateOf(base + 3) : 0.0);
}`;

/** `bias[head][q][k] = attention2dBias[head] + sum_c pair[q][k][c] * W[c][head]` */
export const ipaPairBiasSource = (length, heads, pairChannels) => `${PRELUDE}${ONE_TARGET}
const int L = ${length};
const int HEADS = ${heads};
const int PAIR_C = ${pairChannels};
const int ELEMENTS = ${heads * length * length};
uniform sampler2D pairTex;
uniform sampler2D weightTex;
uniform sampler2D biasTex;

float biasOf(int index) {
  int key = index % L;
  int query = (index / L) % L;
  int head = index / (L * L);
  float total = fetch1(biasTex, head);
  int pairBase = (query * L + key) * PAIR_C;
  for (int c = 0; c < PAIR_C; ++c) {
    total += fetch1(pairTex, pairBase + c) * fetch1(weightTex, c * HEADS + head);
  }
  return total;
}

void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  result = vec4(
    biasOf(base),
    base + 1 < ELEMENTS ? biasOf(base + 1) : 0.0,
    base + 2 < ELEMENTS ? biasOf(base + 2) : 0.0,
    base + 3 < ELEMENTS ? biasOf(base + 3) : 0.0);
}`;

export const ipaLogitsSource = (shape) => {
  const { length, heads, scalarQk, scalarV, pointQk, pointV } = shape;
  return `${PRELUDE}${ONE_TARGET}
const int L = ${length};
const int HEADS = ${heads};
const int SCALAR_QK = ${scalarQk};
const int KV_SCALAR = ${scalarQk + scalarV};
const int POINT_QK = ${pointQk};
const int KV_POINTS = ${pointQk + pointV};
const int ELEMENTS = ${heads * length * length};
const float SCALAR_FACTOR = ${Math.sqrt(1 / (3 * scalarQk))};
const float POINT_FACTOR = ${Math.sqrt(1 / (3 * pointQk * 4.5))};
const float PAIR_FACTOR = ${Math.sqrt(1 / 3)};
uniform sampler2D queryScalarTex;
uniform sampler2D kvScalarTex;
uniform sampler2D queryPointTex;
uniform sampler2D kvPointTex;
uniform sampler2D pairBiasTex;
uniform sampler2D maskTex;
uniform sampler2D trainableTex;

float logitOf(int index) {
  int key = index % L;
  int query = (index / L) % L;
  int head = index / (L * L);

  float total = 0.0;
  int qBase = (query * HEADS + head) * SCALAR_QK;
  int kBase = (key * HEADS + head) * KV_SCALAR;
  for (int c = 0; c < SCALAR_QK; ++c) {
    total += SCALAR_FACTOR * fetch1(queryScalarTex, qBase + c) * fetch1(kvScalarTex, kBase + c);
  }

  float distance = 0.0;
  int qPoint = (query * HEADS + head) * POINT_QK * 3;
  int kPoint = (key * HEADS + head) * KV_POINTS * 3;
  for (int point = 0; point < POINT_QK; ++point) {
    for (int axis = 0; axis < 3; ++axis) {
      float delta = fetch1(queryPointTex, qPoint + point * 3 + axis)
        - fetch1(kvPointTex, kPoint + point * 3 + axis);
      distance += delta * delta;
    }
  }
  // A softplus, so the weight is positive however the parameter trained, and
  // the term is SUBTRACTED: the logit falls as the points separate.
  float pointWeight = POINT_FACTOR * log(1.0 + exp(fetch1(trainableTex, head)));
  total -= 0.5 * pointWeight * distance;
  total += PAIR_FACTOR * fetch1(pairBiasTex, index);
  total -= 1e5 * (1.0 - fetch1(maskTex, query) * fetch1(maskTex, key));
  return total;
}

void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  result = vec4(
    logitOf(base),
    base + 1 < ELEMENTS ? logitOf(base + 1) : 0.0,
    base + 2 < ELEMENTS ? logitOf(base + 2) : 0.0,
    base + 3 < ELEMENTS ? logitOf(base + 3) : 0.0);
}`;
};

/** Softmax over the key axis, into its own tensor - in place is not available. */
export const ipaSoftmaxSource = (length, heads) => `${PRELUDE}${ONE_TARGET}
const int L = ${length};
const int ELEMENTS = ${heads * length * length};
uniform sampler2D logitsTex;

float weightOf(int index) {
  int key = index % L;
  int base = index - key;
  float top = -1e30;
  for (int k = 0; k < L; ++k) top = max(top, fetch1(logitsTex, base + k));
  float total = 0.0;
  for (int k = 0; k < L; ++k) total += exp(fetch1(logitsTex, base + k) - top);
  return exp(fetch1(logitsTex, index) - top) / total;
}

void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  result = vec4(
    weightOf(base),
    base + 1 < ELEMENTS ? weightOf(base + 1) : 0.0,
    base + 2 < ELEMENTS ? weightOf(base + 2) : 0.0,
    base + 3 < ELEMENTS ? weightOf(base + 3) : 0.0);
}`;

/** `scalar[query][head][v] = sum_k attention . kv value channels` */
export const ipaScalarFeatureSource = (length, heads, scalarQk, scalarV) => `${PRELUDE}${ONE_TARGET}
const int L = ${length};
const int HEADS = ${heads};
const int SCALAR_V = ${scalarV};
const int SCALAR_QK = ${scalarQk};
const int KV_SCALAR = ${scalarQk + scalarV};
const int ELEMENTS = ${length * heads * scalarV};
uniform sampler2D attentionTex;
uniform sampler2D kvScalarTex;

float featureOf(int index) {
  int channel = index % SCALAR_V;
  int head = (index / SCALAR_V) % HEADS;
  int query = index / (SCALAR_V * HEADS);
  int attentionBase = (head * L + query) * L;
  float total = 0.0;
  for (int key = 0; key < L; ++key) {
    total += fetch1(attentionTex, attentionBase + key)
      * fetch1(kvScalarTex, (key * HEADS + head) * KV_SCALAR + SCALAR_QK + channel);
  }
  return total;
}

void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  result = vec4(
    featureOf(base),
    base + 1 < ELEMENTS ? featureOf(base + 1) : 0.0,
    base + 2 < ELEMENTS ? featureOf(base + 2) : 0.0,
    base + 3 < ELEMENTS ? featureOf(base + 3) : 0.0);
}`;

/**
 * The point feature: four planes of `heads * pointV` - x, y, z and the norm -
 * with the summed global vector rotated back into the QUERY's frame.
 */
export const ipaPointFeatureSource = (length, heads, pointQk, pointV) => `${PRELUDE}${ONE_TARGET}
const int L = ${length};
const int HEADS = ${heads};
const int POINT_V = ${pointV};
const int POINT_QK = ${pointQk};
const int KV_POINTS = ${pointQk + pointV};
const int PLANE = ${heads * pointV};
const int ELEMENTS = ${4 * heads * pointV * length};
uniform sampler2D attentionTex;
uniform sampler2D kvPointTex;
uniform sampler2D affineTex;
${QUATERNION}

float featureOf(int element) {
  int residueStride = 4 * PLANE;
  int query = element / residueStride;
  int within = element - query * residueStride;
  int plane = within / PLANE;
  int slot = within - plane * PLANE;
  int point = slot % POINT_V;
  int head = slot / POINT_V;

  int attentionBase = (head * L + query) * L;
  vec3 global = vec3(0.0);
  for (int key = 0; key < L; ++key) {
    int pointBase = ((key * HEADS + head) * KV_POINTS + POINT_QK + point) * 3;
    global += fetch1(attentionTex, attentionBase + key) * vec3(
      fetch1(kvPointTex, pointBase),
      fetch1(kvPointTex, pointBase + 1),
      fetch1(kvPointTex, pointBase + 2));
  }
  vec3 local = inverseRotateBy(quaternionOf(affineTex, query),
    global - translationOf(affineTex, query));
  // The fourth plane is the norm, guarded so a zero vector gives 0 and not NaN.
  return plane == 0 ? local.x : plane == 1 ? local.y : plane == 2 ? local.z
    : sqrt(1e-8 + dot(local, local));
}

void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  result = vec4(
    featureOf(base),
    base + 1 < ELEMENTS ? featureOf(base + 1) : 0.0,
    base + 2 < ELEMENTS ? featureOf(base + 2) : 0.0,
    base + 3 < ELEMENTS ? featureOf(base + 3) : 0.0);
}`;

/** `pair[query][head][c] = sum_k attention[head][query][k] * pair[query][k][c]` */
export const ipaPairFeatureSource = (length, heads, pairChannels) => `${PRELUDE}${ONE_TARGET}
const int L = ${length};
const int HEADS = ${heads};
const int PAIR_C = ${pairChannels};
const int ELEMENTS = ${length * heads * pairChannels};
uniform sampler2D attentionTex;
uniform sampler2D pairTex;

float featureOf(int index) {
  int channel = index % PAIR_C;
  int head = (index / PAIR_C) % HEADS;
  int query = index / (PAIR_C * HEADS);
  int attentionBase = (head * L + query) * L;
  int rowBase = query * L * PAIR_C + channel;
  float total = 0.0;
  for (int key = 0; key < L; ++key) {
    total += fetch1(attentionTex, attentionBase + key)
      * fetch1(pairTex, rowBase + key * PAIR_C);
  }
  return total;
}

void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  result = vec4(
    featureOf(base),
    base + 1 < ELEMENTS ? featureOf(base + 1) : 0.0,
    base + 2 < ELEMENTS ? featureOf(base + 2) : 0.0,
    base + 3 < ELEMENTS ? featureOf(base + 3) : 0.0);
}`;

/** The three blocks gathered into one row, in the reference's order. */
export const ipaFeatureRowSource = (length, scalarSize, pointSize, pairSize) => `${PRELUDE}${ONE_TARGET}
const int SCALAR = ${scalarSize};
const int POINT4 = ${4 * pointSize};
const int PAIR = ${pairSize};
const int WIDTH = ${scalarSize + 4 * pointSize + pairSize};
const int ELEMENTS = ${length * (scalarSize + 4 * pointSize + pairSize)};
uniform sampler2D scalarTex;
uniform sampler2D pointTex;
uniform sampler2D pairTex;

float rowOf(int element) {
  int residue = element / WIDTH;
  int within = element - residue * WIDTH;
  if (within < SCALAR) return fetch1(scalarTex, residue * SCALAR + within);
  if (within < SCALAR + POINT4) {
    return fetch1(pointTex, residue * POINT4 + (within - SCALAR));
  }
  return fetch1(pairTex, residue * PAIR + (within - SCALAR - POINT4));
}

void main() {
  int base = linearTexel() * 4;
  if (base >= ELEMENTS) { result = vec4(0.0); return; }
  result = vec4(
    rowOf(base),
    base + 1 < ELEMENTS ? rowOf(base + 1) : 0.0,
    base + 2 < ELEMENTS ? rowOf(base + 2) : 0.0,
    base + 3 < ELEMENTS ? rowOf(base + 3) : 0.0);
}`;

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 */
export function uploadIpaWeights(execution, weights) {
  /** @type {Record<string, Tensor>} */
  const out = {};
  for (const name of IPA_WEIGHTS) {
    const values = weights[name];
    if (values === undefined) throw new Error(`ipa weights: ${name} is missing`);
    out[name] = execution.upload(values, `w.ipa.${name}`);
  }
  return out;
}

export function releaseIpaWeights(execution, weights) {
  for (const name of IPA_WEIGHTS) execution.release(weights[name]);
}

/**
 * The pair LayerNorm, which is the same for all eight iterations and is
 * therefore the caller's to hold.
 *
 * @param {import("../../runtime/execution.js").Execution} execution
 */
export function normalisePair(execution, { pair, weights, length, pairChannels, epsilon = 1e-5 }) {
  return layerNorm(execution, {
    x: pair, weight: weights.pairNormScale, bias: weights.pairNormOffset,
    rows: length * length, channels: pairChannels, epsilon,
  });
}

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{activations: Tensor, normalisedPair: Tensor, mask: Tensor, affine: Tensor,
 *   weights: Record<string, Tensor>, length: number, channels: number,
 *   pairChannels: number, heads: number, scalarQk: number, scalarV: number,
 *   pointQk: number, pointV: number}} call
 * @returns {Tensor} `length x channels`, the update; the caller owns it
 */
export function invariantPointAttention(execution, call) {
  const {
    activations, normalisedPair, mask, affine, weights: w,
    length, channels, pairChannels, heads, scalarQk, scalarV, pointQk, pointV,
  } = call;
  const scalarSize = heads * scalarV;
  const pointSize = heads * pointV;
  const pairSize = heads * pairChannels;
  const released = [];
  const temp = (tensor) => { released.push(tensor); return tensor; };

  const project = (weight, bias, outChannels) => temp(linear(execution, {
    x: activations, weight, bias, rows: length, inChannels: channels, outChannels,
    weightLayout: "in-major",
  }));
  const queryScalar = project(w.queryScalarWeight, w.queryScalarBias, heads * scalarQk);
  const kvScalar = project(w.keyValueScalarWeight, w.keyValueScalarBias, heads * (scalarQk + scalarV));
  const queryPointLocal = project(w.queryPointWeight, w.queryPointBias, heads * pointQk * 3);
  const kvPointLocal = project(
    w.keyValuePointWeight, w.keyValuePointBias, heads * (pointQk + pointV) * 3);

  const toGlobal = (local, points) => {
    const out = temp(execution.allocate(length * heads * points * 3, "ipa.points"));
    execution.dispatch({
      source: transformPointsSource(length, heads, points),
      output: out,
      inputs: { localTex: local, affineTex: affine },
      label: `ipa.points:${length}x${heads}x${points}`,
    });
    return out;
  };
  const queryPoint = toGlobal(queryPointLocal, pointQk);
  const kvPoint = toGlobal(kvPointLocal, pointQk + pointV);

  const pairBias = temp(execution.allocate(heads * length * length, "ipa.pair-bias"));
  execution.dispatch({
    source: ipaPairBiasSource(length, heads, pairChannels),
    output: pairBias,
    inputs: {
      pairTex: normalisedPair, weightTex: w.attention2dWeight, biasTex: w.attention2dBias,
    },
    label: `ipa.pair-bias:${heads}x${length}`,
  });

  const logits = temp(execution.allocate(heads * length * length, "ipa.logits"));
  execution.dispatch({
    source: ipaLogitsSource({ length, heads, scalarQk, scalarV, pointQk, pointV }),
    output: logits,
    inputs: {
      queryScalarTex: queryScalar, kvScalarTex: kvScalar,
      queryPointTex: queryPoint, kvPointTex: kvPoint,
      pairBiasTex: pairBias, maskTex: mask, trainableTex: w.trainablePointWeights,
    },
    label: `ipa.logits:${heads}x${length}`,
  });

  const attention = temp(execution.allocate(heads * length * length, "ipa.attention"));
  execution.dispatch({
    source: ipaSoftmaxSource(length, heads),
    output: attention,
    inputs: { logitsTex: logits },
    label: `ipa.softmax:${heads}x${length}`,
  });

  const scalarFeature = temp(execution.allocate(length * scalarSize, "ipa.scalar"));
  execution.dispatch({
    source: ipaScalarFeatureSource(length, heads, scalarQk, scalarV),
    output: scalarFeature,
    inputs: { attentionTex: attention, kvScalarTex: kvScalar },
    label: `ipa.scalar:${length}x${scalarSize}`,
  });

  const pointFeature = temp(execution.allocate(length * 4 * pointSize, "ipa.point"));
  execution.dispatch({
    source: ipaPointFeatureSource(length, heads, pointQk, pointV),
    output: pointFeature,
    inputs: { attentionTex: attention, kvPointTex: kvPoint, affineTex: affine },
    label: `ipa.point:${length}x${pointSize}`,
  });

  const pairFeature = temp(execution.allocate(length * pairSize, "ipa.pair"));
  execution.dispatch({
    source: ipaPairFeatureSource(length, heads, pairChannels),
    output: pairFeature,
    inputs: { attentionTex: attention, pairTex: normalisedPair },
    label: `ipa.pair:${length}x${pairSize}`,
  });

  const featureChannels = scalarSize + 4 * pointSize + pairSize;
  const features = temp(execution.allocate(length * featureChannels, "ipa.features"));
  execution.dispatch({
    source: ipaFeatureRowSource(length, scalarSize, pointSize, pairSize),
    output: features,
    inputs: { scalarTex: scalarFeature, pointTex: pointFeature, pairTex: pairFeature },
    label: `ipa.features:${length}x${featureChannels}`,
  });

  const output = linear(execution, {
    x: features, weight: w.outputWeight, bias: w.outputBias,
    rows: length, inChannels: featureChannels, outChannels: channels,
    weightLayout: "in-major",
  });
  for (const tensor of released) execution.release(tensor);
  output.label = "ipa.output";
  return output;
}
