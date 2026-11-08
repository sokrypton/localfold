/**
 * AF3's per-atom conditioning: chemistry in, 128 channels per atom out.
 *
 * "This is the bottom of the model. Everything else in the trunk reads token
 * representations; this reads a REFERENCE CONFORMER - where each atom of each
 * residue sits in an idealised copy of it, what element it is, what charge it
 * carries and what it is called - and turns that into the per-atom features the
 * atom transformer runs on. Its output is where 384 of `target_feat`'s 447
 * columns come from."
 *
 * 🔴 FIVE EMBEDDINGS ARE SUMMED AND ALL FIVE ARE BIAS-FREE, "so a missing term
 * is not a constant offset that a later LayerNorm absorbs - it is a direction
 * removed from every atom."
 *
 * 🔴 AND THE CHARGE IS `arcsinh(charge)`, NOT THE CHARGE - WHICH NO
 * PROTEIN-ONLY CHECK CAN SEE. "The two are IDENTICAL AT ZERO. Every standard
 * amino acid conformer in the CCD is neutral, so a test on protein alone can
 * never tell them apart; it first matters on a ligand." boltz2, chai1,
 * rosettafold3 and ESMFold2 use the raw charge.
 *
 * 🔴 AND THE FOUR NAME CHARACTERS ARE SUMMED BEFORE THEY REACH THE RUNNING
 * VALUE, which is a ROUNDING fact rather than an algebraic one: the reference
 * records that "the matmul committed `act + (a+b+c+d)` and `(((act+a)+b)+c)+d`
 * differed in 169,390 of 696,512 floats". The kernel below accumulates the four
 * separately and adds the sum once, in character order.
 *
 * 🔴 AND IT IS MASKED LAST, "so an absent atom contributes nothing downstream
 * even though four of the five embeddings above are non-zero for it".
 *
 * The element is a one-hot over the periodic table and the name characters are
 * 64-way one-hots of ASCII minus 32, so all three are weight-row gathers rather
 * than matrix products - which is also why `kernels/linear.js` cannot serve
 * here: its input widths would be 128 and 64 one-hot columns.
 */
import { ONE_TARGET, PRELUDE, sized, part, memoSource } from "../../runtime/execution.js";
import { linear } from "../../kernels/linear.js";
import { relu } from "../../runtime/elementwise.js";

/** @typedef {import("../../runtime/storage.js").Tensor} Tensor */

export const ELEMENTS = 128;
export const NAME_CHARACTERS = 4;
export const NAME_CLASSES = 64;

export const CONDITIONING_WEIGHTS = [
  "embedRefPos", "embedRefMask", "embedRefCharge", "embedRefElement", "embedRefAtomName",
];
/** ...and the pair conditioning's four, which the same upload carries. */
export const PAIR_CONDITIONING_WEIGHTS = [
  "singleToPairCondRow", "singleToPairCondCol", "embedPairOffsets", "embedPairDistances",
];

const perAtomSourceUncached = (rows, channels, rawCharge, hasBias) => part({ ELEMENTS: rows * channels, C4: channels / 4, C: channels }, {
  uniforms: ["sampler2D positionsTex", "sampler2D maskTex", "sampler2D chargeTex", "sampler2D elementTex", "sampler2D nameTex", "sampler2D posWeightTex", "sampler2D maskWeightTex", "sampler2D chargeWeightTex", "sampler2D elementWeightTex", "sampler2D nameWeightTex", "sampler2D biasTex"],
  body: `  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % C4;
  int row = texel / C4;

  // The position: three columns, so three weight rows scaled.
  vec3 p = vec3(fetch1(positionsTex, row * 3), fetch1(positionsTex, row * 3 + 1),
    fetch1(positionsTex, row * 3 + 2));
  vec4 value = p.x * fetch4(posWeightTex, group)
    + p.y * fetch4(posWeightTex, C4 + group)
    + p.z * fetch4(posWeightTex, 2 * C4 + group);
${hasBias ? "  value += fetch4(biasTex, group);" : "  // boltz2 alone has a bias over the concatenation."}

  float mask = fetch1(maskTex, row);
  value += mask * fetch4(maskWeightTex, group);

  // The element: a one-hot over the periodic table, so one weight row.
  int atomicNumber = int(fetch1(elementTex, row));
  if (atomicNumber >= 0 && atomicNumber < ${ELEMENTS}) {
    value += fetch4(elementWeightTex, atomicNumber * C4 + group);
  }

  // 🔴 arcsinh OF THE CHARGE, not the charge - identical at zero, so no
  // protein-only input can tell the two apart. GLSL ES has no asinh, and the
  // identity log(x + sqrt(x*x + 1)) is what it would be implemented as.
  float charge = fetch1(chargeTex, row);
${rawCharge ? "" : "  charge = log(charge + sqrt(charge * charge + 1.0));"}
  value += charge * fetch4(chargeWeightTex, group);

  // The four name characters, summed in CHARACTER ORDER and added ONCE.
  vec4 nameSum = vec4(0.0);
  for (int character = 0; character < ${NAME_CHARACTERS}; ++character) {
    int code = int(fetch1(nameTex, row * ${NAME_CHARACTERS} + character));
    if (code < 0 || code >= ${NAME_CLASSES}) continue;
    nameSum += fetch4(nameWeightTex, (character * ${NAME_CLASSES} + code) * C4 + group);
  }
  value += nameSum;

  // ...and masked LAST.
  result = value * mask;`,
});

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 */
export function uploadConditioningWeights(execution, weights) {
  /** @type {any} */
  const out = { channels: weights.channels, pairChannels: weights.pairChannels };
  // The pair weights are optional: the target-feat encoder's conditioning needs
  // them and a caller that only wants the single conditioning does not.
  const names = [...CONDITIONING_WEIGHTS,
    ...PAIR_CONDITIONING_WEIGHTS.filter((name) => weights[name] !== undefined)];
  for (const name of names) {
    const values = weights[name];
    if (values === undefined) {
      throw new Error(`af3 atom conditioning weights: ${name} is missing`);
    }
    out[name] = execution.uploadWeight(values, `w.af3.cond.${name}`);
  }
  if (weights.embedAtomFeaturesBias !== undefined
      && weights.embedAtomFeaturesBias !== null) {
    out.embedAtomFeaturesBias = execution.uploadWeight(
      weights.embedAtomFeaturesBias, "w.af3.cond.embedAtomFeaturesBias");
  }
  return out;
}

export function releaseConditioningWeights(execution, weights) {
  for (const name of [...CONDITIONING_WEIGHTS, ...PAIR_CONDITIONING_WEIGHTS,
    "embedAtomFeaturesBias"]) {
    if (weights[name] !== undefined) execution.release(weights[name]);
  }
}

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{positions: Tensor, mask: Tensor, element: Tensor, charge: Tensor,
 *   atomNameChars: Tensor, weights: any, tokens: number, dense: number,
 *   rawRefCharge: boolean}} call
 * @returns {Tensor} `tokens * dense * channels`
 */
export function perAtomConditioning(execution, call) {
  const {
    positions, mask, element, charge, atomNameChars, weights: w, tokens, dense,
  } = call;
  if (typeof call.rawRefCharge !== "boolean") {
    throw new Error(
      "af3 atom conditioning: rawRefCharge has no default - AlphaFold 3 feeds"
      + " arcsinh(charge) and boltz2, chai1, rosettafold3 and ESMFold2 the raw charge,"
      + " and the two are identical at zero so no protein-only input separates them");
  }
  const channels = w.channels;
  const rows = tokens * dense;
  const out = execution.allocate(rows * channels, "af3.cond.per-atom");
  execution.dispatch({
    source: perAtomSource(
      rows, channels, call.rawRefCharge, w.embedAtomFeaturesBias !== undefined),
    output: out,
    inputs: {
      positionsTex: positions, maskTex: mask, chargeTex: charge,
      elementTex: element, nameTex: atomNameChars,
      posWeightTex: w.embedRefPos, maskWeightTex: w.embedRefMask,
      chargeWeightTex: w.embedRefCharge, elementWeightTex: w.embedRefElement,
      nameWeightTex: w.embedRefAtomName,
      ...(w.embedAtomFeaturesBias === undefined
        ? {} : { biasTex: w.embedAtomFeaturesBias }),
    },
    label: `af3.cond.per-atom:${rows}x${channels}${call.rawRefCharge ? ":raw" : ":asinh"}`,
  });
  return out;
}

/**
 * The per-atom PAIR conditioning, within each token's own atoms.
 *
 * 🔴 THROUGH A RELU FIRST, "which is not a detail: the single conditioning is
 * signed and roughly centred, so the relu discards about half of it".
 */
const pairConditioningSourceUncached = (tokens, dense, pairChannels) => sized({ ELEMENTS: tokens * dense * dense * pairChannels }, `${PRELUDE}${ONE_TARGET}
const int DENSE = ${dense};
const int P4 = ${pairChannels / 4};
uniform sampler2D rowTex;
uniform sampler2D columnTex;
uniform sampler2D positionsTex;
uniform sampler2D offsetWeightTex;
uniform sampler2D distanceWeightTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  int group = texel % P4;
  int rest = texel / P4;
  int column = rest % DENSE;
  int rest2 = rest / DENSE;
  int row = rest2 % DENSE;
  int token = rest2 / DENSE;

  int rowAtom = token * DENSE + row;
  int columnAtom = token * DENSE + column;
  vec4 total = fetch4(rowTex, rowAtom * P4 + group)
    + fetch4(columnTex, columnAtom * P4 + group);

  // The offset between the two atoms in the reference conformer, projected.
  vec3 offset = vec3(
    fetch1(positionsTex, rowAtom * 3) - fetch1(positionsTex, columnAtom * 3),
    fetch1(positionsTex, rowAtom * 3 + 1) - fetch1(positionsTex, columnAtom * 3 + 1),
    fetch1(positionsTex, rowAtom * 3 + 2) - fetch1(positionsTex, columnAtom * 3 + 2));
  total += offset.x * fetch4(offsetWeightTex, group)
    + offset.y * fetch4(offsetWeightTex, P4 + group)
    + offset.z * fetch4(offsetWeightTex, 2 * P4 + group);

  // 🔴 THE INVERSE SQUARED DISTANCE, "so nearby atoms give a large feature and
  // distant ones tend to zero rather than growing without bound". The plain
  // squared distance is the obvious reading and grows instead.
  total += fetch4(distanceWeightTex, group) / (1.0 + dot(offset, offset));
  result = total;
}`);

/**
 * @param {import("../../runtime/execution.js").Execution} execution
 * @param {{act: Tensor, positions: Tensor, weights: any, tokens: number,
 *   dense: number, channels: number, pairChannels: number}} call
 * @returns {Tensor} `tokens * dense * dense * pairChannels`
 */
export function perAtomPairConditioning(execution, call) {
  const { act, positions, weights: w, tokens, dense, channels, pairChannels } = call;
  const rows = tokens * dense;
  const rectified = relu(execution, act);
  const row = linear(execution, {
    x: rectified, weight: w.singleToPairCondRow, rows,
    inChannels: channels, outChannels: pairChannels, weightLayout: "in-major",
  });
  const column = linear(execution, {
    x: rectified, weight: w.singleToPairCondCol, rows,
    inChannels: channels, outChannels: pairChannels, weightLayout: "in-major",
  });
  execution.release(rectified);
  const out = execution.allocate(
    tokens * dense * dense * pairChannels, "af3.cond.per-atom-pair");
  execution.dispatch({
    source: pairConditioningSource(tokens, dense, pairChannels),
    output: out,
    inputs: {
      rowTex: row, columnTex: column, positionsTex: positions,
      offsetWeightTex: w.embedPairOffsets, distanceWeightTex: w.embedPairDistances,
    },
    label: `af3.cond.per-atom-pair:${tokens}x${dense}x${pairChannels}`,
  });
  execution.release(row);
  execution.release(column);
  return out;
}

// Every builder above, memoised on its arguments (`memoSource` in webgl/runtime/execution.js).
export const perAtomSource = memoSource(perAtomSourceUncached);
export const pairConditioningSource = memoSource(pairConditioningSourceUncached);
