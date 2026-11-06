/**
 * pLDDT, PAE and pTM - the numbers the page shows beside a fold.
 *
 * `webgpu/heads/confidence.js` is the WebGPU head and the conventions are all
 * its; what is new here is only that every step is a draw.
 *
 *     plddt[i] = E[centre] over softmax(Linear(relu(Linear(relu(Linear(LayerNorm(s)))))))
 *     pae[i][j] = E[centre] over softmax(Linear(pair[i][j]))
 *     ptm       = max over anchors of the mean TM term, the term being
 *                 E[1 / (1 + centre^2 / d0^2)] over the same softmax
 *
 * 🔴 THE TWO BIN-CENTRE RULES ARE DIFFERENT AND NEITHER IS OBVIOUS. pLDDT's are
 * `(bin + 0.5) / bins * 100`, evenly spaced over a percentage; PAE's come from
 * the model's own `breaks` as `breaks[bin] + step / 2`, with the LAST bin taking
 * `breaks[last] + step / 2` rather than extending to infinity. Getting either
 * wrong produces a confidence map that looks entirely reasonable.
 *
 * 🔴 AND THE REDUCTION FOR pTM STAYS ON THE HOST, DELIBERATELY. It is a max over
 * anchors of a mean over a selection - `reduceTmScore` in
 * shared/heads/tm-score.js - and the WebGPU head leaves it there too, because
 * every score over one prediction (pTM, ipTM, one per chain pair) is the same
 * `tokens^2` expectations reduced differently. The expensive part is the
 * expectation, which is a draw; the reduction is `tokens^2` adds on an array
 * that is already back. `tmScoreD0` and `tmPerBinFor` are shared code and are
 * reused rather than restated: d0 is drawn from the whole prediction for both
 * scores, which is AF2's own choice and the sort of thing a second copy gets
 * wrong.
 */
import { ONE_TARGET, PRELUDE, sized } from "../runtime/execution.js";
import { layerNorm } from "../kernels/layer-norm.js";
import { linear } from "../kernels/linear.js";
import { tmPerBinFor, tmScoreD0 } from "../../shared/heads/tm-score.js";

/** @typedef {import("../runtime/storage.js").Tensor} Tensor */

/**
 * `out[row] = sum_b softmax(logits[row])[b] * centres[b]`
 *
 * One fragment owns four rows, each its own softmax. `bins` need not be a
 * multiple of four - every read is a `fetch1` - because `lddtBins` and `paeBins`
 * are the model's and not this port's to round.
 */
export const softmaxExpectationSource = (rows, bins) => sized({ ROWS: rows }, `${PRELUDE}${ONE_TARGET}
const int BINS = ${bins};
uniform sampler2D logitsTex;
uniform sampler2D centresTex;

float expectationOf(int row) {
  int base = row * BINS;
  float top = -1e30;
  for (int b = 0; b < BINS; ++b) top = max(top, fetch1(logitsTex, base + b));
  float denominator = 0.0;
  float total = 0.0;
  for (int b = 0; b < BINS; ++b) {
    float probability = exp(fetch1(logitsTex, base + b) - top);
    denominator += probability;
    total += probability * fetch1(centresTex, b);
  }
  return total / denominator;
}

void main() {
  int texel = linearTexel();
  int base = texel * 4;
  if (base >= ROWS) { result = vec4(0.0); return; }
  result = vec4(
    expectationOf(base),
    base + 1 < ROWS ? expectationOf(base + 1) : 0.0,
    base + 2 < ROWS ? expectationOf(base + 2) : 0.0,
    base + 3 < ROWS ? expectationOf(base + 3) : 0.0);
}`);

/** pLDDT's bin centres: evenly spaced over 0-100. */
export const lddtCentres = (bins) => Float32Array.from(
  { length: bins }, (_, bin) => ((bin + 0.5) / bins) * 100);

/**
 * PAE's bin centres, from the model's breaks.
 *
 * The last bin is `breaks[last] + step / 2` and not an open interval, which is
 * AF2's own convention and is why `maxPredictedAlignedError` is a finite number.
 */
export function paeCentres(breaks) {
  const bins = breaks.length + 1;
  const step = breaks[1] - breaks[0];
  const centres = new Float32Array(bins);
  for (let bin = 0; bin < bins - 1; bin += 1) centres[bin] = breaks[bin] + step / 2;
  centres[bins - 1] = breaks[breaks.length - 1] + step / 2;
  return centres;
}

const LDDT_WEIGHTS = [
  "normScale", "normOffset", "act0Weight", "act0Bias",
  "act1Weight", "act1Bias", "logitsWeight", "logitsBias",
];

/**
 * @param {import("../runtime/execution.js").Execution} execution
 */
export function uploadConfidenceWeights(execution, lddtWeights, paeWeights) {
  /** @type {{lddt: Record<string, Tensor>, pae: Record<string, Tensor>}} */
  const out = { lddt: {}, pae: {} };
  for (const name of LDDT_WEIGHTS) {
    const values = lddtWeights[name];
    if (values === undefined) throw new Error(`confidence weights: lddt ${name} is missing`);
    out.lddt[name] = execution.uploadWeight(values, `w.confidence.lddt.${name}`);
  }
  for (const name of ["logitsWeight", "logitsBias"]) {
    const values = paeWeights[name];
    if (values === undefined) throw new Error(`confidence weights: pae ${name} is missing`);
    out.pae[name] = execution.uploadWeight(values, `w.confidence.pae.${name}`);
  }
  return out;
}

export function releaseConfidenceWeights(execution, weights) {
  for (const group of [weights.lddt, weights.pae]) {
    for (const tensor of Object.values(group)) execution.release(tensor);
  }
}

/**
 * @param {import("../runtime/execution.js").Execution} execution
 * @param {{structure: Tensor, pair: Tensor, length: number, structureChannels: number,
 *   pairChannels: number, weights: {lddt: Record<string, Tensor>, pae: Record<string, Tensor>},
 *   hiddenChannels: number, lddtBins: number, breaks: ArrayLike<number>,
 *   epsilon?: number}} call
 * @returns {{plddt: Float32Array, meanPlddt: number,
 *   predictedAlignedError: Float32Array, tmTerm: Float32Array,
 *   maxPredictedAlignedError: number}}
 */
export function confidenceHeads(execution, call) {
  const {
    structure, pair, length, structureChannels, pairChannels, weights: w,
    hiddenChannels, lddtBins, breaks, epsilon = 1e-5,
  } = call;
  const pairs = length * length;
  const paeBins = breaks.length + 1;
  if (structure.elements !== length * structureChannels) {
    throw new RangeError(
      `confidence: structure holds ${structure.elements} floats, not `
      + `${length} x ${structureChannels}`);
  }
  if (pair.elements !== pairs * pairChannels) {
    throw new RangeError(
      `confidence: pair holds ${pair.elements} floats, not ${length}^2 x ${pairChannels}`);
  }

  const expectation = (logits, rows, bins, centres, label) => {
    const centresTensor = execution.upload(centres, `${label}.centres`);
    const output = execution.allocate(rows, label);
    execution.dispatch({
      source: softmaxExpectationSource(rows, bins),
      output,
      inputs: { logitsTex: logits, centresTex: centresTensor },
      label: `confidence.${label}:${rows}x${bins}`,
    });
    execution.release(centresTensor);
    const values = execution.read(output).slice();
    execution.release(output);
    return values;
  };

  // --- pLDDT -------------------------------------------------------------
  const normalised = layerNorm(execution, {
    x: structure, weight: w.lddt.normScale, bias: w.lddt.normOffset,
    rows: length, channels: structureChannels, epsilon,
  });
  const act0 = linear(execution, {
    x: normalised, weight: w.lddt.act0Weight, bias: w.lddt.act0Bias,
    rows: length, inChannels: structureChannels, outChannels: hiddenChannels,
    activation: "relu", weightLayout: "in-major",
  });
  execution.release(normalised);
  const act1 = linear(execution, {
    x: act0, weight: w.lddt.act1Weight, bias: w.lddt.act1Bias,
    rows: length, inChannels: hiddenChannels, outChannels: hiddenChannels,
    activation: "relu", weightLayout: "in-major",
  });
  execution.release(act0);
  const lddtLogits = linear(execution, {
    x: act1, weight: w.lddt.logitsWeight, bias: w.lddt.logitsBias,
    rows: length, inChannels: hiddenChannels, outChannels: lddtBins,
    weightLayout: "in-major",
  });
  execution.release(act1);
  const plddt = expectation(lddtLogits, length, lddtBins, lddtCentres(lddtBins), "plddt");
  execution.release(lddtLogits);

  // --- PAE and the pTM term, two expectations of one softmax -------------
  const paeLogits = linear(execution, {
    x: pair, weight: w.pae.logitsWeight, bias: w.pae.logitsBias,
    rows: pairs, inChannels: pairChannels, outChannels: paeBins,
    weightLayout: "in-major",
  });
  const centres = paeCentres(breaks);
  const predictedAlignedError = expectation(paeLogits, pairs, paeBins, centres, "pae");
  const tmTerm = expectation(
    paeLogits, pairs, paeBins, tmPerBinFor(centres, tmScoreD0(length)), "tm-term");
  execution.release(paeLogits);

  let total = 0;
  for (const value of plddt) total += value;
  return {
    plddt,
    meanPlddt: total / length,
    predictedAlignedError,
    tmTerm,
    maxPredictedAlignedError: centres[paeBins - 1],
  };
}
