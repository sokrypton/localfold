/**
 * Which activation order, and which OUTPUT order, does the WGSL attention block
 * use when `transpose` is set?
 *
 *     node tools/gpu-chrome.mjs tools/gpu/probe-webgl-column-layout.js
 *
 * 🔴 WRITTEN BECAUSE TWO READINGS OF THE SHADER BOTH FAILED, AND KEPT BECAUSE
 * OF WHAT IT FOUND. `source_row(row)` in the normalise shader maps a
 * destination row to a source row under `transpose`, `mask_index` re-indexes
 * the mask, and `run` does plumb the flag into both - so `transpose: true` with
 * AF2's `[sequence][residue]` activations and `[sequence][residue]` mask ought
 * to be column attention, returning `[residue][sequence]`. It is not: that
 * combination reads relRMS ~1.3.
 *
 * So rather than guess a fourth time, this enumerates the whole space -
 * activation order, MASK order, the flag, and the output taken as-given or
 * swapped - against one f64 reference under both mask conventions, and prints
 * the table. The combination that lands at 1e-7 is the contract; the rest are
 * the evidence that it is the only one, and whether any `transpose: true`
 * combination lands at all is the question this tool exists to answer.
 */
import { AttentionGpu } from "../../webgpu/kernels/attention.js";
import { relativeRms } from "./relative-rms.js";

function generator(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 0x1_0000_0000;
  };
}
const noise = (count, spread, random, centre = 0) => {
  const out = new Float32Array(count);
  for (let i = 0; i < count; i += 1) out[i] = centre + (random() * 2 - 1) * spread;
  return out;
};

/** Gated attention block, f64, with the mask order given explicitly. */
function reference({ activations, mask, weights: w, batch, queryLength, channels, heads, maskOrder }) {
  const headDim = channels / heads;
  const rows = batch * queryLength;
  const scale = 1 / Math.sqrt(headDim);
  const normalised = new Float64Array(rows * channels);
  for (let row = 0; row < rows; row += 1) {
    const at = row * channels;
    let mean = 0;
    for (let c = 0; c < channels; c += 1) mean += activations[at + c];
    mean /= channels;
    let spread = 0;
    for (let c = 0; c < channels; c += 1) spread += (activations[at + c] - mean) ** 2;
    const inverseStd = 1 / Math.sqrt(spread / channels + 1e-5);
    for (let c = 0; c < channels; c += 1) {
      normalised[at + c] = (activations[at + c] - mean) * inverseStd * w.queryNormScale[c]
        + w.queryNormOffset[c];
    }
  }
  const project = (weight, bias, sigmoid) => {
    const out = new Float64Array(rows * channels);
    for (let row = 0; row < rows; row += 1) {
      for (let o = 0; o < channels; o += 1) {
        let sum = bias === undefined ? 0 : bias[o];
        for (let i = 0; i < channels; i += 1) {
          sum += normalised[row * channels + i] * weight[i * channels + o];
        }
        out[row * channels + o] = sigmoid ? 1 / (1 + Math.exp(-sum)) : sum;
      }
    }
    return out;
  };
  const query = project(w.queryWeight, undefined, false);
  const key = project(w.keyWeight, undefined, false);
  const value = project(w.valueWeight, undefined, false);
  const gate = project(w.gatingWeight, w.gatingBias, true);
  const attended = new Float64Array(rows * channels);
  for (let b = 0; b < batch; b += 1) {
    for (let head = 0; head < heads; head += 1) {
      for (let q = 0; q < queryLength; q += 1) {
        const qAt = (b * queryLength + q) * channels + head * headDim;
        const logits = new Float64Array(queryLength);
        for (let k = 0; k < queryLength; k += 1) {
          const kAt = (b * queryLength + k) * channels + head * headDim;
          let score = 0;
          for (let d = 0; d < headDim; d += 1) score += query[qAt + d] * key[kAt + d];
          const maskIndex = maskOrder === "key-major" ? k * batch + b : b * queryLength + k;
          logits[k] = score * scale + 1e9 * (mask[maskIndex] - 1);
        }
        let top = -Infinity;
        for (const logit of logits) top = Math.max(top, logit);
        let total = 0;
        const weight = new Float64Array(queryLength);
        for (let k = 0; k < queryLength; k += 1) {
          weight[k] = Math.exp(logits[k] - top);
          total += weight[k];
        }
        for (let d = 0; d < headDim; d += 1) {
          let acc = 0;
          for (let k = 0; k < queryLength; k += 1) {
            acc += weight[k] * value[(b * queryLength + k) * channels + head * headDim + d];
          }
          attended[qAt + d] = (acc / total) * gate[qAt + d];
        }
      }
    }
  }
  const out = new Float32Array(rows * channels);
  for (let row = 0; row < rows; row += 1) {
    for (let o = 0; o < channels; o += 1) {
      let sum = w.outputBias[o];
      for (let i = 0; i < channels; i += 1) {
        sum += attended[row * channels + i] * w.outputWeight[i * channels + o];
      }
      out[row * channels + o] = sum;
    }
  }
  return out;
}

const swapLeading = (values, outer, inner, channels) => {
  const out = new Float32Array(values.length);
  for (let o = 0; o < outer; o += 1) {
    for (let i = 0; i < inner; i += 1) {
      for (let c = 0; c < channels; c += 1) {
        out[(i * outer + o) * channels + c] = values[(o * inner + i) * channels + c];
      }
    }
  }
  return out;
};

export async function main(device) {
  const sequences = 5;
  const length = 9;
  const channels = 32;
  const heads = 4;
  const random = generator(777);
  const rows = sequences * length;
  const bySequence = noise(rows * channels, 1.5, random);
  const byResidue = swapLeading(bySequence, sequences, length, channels);
  const mask = new Float32Array(rows).fill(1);
  mask[(sequences - 1) * length + 2] = 0;
  mask[1 * length + (length - 1)] = 0;
  const weights = {
    queryNormScale: noise(channels, 0.2, random, 1),
    queryNormOffset: noise(channels, 0.1, random),
    queryWeight: noise(channels * channels, 1 / Math.sqrt(channels), random),
    keyWeight: noise(channels * channels, 1 / Math.sqrt(channels), random),
    valueWeight: noise(channels * channels, 1 / Math.sqrt(channels), random),
    gatingWeight: noise(channels * channels, 1 / Math.sqrt(channels), random),
    gatingBias: noise(channels, 0.1, random, 1),
    outputWeight: noise(channels * channels, 1 / Math.sqrt(channels), random),
    outputBias: noise(channels, 0.05, random),
  };
  const options = {
    flashPrecision: "f32",
    projectPrecision: "f32",
    outputPrecision: "f32",
    storage: { input: "f32", value: "f32", output: "f32" },
  };

  // The reference, computed on the residue-major activations - column attention
  // attending over `sequences` for each residue - under both mask orders.
  const expected = {};
  for (const maskOrder of ["key-major", "batch-major"]) {
    expected[maskOrder] = reference({
      activations: byResidue, mask, weights,
      batch: length, queryLength: sequences, channels, heads, maskOrder,
    });
  }

  // The mask, in AF2's [sequence][residue] order and transposed.
  const maskByResidue = new Float32Array(rows).fill(1);
  for (let s = 0; s < sequences; s += 1) {
    for (let i = 0; i < length; i += 1) maskByResidue[i * sequences + s] = mask[s * length + i];
  }

  const table = [];
  for (const [inputName, activations] of [["bySequence", bySequence], ["byResidue", byResidue]]) {
   for (const [maskName, maskValues] of [["bySequence", mask], ["byResidue", maskByResidue]]) {
    for (const transpose of [true, false]) {
      let got;
      try {
        got = await new AttentionGpu(device, options).run({
          batch: length, queryLength: sequences, channels, heads,
          activations, mask: maskValues, weights, transpose,
        });
      } catch (error) {
        table.push({ input: inputName, mask: maskName, transpose,
          error: String(error.message).slice(0, 90) });
        continue;
      }
      const asGiven = got.output;
      const swapped = swapLeading(asGiven, length, sequences, channels);
      for (const [outputName, values] of [["as-given", asGiven], ["swapped", swapped]]) {
        for (const maskOrder of ["key-major", "batch-major"]) {
          table.push({
            input: inputName,
            mask: maskName,
            transpose,
            output: outputName,
            reference: maskOrder,
            relativeRms: Number(relativeRms(values, expected[maskOrder], "probe").toPrecision(4)),
          });
        }
      }
    }
   }
  }
  table.sort((a, b) => (a.relativeRms ?? 9) - (b.relativeRms ?? 9));
  const agreeing = table.filter((row) => (row.relativeRms ?? 9) < 1e-5);
  return {
    shape: { sequences, length, channels, heads },
    combinations: table.length,
    agreeing,
    // 🔴 THE ANSWER THIS TOOL EXISTS FOR: does ANY `transpose: true` arrangement
    // reproduce column attention through this class?
    anyTransposeTrueAgrees: agreeing.some((row) => row.transpose === true),
    best: table[0],
    worstAgreeing: agreeing.length === 0 ? null : agreeing[agreeing.length - 1],
  };
}
