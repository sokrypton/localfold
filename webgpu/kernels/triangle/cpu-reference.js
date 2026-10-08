import { validateTriangleInput } from "./types.js";

const sigmoid = (value) => 1 / (1 + Math.exp(-value));

function normalizeRows(
  input,
  rowCount,
  channels,
  weight,
  bias,
  epsilon,
) {
  const output = new Float32Array(input.length);
  for (let row = 0; row < rowCount; row += 1) {
    const offset = row * channels;
    let mean = 0;
    for (let c = 0; c < channels; c += 1) mean += input[offset + c];
    mean /= channels;
    let variance = 0;
    for (let c = 0; c < channels; c += 1) {
      const centered = input[offset + c] - mean;
      variance += centered * centered;
    }
    const inverseStd = 1 / Math.sqrt(variance / channels + epsilon);
    for (let c = 0; c < channels; c += 1) {
      output[offset + c] = (input[offset + c] - mean) * inverseStd * weight[c] + bias[c];
    }
  }
  return output;
}

function project(
  input,
  rows,
  inChannels,
  outChannels,
  weight,
  bias,
) {
  const output = new Float32Array(rows * outChannels);
  for (let row = 0; row < rows; row += 1) {
    for (let out = 0; out < outChannels; out += 1) {
      let value = bias[out];
      const weightOffset = out * inChannels;
      const inputOffset = row * inChannels;
      for (let c = 0; c < inChannels; c += 1) {
        value += input[inputOffset + c] * weight[weightOffset + c];
      }
      output[row * outChannels + out] = value;
    }
  }
  return output;
}

export function triangleMultiplicationOutgoingReference(input) {
  validateTriangleInput(input);
  const { length, cZ, cHidden } = input.shape;
  const pairs = length * length;
  const epsilon = input.epsilon ?? 1e-5;
  const w = input.weights;
  const z = normalizeRows(input.z, pairs, cZ, w.layerNormInWeight, w.layerNormInBias, epsilon);

  const ap = project(z, pairs, cZ, cHidden, w.linearAPWeight, w.linearAPBias);
  const ag = project(z, pairs, cZ, cHidden, w.linearAGWeight, w.linearAGBias);
  const bp = project(z, pairs, cZ, cHidden, w.linearBPWeight, w.linearBPBias);
  const bg = project(z, pairs, cZ, cHidden, w.linearBGWeight, w.linearBGBias);
  for (let pair = 0; pair < pairs; pair += 1) {
    const mask = input.mask[pair];
    for (let h = 0; h < cHidden; h += 1) {
      const index = pair * cHidden + h;
      ap[index] = ap[index] * sigmoid(ag[index]) * mask;
      bp[index] = bp[index] * sigmoid(bg[index]) * mask;
    }
  }

  const contracted = new Float32Array(pairs * cHidden);
  for (let i = 0; i < length; i += 1) {
    for (let j = 0; j < length; j += 1) {
      for (let h = 0; h < cHidden; h += 1) {
        let value = 0;
        for (let k = 0; k < length; k += 1) {
          value += ap[(i * length + k) * cHidden + h]
            * bp[(j * length + k) * cHidden + h];
        }
        contracted[(i * length + j) * cHidden + h] = value;
      }
    }
  }

  const normalized = normalizeRows(
    contracted, pairs, cHidden, w.layerNormOutWeight, w.layerNormOutBias, epsilon,
  );
  const output = project(normalized, pairs, cHidden, cZ, w.linearZWeight, w.linearZBias);
  const gate = project(z, pairs, cZ, cZ, w.linearGWeight, w.linearGBias);
  for (let index = 0; index < output.length; index += 1) {
    output[index] = output[index] * sigmoid(gate[index]);
  }
  return output;
}
