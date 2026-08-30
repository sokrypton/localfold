import { float32ToFloat16Array } from "../runtime/float16.js";

const ORDER = [
  "layerNormInWeight", "layerNormInBias",
  "linearAPWeight", "linearAPBias", "linearAGWeight", "linearAGBias",
  "linearBPWeight", "linearBPBias", "linearBGWeight", "linearBGBias",
  "layerNormOutWeight", "layerNormOutBias",
  "linearZWeight", "linearZBias", "linearGWeight", "linearGBias",
];

export function packWeights(weights, precision) {
  const offsets = {};
  let elementCount = 0;
  for (const name of ORDER) {
    offsets[name] = elementCount;
    elementCount += weights[name].length;
  }

  const f32 = new Float32Array(elementCount);
  for (const name of ORDER) f32.set(weights[name], offsets[name]);
  return {
    data: precision === "f16" ? float32ToFloat16Array(f32) : f32,
    offsets,
  };
}

export function expectedWeightElementCount(shape) {
  const { cZ, cHidden } = shape;
  return 2 * cZ + 4 * (cHidden * cZ + cHidden) + 2 * cHidden
    + (cZ * cHidden + cZ) + (cZ * cZ + cZ);
}
