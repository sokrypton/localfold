const positiveInteger = (name, value) => {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`${name} must be a positive safe integer; received ${value}`);
  }
};

const expectLength = (name, value, length) => {
  if (value.length !== length) {
    throw new RangeError(`${name} has ${value.length} values; expected ${length}`);
  }
};

export function validateTriangleInput(input) {
  const { length, cZ, cHidden } = input.shape;
  positiveInteger("length", length);
  positiveInteger("cZ", cZ);
  positiveInteger("cHidden", cHidden);

  const pairCount = length * length;
  if (!Number.isSafeInteger(pairCount * Math.max(cZ, cHidden))) {
    throw new RangeError("tensor dimensions exceed JavaScript's safe integer range");
  }

  expectLength("z", input.z, pairCount * cZ);
  expectLength("mask", input.mask, pairCount);

  const w = input.weights;
  expectLength("layerNormInWeight", w.layerNormInWeight, cZ);
  expectLength("layerNormInBias", w.layerNormInBias, cZ);
  expectLength("linearAPWeight", w.linearAPWeight, cHidden * cZ);
  expectLength("linearAPBias", w.linearAPBias, cHidden);
  expectLength("linearAGWeight", w.linearAGWeight, cHidden * cZ);
  expectLength("linearAGBias", w.linearAGBias, cHidden);
  expectLength("linearBPWeight", w.linearBPWeight, cHidden * cZ);
  expectLength("linearBPBias", w.linearBPBias, cHidden);
  expectLength("linearBGWeight", w.linearBGWeight, cHidden * cZ);
  expectLength("linearBGBias", w.linearBGBias, cHidden);
  expectLength("layerNormOutWeight", w.layerNormOutWeight, cHidden);
  expectLength("layerNormOutBias", w.layerNormOutBias, cHidden);
  expectLength("linearZWeight", w.linearZWeight, cZ * cHidden);
  expectLength("linearZBias", w.linearZBias, cZ);
  expectLength("linearGWeight", w.linearGWeight, cZ * cZ);
  expectLength("linearGBias", w.linearGBias, cZ);

  if (input.epsilon !== undefined && (!(input.epsilon > 0) || !Number.isFinite(input.epsilon))) {
    throw new RangeError(`epsilon must be finite and positive; received ${input.epsilon}`);
  }
}

export function errorMetrics(actual, expected) {
  if (actual.length !== expected.length || actual.length === 0) {
    throw new RangeError("error metrics require equally sized, non-empty arrays");
  }

  let absoluteSum = 0;
  let squareSum = 0;
  let maximum = 0;
  for (let i = 0; i < actual.length; i += 1) {
    const delta = Math.abs(actual[i] - expected[i]);
    absoluteSum += delta;
    squareSum += delta * delta;
    maximum = Math.max(maximum, delta);
  }
  return {
    meanAbsoluteError: absoluteSum / actual.length,
    maxAbsoluteError: maximum,
    rootMeanSquareError: Math.sqrt(squareSum / actual.length),
  };
}
