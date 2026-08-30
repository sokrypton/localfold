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

export function createDeterministicTriangleInput(
  shape,
  seed = 7,
) {
  const random = generator(seed);
  const values = (length, scale, center = 0) => {
    const output = new Float32Array(length);
    for (let i = 0; i < length; i += 1) output[i] = center + (random() * 2 - 1) * scale;
    return output;
  };
  const { length, cZ, cHidden } = shape;
  const pairs = length * length;
  const weights = {
    layerNormInWeight: values(cZ, 0.1, 1),
    layerNormInBias: values(cZ, 0.05),
    linearAPWeight: values(cHidden * cZ, 0.15 / Math.sqrt(cZ)),
    linearAPBias: values(cHidden, 0.03),
    linearAGWeight: values(cHidden * cZ, 0.15 / Math.sqrt(cZ)),
    linearAGBias: values(cHidden, 0.03, 1),
    linearBPWeight: values(cHidden * cZ, 0.15 / Math.sqrt(cZ)),
    linearBPBias: values(cHidden, 0.03),
    linearBGWeight: values(cHidden * cZ, 0.15 / Math.sqrt(cZ)),
    linearBGBias: values(cHidden, 0.03, 1),
    layerNormOutWeight: values(cHidden, 0.1, 1),
    layerNormOutBias: values(cHidden, 0.05),
    linearZWeight: values(cZ * cHidden, 0.15 / Math.sqrt(cHidden)),
    linearZBias: values(cZ, 0.03),
    linearGWeight: values(cZ * cZ, 0.15 / Math.sqrt(cZ)),
    linearGBias: values(cZ, 0.03, 1),
  };
  const mask = new Float32Array(pairs);
  for (let i = 0; i < pairs; i += 1) mask[i] = random() > 0.15 ? 1 : 0;
  return { shape, z: values(pairs * cZ, 1), mask, weights };
}
