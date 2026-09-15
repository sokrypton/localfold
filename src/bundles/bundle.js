const WEIGHT_NAMES = [
  "layerNormInWeight", "layerNormInBias", "linearAPWeight", "linearAPBias",
  "linearAGWeight", "linearAGBias", "linearBPWeight", "linearBPBias",
  "linearBGWeight", "linearBGBias", "layerNormOutWeight", "layerNormOutBias",
  "linearZWeight", "linearZBias", "linearGWeight", "linearGBias",
];

export async function loadTriangleReferenceBundle(manifestUrl) {
  const response = await fetch(manifestUrl);
  if (!response.ok) throw new Error(`failed to load ${manifestUrl}: ${response.status}`);
  const manifest = await response.json();
  if (manifest.formatVersion !== 1 || manifest.operator !== "TriangleMultiplicationOutgoing") {
    throw new Error("unsupported triangle reference manifest");
  }
  const loadTensor = async(name) => {
    const record = manifest.tensors[name];
    if (record === undefined || record.dtype !== "float32") throw new Error(`missing float32 tensor ${name}`);
    const tensorResponse = await fetch(new URL(record.file, manifestUrl));
    if (!tensorResponse.ok) throw new Error(`failed to load tensor ${name}: ${tensorResponse.status}`);
    const buffer = await tensorResponse.arrayBuffer();
    const expectedElements = record.shape.reduce((product, value) => product * value, 1);
    if (buffer.byteLength !== expectedElements * 4) {
      throw new Error(`${name} has ${buffer.byteLength} bytes; expected ${expectedElements * 4}`);
    }
    return new Float32Array(buffer);
  };

  const weights = {};
  await Promise.all(WEIGHT_NAMES.map(async(name) => { weights[name] = await loadTensor(name); }));
  return {
    source: manifest.source,
    input: {
      shape: manifest.shape,
      epsilon: manifest.epsilon,
      z: await loadTensor("z"),
      mask: await loadTensor("mask"),
      weights,
    },
    expected: await loadTensor("expected"),
  };
}
