// The distogram's bins - out of src/af3/trunk/trunk-webgpu.js (which re-exports them), so the featuriser's contact
// classes and the CUDA port's exporter take them without the WebGPU trunk behind them.
export const NUM_BINS = 64;
const FIRST_BREAK = 2.3125;
const LAST_BREAK = 21.6875;

/** The distogram bin edges: 63 of them, evenly spaced. */
export function binEdges(bins = NUM_BINS) {
  const breaks = new Float32Array(bins - 1);
  for (let index = 0; index < bins - 1; index += 1) {
    breaks[index] = FIRST_BREAK + (LAST_BREAK - FIRST_BREAK) * index / (bins - 2);
  }
  return breaks;
}
