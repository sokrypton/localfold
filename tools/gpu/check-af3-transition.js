/**
 * AF3's transition block: GPU against src/af3/pairformer-reference.js.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-af3-transition.js
 *     node tools/gpu-chrome.mjs tools/gpu/check-af3-transition.js --track=single
 *
 * Real trunk weights, both tracks. The pair track is 128 channels widened by 4,
 * the single track 384 by 4, and they exercise different corners of the kernel:
 * the pair track's channel count equals the workgroup size, so its strided
 * loops run exactly one iteration, and the single track's does not.
 */
import { transition } from "../../src/af3/pairformer-reference.js";
import { Af3TransitionGpu } from "../../src/af3/transition-webgpu.js";
import { HttpTensorStore } from "../../src/reference/http-tensor-store.js";

const MANIFEST = "/model-af3-full-f32/manifest.json";
const STACK = "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer";
const TRACKS = {
  pair: { leaf: "pair_transition", channels: 128 },
  single: { leaf: "single_transition", channels: 384 },
};

function option(args, name, fallback) {
  const prefix = `--${name}=`;
  return args.find((argument) => argument.startsWith(prefix))?.slice(prefix.length) ?? fallback;
}

function deterministic(length, seed) {
  let state = seed >>> 0;
  const output = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    output[index] = (((value ^ (value >>> 14)) >>> 0) / 0x1_0000_0000) * 2 - 1;
  }
  return output;
}

function relativeRms(actual, expected) {
  let error = 0;
  let scale = 0;
  for (let index = 0; index < expected.length; index += 1) {
    const difference = actual[index] - expected[index];
    error += difference * difference;
    scale += expected[index] * expected[index];
  }
  return Math.sqrt(error / Math.max(scale, 1e-30));
}

export async function main(device, args) {
  const rows = Number(option(args, "rows", "512"));
  const block = Number(option(args, "block", "0"));
  const wanted = option(args, "track", "pair,single").split(",");
  const store = await HttpTensorStore.open(MANIFEST);
  const runner = new Af3TransitionGpu(device);

  const layer = async (leaf) => {
    const name = `${STACK}/${leaf}`;
    const whole = await store.tensor(name);
    const stride = whole.length / store.shape(name)[0];
    return whole.subarray(block * stride, (block + 1) * stride);
  };

  const results = {};
  let worst = 0;
  for (const track of wanted) {
    const { leaf, channels } = TRACKS[track];
    const weights = {
      inputLayerNormScale: await layer(`${leaf}/input_layer_norm/scale`),
      inputLayerNormOffset: await layer(`${leaf}/input_layer_norm/offset`),
      transition1: await layer(`${leaf}/transition1/weights`),
      transition2: await layer(`${leaf}/transition2/weights`),
    };
    const input = deterministic(rows * channels, 31337 + channels);
    const expected = transition(input, rows, channels, weights, 4);
    const { output, elapsedMilliseconds, memory } = await runner.run(
      input, { rows, channels, factor: 4 }, weights);
    const relRms = relativeRms(output, expected);
    worst = Math.max(worst, relRms);
    results[track] = { relRms, ms: Number(elapsedMilliseconds.toFixed(2)),
                       peakMiB: Number((memory.peakBytes / 2 ** 20).toFixed(2)) };
    console.log(`${track}\tC=${channels}\trelRMS ${relRms.toExponential(2)}`
      + `\t${elapsedMilliseconds.toFixed(1)} ms`
      + `\t${(memory.peakBytes / 2 ** 20).toFixed(1)} MiB`);
  }

  // A wrong split convention or a swapped gate/value half lands near 1, not
  // near tolerance.
  const bound = 1e-5;
  if (worst > bound) throw new Error(`relRMS ${worst.toExponential(2)} exceeds ${bound}`);
  return { rows, block, results };
}
