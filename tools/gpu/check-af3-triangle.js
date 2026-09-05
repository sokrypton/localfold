/**
 * AF3 triangle multiplication: GPU against src/af3/pairformer-reference.js.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-af3-triangle.js
 *     node tools/gpu-chrome.mjs tools/gpu/check-af3-triangle.js --n=48 --block=17
 *
 * The first differential test of the AF3 port. It runs REAL trunk weights -
 * block `--block` of the 48 - against a deterministic pair representation, in
 * both directions, and compares to the CPU reference that was itself checked
 * against DeepMind's implementation at 4.7e-7.
 *
 * 🔴 REAL WEIGHTS, RANDOM ACTIVATIONS. The weights have to be real because the
 * bug this is looking for is a layout one - an interleave read as a block, a
 * matrix not transposed - and c_z and c_hidden are both 128, so every wrong
 * layout still has conforming shapes. The activations are random because a real
 * pair representation is not needed to find that, and would drag the whole
 * embedder in.
 */
import { af3TriangleMultiplication } from "../../src/af3/triangle-webgpu.js";
import { triangleMultiplication } from "../../src/af3/pairformer-reference.js";
import { HttpTensorStore } from "../../src/reference/http-tensor-store.js";

const MANIFEST = "/model-af3-full-f32/manifest.json";
const STACK = "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer";
const CHANNELS = 128;

function option(args, name, fallback) {
  const prefix = `--${name}=`;
  return args.find((argument) => argument.startsWith(prefix))?.slice(prefix.length) ?? fallback;
}

/** A deterministic pair representation and mask. */
function deterministicPair(n, channels, seed) {
  let state = seed >>> 0;
  const next = () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 0x1_0000_0000;
  };
  const pair = new Float32Array(n * n * channels);
  for (let index = 0; index < pair.length; index += 1) pair[index] = next() * 2 - 1;
  const mask = new Float32Array(n * n);
  // A ragged mask, so a kernel that ignores masking fails rather than passing
  // on an all-ones input.
  const sequence = new Float32Array(n);
  for (let i = 0; i < n; i += 1) sequence[i] = i < Math.ceil(n * 0.8) ? 1 : 0;
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) mask[i * n + j] = sequence[i] * sequence[j];
  }
  return { pair, mask };
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
  const n = Number(option(args, "n", "32"));
  const block = Number(option(args, "block", "0"));
  const precisions = option(args, "precision", "f32").split(",");
  // 🔴 --grid-width IS THE ONLY WAY THE z PATH GETS CHECKED. The two
  // projections fold their row tile over y AND z, because n^2 rows over a
  // 32-row tile passes the 65535-per-dimension limit at about 1450 residues -
  // and the contraction is O(n^3), so no CPU reference can follow a
  // differential there. Lowering the width puts group.z > 0 at n=32, against
  // this file's independent reference. `--grid-width=1` is one z slice per
  // tile, which is the most the arithmetic can be asked to survive.
  const projectGridWidth = Number(option(args, "grid-width", "32768"));
  const store = await HttpTensorStore.open(MANIFEST);

  // Each trunk tensor is stacked over the 48 blocks; take one layer's slice.
  const layer = async (leaf) => {
    const name = `${STACK}/${leaf}`;
    const whole = await store.tensor(name);
    const shape = store.shape(name);
    if (shape[0] !== 48) throw new Error(`${name} is not stacked over 48 blocks`);
    const stride = whole.length / shape[0];
    return whole.subarray(block * stride, (block + 1) * stride);
  };

  const results = {};
  let worst = 0;
  for (const direction of ["outgoing", "incoming"]) {
    const at = (leaf) => layer(`triangle_multiplication_${direction}/${leaf}`);
    const weights = {
      leftNormInputScale: await at("left_norm_input/scale"),
      leftNormInputOffset: await at("left_norm_input/offset"),
      projection: await at("projection/weights"),
      gate: await at("gate/weights"),
      centerNormScale: await at("center_norm/scale"),
      centerNormOffset: await at("center_norm/offset"),
      outputProjection: await at("output_projection/weights"),
      gatingLinear: await at("gating_linear/weights"),
    };
    const { pair, mask } = deterministicPair(n, CHANNELS, 20260831 + block);
    const expected = triangleMultiplication(pair, mask, n, CHANNELS, direction, weights);

    for (const precision of precisions) {
      if (precision === "f16" && !device.features.has("shader-f16")) continue;
      const { output } = await af3TriangleMultiplication(
        device, pair, mask, n, CHANNELS, direction, weights,
        { precision, projectGridWidth });
      const relRms = relativeRms(output, expected);
      worst = Math.max(worst, precision === "f32" ? relRms : 0);
      results[`${direction}/${precision}`] = relRms;
      console.log(`${direction}\t${precision}\trelRMS ${relRms.toExponential(2)}`);
    }
  }

  // f32 should be at rounding. Anything looser is a layout bug, not precision:
  // a wrong transpose or a blocked-instead-of-interleaved split lands near 1.
  const bound = 1e-5;
  if (worst > bound) throw new Error(`f32 relRMS ${worst.toExponential(2)} exceeds ${bound}`);
  return { n, block, channels: CHANNELS, results };
}
