/**
 * WebGL2's pairformer block on a REAL bundle's block - any family, any stack - against cpu/af3/trunk/pairformer.js.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-webgl-af3-block-bundle.js --model=/model-mol1-int5/manifest.json \
 *       --stack=after-mixer [--n=24] [--index=0]
 *
 * `--stack` is `trunk` (the pairformer), `mixer` (Mol-1's PairMixer, no grid attention) or `after-mixer` (Mol-1's
 * narrow pairformer). tools/gpu/check-af3-block-any.js is WebGPU's and reads the trunk scope alone; the synthetic
 * check-webgl-af3-block.js picks its own widths, so neither could see a block of a bundle whose shapes no other bundle
 * has. The bar is f32's: both sides are f32 here.
 */
import {
  gridSelfAttention as cpuGrid, pairformerBlock as cpuBlock, transition as cpuTransition,
  triangleMultiplication as cpuTriangle,
} from "../../cpu/af3/trunk/pairformer.js";
import { triangleMultiplication } from "../../webgl/af3/trunk/triangle.js";
import { gridSelfAttention } from "../../webgl/af3/trunk/grid-attention.js";
import { transition } from "../../webgl/af3/trunk/transition.js";
import { af3Dialect, openAf3Store, pairformerBlockWeights } from "../../shared/af3/weights/weights.js";
import {
  pairformerBlock, releasePairformerBlockWeights, uploadPairformerBlockWeights,
} from "../../webgl/af3/trunk/pairformer-block.js";
import { Execution } from "../../webgl/runtime/execution.js";
import { createWebGL2Context } from "../../webgl/index.js";
import { relativeRms } from "./relative-rms.js";
import { Af3PairformerStackGpu } from "../../webgpu/af3/trunk/pairformer-block.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};
const STACKS = {
  trunk: "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer",
  mixer: "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairmixer",
  "after-mixer": "diffuser/evoformer/__layer_stack_no_per_layer_2/trunk_pairformer",
  // the confidence head's pairformer - Mol-1's is 448 wide with 14 heads, a shape no synthetic arm builds
  confidence: "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer",
};

function deterministic(length, seed) {
  const out = new Float32Array(length);
  let state = seed;
  for (let index = 0; index < length; index += 1) {
    state = (state * 1103515245 + 12345) % 2147483648;
    out[index] = (state / 2147483648 - 0.5) * 2;
  }
  return out;
}

export async function main(device, args) {
  const n = Number(option(args, "n", "24"));
  const index = Number(option(args, "index", "0"));
  const bar = Number(option(args, "bar", "2e-4"));
  const stack = option(args, "stack", "trunk");
  const store = await openAf3Store(option(args, "model", "/model-af3-int5/manifest.json"));
  const dialect = af3Dialect(store);
  const host = await pairformerBlockWeights(store, index, STACKS[stack]);
  const pairChannels = host.pairChannels;
  const singleChannels = host.singleChannels ?? 384;
  const pairs = n * n;
  const state = {
    tokens: n, pair: deterministic(pairs * pairChannels, 101), single: deterministic(n * singleChannels, 202),
    pairMask: new Float32Array(pairs).fill(1), seqMask: new Float32Array(n).fill(1),
  };
  const execution = new Execution(createWebGL2Context());
  const weights = uploadPairformerBlockWeights(execution, host);
  const up = (values, label) => execution.upload(values, label);
  // `--parts`: each of the five pair sub-layers alone, residual included, on the same pair - which of them disagrees
  if (args?.includes("--parts")) {
    const swap = dialect.swapTransposedBias === true;
    const divide = dialect.triangleMulDivideByLength === true;
    const pairT = up(state.pair, "pair");
    const maskT = up(state.pairMask, "pair-mask");
    const plus = (delta) => Float32Array.from(state.pair, (v, i) => v + delta[i]);
    const parts = {
      "tri-out": [() => triangleMultiplication(execution, { pair: pairT, mask: maskT, weights: weights.triangleMultiplicationOutgoing,
        length: n, channels: pairChannels, direction: "outgoing", divideByLength: divide, residualOf: pairT }),
      () => plus(cpuTriangle(state.pair, state.pairMask, n, pairChannels, "outgoing", host.triangleMultiplicationOutgoing, dialect))],
      "tri-in": [() => triangleMultiplication(execution, { pair: pairT, mask: maskT, weights: weights.triangleMultiplicationIncoming,
        length: n, channels: pairChannels, direction: "incoming", divideByLength: divide, residualOf: pairT }),
      () => plus(cpuTriangle(state.pair, state.pairMask, n, pairChannels, "incoming", host.triangleMultiplicationIncoming, dialect))],
      grid1: [() => gridSelfAttention(execution, { pair: pairT, mask: maskT, weights: weights.pairAttention1, length: n,
        channels: pairChannels, transpose: false, swapTransposedBias: swap, residualOf: pairT }),
      () => plus(cpuGrid(state.pair, state.pairMask, n, pairChannels, false, host.pairAttention1, dialect))],
      grid2: [() => gridSelfAttention(execution, { pair: pairT, mask: maskT, weights: weights.pairAttention2, length: n,
        channels: pairChannels, transpose: true, swapTransposedBias: swap, residualOf: pairT }),
      () => plus(cpuGrid(state.pair, state.pairMask, n, pairChannels, true, host.pairAttention2, dialect))],
      transition: [() => transition(execution, { x: pairT, weights: weights.pairTransition, rows: n * n, channels: pairChannels,
        residualOf: pairT }),
      () => plus(cpuTransition(state.pair, n * n, pairChannels, host.pairTransition,
        host.pairTransition.transition1.length / pairChannels / pairChannels / 2))],
    };
    const report = {};
    for (const [name, [ours, theirs]] of Object.entries(parts)) {
      if (name.startsWith("grid") && weights.pairAttention1 === undefined) continue;
      report[name] = Number(relativeRms(Float32Array.from(execution.read(ours())), theirs(), name).toPrecision(4));
    }
    return { stack, pairChannels, parts: report };
  }
  // the CPU block wants plain arrays where the bundle hands it thunks
  const expected = cpuBlock(state, host, dialect);
  const out = pairformerBlock(execution, {
    pair: up(state.pair, "pair"), single: up(state.single, "single"),
    pairMask: up(state.pairMask, "pair-mask"), seqMask: up(state.seqMask, "seq-mask"),
    tokens: n, weights,
    swapTransposedBias: dialect.swapTransposedBias === true,
    divideByLength: dialect.triangleMulDivideByLength === true,
  });
  const oursPair = Float32Array.from(execution.read(out.pair));
  const oursSingle = Float32Array.from(execution.read(out.single));
  const pair = relativeRms(oursPair, expected.pair, "pair");
  const single = relativeRms(oursSingle, expected.single, "single");
  // ...and WebGPU's block in f32 on the same state, the third opinion when the CPU and WebGL2 disagree
  const wgsl = await new Af3PairformerStackGpu(device, {
    stagedPrecision: "f32", accumulatePrecision: "f32", weightPrecision: "f32", pairWeightPrecision: "f32",
  }).run({ ...state }, [host], dialect, {});
  const againstWgsl = {
    pair: Number(relativeRms(oursPair, wgsl.pair, "pair").toPrecision(4)),
    single: Number(relativeRms(oursSingle, wgsl.single, "single").toPrecision(4)),
    wgslAgainstCpuPair: Number(relativeRms(wgsl.pair, expected.pair, "pair").toPrecision(4)),
  };
  releasePairformerBlockWeights(execution, weights);
  const worst = Math.max(pair, single);
  return {
    stack, index, n, pairChannels, singleChannels,
    pair: Number(pair.toPrecision(4)), single: Number(single.toPrecision(4)), againstWgsl,
    verdict: worst <= bar ? `ok: within ${bar}` : `FAILED: relRMS ${worst.toPrecision(4)} over ${bar}`,
  };
}
