/**
 * The five pair updates, shared by AF3's two stacks.
 *
 *     pair += triangle_multiplication_outgoing
 *     pair += triangle_multiplication_incoming
 *     pair += grid_self_attention(row)
 *     pair += grid_self_attention(column)
 *     pair += transition
 *
 * The MSA stack and the pairformer stack run exactly this, at the same shapes,
 * with different weights - the MSA stack adds an outer product mean and two MSA
 * updates around it, and the pairformer stack adds the single track. So the
 * ORDER lives here once rather than in both, which matters because the order is
 * the part that is silently wrong when it is wrong: every one of these returns a
 * pair-shaped tensor, so a stack that runs them in the wrong sequence converges
 * to something plausible.
 *
 * 🔴 EACH UPDATE READS THE PAIR AS THE PREVIOUS ONE LEFT IT. They are not five
 * deltas against a common input to be summed at the end. Batching them that way
 * is a natural-looking optimisation and a different function.
 */
import { createTriangleShaders } from "../triangle/shaders.js";
import { packWeights as packTriangleWeights } from "../triangle/weights.js";
import { af3TriangleWeights } from "./triangle-webgpu.js";
import { createGridAttentionShaders, packGridAttentionWeights, PROJECT_ROWS }
  from "./grid-attention-webgpu.js";
import { createTransitionShader, packTransitionWeights } from "./transition-webgpu.js";

export const PAIR_CHANNELS = 128;
export const GRID_WIDTH = 32_768;

/** `accumulator += delta`, elementwise. The residual chain. */
export function createAddShader(elements) {
  return `
const ELEMENTS: u32 = ${elements}u;
const GRID_WIDTH: u32 = ${GRID_WIDTH}u;
@group(0) @binding(0) var<storage, read_write> accumulator: array<f32>;
@group(0) @binding(1) var<storage, read> delta: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index >= ELEMENTS) { return; }
  accumulator[index] = accumulator[index] + delta[index];
}`;
}

/**
 * Compile the pair track's pipelines. Every block has the same shapes, so this
 * runs once per stack and the per-block work is a weight upload.
 *
 * @param {{get: (key: string, source: string) => Promise<GPUComputePipeline>}} cache
 * @param {object} options `sample` is any one block's weights, read only for
 *   its shapes and packing offsets.
 */
export async function compilePairTrack(cache, options) {
  const { n, sample, epsilon, variance, dialect, base } = options;
  // 🔴 THE TEMPLATE STACK IS THIS TRACK AT 64 CHANNELS WITH A FACTOR-2
  // TRANSITION, where the trunk runs 128 and factor 4. Both are "a pairformer
  // block"; only the weight shapes say which, so a wrong factor reads
  // transition1 at the wrong stride rather than failing.
  const channels = options.channels ?? PAIR_CHANNELS;
  const transitionFactor = options.transitionFactor ?? 4;
  const pairs = n * n;
  const shape = { length: n, cZ: channels, cHidden: channels };
  const triangleOffsets = packTriangleWeights(
    af3TriangleWeights(sample.triangleMultiplicationOutgoing, channels), "f32").offsets;
  const gridOffsets = packGridAttentionWeights(sample.pairAttention1).offsets;
  const transitionOffsets = packTransitionWeights(sample.pairTransition).offsets;

  const pipelines = {};
  for (const direction of ["outgoing", "incoming"]) {
    const sources = createTriangleShaders(
      shape, "f32", triangleOffsets, epsilon, direction, variance);
    for (const [name, source] of Object.entries(sources)) {
      pipelines[`tri:${direction}:${name}`] =
        await cache.get(`${base}:tri:${direction}:${name}`, source);
    }
  }
  for (const [key, attention, transpose] of
       [["false", sample.pairAttention1, false], ["true", sample.pairAttention2, true]]) {
    const sources = createGridAttentionShaders(
      { n, channels, heads: attention.heads, dimension: attention.dimension, transpose },
      gridOffsets, epsilon, variance, dialect);
    for (const [name, source] of Object.entries(sources)) {
      pipelines[`grid:${key}:${name}`] = await cache.get(`${base}:grid:${key}:${name}`, source);
    }
  }
  pipelines.pairTransition = await cache.get(`${base}:pair-transition`,
    createTransitionShader({ rows: pairs, channels, factor: transitionFactor },
                           transitionOffsets, epsilon, variance));
  pipelines.addPair = await cache.get(`${base}:add-pair`, createAddShader(pairs * channels));
  return pipelines;
}

/** Pack one block's pair-track weights, ready to upload. */
export function packPairTrackWeights(block, channels = PAIR_CHANNELS) {
  return {
    outgoing: packTriangleWeights(
      af3TriangleWeights(block.triangleMultiplicationOutgoing, channels), "f32").data,
    incoming: packTriangleWeights(
      af3TriangleWeights(block.triangleMultiplicationIncoming, channels), "f32").data,
    grid1: packGridAttentionWeights(block.pairAttention1).data,
    grid2: packGridAttentionWeights(block.pairAttention2).data,
    transition: packTransitionWeights(block.pairTransition).data,
  };
}

/**
 * Record the five pair updates into an open command encoder.
 *
 * @param {object} context `run(label, pipeline, buffers, x, y, z)` records one
 *   pass; `scratch` is seven pair-sized buffers, reused by every operation.
 */
export function encodePairTrack(context) {
  const { run, pipelines, n, gridHeads, pair, pairMask, scratch, biasBuffer, weights } = context;
  const channels = context.channels ?? PAIR_CHANNELS;
  const pairs = n * n;
  const spread = (groups) => [Math.min(groups, GRID_WIDTH), Math.ceil(groups / GRID_WIDTH)];
  const ceil = (value, divisor) => Math.ceil(value / divisor);
  const addGroups = spread(ceil(pairs * channels, 64));
  const addPair = (delta) =>
    run("add", pipelines.addPair, [pair, delta], addGroups[0], addGroups[1]);

  for (const direction of ["outgoing", "incoming"]) {
    const w = weights[direction];
    const p = (name) => pipelines[`tri:${direction}:${name}`];
    run("tri.normalize", p("normalizeInput"), [pair, w, scratch[0]], ceil(pairs, 64));
    run("tri.project", p("projectAB"), [scratch[0], pairMask, w, scratch[1], scratch[2]],
        ceil(channels, 16), ceil(pairs, 16));
    run("tri.contract", p("contract"), [scratch[1], scratch[2], scratch[3]],
        ceil(n, 8), ceil(n, 8), channels);
    run("tri.normalize-hidden", p("normalizeHidden"), [scratch[3], w, scratch[4]], ceil(pairs, 64));
    run("tri.project-out", p("projectOutput"), [scratch[0], scratch[4], w, scratch[5]],
        ceil(channels, 16), ceil(pairs, 16));
    addPair(scratch[5]);
  }

  for (const [key, w] of [["false", weights.grid1], ["true", weights.grid2]]) {
    const p = (name) => pipelines[`grid:${key}:${name}`];
    const linear = spread(ceil(pairs, 64));
    run("grid.normalize", p("normalize"), [pair, w, scratch[0]], linear[0], linear[1]);
    run("grid.bias", p("bias"), [scratch[0], w, biasBuffer], linear[0], linear[1]);
    const perPair = spread(pairs);
    // One workgroup per tile of PROJECT_ROWS pair rows - see the kernel.
    const perTile = spread(ceil(pairs, PROJECT_ROWS));
    run("grid.project", p("project"),
        [scratch[0], w, scratch[1], scratch[2], scratch[3], scratch[4]], perTile[0], perTile[1]);
    // 🔴 THE GRID TRACK'S HEAD COUNT, not the single track's. They differ, 4
    // against 16, and the shader's bounds check makes the wrong one correct but
    // oversubscribed.
    //
    // One thread per (query, row, head), which is ceil(N/64) x N x heads
    // workgroups - see the note on the attend kernel.
    run("grid.attend", p("attend"),
        [scratch[1], scratch[2], scratch[3], biasBuffer, pairMask, scratch[5]],
        ceil(n, 64), n, gridHeads);
    run("grid.project-out", p("project_out"), [scratch[5], scratch[4], w, scratch[6]],
        perPair[0], perPair[1]);
    addPair(scratch[6]);
  }

  const perPair = spread(pairs);
  run("pair-transition", pipelines.pairTransition, [pair, weights.transition, scratch[0]],
      perPair[0], perPair[1]);
  addPair(scratch[0]);
}
