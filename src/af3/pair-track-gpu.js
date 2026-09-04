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
import { createGridAttentionShaders, packGridAttentionWeights }
  from "./grid-attention-webgpu.js";
import { createTransitionShader, packTransitionWeights, transitionRowTile }
  from "./transition-webgpu.js";

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
  // f16 wherever the device has it; see grid-attention-webgpu.js's staged tile.
  const stagedPrecision = options.stagedPrecision ?? "f32";
  // The element the RESIDENT weight buffers hold. Memory, not time; see the
  // note in pairformer-block-webgpu.js.
  const weightPrecision = options.weightPrecision ?? "f32";
  // 🔴 THE TEMPLATE STACK IS THIS TRACK AT 64 CHANNELS WITH A FACTOR-2
  // TRANSITION, where the trunk runs 128 and factor 4. Both are "a pairformer
  // block"; only the weight shapes say which, so a wrong factor reads
  // transition1 at the wrong stride rather than failing.
  const channels = options.channels ?? PAIR_CHANNELS;
  const transitionFactor = options.transitionFactor ?? 4;
  const pairs = n * n;
  const shape = { length: n, cZ: channels, cHidden: channels, weightPrecision };
  const triangleOffsets = packTriangleWeights(
    af3TriangleWeights(sample.triangleMultiplicationOutgoing, channels),
    weightPrecision).offsets;
  const gridOffsets = packGridAttentionWeights(sample.pairAttention1).offsets;
  const transitionOffsets = packTransitionWeights(sample.pairTransition).offsets;

  const pipelines = {};
  for (const direction of ["outgoing", "incoming"]) {
    // 🔴 THE RESIDUAL FORM, so project-out adds into the pair representation
    // rather than writing a delta for a separate add pass to fold in. All five
    // of this track's updates do that now; see the note in
    // src/af3/transition-webgpu.js for what the add pass was costing.
    const { projectTile, contractTile, normalizeRows, ...sources } = createTriangleShaders(
      shape, "f32", triangleOffsets, epsilon, direction, variance, undefined, true);
    // 🔴 THE PROJECTION TILE TRAVELS WITH THE SHADERS. encodePairTrack divides
    // its dispatch by exactly this, so the two cannot drift apart the way a
    // constant repeated in both places did once - see src/triangle/shaders.js.
    pipelines.projectTile = projectTile;
    pipelines.normalizeRows = normalizeRows;
    pipelines.contractTile = contractTile;
    for (const [name, source] of Object.entries(sources)) {
      pipelines[`tri:${direction}:${name}`] =
        await cache.get(`${base}:tri:${direction}:${weightPrecision}:${name}`, source);
    }
  }
  for (const [key, attention, transpose] of
       [["false", sample.pairAttention1, false], ["true", sample.pairAttention2, true]]) {
    const { tiles, ...sources } = createGridAttentionShaders(
      { n, channels, heads: attention.heads, dimension: attention.dimension, transpose,
        residual: true, stagedPrecision },
      gridOffsets, epsilon, variance, dialect);
    pipelines.gridTiles = tiles;
    for (const [name, source] of Object.entries(sources)) {
      pipelines[`grid:${key}:${name}`] =
        await cache.get(`${base}:grid:${key}:${stagedPrecision}:${name}`, source);
    }
  }
  // The transition stages two blocks of its own - the layer-normed rows and the
  // gated intermediate - and narrowing them is the same trade as the attention
  // tile above, on the largest kernel in the trunk. See transition-webgpu.js.
  pipelines.pairTransition = await cache.get(
    `${base}:pair-transition:${stagedPrecision}:${weightPrecision}`,
    createTransitionShader(
      { rows: pairs, channels, factor: transitionFactor, residual: true,
        stagePrecision: stagedPrecision, weightPrecision },
      transitionOffsets, epsilon, variance));
  // 🔴 STILL ONE ADD PASS, and it belongs to the MSA stack rather than to this
  // track: the outer product mean is the one producer whose kernel does not
  // write the pair representation itself. See msa-stack-webgpu.js's "opm.add".
  pipelines.addPair = await cache.get(`${base}:add-pair`, createAddShader(pairs * channels));
  return pipelines;
}

/** Pack one block's pair-track weights, ready to upload. */
export function packPairTrackWeights(block, channels = PAIR_CHANNELS, weightPrecision = "f32") {
  return {
    outgoing: packTriangleWeights(
      af3TriangleWeights(block.triangleMultiplicationOutgoing, channels), weightPrecision).data,
    incoming: packTriangleWeights(
      af3TriangleWeights(block.triangleMultiplicationIncoming, channels), weightPrecision).data,
    grid1: packGridAttentionWeights(block.pairAttention1).data,
    grid2: packGridAttentionWeights(block.pairAttention2).data,
    transition: packTransitionWeights(block.pairTransition, weightPrecision).data,
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
  for (const direction of ["outgoing", "incoming"]) {
    const w = weights[direction];
    const p = (name) => pipelines[`tri:${direction}:${name}`];
    const perNormalizeTile = spread(ceil(pairs, pipelines.normalizeRows));
    run("tri.normalize", p("normalizeInput"), [pair, w, scratch[0]],
        perNormalizeTile[0], perNormalizeTile[1]);
    run("tri.project", p("projectAB"), [scratch[0], pairMask, w, scratch[1], scratch[2]],
        ceil(channels, pipelines.projectTile.columns), ceil(pairs, pipelines.projectTile.rows));
    run("tri.contract", p("contract"), [scratch[1], scratch[2], scratch[3]],
        ceil(n, pipelines.contractTile.columns), ceil(n, pipelines.contractTile.rows), channels);
    run("tri.normalize-hidden", p("normalizeHidden"), [scratch[3], w, scratch[4]],
        perNormalizeTile[0], perNormalizeTile[1]);
    // ...straight into the pair representation, which nothing has read since
    // tri.normalize consumed it into scratch[0].
    run("tri.project-out", p("projectOutput"), [scratch[0], scratch[4], w, pair],
        ceil(channels, pipelines.projectTile.columns), ceil(pairs, pipelines.projectTile.rows));
  }

  for (const [key, w] of [["false", weights.grid1], ["true", weights.grid2]]) {
    const p = (name) => pipelines[`grid:${key}:${name}`];
    const linear = spread(ceil(pairs, 64));
    const perNormalize = spread(ceil(pairs, pipelines.gridTiles.normalizeRows));
    run("grid.normalize", p("normalize"), [pair, w, scratch[0]], perNormalize[0], perNormalize[1]);
    run("grid.bias", p("bias"), [scratch[0], w, biasBuffer], linear[0], linear[1]);
    const perOutTile = spread(ceil(pairs, pipelines.gridTiles.projectOutRows));
    // One workgroup per tile of pair rows - see the kernel.
    const perTile = spread(ceil(pairs, pipelines.gridTiles.projectRows));
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
    run("grid.project-out", p("project_out"), [scratch[5], scratch[4], w, pair],
        perOutTile[0], perOutTile[1]);
  }

  // 🔴 A TILE OF PAIRS A WORKGROUP. This was 241 ms of a 632 ms pairformer pass
  // - the largest single kernel in the trunk - because each workgroup read the
  // whole 196k-float weight set for one row.
  const perTransition = spread(Math.ceil(pairs / transitionRowTile(pairs)));
  // ...reads every row it writes into workgroup memory before writing any of
  // them, and no other workgroup touches those rows, so this is in place.
  run("pair-transition", pipelines.pairTransition, [pair, weights.transition],
      perTransition[0], perTransition[1]);
}
