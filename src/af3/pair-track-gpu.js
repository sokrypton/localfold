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
import { LINEAR_GRID_WIDTH, createTriangleShaders } from "../triangle/shaders.js";
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
  // 🔴 THE SCRATCH LAYOUT IS THIS STACK'S, NOT THE MODULE'S. See
  // PAIR_SCRATCH_STORAGE for what packing buys and UNPACKED_PAIR_SCRATCH for
  // the stack that measured it as a bad trade.
  const scratchStorage = options.scratchStorage ?? UNPACKED_PAIR_SCRATCH;
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
  // 🔴 THE TRIANGLE PROJECTION'S ACCUMULATORS, WHICH ARE A THIRD FORMAT AGAIN.
  // It holds eight vec4 in a WGSL array - the thing a driver spills first - and
  // in f16 they are half that. Worth 1.55x on the kernel at the tile it already
  // had (bench-triangle-project.js at 118 tokens: 1.688 -> 1.087 ms), and
  // tri.project WAS 13% of the trunk's GPU time before this and is 10% after,
  // which is the point rather than a correction.
  const accumulatePrecision = options.accumulatePrecision ?? "f32";
  const shape = {
    length: n, cZ: channels, cHidden: channels, weightPrecision, accumulatePrecision,
  };
  const triangleOffsets = packTriangleWeights(
    af3TriangleWeights(sample.triangleMultiplicationOutgoing, channels),
    weightPrecision).offsets;
  const gridOffsets = packGridAttentionWeights(sample.pairAttention1).offsets;
  const transitionOffsets = packTransitionWeights(sample.pairTransition).offsets;

  const pipelines = {};
  // 🔴 COMPILED CONCURRENTLY, NOT ONE AT A TIME. `createComputePipelineAsync`
  // runs off the main thread, so awaiting each of this track's ~20 shaders in
  // turn serialises compilations that overlap for free. It is paid on the
  // trunk's FIRST pass, which bench-trunk.js reports at 588 ms against a steady
  // 379. The cache stores the promise, so a key asked for twice is still one
  // compilation.
  const pending = [];
  const compileInto = (slot, key, source) => {
    pending.push(cache.get(key, source).then((pipeline) => { pipelines[slot] = pipeline; }));
  };
  for (const direction of ["outgoing", "incoming"]) {
    // 🔴 THE RESIDUAL FORM, so project-out adds into the pair representation
    // rather than writing a delta for a separate add pass to fold in. All five
    // of this track's updates do that now; see the note in
    // src/af3/transition-webgpu.js for what the add pass was costing.
    const { projectTile, contractTile, normalizeRows, projectGridWidth, ...sources } = createTriangleShaders(
      shape, "f32", triangleOffsets, epsilon, direction, variance, undefined, true,
      undefined,
      { normalized: scratchStorage[0], hidden: scratchStorage[4],
        // a is scratch[1] and b is scratch[2]; they share one storage because
        // the incoming direction reads them the other way round.
        ab: scratchStorage[1] });
    // 🔴 THE PROJECTION TILE TRAVELS WITH THE SHADERS. encodePairTrack divides
    // its dispatch by exactly this, so the two cannot drift apart the way a
    // constant repeated in both places did once - see src/triangle/shaders.js.
    pipelines.projectTile = projectTile;
    pipelines.projectGridWidth = projectGridWidth;
    pipelines.normalizeRows = normalizeRows;
    pipelines.contractTile = contractTile;
    for (const [name, source] of Object.entries(sources)) {
      compileInto(`tri:${direction}:${name}`,
                  `${base}:tri:${direction}:${weightPrecision}:${accumulatePrecision}`
                  + `:${scratchStorage.join("")}:${name}`,
                  source);
    }
  }
  for (const [key, attention, transpose] of
       [["false", sample.pairAttention1, false], ["true", sample.pairAttention2, true]]) {
    const { tiles, ...sources } = createGridAttentionShaders(
      { n, channels, heads: attention.heads, dimension: attention.dimension, transpose,
        residual: true, stagedPrecision },
      gridOffsets, epsilon, variance, dialect,
      // 🔴 THE ATTENTION WRITES BACK INTO `normalized`. See encodePairTrack:
      // `grid.project` is the last pass that reads scratch[0], and it runs
      // before `grid.attend` writes it, so the two can share one tensor - and
      // at 300 tokens that is 43.9 MiB of the largest tensor group a trunk
      // holds. They are one storage now because they are one buffer.
      scratchStorage[0], scratchStorage[0],
      // q, k, v and the gate are scratch 1, 2, 3 and 4 in that order.
      { q: scratchStorage[1], k: scratchStorage[2],
        v: scratchStorage[3], gate: scratchStorage[4] });
    pipelines.gridTiles = tiles;
    for (const [name, source] of Object.entries(sources)) {
      compileInto(`grid:${key}:${name}`,
                  `${base}:grid:${key}:${stagedPrecision}`
                  + `:${scratchStorage.join("")}:${name}`,
                  source);
    }
  }
  // The transition stages two blocks of its own - the layer-normed rows and the
  // gated intermediate - and narrowing them is the same trade as the attention
  // tile above, on the largest kernel in the trunk. See transition-webgpu.js.
  compileInto("pairTransition",
    `${base}:pair-transition:${stagedPrecision}:${weightPrecision}`,
    createTransitionShader(
      // 🔴 THE TRANSITION'S RUNNING SUM STAYS f32, WHERE THE TRIANGLE'S DOES
      // NOT, AND THE DIFFERENCE IS THE RATIO. Narrowing it measures 3.938 ->
      // 3.769 ms on bench-transition.js at 118 tokens - 4.3% of a kernel that
      // is 18% of the trunk, so 0.8% - and takes the kernel's own relRMS from
      // 3.55e-4 to 3.33e-3. Ten times the error for eight tenths of a percent
      // is the wrong side of the trade; the triangle's two projections are
      // 1.55x and 1.43x for the same class of change. The option is still in
      // the shader and the bench still reaches it (`@f16+f16`), which is where
      // that measurement lives.
      { rows: pairs, channels, factor: transitionFactor, residual: true,
        stagePrecision: stagedPrecision, weightPrecision },
      transitionOffsets, epsilon, variance));
  // 🔴 STILL ONE ADD PASS, and it belongs to the MSA stack rather than to this
  // track: the outer product mean is the one producer whose kernel does not
  // write the pair representation itself. See msa-stack-webgpu.js's "opm.add".
  compileInto("addPair", `${base}:add-pair`, createAddShader(pairs * channels));
  await Promise.all(pending);
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
/**
 * How each of a pair-track stack's six pair-sized scratch buffers is stored,
 * PACKED - two f16 halves to a word, for five of the six.
 *
 * 🔴 NOTHING USES THIS ANY MORE, AND WHAT IT COST IS WHY. It landed as a
 * memory optimisation and was measured on the pairformer's own differential
 * checker, which passes either way. Every OTHER checker that reaches a pair
 * track was over its bound the whole time, and nobody ran them:
 *
 * | | packed | unpacked | bound |
 * |---|---|---|---|
 * | `check-af3-confidence` stack pair | 3.71e-3 | **3.12e-6** | |
 * | ...its PAE head | 2.88e-3 | **5.75e-6** | 7.1x envelope |
 * | ...its PDE head | 3.29e-3 | **7.47e-6** | |
 * | ...its pLDDT head | 6.88e-4 | **1.16e-4** | |
 * | `check-af3-msa-block` pair | 1.82e-3 | **7.16e-6** | 1e-5 |
 * | `check-af3-template` | 3.79e-5 | **2.52e-7** | 2e-5 |
 * | `check-af3-trunk` pair | 1.04e-4 | **2.18e-5** | 4e-5 |
 *
 * A factor of 1200 on the pair representation that feeds pLDDT and PAE. The
 * confidence head is where it shows because its four blocks amplify and its
 * heads have the tightest envelopes in the repository; the trunk's own checker
 * at n=24 barely moves, which is exactly why one checker is not enough.
 *
 * 🔴 AND HALF-PACKING IS NOT A COMPROMISE, IT IS A SMALLER VERSION OF THE SAME
 * FAULT. Bisected: `normalized` and the triangle's `a` and `b` are where it
 * hurts - a and b are MULTIPLIED against each other in the contraction, so
 * their rounding squares - while `hidden` and grid attention's output cost
 * nothing measurable anywhere. Packing only those two still fails the
 * confidence head's PAE and PDE (9.48e-4 and 1.12e-3) and still misses the MSA
 * block's bound by 50x.
 *
 * It is kept, exported and unused so that a caller who needs the memory more
 * than the accuracy can ask compilePairTrack for it and know what it buys.
 * See src/runtime/storage.js for what a packed word costs and for the rule
 * that one invocation must own both of its halves.
 *
 * 🔴 THERE ARE SIX OF THESE AND THERE WERE SEVEN. Nothing in the repository
 * ever read `scratch[6]`: encodePairTrack indexes 0 to 5, and so does every
 * caller. All three stacks allocated it anyway - a pair-sized tensor per
 * stack, 19.5 MiB in the MSA stack at 200 tokens and 10.2 in the template,
 * held for the length of a pass and touched by nothing.
 *
 * 🔴 AND THERE ARE FIVE NOW, NOT SIX. The sixth was the grid attention's
 * output, and `grid.project` - the last pass that reads scratch[0] - runs
 * before the pass that writes it, so the two share one tensor. That is a sixth
 * of the largest tensor group an AF3 trunk holds: 43.9 MiB at 300 tokens.
 */
export const PAIR_SCRATCH_STORAGE = ["f16", "f16", "f16", "f32", "f16"];

/**
 * What every stack actually uses: one word an element.
 *
 * 🔴 ONE STATEMENT OF THE LAYOUT, because two would be a buffer of the right
 * element count and the wrong byte length - which nothing validates and
 * nothing throws on. The allocation reads this and so does every shader that
 * touches the buffer.
 */
export const UNPACKED_PAIR_SCRATCH = ["f32", "f32", "f32", "f32", "f32"];

/** How many pair-sized scratch tensors a pair-track stack needs. */
export const PAIR_SCRATCH_COUNT = PAIR_SCRATCH_STORAGE.length;

/**
 * Record the five pair updates into an open command encoder.
 *
 * @param {object} context `run(label, pipeline, buffers, x, y, z)` records one
 *   pass; `scratch` is five pair-sized buffers, reused by every operation.
 */
export function encodePairTrack(context) {
  const { run, pipelines, n, gridHeads, pair, pairMask, scratch, biasBuffer, weights } = context;
  const channels = context.channels ?? PAIR_CHANNELS;
  const pairs = n * n;
  const spread = (groups) => [Math.min(groups, GRID_WIDTH), Math.ceil(groups / GRID_WIDTH)];
  // 🔴 THE TRIANGLE KERNELS FOLD AT THEIR OWN WIDTH, NOT THIS FILE'S. They
  // happen to be the same number, and agreeing by coincidence is how a caller
  // silently addresses rows that do not exist. See LINEAR_GRID_WIDTH.
  const triangleWidth = pipelines.projectGridWidth ?? LINEAR_GRID_WIDTH;
  const spreadTriangle = (groups) =>
    [Math.min(groups, triangleWidth), Math.ceil(groups / triangleWidth)];
  const ceil = (value, divisor) => Math.ceil(value / divisor);

  /**
   * The rows of the pair this track processes at a time.
   *
   * 🔴 THE SCRATCH IS 62% OF A LARGE FOLD'S PEAK AND HAD NO CHEAPER ROUTE.
   * Five pair-sized buffers is 5987 MiB at 1530 tokens, of a 9662 MiB fold -
   * and the only thing the budget's retry could give up was WEIGHT residency,
   * which is about 567 MiB and does not grow with the protein. Chunking is the
   * route that does: everything here except the contraction's `b` and the
   * normalised input needs only the rows it is working on.
   *
   * 🔴 AND IT NEEDS NO KERNEL CHANGES, WHICH IS WHY IT IS WORTH DOING. Every
   * pair-shaped buffer is indexed row-major, so rows [r0, r1) are a contiguous
   * byte range and binding that SLICE makes the shader's own indexing address
   * the chunk. A pair row is `n * channels * 4` bytes, always a multiple of the
   * 256-byte binding alignment, so the offsets are always legal.
   *
   * Defaults to the whole track, which emits exactly the passes it always did.
   */
  const rowChunk = Math.min(context.rowChunk ?? n, n);
  if (!Number.isInteger(rowChunk) || rowChunk < 1) {
    throw new RangeError(`rowChunk ${context.rowChunk} is not a positive integer`);
  }
  /** Rows [from, from + rows) of a pair-shaped buffer of `width` channels. */
  const slice = (allocation, from, rows, width) => (from === 0 && rows === n
    ? allocation
    : { buffer: allocation.buffer,
        byteOffset: from * n * width * 4,
        byteSize: rows * n * width * 4 });
  const chunks = [];
  for (let from = 0; from < n; from += rowChunk) {
    chunks.push({ from, rows: Math.min(rowChunk, n - from) });
  }
  for (const direction of ["outgoing", "incoming"]) {
    const w = weights[direction];
    const p = (name) => pipelines[`tri:${direction}:${name}`];
    const perNormalizeTile = spread(ceil(pairs, pipelines.normalizeRows));
    run("tri.normalize", p("normalizeInput"), [pair, w, scratch[0]],
        perNormalizeTile[0], perNormalizeTile[1]);
    // ...rows folded over y and z: x is the channel tile, so the pair rows have
    // nowhere else to go and there are n^2 of them. See the note in the kernel.
    const perProjectTile = spreadTriangle(ceil(pairs, pipelines.projectTile.rows));
    run("tri.project", p("projectAB"), [scratch[0], pairMask, w, scratch[1], scratch[2]],
        ceil(channels, pipelines.projectTile.columns), perProjectTile[0], perProjectTile[1]);
    run("tri.contract", p("contract"), [scratch[1], scratch[2], scratch[3]],
        ceil(n, pipelines.contractTile.columns), ceil(n, pipelines.contractTile.rows), channels);
    run("tri.normalize-hidden", p("normalizeHidden"), [scratch[3], w, scratch[4]],
        perNormalizeTile[0], perNormalizeTile[1]);
    // ...straight into the pair representation, which nothing has read since
    // tri.normalize consumed it into scratch[0].
    run("tri.project-out", p("projectOutput"), [scratch[0], scratch[4], w, pair],
        ceil(channels, pipelines.projectTile.columns), perProjectTile[0], perProjectTile[1]);
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
    // ...into scratch[0], which `grid.project` above was the last pass to
    // read. See the note where the shaders are compiled.
    run("grid.attend", p("attend"),
        [scratch[1], scratch[2], scratch[3], biasBuffer, pairMask, scratch[0]],
        ceil(n, 64), n, gridHeads);
    run("grid.project-out", p("project_out"), [scratch[0], scratch[4], w, pair],
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
