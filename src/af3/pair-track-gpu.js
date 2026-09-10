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
import { stagedMatrixStorage } from "../runtime/matrix-linear.js";
import {
  createTriangleProjectMatrixShader, createTriangleProjectOutMatrixShaders,
  createTriangleContractMatrixShader, triangleContractMatrixDispatch,
  triangleProjectMatrixDispatch, triangleProjectOutMatrixDispatch,
  triangleProjectMatrixFits, TRIANGLE_PROJECT_MATRIX_GEOMETRY,
} from "../triangle/project-matrix.js";
import { createGridAttentionShaders, packGridAttentionWeights }
  from "./grid-attention-webgpu.js";
import {
  createGridProjectMatrixShader, createGridProjectOutMatrixShader,
  gridProjectMatrixDispatch, gridProjectMatrixFits, gridProjectOutMatrixDispatch,
} from "./grid-project-matrix.js";
import {
  createTransitionShader, createTransitionSplitShaders, packTransitionWeights,
  transitionRowTile, transitionSplitChunkRows,
} from "./transition-webgpu.js";

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
  // 🔴 THE RESIDENT PAIR WEIGHTS COULD BE f16 AND ARE NOT, AND THE REASON IS
  // MEASURED. These buffers are read one scalar at a time, so halving their
  // bytes buys no bandwidth - this file's own table records -2% on AlphaFold
  // 3's trunk. What it WOULD buy is the peak, and only where the pair track is
  // wide, because these weights go as the SQUARE of the channel count:
  //
  //   OpenDDE, 68 residues   1198.3 -> 873.5 MiB   (-27%)   20.4 -> 20.4 s
  //   AlphaFold 3, the same   476.0 -> 476.0       (  0%)    4.2 ->  4.1
  //
  // AlphaFold 3 does not move because its peak is the diffusion transformer's
  // 378 MiB of resident weights, not the trunk's. Accuracy is unmoved on both:
  // AF3 reads RMSD 0.682 -> 0.683 and pLDDT 85.621 -> 85.639, OpenDDE 1.680 ->
  // 1.681 with pLDDT identical to four decimals.
  //
  // 🔴 AND TURNING IT ON AS A DEFAULT PRODUCES NaN, WHICH IS WHY IT IS STILL
  // OPT-IN. `--pair-weights=f16` reaches the TRUNK's stack only; the default
  // also reaches OpenDDE's structural-token refiner and its confidence head,
  // which build their own `Af3PairformerStackGpu` with no options - and the
  // fold comes back with every coordinate NaN. So the 27% is real and the
  // route to it is not a one-line default. Open: which of those two stacks,
  // and why f16 weights are safe in the trunk and not there.
  //
  // The old comment, still true of the trade itself:
  //
  //   OpenDDE, 68 residues   1198.3 -> 873.5 MiB   (-27%)   20.4 -> 20.4 s
  //   AlphaFold 3, the same   476.0 -> 476.0       ( 0%)     4.2 ->  4.1
  //
  // AlphaFold 3 does not move because its peak is the diffusion transformer's
  // 378 MiB of resident weights, not the trunk's - its pair track is 128
  // channels against OpenDDE's 384, and these weights go as the SQUARE of that.
  // Accuracy is unmoved either way: on 6MRR, AF3 reads RMSD 0.682 -> 0.683 and
  // pLDDT 85.621 -> 85.639, OpenDDE 1.680 -> 1.681 with pLDDT identical.
  const weightPrecision = options.weightPrecision ?? "f32";
  // 🔴 THE TEMPLATE STACK IS THIS TRACK AT 64 CHANNELS WITH A FACTOR-2
  // TRANSITION, where the trunk runs 128 and factor 4. Both are "a pairformer
  // block"; only the weight shapes say which, so a wrong factor reads
  // transition1 at the wrong stride rather than failing.
  // Required for the same reason `encodePairTrack` requires it: these two
  // resolve the SAME number, and a default in either is how they drift.
  const { channels } = options;
  if (!(channels > 0)) {
    throw new Error("compilePairTrack needs the track's channel count");
  }
  // 🔴 THE TRANSITION'S FACTOR IS THE WEIGHTS', NOT A DEFAULT. `transition2` is
  // [hidden, channels], so the factor is `hidden / channels` and every stack
  // states its own: AlphaFold 3's pair transition is 4, the template stack's is
  // 2, and OpenDDE's structural-token REFINER is 2 where its trunk and its
  // confidence head are 4. Defaulting to 4 reads a factor-2 `transition1` at
  // twice its stride - which does not fail, because the buffer is merely
  // shorter than the kernel thinks. Only the template stack ever passed one.
  const derivedFactor = sample.pairTransition?.transition2 === undefined ? undefined
    : sample.pairTransition.transition2.length / (channels * channels);
  const transitionFactor = options.transitionFactor ?? derivedFactor ?? 4;
  if (!Number.isInteger(transitionFactor)) {
    throw new Error(`pair transition factor ${transitionFactor} is not an integer; `
      + `transition2 has ${sample.pairTransition?.transition2?.length} elements `
      + `at ${channels} channels`);
  }
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
  // 🔴 A STACK WITHOUT THE GRID ATTENTION IS THIS TRACK MINUS TWO OF ITS FIVE
  // UPDATES, AND THAT IS ESMFold2's TRUNK EXACTLY. Its block is a pairformer
  // block with the two grid attentions and the single track removed - checked
  // at relRMS 1.4e-6 per recycle by tools/check-esmfold2-trunk.js - so the
  // arithmetic transfers whole and only the SEQUENCE changes.
  //
  // 🔴 AND SKIPPING BEATS ZEROING, WHICH IS THE OBVIOUS WAY TO DO IT. An
  // attention whose output projection is zero adds zero, so a zeroed block IS
  // ESMFold2's block and needs no code at all - but grid.attend is the largest
  // kernel in an AF3 trunk (34.6% of its GPU time at 700 tokens) and it would
  // run 24 times a loop, four loops, to add zero. The zeroed arm stays as the
  // thing this is checked bit-identical against; see check-esmfold2-trunk-gpu.js.
  const gridAttention = options.gridAttention ?? true;
  // 🔴 THE TRIANGLE PROJECTION ON THE MATRIX UNITS, WHICH IS THE LARGEST KERNEL
  // LEFT IN AN ESMFold2 TRUNK: 130.6 ms of 494 once the transition is split.
  // It needs no new memory - its source and both its outputs are pair-sized
  // scratch the track already holds - but it does need the four projection
  // matrices INTERLEAVED, so the layout and the offsets below are resolved from
  // the same flag. A packed a/b keeps the vector kernel: see the note in
  // src/triangle/project-matrix.js.
  const abPacked = scratchStorage[1] === "f16" || scratchStorage[2] === "f16";
  const projectMatrix = options.triangleProjectMatrix !== undefined
    && options.triangleProjectMatrix !== false && !abPacked
    ? options.triangleProjectMatrix : false;
  const abLayout = projectMatrix === false ? "blocked" : "interleaved";
  // ...and the output projection's two matrices, which the same knob moves and
  // which need a transpose rather than an interleave. See project-matrix.js.
  const zgLayout = projectMatrix === false ? "blocked" : "transposed";
  const triangleOffsets = packTriangleWeights(
    af3TriangleWeights(sample.triangleMultiplicationOutgoing, channels),
    weightPrecision, { abLayout, zgLayout, cHidden: channels, cZ: channels }).offsets;
  const gridOffsets = gridAttention
    ? packGridAttentionWeights(sample.pairAttention1).offsets : null;
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
      shape, "f32", triangleOffsets, epsilon, direction, variance,
      // 🔴 THE PROJECTION TILE IS THE CALLER'S, because it is an occupancy
      // choice and this file cannot see the device. undefined keeps
      // src/triangle/shaders.js's default, which is every device but ampere.
      options.triangleProjectTile ?? undefined, true,
      undefined,
      // 🔴 THE NORMALISED HIDDEN GOES BACK INTO `a`, WHICH IS DEAD BY THEN.
      // `tri.contract` is the last pass that reads scratch[1] and scratch[2],
      // and it runs before `tri.normalize-hidden` writes one of them - so the
      // triangle needs FOUR pair-sized tensors and not five. Only the grid
      // attention ever wanted a fifth (it holds q, k, v and a gate at once),
      // which is why pairScratchCount asks whether the track runs one.
      { normalized: scratchStorage[0], hidden: scratchStorage[1],
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
      // ...and the vector projection is not compiled at all where the matrix
      // one replaces it, because its weights are no longer in the buffer.
      if (["projectAB", "projectOutput", "contract"].includes(name)
          && projectMatrix !== false) continue;
      compileInto(`tri:${direction}:${name}`,
                  `${base}:tri:${direction}:${weightPrecision}:${accumulatePrecision}`
                  + `:${scratchStorage.join("")}:${name}`,
                  source);
    }
    if (projectMatrix !== false) {
      const geometry = { ...TRIANGLE_PROJECT_MATRIX_GEOMETRY, ...projectMatrix };
      if (!triangleProjectMatrixFits(
        geometry, options.maxComputeWorkgroupStorageSize ?? 49152)) {
        throw new RangeError("the matrix triangle projection does not fit this device");
      }
      pipelines.projectMatrix = triangleProjectMatrixDispatch(
        { rows: pairs, cHidden: channels }, projectMatrix);
      const matrixKey = `${base}:tri:${direction}:${weightPrecision}:${accumulatePrecision}`
        + `:${scratchStorage.join("")}:project-matrix:${JSON.stringify(projectMatrix)}`;
      compileInto(`tri:${direction}:projectAB`, `${matrixKey}:ab`,
                  createTriangleProjectMatrixShader(
                    { cZ: channels, cHidden: channels },
                    { normalized: scratchStorage[0], ab: "f32", weight: weightPrecision },
                    projectMatrix));
      const outSources = createTriangleProjectOutMatrixShaders(
        { cZ: channels, cHidden: channels },
        { normalized: scratchStorage[0], hidden: scratchStorage[1], gate: "f32",
          weight: weightPrecision },
        projectMatrix);
      pipelines.projectOutMatrix = triangleProjectOutMatrixDispatch(
        { rows: pairs, cZ: channels }, projectMatrix);
      compileInto(`tri:${direction}:projectOutGate`, `${matrixKey}:out-gate`, outSources.gate);
      compileInto(`tri:${direction}:projectOut`, `${matrixKey}:out`, outSources.project);
      // ...and the contraction, which is a BATCHED product over the channels
      // and transposes a different operand in each direction.
      pipelines.contractMatrix = triangleContractMatrixDispatch(
        { length: n, channels }, projectMatrix);
      compileInto(`tri:${direction}:contract`, `${matrixKey}:contract`,
                  // 🔴 THE CONTRACTION'S ACCUMULATOR IS ITS OWN, because its K
                  // is the PROTEIN'S LENGTH where every other staged GEMM here
                  // contracts a channel count. A caller narrowing the results
                  // to halves for the occupancy must not narrow this one - it
                  // overflows, and the fold comes back all NaN rather than
                  // slightly wrong. See stagedMatrixResult in
                  // src/esmfold2/trunk-webgpu.js.
                  createTriangleContractMatrixShader(
                    { length: n, channels }, direction, { ab: "f32" }, projectMatrix));
    }
  }
  for (const [key, attention, transpose] of (gridAttention
       ? [["false", sample.pairAttention1, false], ["true", sample.pairAttention2, true]]
       : [])) {
    const { tiles, ...sources } = createGridAttentionShaders(
      { n, channels, heads: attention.heads, dimension: attention.dimension, transpose,
        residual: true, stagedPrecision,
        // 🔴 `grid.attend` IS THE LARGEST KERNEL IN THE TRUNK AND THE ONLY
        // CUBIC ONE, which is why the matrix units are pointed at this one
        // first. Off unless the caller asks; the device profile decides, and
        // the geometry is swept per kernel and never inherited. See
        // src/af3/grid-attention-matrix.js.
        attendMatrix: options.attendMatrix ?? false },
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
    // 🔴 THE PROJECTION ON THE UNITS, off unless the caller asks. It is the
    // biggest pass in an OpenDDE trunk - 386 ms of 1876 at 256 tokens - and its
    // weights are ALREADY interleaved [k][4w + role] for the vector kernel's
    // vec4 read, which is the layout the staged epilogue wants. See
    // src/af3/grid-project-matrix.js.
    if (options.gridProjectMatrix !== undefined && options.gridProjectMatrix !== false
        && scratchStorage.slice(1, 5).every((s) => s !== "f16")) {
      const width = attention.heads * attention.dimension;
      const geometry = options.gridProjectMatrix === true ? {} : options.gridProjectMatrix;
      if (!gridProjectMatrixFits(geometry, options.maxComputeWorkgroupStorageSize ?? 49152)) {
        throw new RangeError("the matrix grid projection does not fit this device");
      }
      pipelines.gridProjectMatrix = {
        ...gridProjectMatrixDispatch({ rows: n * n, width }, geometry),
        out: gridProjectOutMatrixDispatch({ rows: n * n, channels }, geometry),
        // The uniforms the caller must allocate; carried so a caller that reads
        // the constants instead cannot disagree with the compiled shader.
        rows: n * n, channels, width,
        weightOffset: gridOffsets.qkvgProjection,
        outWeightOffset: gridOffsets.outputProjection,
      };
      compileInto(`grid:${key}:projectOutMatrix`,
                  `${base}:grid:${key}:${stagedPrecision}:${scratchStorage.join("")}`
                  + `:project-out-matrix:${JSON.stringify(geometry)}:${transpose}`,
                  createGridProjectOutMatrixShader(
                    { n, channels, width, transpose },
                    // ...f32 for the same reason - see the note above.
                    { gathered: scratchStorage[0], gate: scratchStorage[4],
                      weight: "f32" },
                    geometry, true));
      compileInto(`grid:${key}:projectMatrix`,
                  `${base}:grid:${key}:${stagedPrecision}:${scratchStorage.join("")}`
                  + `:project-matrix:${JSON.stringify(geometry)}:${transpose}`,
                  createGridProjectMatrixShader(
                    { n, channels, width, transpose },
                    // 🔴 f32 AND NOT `weightPrecision`, BECAUSE THIS PACK HAS
                    // NO PRECISION. packGridAttentionWeights takes a shape and
                    // nothing else and always writes a Float32Array, where
                    // packTriangleWeights and packTransitionWeights take the
                    // element. Telling the shader otherwise reads f32 bytes as
                    // pairs of halves: check-af3-block-any.js returned NaN.
                    { normalized: scratchStorage[0], qkvg: scratchStorage[1],
                      weight: "f32" },
                    geometry));
    }
    for (const [name, source] of Object.entries(sources)) {
      compileInto(`grid:${key}:${name}`,
                  `${base}:grid:${key}:${stagedPrecision}`
                  + `:${scratchStorage.join("")}:m${options.attendMatrix ?? 0}:${name}`,
                  source);
    }
  }
  // The transition stages two blocks of its own - the layer-normed rows and the
  // gated intermediate - and narrowing them is the same trade as the attention
  // tile above, on the largest kernel in the trunk. See transition-webgpu.js.
  // 🔴 THE FACTOR IS IN THE KEY, BECAUSE TWO STACKS OF ONE MODEL DIFFER ON IT.
  // OpenDDE's structural-token refiner and its confidence head are both four
  // pairformer blocks at 384 channels over the same token count with the same
  // extra pair bias - and their pair transitions are factor 2 and factor 4. The
  // key could not tell them apart, so the second asked the cache for the first
  // one's shader and the cache reported a COLLISION rather than serving it,
  // which is the whole reason that check exists.
  // 🔴 THE SPLIT TRANSITION, WHICH IS A CHANNEL-WIDTH DECISION AND NOT A DEVICE
  // ONE ALONE. The fused kernel below holds the WIDENED row in workgroup
  // memory, so its row tile - all of its weight-read amortisation - halves each
  // time the channels double: measured against its own best tile at 200 tokens,
  // the split is 1.13x at 128 channels, 2.77x at 256 and 3.71x at 384. AF3's
  // trunk is 128 and ESMFold2's is 256 and OpenDDE's is 384, so the same knob
  // means something different in each. See createTransitionSplitShaders.
  const splitTransition = options.pairTransitionSplit ?? false;
  if (splitTransition !== false) {
    const matrix = splitTransition === true ? {} : splitTransition;
    const split = createTransitionSplitShaders(
      { rows: pairs, channels, factor: transitionFactor },
      transitionOffsets, epsilon, variance,
      { weightPrecision, matrix });
    // The caller must not have asked for a geometry this device cannot stage.
    const bytes = stagedMatrixStorage({ ...split.geometry, ...matrix });
    if (bytes > (options.maxComputeWorkgroupStorageSize ?? 49152)) {
      throw new RangeError(`a split transition stages ${bytes} B, over the limit`);
    }
    pipelines.transitionSplit = {
      tiles: split.tiles,
      chunkRows: transitionSplitChunkRows(pairs, channels, transitionFactor, {
        // 🔴 THE KNOB WAS PLUMBED IN AND NEVER READ. `pairTransitionChunkBytes`
        // is put into these options by pairformer-block-webgpu.js and by
        // msa-stack-webgpu.js, and this call - the only caller of
        // transitionSplitChunkRows - did not pass it, so the rule fell back to
        // its own 64 MiB default. Every AF3 and OpenDDE arm ever measured with
        // that knob was measured at 64 MiB, and a sweep of 64, 128 and 256
        // moved the `down` pass by 0.4 ms of 77.5 and its group count not at
        // all. Same shape as the `matrixLinear: false` that fell through into
        // the matrix path: a knob with no effect is worse than no knob.
        targetBytes: options.pairTransitionChunkBytes ?? undefined,
        maxStorageBufferBindingSize: options.maxStorageBufferBindingSize,
        minStorageBufferOffsetAlignment: options.minStorageBufferOffsetAlignment,
        blockRows: split.tiles.blockRows,
      }),
      intermediate: channels * transitionFactor,
    };
    for (const [name, source] of Object.entries(
      { normalize: split.normalize, wide: split.wide, down: split.down })) {
      compileInto(`pairTransitionSplit:${name}`,
        `${base}:pair-transition-split:${transitionFactor}:${weightPrecision}`
        + `:${JSON.stringify(matrix)}:${name}`,
        source);
    }
  }
  compileInto("pairTransition",
    `${base}:pair-transition:${transitionFactor}:${stagedPrecision}:${weightPrecision}`,
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
export function packPairTrackWeights(block, channels = PAIR_CHANNELS, weightPrecision = "f32",
                                    gridAttention = true, abLayout = "blocked",
                                    want = {}) {
  // 🔴 WHAT THE CALLER STILL NEEDS, because some of these may already be on the
  // device. This packer returns all five from one call, so a caller that stopped
  // BINDING a buffer was still paying to build it - see
  // src/af3/pair-track-device-weights.js.
  const { transition = true, triangles = true, grids = true } = want;
  // 🔴 THE LAYOUT IS THE PATH'S, AND THE TWO MUST AGREE. `interleaved` swaps the
  // four projection matrices for one transposed, interleaved block, which is
  // what the matrix `tri.project` reads and the vector one cannot; the element
  // count is identical either way, so this is a reshape rather than a cost. The
  // offsets `compilePairTrack` resolves come from the SAME call with the SAME
  // flag - see triangleOffsets there - because a pack and an offset table that
  // disagree is a finite, plausible tensor.
  const triangle = (weights) => packTriangleWeights(
    af3TriangleWeights(weights, channels), weightPrecision,
    { abLayout, zgLayout: abLayout === "interleaved" ? "transposed" : "blocked",
      cHidden: channels, cZ: channels }).data;
  return {
    ...(triangles ? {
      outgoing: triangle(block.triangleMultiplicationOutgoing),
      incoming: triangle(block.triangleMultiplicationIncoming),
    } : {}),
    // ...and a block with no grid attention has no such tensors to pack. See
    // compilePairTrack's gridAttention.
    ...(gridAttention && grids ? {
      grid1: packGridAttentionWeights(block.pairAttention1).data,
      grid2: packGridAttentionWeights(block.pairAttention2).data,
    } : {}),
    // 🔴 SKIPPED WHEN THE DEVICE DECODER HAS ALREADY FILLED IT. This packer
    // returns all five in one call, so a caller that stopped USING the
    // transition's buffer was still paying for it - 229.6 MiB of f16 over 48
    // blocks at 384 channels, decoded out of int5 and narrowed on the main
    // thread for a buffer nothing bound.
    ...(transition
      ? { transition: packTransitionWeights(block.pairTransition, weightPrecision).data }
      : {}),
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
 * How many of them a track that may or may not run the grid attention needs.
 *
 * 🔴 THE GRID IS WHAT WANTS THE FIFTH, AND ESMFold2's TRUNK DOES NOT RUN ONE.
 * `grid.project` writes q, k, v and a gate and all four are live at once; the
 * triangle's longest overlap is the normalised pair, a, b and the contraction's
 * output, which is four. So a trunk with the grid attention off holds one
 * pair-sized tensor fewer - 43.9 MiB at 300 tokens, on the largest tensor group
 * this model has.
 *
 * 🔴 AND IT IS A FUNCTION AND NOT A SECOND CONSTANT, because the count and the
 * STORAGE list have to describe the same buffers: a caller that allocated four
 * and compiled shaders against a five-entry storage array would bind a tensor
 * of the right element count and the wrong byte length, which nothing
 * validates. Both come off the same array.
 */
export function pairScratchCount(gridAttention = true) {
  return gridAttention ? PAIR_SCRATCH_COUNT : PAIR_SCRATCH_COUNT - 1;
}

/**
 * Record the five pair updates into an open command encoder.
 *
 * @param {object} context `run(label, pipeline, buffers, x, y, z)` records one
 *   pass; `scratch` is five pair-sized buffers, reused by every operation.
 */
export function encodePairTrack(context) {
  const { run, pipelines, n, gridHeads, pair, pairMask, scratch, biasBuffer, weights } = context;
  // 🔴 THE WIDTH IS REQUIRED, AND USED TO DEFAULT TO 128. That default is what
  // broke OpenDDE: `compilePairTrack` was given 384 and generated kernels for
  // it, while THIS function sized every dispatch for AlphaFold 3's 128 - so two
  // thirds of every pair row went unprocessed, on a track whose kernels each
  // check out at 6e-7 in isolation. The block's relRMS against its own CPU
  // reference was 1.293 and the contact map scored BELOW chance, with nothing
  // out of range and no validation error anywhere.
  //
  // This is the second time a shape resolved in two places has done this here;
  // CLAUDE.md's closing habit records the first, where "shaders tiling by four
  // under a dispatch dividing by eight" left half the tokens unprocessed and
  // read as a 30% speedup. A default is what let the two drift, so there is
  // none: every caller states it, and a caller that forgets is an error rather
  // than a silently truncated track.
  const { channels } = context;
  if (!(channels > 0)) {
    throw new Error("encodePairTrack needs the track's channel count; it is the "
      + "bundle's, not a constant, and sizing the dispatch for the wrong one "
      + "leaves rows unprocessed with no validation error");
  }
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
    // ...the vector row tile, which `tri.project-out` still uses whichever
    // kernel does the projection: only projectAB moves to the units.
    const perProjectTile = spreadTriangle(ceil(pairs, pipelines.projectTile.rows));
    if (pipelines.projectMatrix !== undefined) {
      // 🔴 THE SAME THREE BUFFERS, A DIFFERENT KERNEL. Source, `a` and `b` are
      // the pair-sized scratch the vector kernel used; only the weight LAYOUT
      // and the dispatch change. See src/triangle/project-matrix.js.
      const uniform = context.projectMatrix;
      if (uniform === undefined) {
        throw new Error("the matrix triangle projection needs its uniform; "
          + "see allocateTriangleProjectMatrix");
      }
      run("tri.project", p("projectAB"),
          [scratch[0], w, uniform.parameters, scratch[1], pairMask, scratch[2]],
          pipelines.projectMatrix.x, pipelines.projectMatrix.y);
    } else {
      run("tri.project", p("projectAB"), [scratch[0], pairMask, w, scratch[1], scratch[2]],
          ceil(channels, pipelines.projectTile.columns), perProjectTile[0], perProjectTile[1]);
    }
    if (pipelines.contractMatrix !== undefined) {
      // 🔴 a IS THE LEFT OPERAND OUTGOING AND THE RIGHT ONE INCOMING, which is
      // the whole difference between the two directions here - the shader
      // transposes whichever of the two it has to. Swapping them returns a
      // finite tensor of the same shape.
      const swap = direction === "incoming";
      const c = pipelines.contractMatrix;
      run("tri.contract", p("contract"),
          [swap ? scratch[2] : scratch[1], swap ? scratch[1] : scratch[2],
           context.projectMatrix.contract, scratch[3]],
          c.x, c.y, c.z);
    } else {
      run("tri.contract", p("contract"), [scratch[1], scratch[2], scratch[3]],
          ceil(n, pipelines.contractTile.columns), ceil(n, pipelines.contractTile.rows), channels);
    }
    // ...into scratch[1], which `tri.contract` was the last pass to read; see
    // the note where the triangle's shaders are compiled.
    run("tri.normalize-hidden", p("normalizeHidden"), [scratch[3], w, scratch[1]],
        perNormalizeTile[0], perNormalizeTile[1]);
    // ...straight into the pair representation, which nothing has read since
    // tri.normalize consumed it into scratch[0].
    if (pipelines.projectOutMatrix !== undefined) {
      // 🔴 THE GATE GOES IN scratch[2], WHICH IS DEAD BY HERE. `tri.contract`
      // was the last reader of `b` and `tri.normalize-hidden` has already
      // consumed the contraction, so the second GEMM's target costs nothing.
      const uniform = context.projectMatrix;
      const out = pipelines.projectOutMatrix;
      run("tri.project-out.gate", p("projectOutGate"),
          [scratch[0], w, uniform.outGate, scratch[2]], out.x, out.y);
      run("tri.project-out", p("projectOut"),
          [scratch[1], w, uniform.outProject, pair, scratch[2]], out.x, out.y);
    } else {
      run("tri.project-out", p("projectOutput"), [scratch[0], scratch[1], w, pair],
          ceil(channels, pipelines.projectTile.columns), perProjectTile[0], perProjectTile[1]);
    }
  }

  // Two of the five updates, or none of them - see compilePairTrack.
  for (const [key, w] of ((context.gridAttention ?? true)
       ? [["false", weights.grid1], ["true", weights.grid2]] : [])) {
    const p = (name) => pipelines[`grid:${key}:${name}`];
    const linear = spread(ceil(pairs, 64));
    const perNormalize = spread(ceil(pairs, pipelines.gridTiles.normalizeRows));
    run("grid.normalize", p("normalize"), [pair, w, scratch[0]], perNormalize[0], perNormalize[1]);
    run("grid.bias", p("bias"), [scratch[0], w, biasBuffer], linear[0], linear[1]);
    const perOutTile = spread(ceil(pairs, pipelines.gridTiles.projectOutRows));
    // One workgroup per tile of pair rows - see the kernel.
    const perTile = spread(ceil(pairs, pipelines.gridTiles.projectRows));
    if (pipelines.gridProjectMatrix !== undefined) {
      // The same five buffers; only the weight READ and the dispatch change.
      const uniform = context.gridProjectMatrix;
      if (uniform === undefined) {
        throw new Error("the matrix grid projection needs its uniform; "
          + "see allocateGridProjectMatrix");
      }
      const g = pipelines.gridProjectMatrix;
      run("grid.project", p("projectMatrix"),
          [scratch[0], w, uniform.project, scratch[1], scratch[2], scratch[3], scratch[4]],
          g.x, g.y);
    } else {
      run("grid.project", p("project"),
          [scratch[0], w, scratch[1], scratch[2], scratch[3], scratch[4]], perTile[0], perTile[1]);
    }
    // 🔴 THE GRID TRACK'S HEAD COUNT, not the single track's. They differ, 4
    // against 16, and the shader's bounds check makes the wrong one correct but
    // oversubscribed.
    //
    // One thread per (query, row, head) on the scalar kernel, one WORKGROUP per
    // tile of `attendRows` of them on the matrix one - so the x extent comes
    // from the tile the shaders were generated with and is not a constant here.
    // See the note on the attend kernel and src/af3/grid-attention-matrix.js.
    // ...into scratch[0], which `grid.project` above was the last pass to
    // read. See the note where the shaders are compiled.
    run("grid.attend", p("attend"),
        [scratch[1], scratch[2], scratch[3], biasBuffer, pairMask, scratch[0]],
        ceil(n, pipelines.gridTiles.attendRows), n, gridHeads);
    if (pipelines.gridProjectMatrix !== undefined) {
      // The same four buffers, plus the gate as its own binding rather than as
      // the second one - see sourceModulate in src/runtime/matrix-linear.js.
      const out = pipelines.gridProjectMatrix.out;
      run("grid.project-out", p("projectOutMatrix"),
          [scratch[0], w, context.gridProjectMatrix.out, pair, scratch[4]],
          out.x, out.y);
    } else {
      run("grid.project-out", p("project_out"), [scratch[0], scratch[4], w, pair],
          perOutTile[0], perOutTile[1]);
    }
  }

  // 🔴 A TILE OF PAIRS A WORKGROUP. This was 241 ms of a 632 ms pairformer pass
  // - the largest single kernel in the trunk - because each workgroup read the
  // whole 196k-float weight set for one row.
  if (pipelines.transitionSplit !== undefined) {
    // 🔴 THREE PASSES OVER A CHUNK OF ROWS, AND THE PAIR IS BOUND AT AN OFFSET.
    // The widened activation is 369 MiB at 300 ESMFold2 tokens, so it is sized
    // for ONE chunk and the pair moves past it - which is only legal because
    // transitionSplitChunkRows aligns the chunk to the binding alignment as
    // well as to the block. See createTransitionSplitShaders for why this is
    // three passes and not two.
    const split = pipelines.transitionSplit;
    const scratchSplit = context.transitionSplit;
    if (scratchSplit === undefined) {
      throw new Error("a split pair transition needs its buffers; see allocateTransitionSplit");
    }
    const p = (name) => pipelines[`pairTransitionSplit:${name}`];
    for (let start = 0; start < pairs; start += split.chunkRows) {
      const count = Math.min(split.chunkRows, pairs - start);
      const parameters = scratchSplit.parameters.get(count);
      if (parameters === undefined) {
        throw new Error(`no split transition uniform for a chunk of ${count} rows`);
      }
      // 🔴 AN ALLOCATION, NOT A BUFFER. `run` takes the pair track's allocation
      // shape and reads `byteOffset`/`byteSize` off it, and an allocation is
      // itself often a sub-range of a pooled buffer - so the chunk's offset is
      // added to the one the allocation already has rather than replacing it.
      const at = (allocation) => ({
        buffer: allocation.buffer,
        byteOffset: (allocation.byteOffset ?? 0) + start * channels * 4,
        byteSize: count * channels * 4,
      });
      const perNormalize = spread(Math.ceil(count / split.tiles.normalizeRows));
      run("pair-transition.normalize", p("normalize"),
          [at(pair), weights.transition, scratchSplit.normalized, parameters.normalize],
          perNormalize[0], perNormalize[1]);
      run("pair-transition.wide", p("wide"),
          [scratchSplit.normalized, weights.transition, parameters.wide, scratchSplit.wide],
          Math.ceil(split.intermediate * 2 / split.tiles.blockColumns),
          Math.ceil(count / split.tiles.blockRows));
      // ...residual, straight into the pair, which is what the fused kernel does.
      run("pair-transition.down", p("down"),
          [scratchSplit.wide, weights.transition, parameters.down, at(pair)],
          Math.ceil(channels / split.tiles.blockColumns),
          Math.ceil(count / split.tiles.blockRows));
    }
    return;
  }
  const perTransition = spread(Math.ceil(pairs / transitionRowTile(pairs, channels)));
  // ...reads every row it writes into workgroup memory before writing any of
  // them, and no other workgroup touches those rows, so this is in place.
  run("pair-transition", pipelines.pairTransition, [pair, weights.transition],
      perTransition[0], perTransition[1]);
}
