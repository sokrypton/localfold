import { concatenateAs, writeInto } from "../runtime/float16.js";
import { deviceTuning, halfPrecisionAvailable } from "../runtime/device-profile.js";
import { residentPackedOnDevice } from "./device-weights.js";
import { SOURCES } from "./weights.js";
/**
 * AF3's diffusion token transformer: 24 blocks, AdaLN-conditioned, pair-biased.
 *
 * The bulk of the diffusion head's parameters. Each block is
 *
 *     act += adaptiveZeroInit(attention(adaptiveLayerNorm(act, cond)))
 *     act += adaptiveZeroInit(swiglu(adaptiveLayerNorm(act, cond)))
 *
 * where AdaLN is `sigmoid(scale(cond)) * layerNorm(act) + bias(cond)` and the
 * zero-init gate is `sigmoid(zeroCond(cond))`, whose bias is initialised at -2
 * so an untrained block starts near the identity.
 *
 * 🔴 THE BLOCKS ARE NESTED SIX BY FOUR AND THE PAIR LOGITS FOLLOW THAT NESTING.
 * The LayerNorm over the pair conditioning is computed ONCE and shared, but each
 * of the six SUPER-BLOCKS projects it to its own four blocks' worth of head
 * biases - so there are six projections, not one and not twenty-four. The
 * checkpoint says so: pair_logits_projection is (6, 128, 4, 16). A flat reading
 * of the stack indexes the wrong weights for every block after the fourth.
 *
 * 🔴 THE CONDITIONING IS NARROWER THAN THE ACTIVATION. cond is 384 and act is
 * 768, so every AdaLN projection is 384->768 rather than square. In the atom
 * stacks both are 128 and the distinction is invisible, which is exactly how a
 * square assumption survives to here and then reads at the wrong stride.
 *
 * 🔴 THE ATTENTION SCALE IS THE PER-HEAD DIMENSION, taken AFTER dividing by the
 * head count: AF3 writes `key_dim = key_dim // num_head` and only then
 * `key_dim ** -0.5`. That is 48, not 768. Using the full width is a factor of
 * four on every logit, which softmax turns into much flatter attention - and
 * flatter attention still folds proteins, just worse.
 *
 * 🔴 THE LayerNorms HERE ARE TWO-PASS, not the trunk's fast variance. AF3 sets
 * use_fast_variance=False for the diffusion and atom stacks. See
 * src/triangle/shaders.js for why that cannot be a global.
 */
import { GpuBufferAllocator } from "../runtime/allocator.js";
import { noteAllocation, noteDestroy } from "../runtime/device-memory.js";
import { GpuMemoryBudgetError, noteResidencyRefused, residencyAllowed }
  from "../runtime/device-memory.js";
import { releaseResidentWeights, residentWeightBuffer } from "../runtime/resident.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";
import { DeferredValidation } from "../runtime/validation.js";
import { releaseWeights } from "./weights.js";
/**
 * How much a schedule may hold in cached pair attention biases.
 *
 * 🔴 THE CACHE IS 64 x tokens^2 BYTES A BLOCK, so all twenty-four are 61 MiB at
 * 200 tokens and 246 at 400 - and a long fold is where memory binds, not where
 * it is spare. The saving is per block and so is the cost, so this caps how
 * many blocks keep theirs: all of them at 208 tokens or fewer, six of the
 * twenty-four at 400. Measured at 200 tokens, 16 flow steps, tools/gpu/fold.js
 * --folds=2: a denoiser call 204 -> 196 ms, a fold 7.5 -> 7.2 s, peak 646 ->
 * 688 MiB.
 */
export const PAIR_LOGITS_CACHE_BYTES = 64 * 1024 * 1024;

const GRID_WIDTH = 32_768;

export const BLOCK_ORDER = [
  "SingleCondLayerNormScale", "SingleCondScaleWeights", "SingleCondScaleBias", "SingleCondBias",
  "qProjection", "qBias", "kProjection", "vProjection", "gatingQuery",
  "Transition2", "AdaptiveZeroCondWeights", "AdaptiveZeroCondBias",
  "ffwSingleCondLayerNormScale", "ffwSingleCondScaleWeights", "ffwSingleCondScaleBias",
  "ffwSingleCondBias", "ffwTransition1", "ffwTransition2",
  "ffwAdaptiveZeroCondWeights", "ffwAdaptiveZeroCondBias",
];

/**
 * 🔴 THE UPLOAD WAS THE FLOOR, NOT THE ARITHMETIC. A block is ~26 MB, so the
 * loop wrote ~630 MB to the device per call - and at eight tokens, where the
 * matmuls are nothing, twenty-four blocks still cost 174 ms, which is that
 * write at about 3.6 GB/s. A 200-step fold did it two hundred times over
 * weights that never change.
 *
 * 🔴 SO THIS TRADES DEVICE MEMORY FOR IT, DELIBERATELY - ~630 MB at f32, the
 * same order as the checkpoint itself. It goes through src/runtime/resident.js
 * rather than a WeakMap of its own so that ONE call hands back every weight
 * buffer on a device, which is what the budget fallback below needs.
 */
function residentBlockBuffer(device, block, pack, variant = "") {
  return residentWeightBuffer(device, block, "difftx.block.resident", () => pack().data, variant);
}

/**
 * The same buffer, decoded on the device when the weights allow it.
 *
 * 🔴 IT IS ~320 ms OF A SESSION'S FIRST FOLD. Packing these 24 blocks on the
 * host measures 440 ms cold against 119 for the GPU, and the two agree on every
 * one of 198 million elements - see tools/gpu/check-block-upload.js. The shared
 * machinery is in src/af3/device-weights.js, because the trunk's packers have
 * the same shape and the same problem.
 */
function residentBlockOnDevice(device, block, precision) {
  if (precision !== "f16") return Promise.resolve(undefined);
  return residentPackedOnDevice(device, {
    key: block, label: "difftx.block.resident", order: BLOCK_ORDER,
    weights: block, variant: precision,
  });
}

/**
 * Every block's zero-init gate weights, concatenated, so ONE dispatch can do
 * all twenty-four.
 *
 * 🔴 THE GATE IS A PROJECTION OF THE CONDITIONING, AND THE CONDITIONING DOES
 * NOT DEPEND ON THE BLOCK. `attention-output`'s epilogue computes
 * `bias + cond @ W_zero` and then divides the projection by `1 + exp(-that)`.
 * Nothing in it reads the block's activation, so it is not part of the
 * sequential chain the blocks form - it only sits inside the block loop
 * because that is where its weights are.
 *
 * 🔴 AND THAT LOOP IS WHERE THE OCCUPANCY GOES. `bench-head.js --profile` at 68
 * tokens: `attention-output` is **2.13 ms at 204 workgroups a pass**, the
 * largest pass in a denoiser call and the most underfilled, on a device with
 * 108 SMs. The K-split measurement in docs/A100.md prices the epilogue alone at
 * **~1.09 ms** of that. Batched over the blocks the same arithmetic dispatches
 * `blocks x tokens x C/lanes` groups - 1224 rather than 204 at 68 tokens - and
 * runs once instead of twenty-four times.
 *
 * 🔴 IT IS ALSO INDEPENDENT OF THE SAMPLE. `perToken` reduces a row to
 * `row % TOKENS` for the conditioning, so a batched sampler recomputed this
 * `samples` times over as well. The buffer is per TOKEN, not per row.
 *
 * 🔴 A DUPLICATE OF WEIGHTS THAT ARE ALREADY RESIDENT, DELIBERATELY. Per block
 * this is C_COND x C + C - about 1.2 MB at f32 and 0.6 at f16 - so 28 MB / 14
 * MB across twenty-four, against the ~630 MB the blocks already hold. Making
 * the block buffers one contiguous allocation instead would avoid the copy and
 * is the larger refactor; this is the cheap half.
 *
 * @param {readonly object[]} blocks in dispatch order - z indexes this array
 * @param {"f32"|"f16"} precision must match the shader's weight word
 */
/**
 * A block's twelve zero-gate tensors, in the order the kernel addresses them.
 *
 * 🔴 EXPORTED SO THE DEVICE PATH CANNOT DRIFT FROM THE HOST ONE. ZG_* in the
 * shader are running sums of these lengths; a list written twice is a list that
 * ends up written differently, and here that reads a neighbouring tensor -
 * a wrong fold and not a crash.
 *   0-3  the two zero gates      (attention-output's, ffw-out's)
 *   4-7  adaln's conditioned norm    ln scale, scale weights, bias, scale bias
 *   8-11 ffw-adaln's, the same four
 */
export const ZERO_GATE_ORDER = ["AdaptiveZeroCondWeights", "AdaptiveZeroCondBias",
  "ffwAdaptiveZeroCondWeights", "ffwAdaptiveZeroCondBias",
  "SingleCondLayerNormScale", "SingleCondScaleWeights",
  "SingleCondBias", "SingleCondScaleBias",
  "ffwSingleCondLayerNormScale", "ffwSingleCondScaleWeights",
  "ffwSingleCondBias", "ffwSingleCondScaleBias"];

export function packZeroGateWeights(blocks, precision = "f32") {
  const first = blocks[0];
  if (first === undefined) throw new Error("no diffusion blocks to pack");
  // 🔴 BOTH GATES IN ONE BLOCK'S SPAN, in this order, because one dispatch
  // computes both: attention-output's and ffw-out's epilogues are the same
  // projection of the same conditioning against different weights, and the
  // kernel stages the conditioning once for the pair.
  // 🔴 THE ORDER IS THE SHADER'S ADDRESS MAP. ZG_* in the kernel are running
  // sums of these lengths; changing this list without changing those silently
  // reads a neighbouring tensor, which is a wrong fold and not a crash.
  //   0-3  the two zero gates      (attention-output's, ffw-out's)
  //   4-7  adaln's conditioned norm    ln scale, scale weights, bias, scale bias
  //   8-11 ffw-adaln's, the same four
  const NAMES = ZERO_GATE_ORDER;
  const span = NAMES.reduce((total, name) => total + first[name].length, 0);
  // 🔴 CHECKED BEFORE ANYTHING IS WRITTEN, NOT AFTER. The shader finds a block
  // by multiplying this span by the block index, so a block that disagrees with
  // it shifts every later one - and a block that is LONGER would run off the
  // end of the buffer on its way to being caught. Validate the whole list, then
  // write.
  for (const [at, block] of blocks.entries()) {
    const length = NAMES.reduce((total, name) => total + (block[name]?.length ?? NaN), 0);
    if (length !== span) {
      throw new Error(`block ${at} zero-gate is ${length}, not ${span}`);
    }
  }
  return concatenateAs(precision, blocks.length * span, (target) => {
    for (const [at, block] of blocks.entries()) {
      let offset = at * span;
      for (const name of NAMES) {
        writeInto(target, block[name], offset);
        offset += block[name].length;
      }
    }
  });
}

/**
 * @param {"f32"|"f16"} precision the element the packed buffer holds. Offsets
 *   are in elements and do not depend on it; the shader must be built for the
 *   same word or it reads half the values at twice the stride.
 */
/**
 * The packing offsets alone, without building the buffer.
 *
 * 🔴 THE SHADERS NEED THE OFFSETS AND NOTHING ELSE, and `packBlockWeights` was
 * being called on a sample block to get them - concatenating 31.5 MiB, and in
 * f16 converting it, to read a dozen numbers that are a running sum of lengths.
 */
export function blockWeightOffsets(block) {
  // 🔴 THE LENGTHS COME FROM THE THUNKS WHEN THERE ARE ANY. Reading
  // `block[name].length` MATERIALISES that tensor, so asking a sample block for
  // a dozen running sums decoded the whole of it - 8.3 million elements out of
  // int5, once per denoiser call before #compile was memoised and once per fold
  // after. `stacked` records the range it will read, and that is the length.
  const sources = block[SOURCES];
  const offsets = {};
  let total = 0;
  for (const name of BLOCK_ORDER) {
    const thunk = sources?.[name];
    const length = Number.isInteger(thunk?.count) ? thunk.count : block[name]?.length;
    if (length === undefined) throw new Error(`diffusion block missing ${name}`);
    offsets[name] = total;
    total += length;
  }
  return offsets;
}

export function packBlockWeights(block, precision = "f32") {
  const offsets = {};
  let total = 0;
  for (const name of BLOCK_ORDER) {
    if (block[name] === undefined) throw new Error(`diffusion block missing ${name}`);
    offsets[name] = total;
    total += block[name].length;
  }
  const data = concatenateAs(precision, total, (target) => {
    for (const name of BLOCK_ORDER) writeInto(target, block[name], offsets[name]);
  });
  return { data, offsets };
}

/**
 * How much of the intermediate `ffw-out` stages at once.
 *
 * 🔴 RESOLVED IN ONE PLACE BECAUSE TWO PLACES NEED IT AND THEY MUST AGREE. The
 * factory sizes its workgroup array from this; the caller sizes `outTile` from
 * it, since the tile only fits if the chunk is what is staged. Defaulting it
 * twice is the shape of mistake the note on `tile` and `splits` below records
 * as half the tokens going unprojected.
 */
export const DEFAULT_OUT_CHUNK = 384;
export const resolveOutChunk = (intermediate, requested) =>
  Math.min(intermediate, requested ?? DEFAULT_OUT_CHUNK);

export function createDiffusionTransformerShaders(shape, offsets) {
  const { tokens, channels, condChannels, pairChannels, heads, dimension, factor } = shape;
  // 🔴 THE FOUR PER-TOKEN KERNELS RUN ONE WORKGROUP PER TOKEN, so the token
  // count IS the occupancy: a 59-residue chain launched 59 workgroups of 64
  // threads, which is under four thousand threads for a GPU that wants tens of
  // thousands, and each of those threads then walked a 768-long dot product.
  // Widening the workgroup is the cheap half of fixing that - the same work,
  // more lanes over it - and it costs only workgroup memory, which the
  // transition's 1536-wide scratch dominates anyway.
  const lanes = shape.lanes ?? 256;
  // 🔴 THE ATTENTION IS BARRIER-BOUND, NOT ARITHMETIC-BOUND, AND ITS TWO
  // REDUCTION TREES ARE MOST OF THE BARRIERS. One workgroup owns one (token,
  // head): at 240 tokens that is 3840 workgroups of 256 lanes - 983,040
  // threads on a device holding 221,184, so it is oversubscribed rather than
  // starved - and each does about 46 KFLOP, 180 floating-point operations a
  // lane, around **twenty barriers**. Two `for (stride = lanes/2; ...)` trees
  // are sixteen of them.
  //
  // `subgroupMax` and `subgroupAdd` reduce within a subgroup with no barrier at
  // all, so the tree collapses to one reduction per subgroup, one barrier, and
  // a final pass over `lanes / subgroupSize` values. Narrowing the workgroup
  // instead does nothing - 64 and 128 lanes measure exactly what 256 does -
  // which is what says the cost is the barriers and not the idle lanes.
  const subgroups = shape.attendSubgroups === true;
  // The same capability, asked for separately: a device can be good at one
  // kernel's reductions and not another's, and these are measured apart.
  const normSubgroups = shape.normSubgroups === true;
  // 🔴 SPLITTING THE NORM'S OUTPUT CHANNELS. Requires C to divide by the lane
  // count, because a split is exactly one lane's worth of columns and a ragged
  // last split would need a bounds check on every write.
  const normSplit = shape.normSplit === true && channels % lanes === 0;
  const normSplits = normSplit ? channels / lanes : 1;
  // 🔴 THE NORM'S OWN K SPLIT. Its inner extent is the conditioning width, and
  // unlike attention-output's the epilogue left behind is a sigmoid and a
  // normalise rather than a second projection - so the ~1 ms floor that made
  // that split a wash does not apply here.
  // 🔴 THE ZERO-INIT GATE, HOISTED OUT OF THE BLOCK LOOP. See
  // packZeroGateWeights: the gate is a projection of the conditioning and the
  // conditioning does not move with the block, so all twenty-four can run in
  // one dispatch of `blocks x tokens x C/lanes` groups instead of twenty-four
  // of 204. Off by default; the caller supplies the batched buffer or this
  // stays false, because the kernel then has nothing to read.
  // 🔴 REQUIRES C TO DIVIDE BY THE LANES, so `c` is always in range and the
  // kernel needs no bounds check before its barrier. A non-uniform `return`
  // ahead of a workgroupBarrier is invalid WGSL, so the alternative is a
  // predicated write and a clamped weight read; this guard is cheaper and it
  // is the one diffusionNormSplit already uses.
  // 🔴 AND C x 4 MUST BE A MULTIPLE OF 256, because each block reads its slice
  // of the batched buffer through a bind-group offset of `block * tokens * C *
  // 4` and WebGPU aligns those to 256 bytes. C = 768 gives 3072, which divides.
  const batchedGates = shape.batchedGates === true && channels % lanes === 0
    && (channels * 4) % 256 === 0;
  // 🔴 AND ITS OWN TOKEN TILE, FOR THE REASON qkvg HAS ONE. The shared `tile` is
  // 1 below 175 tokens, which for this kernel is 68 token groups each re-reading
  // the SAME six weight slices: 5.8 GB a step of weight traffic at 68 tokens
  // against 0.76 at a tile of 8. Every other kernel here is starved of
  // workgroups and wants a small tile; this one has the block axis to fill the
  // device with and is bound by weight bandwidth instead, so it wants the
  // opposite. Measured below.
  const gateTile = Math.max(1, shape.gateTile ?? 8);
  // 🔴 FORCED TO 1 WHEN THE PROJECTION IS BATCHED, because the split divides a
  // projection this kernel no longer has. Left at 4 it would compile a "split"
  // form with nothing to split and a "reduce" that sums one part.
  const normKSplits = batchedGates ? 1 : Math.max(1, shape.normKSplits ?? 1);
  const normKSpan = condChannels / normKSplits;
  if (normKSplits > 1 && !Number.isInteger(normKSpan)) {
    throw new Error(`normKSplits ${normKSplits} does not divide ${condChannels}`);
  }
  // 🔴 AND THE KEY READ IS THE SHAPE OF THE PROBLEM. The logit loop gives a
  // lane a whole KEY - `k[j * WIDTH + head * DIMENSION + d]` for d = 0..47 -
  // so within a lane it is 48 contiguous floats and BETWEEN lanes it is
  // 3072 bytes. A warp issues 32 separate 192-byte reads spread over 98 KB
  // where it could issue coalesced ones. Staging the chunk first inverts that:
  // the staging loop is indexed by `local` over (key, channel) together, so
  // consecutive lanes read consecutive addresses, and the dot product then
  // reads workgroup memory where the stride costs nothing.
  const stageKeys = shape.attendStageKeys === true;
  const keyChunk = shape.attendKeyChunk ?? 64;
  const subgroupSize = shape.subgroupSize ?? 32;
  const groupsOfLanes = Math.ceil(lanes / subgroupSize);
  // How many tokens one workgroup projects at once, and how many ways its
  // output range is split. `splits` must divide heads*dimension.
  // 🔴 NO DEFAULTS HERE. These used to fall back to their own constants, and a
  // caller that resolved them from the device limits and forgot to pass them
  // down got shaders tiling by four under a dispatch that divided the token
  // count by eight - half the tokens never projected, reported as a speedup.
  // A shape that does not say is a bug, so say so.
  const { tile, splits, outTile } = shape;
  for (const [name, value] of Object.entries({ tile, splits, outTile })) {
    if (!Number.isInteger(value) || value < 1) {
      throw new Error(`the diffusion transformer's ${name} must be a positive integer,`
        + ` not ${value}: the caller resolves it from the device's workgroup storage`);
    }
  }
  if ((heads * dimension) % splits !== 0 || (channels * factor) % splits !== 0) {
    throw new Error(`splits ${splits} must divide both ${heads * dimension} and`
      + ` ${channels * factor}`);
  }
  const attnOutTile = Math.max(1, shape.attnOutTile ?? tile);
  const width = heads * dimension;
  const intermediate = channels * factor;
  const pairs = tokens * tokens;
  const outChunk = resolveOutChunk(intermediate, shape.outChunk);
  if (intermediate % outChunk !== 0) {
    throw new Error(`outChunk ${outChunk} does not divide the intermediate ${intermediate}`);
  }
  const outPerLane = Math.ceil((channels / splits) / lanes);
  // 🔴 THE TOKEN TILE IS A VECTOR IN THE TWO WIDE PROJECTIONS TOO. Both read
  // one activation per token of the tile and multiply it by the same weight, so
  // one workgroup read and one vector multiply-add replace TILE of each - qkvg
  // went from 24 instructions a channel to nine.
  // 🔴 THE WEIGHT BUFFER IS A STORAGE FORMAT, AND THIS IS THE BIGGEST ONE THERE
  // IS. `difftx.block.resident` was 756 MiB when it was f32 - 68% of what a
  // fold then held, more than the trunk's entire pairformer - because 24 blocks
  // of a 768-channel transformer each keep 31.5 MiB resident for the model's
  // lifetime. In f16 it is 378 MiB, and it is still the largest single tensor a
  // fold holds; a fold now holds 798 MiB in total.
  //
  // 🔴 AND HERE IT BUYS TIME AS WELL, WHICH IS TRUE OF NOWHERE ELSE IN EITHER
  // MODEL. This comment said "it buys no time" for a while, on the strength of
  // AF3's trunk, where halving resident weight bytes measured 377 ms against
  // 378. That reasoning does not transfer: this stack STREAMS its whole weight
  // set once per token tile rather than keeping it in cache, so the bytes are
  // the cost. Measured on bench-diffusion-transformer.js: 48 -> 41 ms at 59
  // tokens and 103 -> 89 at 150, and a denoiser call 104-114 -> 86-91.
  //
  // Reads are widened at the point of use and the arithmetic is f32 throughout.
  const weightPrecision = shape.weightPrecision ?? "f32";
  if (!["f32", "f16"].includes(weightPrecision)) {
    throw new RangeError(`unknown diffusion transformer weight precision ${weightPrecision}`);
  }
  const weight16 = weightPrecision === "f16";
  const wf = (e) => (weight16 ? `f32(${e})` : e);

  // 🔴 THE CHANNELS ARE STAGED IN CHUNKS SO THAT THE TOKEN TILE CAN GROW, and
  // the tile is the only thing that matters here: this stack streams all 566 MB
  // of its weights once per tile, so at four tokens a 240-token call makes
  // sixty passes over them. Holding TILE x C activations is what capped it -
  // 12 KB at four tokens, 24 at eight, where residency collapses and tile 8
  // measured 343 ms against 320. A chunk unties the two.
  // 🔴 SPLITTING K IS THE OTHER HALF OF THE TOKEN TILE, AND NEITHER WORKS
  // ALONE. The projections do one multiply-add per weight they load - 8 flops
  // for 16 bytes - which caps them near 2.5 TFLOP/s off L2 whatever else is
  // done. The token tile raises that ratio to TILE/2 and pays for it in
  // workgroups, ceil(tokens/TILE) instead of tokens, which is why the tile is
  // pinned to 1 below a few hundred tokens: halving 204 workgroups measures
  // 0.68x. Splitting the inner extent multiplies the workgroups back.
  //
  // Measured together on this shape (tools/gpu/probe-split-k.js, every arm
  // checked against a known answer): plain 1.97 TFLOP/s, K split alone 2.63,
  // tile 4 alone 2.18, **tile 4 with K split 8 4.54** - 2.30x, where neither
  // lever passes 1.35x on its own. See docs/A100.md.
  // 🔴 THE SAMPLE DIMENSION IS A SECOND PATH, NOT A THREADED PARAMETER. At
  // samples === 1 every kernel below must emit exactly what it emitted before
  // this existed - tools/gpu/check-difftx-samples.js asserts that byte for
  // byte - so the default fold cannot regress no matter what the batched path
  // does. Above 1 the row axis becomes (sample, token) and three things have
  // to know: the conditioning is per TOKEN and shared across samples, the pair
  // bias likewise, and attention must not cross a sample boundary.
  const samples = Math.max(1, shape.samples ?? 1);
  const rows = samples * tokens;
  // What a row-parallel kernel bounds against, and what a per-token lookup
  // reduces a row to. At one sample both are what they always were.
  const ROWS = samples === 1 ? "TOKENS" : `${rows}u`;
  const perToken = (row) => (samples === 1 ? row : `(${row}) % TOKENS`);
  // A key row inside the query's sample, and the token a query row reduces to.
  // Both are the identity at one sample, textually as well as numerically.
  const keyRow = (j) => (samples === 1 ? j : `key_base + ${j}`);
  const tokenI = samples === 1 ? "i" : "token_i";
  void ROWS; void perToken; void keyRow; void tokenI;
  const kSplits = Math.max(1, shape.kSplits ?? 1);
  // Only the split kernel can afford a raised tile; without a split behind it a
  // bigger tile is 0.54x-0.82x on every kernel measured.
  const qkvgTile = kSplits > 1 ? (shape.qkvgTile ?? tile) : tile;
  const wideTile = kSplits > 1 ? (shape.wideTile ?? qkvgTile) : tile;
  // 🔴 ffw-out SPLITS ITS OWN K, AND OVER A DIFFERENT EXTENT. Its inner
  // dimension is the INTERMEDIATE, not the channel count, so it needs its own
  // part count and its own span - and its epilogue is a second projection
  // rather than a bias, which is why the reduction below is most of the
  // original kernel rather than a sum.
  // attention-output's inner extent is WIDTH, and it stages the whole of it in
  // workgroup memory rather than in chunks - so unlike ffw-out its span has no
  // divisibility constraint beyond dividing WIDTH.
  const attnKSplits = Math.max(1, shape.attnKSplits ?? 1);
  const attnKSpan = width / attnKSplits;
  if (attnKSplits > 1 && !Number.isInteger(attnKSpan)) {
    throw new Error(`attnKSplits ${attnKSplits} does not divide the width ${width}`);
  }
  const outKSplits = Math.max(1, shape.outKSplits ?? 1);
  const outKSpan = intermediate / outKSplits;
  // 🔴 A PART WALKS ITS SPAN IN CHUNKS, SO THE CHUNK MUST DIVIDE THE SPAN. With
  // an intermediate of 1536 and a chunk of 384 the only safe part counts are 1,
  // 2 and 4; at 8 the span is 192 and the loop `chunk0 < start + 192` still
  // runs one whole 384-wide iteration, reading the NEXT part's slice and
  // double-counting it. That is a silent wrong answer, not a slow one, so it
  // throws here rather than being clamped.
  if (outKSplits > 1 && (!Number.isInteger(outKSpan) || outKSpan % outChunk !== 0)) {
    throw new Error(`outKSplits ${outKSplits} gives a span of ${outKSpan}, `
      + `which the out chunk ${outChunk} does not divide`);
  }
  if (channels % kSplits !== 0) {
    throw new Error(`kSplits ${kSplits} does not divide ${channels} channels`);
  }
  const kSpan = channels / kSplits;
  // A part walks its own span in chunks, so the chunk can never be longer than
  // the span - otherwise one part would run past the next part's start.
  const channelChunk = Math.min(channels, kSpan, shape.channelChunk ?? 256);
  if (channels % channelChunk !== 0) {
    throw new Error(`channelChunk ${channelChunk} does not divide ${channels}`);
  }
  if (kSpan % channelChunk !== 0) {
    throw new Error(`channelChunk ${channelChunk} does not divide the K span ${kSpan}`);
  }
  // 🔴 THE TILE HELPERS ARE A FACTORY BECAUSE ONE KERNEL WANTS A DIFFERENT
  // TILE FROM THE REST. They used to be plain constants over the single `tile`,
  // which is right while every kernel shares one - and qkvg no longer does: it
  // is the only projection with a K split behind it, so it is the only one that
  // can afford a big tile. Bound to one number, raising it for qkvg raised it
  // for adaln and ffw-adaln too, and those measured 0.54x.
  const tilingFor = (t) => {
    const tileWidth = Math.min(4, t);
    const tileGroups = t / tileWidth;
    const tileLanes = { 1: "f32", 2: "vec2<f32>", 4: "vec4<f32>" }[tileWidth];
    if (tileLanes === undefined || !Number.isInteger(tileGroups)) {
      throw new Error(`tile ${t} is not 1, 2 or a multiple of 4`);
    }
    // Token i of the tile lives in group i/tileWidth, lane i%tileWidth. The
    // staged activations are indexed by group and channel; a register array
    // only by group.
    const lane = (i) => (tileWidth === 1 ? "" : `.${"xyzw"[i % tileWidth]}`);
    const group = (i) => Math.floor(i / tileWidth);
    const overTile = (body) =>
      Array.from({ length: t }, (_, i) => body(i)).join("\n      ");
    const overGroups = (body) =>
      Array.from({ length: tileGroups }, (_, g) => body(g)).join("\n      ");
    const stagedAt = (i) => `xt[${group(i)}u * CHANNEL_CHUNK + cc]${lane(i)}`;
    const stageChunk = `    workgroupBarrier();
    for (var cc = local; cc < CHANNEL_CHUNK; cc += ${lanes}u) {
      ${overTile((i) => `{
        let token = base_token + ${i}u;
        var value = 0.0;
        if (token < ${ROWS}) { value = xbuf[token * C + c0 + cc]; }
        ${stagedAt(i)} = value;
      }`)}
    }
    workgroupBarrier();`;
    return { tile: t, tileWidth, tileGroups, tileLanes, lane, group,
             overTile, overGroups, stagedAt, stageChunk };
  };
  const { tileWidth, tileGroups, tileLanes, lane, group,
          overTile, overGroups, stagedAt, stageChunk } = tilingFor(tile);
  void tileWidth; void stagedAt;

  /**
   * 🔴 ONE OUTPUT AN INVOCATION, WHICH IS WHAT MAKES THE TOKEN TILE AFFORDABLE.
   * The accumulators are (matrices x tile groups x outputs a lane) vectors, and
   * every one of them is live across the whole channel loop - so a second
   * output a lane doubles the registers, and at eight tokens that was enough to
   * spill: tile 8 measured 542 ms against tile 4's 332 at 240 tokens with two
   * outputs a lane, and the traffic model says it should have been ~200.
   *
   * So each kernel splits its own output range to exactly `lanes` wide rather
   * than sharing one `splits`. Their ranges differ - the attention projection
   * is heads*dimension, the widening is the doubled intermediate - so one
   * number cannot make both exact.
   */
  const splitFor = (range, name) => {
    if (range % lanes !== 0) {
      throw new Error(`${name} ${range} is not a multiple of ${lanes} lanes`);
    }
    return range / lanes;
  };
  const qkvgSplits = splitFor(width, "the attention projection's width");
  const wideSplits = splitFor(intermediate, "the transition's intermediate");
  const outSplits = splitFor(channels, "the channel count");



  const common = `${weight16 ? "enable f16;\n" : ""}
const TOKENS: u32 = ${tokens}u;
const PAIRS: u32 = ${pairs}u;
const C: u32 = ${channels}u;
const C_COND: u32 = ${condChannels}u;
const C_PAIR: u32 = ${pairChannels}u;
const HEADS: u32 = ${heads}u;
const DIMENSION: u32 = ${dimension}u;
const WIDTH: u32 = ${width}u;
const INTERMEDIATE: u32 = ${intermediate}u;
const GRID_WIDTH: u32 = ${GRID_WIDTH}u;
const EPSILON: f32 = 1.0e-5;
const SCALE: f32 = ${1 / Math.sqrt(dimension)};
${Object.entries(offsets).map(([name, value]) => `const W_${name}: u32 = ${value}u;`).join("\n")}

fn logistic(value: f32) -> f32 { return 1.0 / (1.0 + exp(-value)); }
fn swish(value: f32) -> f32 { return value / (1.0 + exp(-value)); }
`;

  // The shared LayerNorm over the pair conditioning. Two-pass, no offset.
  const normalisePair = `
const PAIRS: u32 = ${pairs}u;
const C_PAIR: u32 = ${pairChannels}u;
const GRID_WIDTH: u32 = ${GRID_WIDTH}u;
const EPSILON: f32 = 1.0e-5;
@group(0) @binding(0) var<storage, read> pair_cond: array<f32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read_write> normalized: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x + id.y * GRID_WIDTH * 64u;
  if (row >= PAIRS) { return; }
  let base = row * C_PAIR;
  var total = 0.0;
  for (var c = 0u; c < C_PAIR; c += 1u) { total += pair_cond[base + c]; }
  let mean = total / f32(C_PAIR);
  var variance = 0.0;
  for (var c = 0u; c < C_PAIR; c += 1u) {
    let d = pair_cond[base + c] - mean;
    variance += d * d;
  }
  let inverse_std = inverseSqrt(variance / f32(C_PAIR) + EPSILON);
  for (var c = 0u; c < C_PAIR; c += 1u) {
    normalized[base + c] = (pair_cond[base + c] - mean) * inverse_std * scale[c];
  }
}`;

  // One super-block's projection, unpacked into per-block head-major logits.
  // 🔴 THE PROJECTION IS (pair, blocksPerSuper, heads) and the attention wants
  // (head, i, j) - so this pass is where the six-by-four nesting is resolved.
  // 🔴 ONE SHADER PER POSITION IN THE SUPER-BLOCK. The inner index selects a
  // column group of the projection, and the pipeline cache here takes no
  // override constants - so it is baked in rather than passed. Four sources,
  // compiled once each, not twenty-four.
  // 🔴 THE HEADS ARE CONTIGUOUS IN THE PROJECTION, SO THEY ARE THE VECTOR. This
  // is the largest kernel of the stack once a protein is any size - 27 of 108
  // ms at 240 tokens, because it is quadratic in tokens where the token
  // projections are linear. It used to loop heads outside channels, re-reading
  // the normalised pair row for every one of the sixteen and reading one weight
  // per multiply-add: 48 instructions to buy 16. Channels outside, heads as
  // four vec4s, it is nine.
  const headVectors = heads / 4;
  const overHeadVectors = (body) =>
    Array.from({ length: headVectors }, (_, h) => body(h)).join("\n    ");
  const pairLogitsFor = (inner, perSuper) => `${common}
const INNER: u32 = ${inner}u;
const PER_SUPER: u32 = ${perSuper}u;
@group(0) @binding(0) var<storage, read> normalized: array<f32>;
// ...as vec4, which is why the column base must be a multiple of four: it is
// c * PER_SUPER * HEADS + INNER * HEADS, and HEADS is sixteen.
@group(0) @binding(1) var<storage, read> projection: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> logits: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x + id.y * GRID_WIDTH * 64u;
  if (row >= PAIRS) { return; }
  let base = row * C_PAIR;
  ${overHeadVectors((h) => `var total${h} = vec4<f32>(0.0);`)}
  for (var c = 0u; c < C_PAIR; c += 1u) {
    // ...read once, used by every head.
    let x = normalized[base + c];
    let column = (c * PER_SUPER * HEADS + INNER * HEADS) / 4u;
    ${overHeadVectors((h) => `total${h} += x * projection[column + ${h}u];`)}
  }
  ${overHeadVectors((h) => Array.from({ length: 4 }, (_, l) =>
    `logits[(${h * 4 + l}u) * PAIRS + row] = total${h}.${"xyzw"[l]};`).join("\n    "))}
}`;

  // 🔴 THE WEIGHTS WERE THE BANDWIDTH, NOT THE ARITHMETIC. One workgroup per
  // token meant every workgroup read the block's whole weight set: 5.9M floats
  // for 2.4M multiply-adds, an arithmetic intensity of a quarter of a MAC per
  // byte where this device needs about twelve to be compute bound. A call read
  // 33 GB of weights that way, which at ~350 GB/s is the 107 ms it took.
  //
  // So the projection now runs over a TILE of tokens at once and each weight it
  // reads serves all of them. The AdaLN that produces the tile's input is
  // per-token and cannot tile - it is a different shape of work - so it moves
  // into its own pass and hands `x` over through a buffer.
  /**
   * AdaLN: normalise the activation, normalise and project the conditioning
   * into a scale and a shift, and apply them. Both halves of a block use it,
   * with different weights, so it is generated twice from here.
   *
   * 🔴 ONE WEIGHT READ SERVES THE WHOLE TILE, which is the point of tiling it.
   * One workgroup a token read both 384x768 conditioning matrices per token -
   * 2.4 MB each way, 13.6 GB a call at 240 tokens - and the two AdaLN passes
   * were 16.5 ms of a 104 ms stack there. The LayerNorms stay per token,
   * sequential over the tile, because their reductions are not shared.
   */
  // 🔴 THIS KERNEL IS ITS REDUCTIONS, NOT ITS PROJECTION, AND THE TILE SWEEP IS
  // WHAT SAYS SO. A weight-bound kernel gets FASTER with a bigger token tile -
  // that is the whole reason qkvg and ffw-wide wanted one. adaln gets 1.85x
  // SLOWER (112.5 -> 208.3 us at tile 2), because its per-token loop runs four
  // barrier-tree reductions and doubling the tile doubles them while halving
  // the workgroups. So the fix here is the opposite of the one those two
  // needed: collapse the trees, leave the tile alone.
  //
  // Four reductions a token, each a 1 + log2(lanes) barrier tree, is 36
  // barriers at 256 lanes. subgroupAdd reduces within a subgroup with no
  // barrier at all, leaving one write per subgroup and a pass over
  // lanes/subgroupSize values - 2 barriers a reduction, 8 a token. It is the
  // same move `attend` already makes; this kernel never got it.
  const normGroups = Math.ceil(lanes / subgroupSize);
  const reduceBody = normSubgroups
    ? `  // 🔴 THE LEADING BARRIER IS NOT OPTIONAL. reduce_a is reused by every
  // call, and without it a fast subgroup could overwrite a slot the previous
  // reduction's final pass has not read yet.
  workgroupBarrier();
  let wide = subgroupAdd(value);
  if (local % ${subgroupSize}u == 0u) { reduce_a[local / ${subgroupSize}u] = wide; }
  workgroupBarrier();
  var total = 0.0;
  for (var i = 0u; i < ${normGroups}u; i += 1u) { total += reduce_a[i]; }
  return total;`
    : `  reduce_a[local] = value;
  workgroupBarrier();
  for (var stride = ${lanes / 2}u; stride > 0u; stride >>= 1u) {
    if (local < stride) { reduce_a[local] += reduce_a[local + stride]; }
    workgroupBarrier();
  }
  return reduce_a[0];`;
  // 🔴 THE CONDITIONING'S MEAN AND VARIANCE DO NOT DEPEND ON THE BLOCK, and
  // every block was recomputing them. Only the SCALE is per block
  // (SingleCondLayerNormScale, and a separate one for the transition), so
  // `(cond - mean) * inverse` is one tensor for the whole call - and it was
  // being derived 24 blocks x 2 kernels x 3 channel splits = 144 times, by
  // workgroups that each ran two barrier-tree reductions to get it.
  //
  // Hoisting it is what the K-split attempt above says to do: that split lost
  // because it multiplied this phase rather than the projection, and the fix
  // for work that should not repeat is to stop repeating it, not to divide the
  // other half more finely.
  const normaliseCond = `${normSubgroups
    ? "enable subgroups;\nenable subgroup_size_control;\n" : ""}${common}
@group(0) @binding(0) var<storage, read> cond: array<f32>;
@group(0) @binding(1) var<storage, read_write> normalised: array<f32>;

var<workgroup> reduce_c: array<f32, ${lanes}>;

fn reduce_cond(local: u32, value: f32) -> f32 {
${normSubgroups
    ? `  workgroupBarrier();
  let wide = subgroupAdd(value);
  if (local % ${subgroupSize}u == 0u) { reduce_c[local / ${subgroupSize}u] = wide; }
  workgroupBarrier();
  var total = 0.0;
  for (var i = 0u; i < ${Math.ceil(lanes / subgroupSize)}u; i += 1u) { total += reduce_c[i]; }
  return total;`
    : `  reduce_c[local] = value;
  workgroupBarrier();
  for (var stride = ${lanes / 2}u; stride > 0u; stride >>= 1u) {
    if (local < stride) { reduce_c[local] += reduce_c[local + stride]; }
    workgroupBarrier();
  }
  return reduce_c[0];`}
}

@compute @workgroup_size(${lanes})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let token = group.x + group.y * GRID_WIDTH;
  if (token >= TOKENS) { return; }
  let local = local_id.x;
  let base = token * C_COND;
  var total = 0.0;
  for (var c = local; c < C_COND; c += ${lanes}u) { total += cond[base + c]; }
  let mean = reduce_cond(local, total) / f32(C_COND);
  workgroupBarrier();
  var centred = 0.0;
  for (var c = local; c < C_COND; c += ${lanes}u) {
    let d = cond[base + c] - mean;
    centred += d * d;
  }
  let inverse = inverseSqrt(reduce_cond(local, centred) / f32(C_COND) + EPSILON);
  workgroupBarrier();
  for (var c = local; c < C_COND; c += ${lanes}u) {
    normalised[base + c] = (cond[base + c] - mean) * inverse;
  }
}`;

  const conditionedNorm = (prefix, bindingsFor, mode = "fused") => `${normSubgroups
    ? "enable subgroups;\nenable subgroup_size_control;\n" : ""}${common}
const TILE: u32 = ${tile}u;
${bindingsFor(mode)}

${batchedGates ? "// cond_norm is gone: the projection that read it moved to zero-gates."
  : `var<workgroup> cond_norm: array<${tileLanes}, ${tileGroups * (mode === "split" ? normKSpan : condChannels)}>;`}
var<workgroup> reduce_a: array<f32, ${lanes}>;
var<workgroup> act_means: array<f32, ${tile}>;
var<workgroup> act_inverses: array<f32, ${tile}>;

fn reduce_sum(local: u32, value: f32) -> f32 {
${reduceBody}
}

@compute @workgroup_size(${lanes})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let base_token = group.x * TILE;
  if (base_token >= ${ROWS}) { return; }
  let local = local_id.x;

  // 🔴 TWO-PASS VARIANCE, and no scale or offset on the activation's own norm.
  // A token past the end is clamped rather than skipped: every lane has to
  // reach the barriers, and its lane of the vector is dropped at the write.
  for (var t = 0u; t < TILE; t += 1u) {
    let token = min(base_token + t, ${ROWS} - 1u);
${mode === "split" ? "" : `    var total = 0.0;
    for (var c = local; c < C; c += ${lanes}u) { total += act[token * C + c]; }`}
${mode === "split" ? "" : `    let act_mean = reduce_sum(local, total) / f32(C);
    workgroupBarrier();
    var centred = 0.0;
    for (var c = local; c < C; c += ${lanes}u) {
      let d = act[token * C + c] - act_mean;
      centred += d * d;
    }
    let act_inverse = inverseSqrt(reduce_sum(local, centred) / f32(C) + EPSILON);
    workgroupBarrier();
    if (local == 0u) {
      act_means[t] = act_mean;
      act_inverses[t] = act_inverse;
    }
    workgroupBarrier();`}

${mode === "reduce" || batchedGates ? "" : `    // 🔴 THE NORMALISED CONDITIONING IS READ, NOT DERIVED. normalise-cond
    // above produced it once for the whole call; all that is left here is the
    // per-block scale and the staging, which is a copy.
    // 🔴 A SPLIT PART STAGES ONLY ITS SLICE. Staging the whole conditioning in
    // every part is what made the first two attempts at this split lose: the
    // loop below is OUTSIDE the projection that gets sliced, so four parts did
    // four times the staging to do a quarter of the projection each. It is the
    // same trap attention-output's split hit, written down there and not
    // applied here.
    for (var c = local; c < ${mode === "split" ? `${normKSpan}u` : "C_COND"}; c += ${lanes}u) {
      let at = ${mode === "split" ? `group.z * ${normKSpan}u + c` : "c"};
      let value = cond[${perToken("token")} * C_COND + at]
        * ${wf(`weights[W_${prefix}SingleCondLayerNormScale + at]`)};
      ${Array.from({ length: tile }, (_, t) =>
        `if (t == ${t}u) { cond_norm[${group(t)}u * ${mode === "split" ? `${normKSpan}u` : "C_COND"} + c]${lane(t)} = value; }`)
        .join("\n      ")}
    }
    workgroupBarrier();`}
  }

  // 🔴 ONE WORKGROUP PER TOKEN IS 68 OF THEM, AND THIS DEVICE HAS 108 CORES.
  // The projection below is the whole cost of this kernel - its four
  // reductions are 0.4% of its memory traffic, which is why collapsing them
  // with subgroupAdd measured 2.71 -> 2.70 ms, nothing - and it was running at
  // 8% occupancy. Splitting the OUTPUT channels across workgroups multiplies
  // the parallelism by C/lanes and costs only the norms, recomputed once per
  // split: the phase above reads about ten values a lane where the projection
  // reads 2304, so paying for it three times is under 2%.
  //
  // The token tile cannot do this - it goes the wrong way, halving the
  // workgroups - which is why adaln measured 1.85x SLOWER at tile 2 and why
  // this kernel needed the opposite fix from qkvg's.
  ${normSplit
    ? `{
    let c = group.y * ${lanes}u + local;`
    : `for (var c = local; c < C; c += ${lanes}u) {`}
    ${overGroups((g) =>
      // 🔴 ONLY THE FUSED KERNEL SEEDS WITH THE BIAS, AND THIS READ
      // `mode === "split" ? 0.0 : bias` - WHICH GAVE IT TO "reduce" TOO. The
      // reduce then adds the bias again in its epilogue, exactly as its comment
      // says it should, so every conditioned norm came out with TWICE the bias
      // on its scale. That scale feeds a sigmoid gate, so the error is large
      // (relRMS 0.813 at the module level) and - because a doubled constant
      // does not depend on how many parts were summed - it is IDENTICAL at
      // normKSplits 2 and 4. That count-independence is what made it look
      // structural for so long.
      //
      // Present since the split was written (311d7a4), which benchmarked the
      // arms and never folded one. See tools/gpu/check-difftx-splits.js.
      `var scale${g} = ${tileLanes}(${mode === "fused" && !batchedGates
        ? wf(`weights[W_${prefix}SingleCondScaleBias + c]`) : "0.0"});
    var shift${g} = ${tileLanes}(0.0);`)}
${batchedGates
  ? `    // 🔴 BOTH ALREADY COMPUTED, FOR EVERY BLOCK, IN ONE DISPATCH. The
    // projection that stood here - C_COND x C against a conditioning that does
    // not move with the block - is phase B or C of zero-gates now. What is left
    // of this kernel is the activation's OWN LayerNorm, which does read the
    // block's activation, and the elementwise apply.
    //
    // 🔴 PER TOKEN, NOT PER ROW: the scale and shift come from the
    // conditioning, so a batched sampler shares them across its samples.
    ${overTile((t) => `{
      let token = base_token + ${t}u;
      if (token < ${ROWS}) {
        let at = ${perToken("token")} * 2u * C + c;
        scale${group(t)}${lane(t)} = scale_shift[at];
        shift${group(t)}${lane(t)} = scale_shift[at + C];
      }
    }`)}`
  : mode === "reduce"
  ? `    // 🔴 THE BIAS LANDS HERE, ONCE. A part cannot add it - there are
    // normKSplits of them - so the split kernel leaves the scale unbiased and
    // this sums the parts and then adds it.
    {
      let stride = ${rows}u * C;
      for (var part = 0u; part < ${normKSplits}u; part = part + 1u) {
        ${overTile((t) => `{
          let token = base_token + ${t}u;
          if (token < ${ROWS}) {
            let at = (part * 2u) * stride + token * C + c;
            scale${group(t)}${lane(t)} += partials[at];
            shift${group(t)}${lane(t)} += partials[at + stride];
          }
        }`)}
      }
    }
    ${overGroups((g) => `scale${g} += ${tileLanes}(${wf(`weights[W_${prefix}SingleCondScaleBias + c]`)});`)}`
  : `    for (var d = 0u; d < ${mode === "split" ? `${normKSpan}u` : "C_COND"}; d += 1u) {
      let dw = ${mode === "split" ? `group.z * ${normKSpan}u + d` : "d"};
      let ws = ${wf(`weights[W_${prefix}SingleCondScaleWeights + dw * C + c]`)};
      let wb = ${wf(`weights[W_${prefix}SingleCondBias + dw * C + c]`)};
      ${overGroups((g) => `{
        let cn = cond_norm[${g}u * ${mode === "split" ? `${normKSpan}u` : "C_COND"} + d];
        scale${g} += cn * ws;
        shift${g} += cn * wb;
      }`)}
    }`}
${mode === "split"
  ? `    ${overTile((t) => `{
      let token = base_token + ${t}u;
      if (token < ${ROWS}) {
        let stride = ${rows}u * C;
        let at = (group.z * 2u) * stride + token * C + c;
        partials[at] = scale${group(t)}${lane(t)};
        partials[at + stride] = shift${group(t)}${lane(t)};
      }
    }`)}`
  : `    ${overGroups((g) => `let gated${g} =
      ${tileLanes}(1.0) / (${tileLanes}(1.0) + exp(-scale${g}));`)}
    ${overTile((t) => `{
      let token = base_token + ${t}u;
      if (token < ${ROWS}) {
        let normalized = (act[token * C + c] - act_means[${t}u]) * act_inverses[${t}u];
        xbuf[token * C + c] =
          gated${group(t)}${lane(t)} * normalized + shift${group(t)}${lane(t)};
      }
    }`)}`}
  }
}`;

  // 🔴 THE TWO CALLERS BIND IN DIFFERENT ORDERS, so the bindings are a function
  // of the mode rather than one string. The split and reduce forms happen to be
  // the SAME for both - a split part reads the conditioning and the weights, a
  // reduction reads the activation and the weights - so only the fused form
  // differs, which is the form that already differed.
  const splitBindings = `@group(0) @binding(0) var<storage, read> cond: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<${weightPrecision}>;
@group(0) @binding(2) var<storage, read_write> partials: array<f32>;`;
  const reduceBindings = `@group(0) @binding(0) var<storage, read> act: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<${weightPrecision}>;
@group(0) @binding(2) var<storage, read> partials: array<f32>;
@group(0) @binding(3) var<storage, read_write> xbuf: array<f32>;`;
  const normBindings = (fused) => (mode) => (
    mode === "split" ? splitBindings : mode === "reduce" ? reduceBindings : fused);

  const adalnBindings = normBindings(batchedGates ? `@group(0) @binding(0) var<storage, read> act: array<f32>;
// 🔴 THREE BINDINGS, NOT FOUR. The projection was this kernel's ONLY read of
// the weights; hoisted, that binding goes unread and an auto layout DROPS it,
// so a positional bind group of four fails. Same trap as ffw-out-reduce's.
@group(0) @binding(1) var<storage, read> scale_shift: array<f32>;
@group(0) @binding(2) var<storage, read_write> xbuf: array<f32>;`
    : `@group(0) @binding(0) var<storage, read> act: array<f32>;
@group(0) @binding(1) var<storage, read> cond: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<${weightPrecision}>;
@group(0) @binding(3) var<storage, read_write> xbuf: array<f32>;`);
  const adaln = conditionedNorm("", adalnBindings,
    normKSplits > 1 ? "split" : "fused");
  const adalnReduce = normKSplits > 1 ? conditionedNorm("", adalnBindings, "reduce") : null;

  // 🔴 qkvg IS BUILT UNDER ITS OWN TILING, and the template below is unchanged
  // by that: the destructuring shadows the shared helpers for this block only,
  // so the same text emits a tile-4 kernel here and a tile-1 one everywhere
  // else. `qkvgTile` is the token tile the K split pays for.
  const qkvg = (() => {
  const { tileGroups, tileLanes, lane, group, overTile, overGroups, stageChunk } =
    tilingFor(qkvgTile);
  const tile = qkvgTile;
  void tile;
  return `${common}
const TILE: u32 = ${tile}u;
const SPLITS: u32 = ${splits}u;
@group(0) @binding(0) var<storage, read> xbuf: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<${weightPrecision}>;
${kSplits > 1
  ? `// One partial per K part per output, summed by qkvgReduce.
@group(0) @binding(2) var<storage, read_write> partials: array<f32>;`
  : `@group(0) @binding(2) var<storage, read_write> q: array<f32>;
@group(0) @binding(3) var<storage, read_write> k: array<f32>;
@group(0) @binding(4) var<storage, read_write> v: array<f32>;
@group(0) @binding(5) var<storage, read_write> gate: array<f32>;`}

const CHANNEL_CHUNK: u32 = ${channelChunk}u;
var<workgroup> xt: array<${tileLanes}, ${tileGroups * channelChunk}>;

@compute @workgroup_size(${lanes})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let base_token = group.x * TILE;
  let local = local_id.x;
  let out = group.y * ${lanes}u + local;

  ${kSplits > 1 ? `let k_start = group.z * ${kSpan}u;
  let k_stop = k_start + ${kSpan}u;` : ""}
  ${overGroups((g) => `var q${g} = ${tileLanes}(${kSplits > 1 ? "0.0" : wf(`weights[W_qBias + out]`)});   // only q has a bias, and when K is split the reduction adds it
  var k${g} = ${tileLanes}(0.0);
  var v${g} = ${tileLanes}(0.0);
  var g${g} = ${tileLanes}(0.0);`)}

  for (var c0 = ${kSplits > 1 ? "k_start" : "0u"}; c0 < ${kSplits > 1 ? "k_stop" : "C"}; c0 += CHANNEL_CHUNK) {
${stageChunk}
    // 🔴 FOUR WEIGHTS READ ONCE, USED TILE TIMES. That ratio is the whole point
    // of this kernel - and this stack streams all 566 MB of its weights once
    // per TILE of tokens, so it is also the whole of its cost.
    for (var cc = 0u; cc < CHANNEL_CHUNK; cc += 1u) {
      let column = (c0 + cc) * WIDTH + out;
      let wq = ${wf(`weights[W_qProjection + column]`)};
      let wk = ${wf(`weights[W_kProjection + column]`)};
      let wv = ${wf(`weights[W_vProjection + column]`)};
      let wg = ${wf(`weights[W_gatingQuery + column]`)};
      ${overGroups((g) => `{
        let x = xt[${g}u * CHANNEL_CHUNK + cc];
        q${g} += x * wq;
        k${g} += x * wk;
        v${g} += x * wv;
        g${g} += x * wg;
      }`)}
    }
  }

  ${overTile((t) => `{
    let token = base_token + ${t}u;
    if (token < ${ROWS}) {
      let index = token * WIDTH + out;
${kSplits > 1
    ? `      let slot = (group.z * 4u) * ${rows}u * WIDTH + index;
      let stride = ${rows}u * WIDTH;
      partials[slot] = q${group(t)}${lane(t)};
      partials[slot + stride] = k${group(t)}${lane(t)};
      partials[slot + stride * 2u] = v${group(t)}${lane(t)};
      partials[slot + stride * 3u] = g${group(t)}${lane(t)};`
    : `      q[index] = q${group(t)}${lane(t)};
      k[index] = k${group(t)}${lane(t)};
      v[index] = v${group(t)}${lane(t)};
      gate[index] = g${group(t)}${lane(t)};`}
    }
  }`)}
}`;
  })();

  const attend = `${subgroups ? "enable subgroups;\nenable subgroup_size_control;\n" : ""}${common}
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> pair_logits: array<f32>;
@group(0) @binding(4) var<storage, read> mask: array<f32>;
@group(0) @binding(5) var<storage, read_write> gathered: array<f32>;

var<workgroup> logits: array<f32, ${tokens}>;
${stageKeys ? `var<workgroup> k_tile: array<f32, ${keyChunk * dimension}>;
var<workgroup> q_tile: array<f32, ${dimension}>;` : ""}
var<workgroup> reduce: array<f32, ${subgroups ? groupsOfLanes : lanes}>;

@compute @workgroup_size(${lanes})${subgroups ? ` @subgroup_size(${subgroupSize})` : ""}
fn main(@builtin(workgroup_id) group: vec3<u32>,
${subgroups ? "        @builtin(subgroup_invocation_id) subgroup_lane: u32,\n        @builtin(subgroup_id) subgroup_index: u32,\n" : ""}        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let slot = group.x + group.y * GRID_WIDTH;
  if (slot >= ${ROWS} * HEADS) { return; }
  let head = slot % HEADS;
  let i = slot / HEADS;
  let local = local_id.x;
  let query_base = i * WIDTH + head * DIMENSION;
${samples === 1 ? "" : `  // 🔴 THE KEYS STAY INSIDE THE QUERY'S SAMPLE. A wider row axis batches every
  // other kernel in this stack for free; this one would silently attend ACROSS
  // samples, which is not a slower answer but a different model.
  let key_base = (i / TOKENS) * TOKENS;
  let token_i = i % TOKENS;`}

${stageKeys ? `  for (var d = local; d < DIMENSION; d += ${lanes}u) {
    q_tile[d] = q[query_base + d];
  }
  for (var base = 0u; base < TOKENS; base += ${keyChunk}u) {
    let count = min(${keyChunk}u, TOKENS - base);
    workgroupBarrier();
    // Indexed over (key, channel) together, so consecutive lanes are
    // consecutive addresses.
    for (var idx = local; idx < count * DIMENSION; idx += ${lanes}u) {
      k_tile[idx] = k[(${keyRow("base + idx / DIMENSION")}) * WIDTH + head * DIMENSION
        + idx % DIMENSION];
    }
    workgroupBarrier();
    for (var jj = local; jj < count; jj += ${lanes}u) {
      var dot = 0.0;
      for (var d = 0u; d < DIMENSION; d += 1u) {
        dot += q_tile[d] * k_tile[jj * DIMENSION + d];
      }
      let j = base + jj;
      logits[j] = dot * SCALE + 1.0e9 * (mask[j] - 1.0)
        + pair_logits[(head * TOKENS + ${tokenI}) * TOKENS + j];
    }
  }
  workgroupBarrier();` : `  for (var j = local; j < TOKENS; j += ${lanes}u) {
    var dot = 0.0;
    for (var d = 0u; d < DIMENSION; d += 1u) {
      dot += q[query_base + d] * k[${keyRow("j")} * WIDTH + head * DIMENSION + d];
    }
    logits[j] = dot * SCALE + 1.0e9 * (mask[j] - 1.0)
      + pair_logits[(head * TOKENS + ${tokenI}) * TOKENS + j];
  }
  workgroupBarrier();`}

  var local_max = -3.0e38;
  for (var j = local; j < TOKENS; j += ${lanes}u) { local_max = max(local_max, logits[j]); }
${subgroups ? `  let wide_max = subgroupMax(local_max);
  if (subgroup_lane == 0u) { reduce[subgroup_index] = wide_max; }
  workgroupBarrier();
  var largest = reduce[0];
  for (var g = 1u; g < ${groupsOfLanes}u; g += 1u) { largest = max(largest, reduce[g]); }
  workgroupBarrier();` : `  reduce[local] = local_max;
  workgroupBarrier();
  for (var stride = ${lanes / 2}u; stride > 0u; stride >>= 1u) {
    if (local < stride) { reduce[local] = max(reduce[local], reduce[local + stride]); }
    workgroupBarrier();
  }
  let largest = reduce[0];
  workgroupBarrier();`}

  var local_sum = 0.0;
  for (var j = local; j < TOKENS; j += ${lanes}u) {
    let value = exp(logits[j] - largest);
    logits[j] = value;
    local_sum += value;
  }
${subgroups ? `  let wide_sum = subgroupAdd(local_sum);
  if (subgroup_lane == 0u) { reduce[subgroup_index] = wide_sum; }
  workgroupBarrier();
  var total = reduce[0];
  for (var g = 1u; g < ${groupsOfLanes}u; g += 1u) { total = total + reduce[g]; }
  workgroupBarrier();` : `  reduce[local] = local_sum;
  workgroupBarrier();
  for (var stride = ${lanes / 2}u; stride > 0u; stride >>= 1u) {
    if (local < stride) { reduce[local] += reduce[local + stride]; }
    workgroupBarrier();
  }
  let total = reduce[0];
  workgroupBarrier();`}

  for (var d = local; d < DIMENSION; d += ${lanes}u) {
    var sum = 0.0;
    for (var j = 0u; j < TOKENS; j += 1u) {
      sum += logits[j] * v[${keyRow("j")} * WIDTH + head * DIMENSION + d];
    }
    gathered[i * WIDTH + head * DIMENSION + d] = sum / total;
  }
}`;

  // Gate, project back, apply the zero-init gate, and add to the residual.
  // 🔴 A TILE OF TOKENS AND ONE OUTPUT AN INVOCATION. This reads the whole
  // 768x768 output projection to project a token - 2.4 MB each, 3.3 GB a call -
  // so the tile is what divides it. Tiled ALONE it measured worse (69 ms
  // against 65 for the stack), because it also divides the workgroups and this
  // kernel had one per token; splitting the output range to exactly `lanes`
  // wide puts them back, and leaves each invocation a single accumulator.
  // 🔴 SAME SPLIT AS ffw-out AND FOR THE SAME REASON: the zero-init gate, the
  // sigmoid and the residual add are loop-invariant across a K split, so they
  // move into the reduction and run once while only the projection divides. A
  // split part declares three bindings, contiguously - it never reads `cond`,
  // and an unused binding is dropped from an `auto` layout, which a positional
  // bind group then fails to match.
  const { attentionOutput, attentionOutputReduce } = (() => {
  // 🔴 ITS OWN TILE, FOR THE REASON THE FILE ALREADY GIVES FOR qkvg's:
  // "splitting K and the token tile are one decision... the split is what makes
  // a bigger tile affordable". attention-output re-reads a 768x768 projection
  // once per TOKEN GROUP - 1.93 GB a step at a tile of 1 and 68 tokens, against
  // a measured 1.43 ms and a 1.28 ms bandwidth floor, so it is bound by exactly
  // that. The shared tile cannot be raised to fix it because adaln and
  // ffw-adaln measured 0.54x when a raised tile reached them.
  const { tileGroups, tileLanes, lane, group, overTile, overGroups } =
    tilingFor(attnOutTile);
  const attentionOutputFor = (mode) => `${common}
const TILE: u32 = ${attnOutTile}u;
${mode === "split"
  ? `@group(0) @binding(0) var<storage, read> gathered: array<f32>;
@group(0) @binding(1) var<storage, read> gate: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<${weightPrecision}>;
@group(0) @binding(3) var<storage, read_write> partials: array<f32>;`
  : mode === "reduce"
    ? (batchedGates
      // 🔴 THREE BINDINGS ONCE THE GATE IS HOISTED, for the third time in this
      // file: the epilogue was this reduction's only read of `cond` AND of
      // `weights`, and an auto layout drops both. Latent until now because
      // attnSplits is 1 by default, so nothing compiled this form.
      ? `@group(0) @binding(0) var<storage, read> partials: array<f32>;
@group(0) @binding(1) var<storage, read> zero_gate: array<f32>;
@group(0) @binding(2) var<storage, read_write> act: array<f32>;`
      : `@group(0) @binding(0) var<storage, read> partials: array<f32>;
@group(0) @binding(1) var<storage, read> cond: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<${weightPrecision}>;
@group(0) @binding(3) var<storage, read_write> act: array<f32>;`)
    : `@group(0) @binding(0) var<storage, read> gathered: array<f32>;
@group(0) @binding(1) var<storage, read> gate: array<f32>;
${batchedGates
  ? `// 🔴 THIS BLOCK'S SLICE OF THE BATCHED GATE, bound at an offset - the
// kernel is compiled once and reused for every block, so it cannot carry a
// block index of its own.
@group(0) @binding(2) var<storage, read> zero_gate: array<f32>;`
  : `@group(0) @binding(2) var<storage, read> cond: array<f32>;`}
@group(0) @binding(3) var<storage, read> weights: array<${weightPrecision}>;
@group(0) @binding(4) var<storage, read_write> act: array<f32>;`}

// The tile's tokens as one vector, so one weight read serves all of them - and
// then, once the projection loop is done with it, the CONDITIONING.
//
// 🔴 THE ZERO-GATE LOOP BELOW READ cond FROM GLOBAL, ONCE PER TOKEN PER
// CONDITIONING CHANNEL, ON EVERY LANE. It is indexed by (token, d) and not by
// the channel a lane owns, so all 256 lanes wanted the same TILE x C_COND
// floats: one weight read, TILE global reads and TILE scalar multiply-adds a
// step, which at width 768 and C_COND 384 is about 60% of this kernel. Staged
// as a vector over the tile it is one weight read, one workgroup read and one
// vector multiply-add.
//
// 🔴 AND IT REUSES THESE SLOTS RATHER THAN TAKING MORE. gated is dead once
// the projection loop has read it, and it is the larger of the two uses at
// WIDTH against C_COND - so the conditioning costs nothing and this kernel's
// residency is unchanged. ffw-out does the same thing for the same reason.
// 🔴 A SPLIT PART STAGES ONLY ITS SLICE, so the array is the span rather than
// the width - and it needs no room for the conditioning, because it has no
// epilogue. Staging the WHOLE width in every part was the first version and it
// made the split a LOSS: 2.11 -> 3.67 ms, because the gate and the logistic
// are outside the projection loop and were being repeated four times.
var<workgroup> gated: array<${tileLanes}, ${tileGroups * (mode === "split"
  ? attnKSpan : (batchedGates ? width : Math.max(width, condChannels)))}>;

@compute @workgroup_size(${lanes})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let base_token = group.x * TILE;
  if (base_token >= ${ROWS}) { return; }
  let local = local_id.x;
  let c = group.y * ${lanes}u + local;

  ${overGroups((g) => `var projected${g} = ${tileLanes}(0.0);`)}
${mode === "reduce"
  ? `  {
    let stride = ${rows}u * C;
    ${overTile((t) => `{
      let token = base_token + ${t}u;
      if (token < ${ROWS}) {
        var total = 0.0;
        for (var part = 0u; part < ${attnKSplits}u; part = part + 1u) {
          total = total + partials[part * stride + token * C + c];
        }
        projected${group(t)}${lane(t)} = total;
      }
    }`)}
  }`
  : `${mode === "split" ? `  let w_start = group.z * ${attnKSpan}u;` : `  let w_start = 0u;`}
  for (var w = local; w < ${mode === "split" ? `${attnKSpan}u` : "WIDTH"}; w += ${lanes}u) {
    ${overTile((t) => `{
      let token = base_token + ${t}u;
      var value = 0.0;
      if (token < ${ROWS}) {
        let index = token * WIDTH + w_start + w;
        value = gathered[index] * logistic(gate[index]);
      }
      gated[${group(t)}u * ${mode === "split" ? `${attnKSpan}u` : "WIDTH"} + w]${lane(t)} = value;
    }`)}
  }
  workgroupBarrier();
  for (var w = 0u; w < ${mode === "split" ? `${attnKSpan}u` : "WIDTH"}; w += 1u) {
    // ...read once, used by every token of the tile.
    let weight = ${wf(`weights[W_Transition2 + (w_start + w) * C + c]`)};
    ${overGroups((g) => `projected${g} += gated[${g}u * ${mode === "split" ? `${attnKSpan}u` : "WIDTH"} + w] * weight;`)}
  }`}
${mode === "split" ? `  ${overTile((t) => `{
    let token = base_token + ${t}u;
    if (token < ${ROWS}) {
      partials[group.z * ${rows}u * C + token * C + c] = projected${group(t)}${lane(t)};
    }
  }`)}` : ""}
${mode === "split" ? "" : `${batchedGates ? `  // 🔴 ALREADY COMPUTED, FOR EVERY BLOCK, IN ONE DISPATCH. This used to stage
  // the conditioning into the slots the projection loop had finished with and
  // then walk a C_COND x C projection right here - inside a kernel running 204
  // workgroups, twenty-four times a step, and once per SAMPLE on top of that.
  // See packZeroGateWeights and the zero-gates pass.
  //
  // 🔴 THE ARITHMETIC IS UNCHANGED AND SO IS ITS ORDER: seeded with the bias,
  // then accumulating cond[d] * w[d] over increasing d, in f32, from the
  // same precision. That is what makes the move bit-exact rather than merely
  // close, which is the bar check-difftx-splits.js holds a split to.
  ${overGroups((g) => `var zero${g} = ${tileLanes}(0.0);`)}
  ${overTile((t) => `{
    let token = base_token + ${t}u;
    if (token < ${ROWS}) {
      zero${group(t)}${lane(t)} = zero_gate[${perToken("token")} * C + c];
    }
  }`)}
` : `  // 🔴 THE ZERO-INIT GATE READS THE RAW CONDITIONING, not the normalised one.
  // ...staged into the slots the projection loop has finished with. The barrier
  // before is what makes reusing them safe.
  workgroupBarrier();
  for (var d = local; d < C_COND; d += ${lanes}u) {
    ${overTile((t) => `{
      let token = base_token + ${t}u;
      var value = 0.0;
      if (token < ${ROWS}) { value = cond[${perToken("token")} * C_COND + d]; }
      gated[${group(t)}u * C_COND + d]${lane(t)} = value;
    }`)}
  }
  workgroupBarrier();
  ${overGroups((g) => `var zero${g} = ${tileLanes}(${wf(`weights[W_AdaptiveZeroCondBias + c]`)});`)}
  for (var d = 0u; d < C_COND; d += 1u) {
    let w = ${wf(`weights[W_AdaptiveZeroCondWeights + d * C + c]`)};
    ${overGroups((g) => `zero${g} += gated[${g}u * C_COND + d] * w;`)}
  }
`}
  ${overGroups((g) => `let contribution${g} = projected${g}
    / (${tileLanes}(1.0) + exp(-zero${g}));`)}
  ${overTile((t) => `{
    let token = base_token + ${t}u;
    if (token < ${ROWS}) {
      act[token * C + c] = act[token * C + c] + contribution${group(t)}${lane(t)};
    }
  }`)}
`}
}`;
  return {
    attentionOutput: attentionOutputFor(attnKSplits > 1 ? "split" : "fused"),
    attentionOutputReduce: attnKSplits > 1 ? attentionOutputFor("reduce") : null };
  })();

  // 🔴 EVERY BLOCK'S ZERO-INIT GATE, IN ONE DISPATCH, OUTSIDE THE BLOCK LOOP.
  // The gate is `bias + cond @ W_zero` and the conditioning does not move with
  // the block, so nothing here is part of the sequential chain the blocks form.
  // It sat inside the loop only because that is where its weights are, and
  // packZeroGateWeights is what takes that reason away.
  //
  // 🔴 z IS THE BLOCK. At 68 tokens this dispatches 17 x 3 x 24 = 1224
  // workgroups where the epilogue it replaces ran 204, twenty-four times over.
  //
  // 🔴 AND IT IS PER TOKEN, NOT PER ROW. `perToken` reduces a row to
  // `row % TOKENS` for the conditioning, so under a batched sampler the old
  // epilogue recomputed an identical gate for every sample.
  // 🔴 THE OFFSETS INSIDE A BLOCK'S SPAN, running sums of packZeroGateWeights'
  // NAMES in that exact order. Derived here once rather than typed, because the
  // packer and the shader disagreeing is a wrong fold and not a crash - see
  // test/af3-zero-gate-pack.test.js, which pins both ends independently.
  const zgProjection = condChannels * channels;
  const zgGate = zgProjection + channels;              // weights then bias
  const zgAdalnBase = 2 * zgGate;                      // past both zero gates
  const zgAdalnSpan = condChannels + 2 * zgProjection + channels;
  const zgOffsets = {
    ZG_FFW: zgGate,
    AD_LN: zgAdalnBase,
    AD_SW: zgAdalnBase + condChannels,
    AD_CB: zgAdalnBase + condChannels + zgProjection,
    AD_SB: zgAdalnBase + condChannels + 2 * zgProjection,
    FA_LN: zgAdalnBase + zgAdalnSpan,
    FA_SW: zgAdalnBase + zgAdalnSpan + condChannels,
    FA_CB: zgAdalnBase + zgAdalnSpan + condChannels + zgProjection,
    FA_SB: zgAdalnBase + zgAdalnSpan + condChannels + 2 * zgProjection,
  };


  // 🔴 EVERY BLOCK'S CONDITIONING PROJECTIONS, IN ONE DISPATCH, OUTSIDE THE
  // BLOCK LOOP. Six projections live here - two zero gates and two conditioned
  // norms' scale-and-shift pairs - and all six read a conditioning that does
  // not move with the block. None of them is part of the sequential chain the
  // blocks form; they sat inside the loop only because that is where their
  // weights are, and packZeroGateWeights takes that reason away.
  //
  // 🔴 z IS THE BLOCK. At 68 tokens this dispatches 17 x 3 x 24 = 1224
  // workgroups where the four kernels it empties ran 204 apiece, twenty-four
  // times a step - the four most underfilled passes in a denoiser call.
  //
  // 🔴 AND IT IS PER TOKEN, NOT PER ROW. `perToken` reduces a row to
  // `row % TOKENS` for the conditioning, so under a batched sampler all of this
  // was recomputed identically for every sample.
  const zeroGates = !batchedGates ? null : (() => {
  const { tileGroups, tileLanes, lane, group, overTile, overGroups } =
    tilingFor(gateTile);
  // One conditioned norm's scale and shift, from `cond * lnScale`. The two
  // prefixes differ only in which four tensors they read, so this is a factory.
  //
  // 🔴 THE LAYER-NORM SCALE CANNOT BE FOLDED INTO THE PROJECTION WEIGHT, which
  // is the obvious optimisation and is wrong. adaln computes
  // `(cond[d] * lnScale[d]) * ws[d][c]`; folding gives `cond[d] * (lnScale[d] *
  // ws[d][c])`, which rounds differently. Staging cond_norm and projecting it
  // is what keeps this bit-exact, and bit-exact is the bar.
  // 🔴 INSIDE THE GATE TILING'S SCOPE, because it emits overTile/overGroups
  // code and would otherwise be built under the shared tile of 1.
  const zgNorm = (ln, sw, cb, sb, out) => `
  // 🔴 ITS OWN SCOPE, because the two prefixes generate the same names from the
  // same tile helpers and WGSL has no shadowing across a function body.
  {
  // ...the staged conditioning is dead once the loop above has read it.
  workgroupBarrier();
  for (var d = local; d < C_COND; d += ${lanes}u) {
    ${overTile((t) => `{
      let token = base_token + ${t}u;
      var value = 0.0;
      if (token < TOKENS) {
        value = cond_normalised[token * C_COND + d] * ${wf(`weights[wbase + ${ln} + d]`)};
      }
      staged[${group(t)}u * C_COND + d]${lane(t)} = value;
    }`)}
  }
  workgroupBarrier();
  ${overGroups((g) => `var scale${g} = ${tileLanes}(${wf(`weights[wbase + ${sb} + c]`)});`)}
  ${overGroups((g) => `var shift${g} = ${tileLanes}(0.0);`)}
  for (var d = 0u; d < C_COND; d += 1u) {
    let ws = ${wf(`weights[wbase + ${sw} + d * C + c]`)};
    let wb = ${wf(`weights[wbase + ${cb} + d * C + c]`)};
    ${overGroups((g) => `{
      let cn${g} = staged[${g}u * C_COND + d];
      scale${g} += cn${g} * ws;
      shift${g} += cn${g} * wb;
    }`)}
  }
  ${overTile((t) => `{
    let token = base_token + ${t}u;
    if (token < TOKENS) {
      let at = (group.z * TOKENS + token) * 2u * C + c;
      ${out}[at] = scale${group(t)}${lane(t)};
      ${out}[at + C] = shift${group(t)}${lane(t)};
    }
  }`)}
  }`;
  return `${common}
const TILE: u32 = ${gateTile}u;
${Object.entries(zgOffsets).map(([name, at]) => `const ${name}: u32 = ${at}u;`).join("\n")}
const ZG_STRIDE: u32 = ${zgAdalnBase + 2 * zgAdalnSpan}u;
// 🔴 TWO CONDITIONINGS, AND THEY ARE NOT INTERCHANGEABLE. The zero gates read
// the RAW conditioning - attention-output's own comment says so and binds
// condBuffer - while the conditioned norms read the NORMALISED one that
// cond-norm produced, which adaln binds in that slot. Feeding the raw buffer
// to both compiles, runs, and folds a different protein: pLDDT 64.29 against
// 85.57, which is how this was caught.
@group(0) @binding(0) var<storage, read> cond: array<f32>;
@group(0) @binding(1) var<storage, read> cond_normalised: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<${weightPrecision}>;
@group(0) @binding(3) var<storage, read_write> gates: array<f32>;
@group(0) @binding(4) var<storage, read_write> ffw_gates: array<f32>;
@group(0) @binding(5) var<storage, read_write> adaln_scale_shift: array<f32>;
@group(0) @binding(6) var<storage, read_write> ffw_adaln_scale_shift: array<f32>;

// 🔴 ONE STAGING ARRAY FOR ALL THREE PHASES, reused behind barriers rather than
// three of them. Three would be ${3 * tileGroups * condChannels * 4 * (tile < 4 ? tile : 4)} bytes of workgroup
// storage and this kernel has no need of them at once.
var<workgroup> staged: array<${tileLanes}, ${tileGroups * condChannels}>;

@compute @workgroup_size(${lanes})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let base_token = group.x * TILE;
  // Uniform across the workgroup, so this return is legal ahead of the barrier.
  if (base_token >= TOKENS) { return; }
  let local = local_id.x;
  let c = group.y * ${lanes}u + local;
  let wbase = group.z * ZG_STRIDE;
  // The tile's conditioning, staged once so one weight read serves all of it -
  // the same shape the epilogue used, and the reason a tile pays here at all.
  for (var d = local; d < C_COND; d += ${lanes}u) {
    ${overTile((t) => `{
      let token = base_token + ${t}u;
      var value = 0.0;
      if (token < TOKENS) { value = cond[token * C_COND + d]; }
      staged[${group(t)}u * C_COND + d]${lane(t)} = value;
    }`)}
  }
  workgroupBarrier();
  // 🔴 SEEDED WITH THE BIAS AND ACCUMULATED OVER INCREASING d, IN f32, FROM
  // WEIGHTS OF THE SAME PRECISION - which is the epilogue's order exactly, and
  // what makes this bit-exact rather than merely close.
  ${overGroups((g) => `var zero${g} = ${tileLanes}(${wf(`weights[wbase + C_COND * C + c]`)});`)}
  ${overGroups((g) => `var ffw${g} = ${tileLanes}(${wf(`weights[wbase + ZG_FFW + C_COND * C + c]`)});`)}
  // 🔴 ONE PASS OVER d FOR BOTH GATES. The conditioning is staged once and read
  // once per gate; the two weight reads are the only extra traffic, and they
  // are the reads the two epilogues were each making on their own anyway.
  for (var d = 0u; d < C_COND; d += 1u) {
    let w = ${wf(`weights[wbase + d * C + c]`)};
    let wf_ffw = ${wf(`weights[wbase + ZG_FFW + d * C + c]`)};
    ${overGroups((g) => `{
      let staged${g} = staged[${g}u * C_COND + d];
      zero${g} += staged${g} * w;
      ffw${g} += staged${g} * wf_ffw;
    }`)}
  }
  ${overTile((t) => `{
    let token = base_token + ${t}u;
    if (token < TOKENS) {
      let at = (group.z * TOKENS + token) * C + c;
      gates[at] = zero${group(t)}${lane(t)};
      ffw_gates[at] = ffw${group(t)}${lane(t)};
    }
  }`)}
${zgNorm("AD_LN", "AD_SW", "AD_CB", "AD_SB", "adaln_scale_shift")}
${zgNorm("FA_LN", "FA_SW", "FA_CB", "FA_SB", "ffw_adaln_scale_shift")}
}`;
  })();

  // 🔴 THE BATCHED FORM USES THE SAME ORDER AS adaln's, where the unbatched
  // forms differ - see the note on normBindings. There is no reason for two
  // orders once the dispatch is written alongside them, and one order is one
  // fewer thing for a positional bind group to get wrong.
  const ffwAdalnBindings = normBindings(batchedGates ? `@group(0) @binding(0) var<storage, read> act: array<f32>;
// 🔴 THREE BINDINGS, NOT FOUR. The projection was this kernel's ONLY read of
// the weights; hoisted, that binding goes unread and an auto layout DROPS it,
// so a positional bind group of four fails. Same trap as ffw-out-reduce's.
@group(0) @binding(1) var<storage, read> scale_shift: array<f32>;
@group(0) @binding(2) var<storage, read_write> xbuf: array<f32>;`
    : `@group(0) @binding(0) var<storage, read> cond: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<${weightPrecision}>;
@group(0) @binding(2) var<storage, read> act: array<f32>;
@group(0) @binding(3) var<storage, read_write> xbuf: array<f32>;`);
  const ffwAdaln = conditionedNorm("ffw", ffwAdalnBindings,
    normKSplits > 1 ? "split" : "fused");
  const ffwAdalnReduce = normKSplits > 1
    ? conditionedNorm("ffw", ffwAdalnBindings, "reduce") : null;

  // The widening half: x (C) -> gate and value (INTERMEDIATE each) -> SwiGLU.
  // 🔴 THE SWISH GATE MOVES TO THE REDUCTION WHEN K IS SPLIT. This kernel fuses
  // `swish(gate) * value` into its store, and swish is not linear - applying it
  // per part and summing gives a different function. So a split kernel writes
  // the two accumulators raw and the reduction gates them, exactly once.
  const ffwWide = (() => {
  const { tileGroups, tileLanes, lane, group, overTile, overGroups, stageChunk } =
    tilingFor(wideTile);
  const tile = wideTile;
  void tile; void lane; void group; void overTile;
  return `${common}
const TILE: u32 = ${wideTile}u;
@group(0) @binding(0) var<storage, read> xbuf: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<${weightPrecision}>;
${kSplits > 1
  ? `@group(0) @binding(2) var<storage, read_write> partials: array<f32>;`
  : `@group(0) @binding(2) var<storage, read_write> gated: array<f32>;`}

const CHANNEL_CHUNK: u32 = ${channelChunk}u;
var<workgroup> xt: array<${tileLanes}, ${tileGroups * channelChunk}>;

@compute @workgroup_size(${lanes})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let base_token = group.x * TILE;
  let local = local_id.x;
  let i = group.y * ${lanes}u + local;

  ${overGroups((g) => `var gate_acc${g} = ${tileLanes}(0.0);
  var value_acc${g} = ${tileLanes}(0.0);`)}

  ${kSplits > 1 ? `let k_start = group.z * ${kSpan}u;
  let k_stop = k_start + ${kSpan}u;` : ""}
  let wide = INTERMEDIATE * 2u;
  for (var c0 = ${kSplits > 1 ? "k_start" : "0u"}; c0 < ${kSplits > 1 ? "k_stop" : "C"}; c0 += CHANNEL_CHUNK) {
${stageChunk}
    // 🔴 BLOCKED, gate half first - the same convention as the trunk's
    // transition and the opposite of triangle multiplication's interleave.
    for (var cc = 0u; cc < CHANNEL_CHUNK; cc += 1u) {
      let column = W_ffwTransition1 + (c0 + cc) * wide;
      let wg = ${wf(`weights[column + i]`)};
      let wv = ${wf(`weights[column + INTERMEDIATE + i]`)};
      ${overGroups((g) => `{
        let x = xt[${g}u * CHANNEL_CHUNK + cc];
        gate_acc${g} += x * wg;
        value_acc${g} += x * wv;
      }`)}
    }
  }

${kSplits > 1
  ? `  ${overTile((t) => `{
    let token = base_token + ${t}u;
    if (token < ${ROWS}) {
      let index = token * INTERMEDIATE + i;
      let stride = ${rows}u * INTERMEDIATE;
      let slot = (group.z * 2u) * stride + index;
      partials[slot] = gate_acc${group(t)}${lane(t)};
      partials[slot + stride] = value_acc${group(t)}${lane(t)};
    }
  }`)}`
  : `  ${overGroups((g) => `let swished${g} = gate_acc${g}
    / (${tileLanes}(1.0) + exp(-gate_acc${g})) * value_acc${g};`)}
  ${overTile((t) => `{
    let token = base_token + ${t}u;
    if (token < ${ROWS}) {
      gated[token * INTERMEDIATE + i] = swished${group(t)}${lane(t)};
    }
  }`)}`}
}`;
  })();

  // The gate applied once, over the summed parts.
  const ffwWideReduce = kSplits <= 1 ? null : `${common}
@group(0) @binding(0) var<storage, read> partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> gated: array<f32>;
@compute @workgroup_size(${lanes})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x + gid.y * GRID_WIDTH * ${lanes}u;
  if (index >= ${rows}u * INTERMEDIATE) { return; }
  let stride = ${rows}u * INTERMEDIATE;
  var g = 0.0;
  var v = 0.0;
  for (var part = 0u; part < ${kSplits}u; part = part + 1u) {
    let slot = (part * 2u) * stride + index;
    g = g + partials[slot];
    v = v + partials[slot + stride];
  }
  gated[index] = g / (1.0 + exp(-g)) * v;
}`;

  // ...and the way back, INTERMEDIATE -> C, gated by the zero-init conditioning
  // and added to the residual.
  //
  // 🔴 THE LARGEST KERNEL OF THE DENOISER, AND IT IS ITS WEIGHT READS. Every
  // workgroup reads INTERMEDIATE x SLICE of transition2 to produce outTile
  // tokens - 2.4 MB for two of them, 3.4 GB a call, which at the 445 GB/s
  // tools/gpu/probe-alu.js measures for cached global reads is most of the
  // 20 ms it took. A bigger tile divides that traffic and used to cost
  // workgroup memory: outTile 4 wants 4 x INTERMEDIATE floats, 24 KB, and
  // measured 85 ms against outTile 2's 74. Staging a CHUNK of the intermediate
  // instead unties them - the accumulator survives the chunks - so the tile
  // buys its traffic back without spending residency for it.
  //
  // 🔴 AND THE TILE IS THE VECTOR. The inner loop multiplies one weight against
  // every token of the tile, so those tokens are exactly the axis to vectorise:
  // one workgroup read and one vector multiply-add where there were outTile of
  // each. See src/af3/transition-webgpu.js, which is the same kernel shape.
  const outVector = { 1: "f32", 2: "vec2<f32>", 4: "vec4<f32>" }[Math.min(4, outTile)];
  const outWidth = Math.min(4, outTile);
  const outGroups = outTile / outWidth;
  if (outVector === undefined || !Number.isInteger(outGroups)) {
    throw new Error(`outTile ${outTile} is not 1, 2 or a multiple of 4`);
  }
  const outLane = (t) => outWidth === 1 ? "" : `.${"xyzw"[t % outWidth]}`;
  const outGroup = (t) => Math.floor(t / outWidth);
  const overOutTile = (body) =>
    Array.from({ length: outTile }, (_, t) => body(t)).join("\n    ");
  const overOutGroups = (body) =>
    Array.from({ length: outGroups }, (_, g) => body(g)).join("\n    ");
  // 🔴 THE EPILOGUE IS LOOP-INVARIANT ACROSS A K SPLIT, WHICH IS WHY THIS ONE
  // CAN BE SPLIT AT ALL. After its K loop this kernel computes a SECOND
  // projection - the zero-init gate over the conditioning - then a sigmoid and
  // a residual add, about a quarter of its work. None of that depends on which
  // slice of K a part walked, so moving it into the reduction runs it exactly
  // once and costs nothing extra; only the K loop is divided.
  //
  // `mode` is "fused" (no split), "split" (K slice, store partials, no
  // epilogue) or "reduce" (sum partials, then the epilogue verbatim).
  const ffwOutFor = (mode) => `${common}
const TILE: u32 = ${outTile}u;
const OUT_CHUNK: u32 = ${outChunk}u;
${/* 🔴 A SPLIT PART DECLARES ONLY WHAT IT USES, CONTIGUOUSLY. It has no
   epilogue, so it never touches `cond` - and under layout:"auto" an unused
   binding is dropped from the layout, which a positional bind group then
   fails to match. Leaving a hole at binding 1 is not an option; the split
   shader simply has three bindings. */ ""}${mode === "split"
  ? `@group(0) @binding(0) var<storage, read> gated: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<${weightPrecision}>;
@group(0) @binding(2) var<storage, read_write> partials: array<f32>;`
  : mode === "reduce"
    ? (batchedGates
      // 🔴 THREE BINDINGS, NOT FOUR, AND THAT IS NOT COSMETIC. The gate was
      // this reduction's ONLY use of `weights`; precomputed, the binding goes
      // unread, and layout:"auto" DROPS an unread binding - so a positional
      // bind group of four fails with "binding index 2 not present in the bind
      // group layout". The split form three cases up carries the same warning
      // for the same reason, and this is the second time it has been earned.
      ? `@group(0) @binding(0) var<storage, read> partials: array<f32>;
// This block's window onto the batched ffw gate; see packZeroGateWeights.
@group(0) @binding(1) var<storage, read> ffw_zero_gate: array<f32>;
@group(0) @binding(2) var<storage, read_write> act: array<f32>;`
      : `@group(0) @binding(0) var<storage, read> partials: array<f32>;
@group(0) @binding(1) var<storage, read> cond: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<${weightPrecision}>;
@group(0) @binding(3) var<storage, read_write> act: array<f32>;`)
    : `@group(0) @binding(0) var<storage, read> gated: array<f32>;
${batchedGates
  ? `@group(0) @binding(1) var<storage, read> ffw_zero_gate: array<f32>;`
  : `@group(0) @binding(1) var<storage, read> cond: array<f32>;`}
@group(0) @binding(2) var<storage, read> weights: array<${weightPrecision}>;
@group(0) @binding(3) var<storage, read_write> act: array<f32>;`}

// One chunk of the intermediate, holding the tile's tokens as a vector - and
// then, once the chunk loop is done with it, the CONDITIONING.
//
// 🔴 THE ZERO-GATE LOOP READ cond FROM GLOBAL, ONCE PER TOKEN PER CHANNEL,
// ON EVERY LANE. It is indexed by (token, d) and not by c, so all 256 lanes
// of a workgroup want the same 4 x C_COND values - and at C_COND 384 that loop
// was one weight read, four global conditioning reads and four scalar
// multiply-adds a step, about 43% of this kernel's instructions. Staged as a
// vector over the tile it is one weight read, one workgroup read and one vector
// multiply-add: 384 x 3 where it was 384 x 9.
//
// 🔴 AND IT COSTS NO WORKGROUP MEMORY, because wt is dead by then. The chunk
// loop has finished reading it before the gate is computed, so the same slots
// carry the conditioning; the array is sized for whichever use is larger. A
// second array would have taken this kernel from 6 KiB to 12 - two workgroups
// a core against five - which is the trade this repository has lost to four
// times.
var<workgroup> wt: array<${outVector}, ${outGroups * Math.max(outChunk, condChannels)}>;

@compute @workgroup_size(${lanes})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let base_token = group.x * TILE;
  if (base_token >= ${ROWS}) { return; }
  let local = local_id.x;
  let c = group.y * ${lanes}u + local;

  ${overOutGroups((g) => `var acc${g} = ${outVector}(0.0);`)}

${mode === "reduce"
  ? `  {
    let stride = ${rows}u * C;
    ${overOutTile((t) => `{
      let token = base_token + ${t}u;
      if (token < ${ROWS}) {
        var total = 0.0;
        for (var part = 0u; part < ${outKSplits}u; part = part + 1u) {
          total = total + partials[part * stride + token * C + c];
        }
        acc${outGroup(t)}${outLane(t)} = total;
      }
    }`)}
  }
`
  : `  for (var chunk0 = ${mode === "split" ? "group.z * " + outKSpan + "u" : "0u"}; chunk0 < ${mode === "split" ? "group.z * " + outKSpan + "u + " + outKSpan + "u" : "INTERMEDIATE"}; chunk0 += OUT_CHUNK) {
    // ...before overwriting the chunk the previous iteration is still reading.
    workgroupBarrier();
    for (var i = local; i < OUT_CHUNK; i += ${lanes}u) {
      ${overOutTile((t) => `{
        let token = base_token + ${t}u;
        var value = 0.0;
        if (token < ${ROWS}) { value = gated[token * INTERMEDIATE + chunk0 + i]; }
        wt[${outGroup(t)}u * OUT_CHUNK + i]${outLane(t)} = value;
      }`)}
    }
    workgroupBarrier();

    for (var i = 0u; i < OUT_CHUNK; i += 1u) {
      // ...read once, used by every token of the tile.
      let weight = ${wf(`weights[W_ffwTransition2 + (chunk0 + i) * C + c]`)};
      ${overOutGroups((g) => `acc${g} += wt[${g}u * OUT_CHUNK + i] * weight;`)}
    }
  }

`}
${mode === "split" ? `  ${overOutTile((t) => `{
    let token = base_token + ${t}u;
    if (token < ${ROWS}) {
      partials[group.z * ${rows}u * C + token * C + c] = acc${outGroup(t)}${outLane(t)};
    }
  }`)}
` : ""}
${mode === "split" ? "" : `${batchedGates ? `  // 🔴 ALREADY COMPUTED, FOR EVERY BLOCK, IN ONE DISPATCH - the same hoist
  // attention-output's epilogue gets, and the same reason: this projection
  // reads the conditioning and the conditioning does not move with the block.
  // Under outKSplits it is this REDUCE that carried the whole gate, which is
  // why docs/A100.md prices ffw-out's split at a 2x on a loop that was only
  // two thirds of the kernel. Order and precision are unchanged, so bit-exact.
  ${overOutGroups((g) => `var zero${g} = ${outVector}(0.0);`)}
  ${overOutTile((t) => `{
    let token = base_token + ${t}u;
    if (token < ${ROWS}) {
      zero${outGroup(t)}${outLane(t)} = ffw_zero_gate[${perToken("token")} * C + c];
    }
  }`)}
` : `  // 🔴 THE ZERO-INIT GATE READS THE RAW CONDITIONING, not the normalised one.
  // ...the conditioning into the slots the chunk loop has finished with. The
  // barrier before is what makes reusing them safe; the one after is the
  // ordinary staging barrier.
  workgroupBarrier();
  for (var d = local; d < C_COND; d += ${lanes}u) {
    ${overOutTile((t) => `{
      let token = base_token + ${t}u;
      var value = 0.0;
      if (token < ${ROWS}) { value = cond[${perToken("token")} * C_COND + d]; }
      wt[${outGroup(t)}u * C_COND + d]${outLane(t)} = value;
    }`)}
  }
  workgroupBarrier();

  ${overOutGroups((g) => `var zero${g} = ${outVector}(${wf(`weights[W_ffwAdaptiveZeroCondBias + c]`)});`)}
  for (var d = 0u; d < C_COND; d += 1u) {
    let w = ${wf(`weights[W_ffwAdaptiveZeroCondWeights + d * C + c]`)};
    ${overOutGroups((g) => `zero${g} += wt[${g}u * C_COND + d] * w;`)}
  }
`}
  ${overOutGroups((g) => `let contribution${g} = acc${g}
    / (${outVector}(1.0) + exp(-zero${g}));`)}
  ${overOutTile((t) => `{
    let token = base_token + ${t}u;
    if (token < ${ROWS}) {
      act[token * C + c] = act[token * C + c] + contribution${outGroup(t)}${outLane(t)};
    }
  }`)}
`}
}`;
  const ffwOut = ffwOutFor(outKSplits > 1 ? "split" : "fused");
  const ffwOutReduce = outKSplits > 1 ? ffwOutFor("reduce") : null;

  // 🔴 THE REDUCTION IS WHERE THE BIAS LANDS. A K part cannot add it - there
  // are kSplits of them and the bias is one - so qkvg leaves its accumulator
  // unbiased when split and this pass adds it exactly once.
  const qkvgReduce = kSplits <= 1 ? null : `${common}
@group(0) @binding(0) var<storage, read> partials: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<${weightPrecision}>;
@group(0) @binding(2) var<storage, read_write> q: array<f32>;
@group(0) @binding(3) var<storage, read_write> k: array<f32>;
@group(0) @binding(4) var<storage, read_write> v: array<f32>;
@group(0) @binding(5) var<storage, read_write> gate: array<f32>;

@compute @workgroup_size(${lanes})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x + gid.y * GRID_WIDTH * ${lanes}u;
  if (index >= ${rows}u * WIDTH) { return; }
  let stride = ${rows}u * WIDTH;
  var sq = 0.0;
  var sk = 0.0;
  var sv = 0.0;
  var sg = 0.0;
  for (var part = 0u; part < ${kSplits}u; part = part + 1u) {
    let slot = (part * 4u) * stride + index;
    sq = sq + partials[slot];
    sk = sk + partials[slot + stride];
    sv = sv + partials[slot + stride * 2u];
    sg = sg + partials[slot + stride * 3u];
  }
  q[index] = sq + ${wf(`weights[W_qBias + index % WIDTH]`)};
  k[index] = sk;
  v[index] = sv;
  gate[index] = sg;
}`;

  return { normalisePair, normaliseCond, pairLogitsFor, adaln, qkvg, qkvgReduce, attend, attentionOutput,
           adalnReduce, ffwAdalnReduce, normKSplits,
           normPartialFloats: normKSplits > 1 ? normKSplits * 2 * rows * channels : 0,
           attentionOutputReduce, attnKSplits,
           attnPartialFloats: attnKSplits > 1 ? attnKSplits * rows * channels : 0,
           ffwAdaln, ffwWide, ffwWideReduce, ffwOut, ffwOutReduce,
           qkvgSplits, wideSplits, outSplits, outKSplits,
           outPartialFloats: outKSplits > 1 ? outKSplits * rows * channels : 0,
           kSplits, normSplits,
           zeroGates, batchedGates, gateTile, attnOutTile,
           // Per TOKEN and per block, not per row - the gate does not read the
           // sample. See packZeroGateWeights.
           zeroGateFloats: batchedGates ? tokens * channels : 0,
           qkvgPartialFloats: kSplits > 1 ? kSplits * 4 * rows * width : 0,
           widePartialFloats: kSplits > 1 ? kSplits * 2 * rows * intermediate : 0 };
}

export class Af3DiffusionTransformerGpu {
  /**
   * The normalised pair conditioning, kept across calls.
   *
   * 🔴 NEITHER THE PAIR CONDITIONING NOR ITS LAYERNORM READS THE NOISE LEVEL.
   * The stack's twenty-four blocks all read one normalised pair tensor, built
   * by a single pass at the top of the call - and a sampler was uploading the
   * unnormalised tokens^2 x 128 tensor and running that pass again on every
   * step, for the identical bytes. At 59 tokens that is 1.8 MB across the bus
   * and a 3481 x 128 layer norm, two hundred times; at 256 tokens, 34 MB.
   *
   * Keyed on the array's identity, which is the same question the diffusion
   * head asks of its own pair cache: a new fold brings a new array. The buffer
   * lives outside the pooled allocator, because a pooled one is recycled when
   * the call that took it ends.
   */
  #pairNorm;

  /**
   * The per-block pair attention biases, for the whole SCHEDULE.
   *
   * 🔴 THEY WERE RECOMPUTED ONCE A CALL FOR THE IDENTICAL ANSWER. The
   * pair-logits pass reads #pairNorm - which is the trunk's, built once and
   * held - and the super-block's `pairLogitsProjection`, which is a weight.
   * Neither moves with the noise level, so twenty-four passes an hour of the
   * fold produced bit-for-bit what the previous call produced. Measured at 200
   * tokens with tools/gpu/fold.js --profile --profile-from=trunk-done, that is
   * 0.61 ms a pass and 14.6 ms a call, 8% of a denoiser call's GPU time.
   *
   * 🔴 ONE BUFFER PER BLOCK, NOT ONE SLICE OF ONE BUFFER. A slice needs a
   * 256-byte-aligned offset and a block's logits are 64 x tokens^2 bytes,
   * which is not a multiple of 256 at odd token counts - 59 is one of the two
   * sizes checked here. Separate buffers also let the bind-group cache stay as
   * it is, keyed by block.
   *
   * `ready` is set only after the submit that fills them, because run()
   * restarts the whole call on a budget refusal and a cache that merely EXISTS
   * would then be handed to attend uninitialised - the same trap #pairNorm's
   * `pairCond` guards against.
   */
  #pairLogits;

  /**
   * The last shape this instance compiled for, and everything #compile
   * derived from it.
   *
   * 🔴 #compile BUILDS THE LARGEST WGSL IN THE MODEL, AS STRINGS, and did it
   * on every denoiser call - eleven shaders plus one per block in a super
   * block, template-substituted from the shape, and then handed to a pipeline
   * cache that COMPARES the string it was given against the one it stored.
   * None of it reads the noise level. A sampler paid for all of it two hundred
   * times to be told each time that the pipeline was already there.
   */
  #compiled;

  /**
   * Bind groups, keyed by the pass that uses them.
   *
   * 🔴 THE STACK CREATES SEVEN PER BLOCK AND HAS TWENTY-FOUR OF THEM. With the
   * scratch tensors and the resident weights both stable across a schedule,
   * every one of those 168 descriptors names the same buffers on every step -
   * and each carries a `getBindGroupLayout` call of its own. The entry stores
   * the buffers it was built from, so a call whose buffers moved (the
   * uploading weight path, where a super block's weights are pooled and
   * recycled) rebuilds rather than binding something else.
   */
  #bindGroups = new Map();

  /**
   * The stack's scratch tensors, kept across calls.
   *
   * 🔴 THEIR SIZE IS THE SHAPE'S, NOT THE STEP'S, and this allocator does not
   * pool - so a schedule created and destroyed eleven buffers per step and
   * handed every pass a bind group naming addresses that would not exist next
   * time. Keeping them costs nothing at the PEAK, which is inside a call and
   * unchanged, and it is what makes #bindGroups hit. Released by dispose(),
   * which the samplers call in a finally; the confidence head runs after that
   * and is where an AF3 fold's peak actually is.
   */
  #scratch;

  /**
   * @param {{residentWeights?: boolean}} [options] whether the 24 blocks' packed
   *   weights stay on the device between calls. See the note at the upload; a
   *   budget refusal turns this off on its own.
   */
  constructor(device, options = {}) {
    this.device = device;
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
    this.residentWeights = (options.residentWeights ?? true) && residencyAllowed(device);
  }

  /** Give back the normalised pair tensor. Callers that keep an instance own this. */
  /** Give back the normalised pair tensor alone. */
  #releasePairLogits() {
    if (this.#pairLogits === undefined) return;
    for (const buffer of this.#pairLogits.buffers) {
      buffer.destroy();
      noteDestroy(this.device, this.#pairLogits.bytes, "difftx.pair-logits");
    }
    this.#pairLogits = undefined;
    this.#bindGroups.clear();
  }

  #releasePairNorm() {
    this.#releasePairLogits();
    if (this.#pairNorm === undefined) return;
    this.#pairNorm.buffer.destroy();
    noteDestroy(this.device, this.#pairNorm.bytes, "difftx.pair-norm");
    this.#pairNorm = undefined;
    this.#bindGroups.clear();
  }

  dispose() {
    this.#releasePairLogits();
    this.#bindGroups.clear();
    if (this.#scratch !== undefined) {
      for (const [label, entry] of Object.entries(this.#scratch.buffers)) {
        entry.buffer.destroy();
        noteDestroy(this.device, entry.bytes, label);
      }
      this.#scratch = undefined;
    }
    if (this.#pairNorm === undefined) return;
    this.#pairNorm.buffer.destroy();
    noteDestroy(this.device, this.#pairNorm.bytes, "difftx.pair-norm");
    this.#pairNorm = undefined;
  }

  /**
   * A scratch tensor that outlives the call, sized by the shape.
   *
   * The whole set is dropped and rebuilt when the shape moves, because they
   * move together and a half-resized set is a bind group naming two different
   * molecules.
   */
  #scratchBuffer(key, label, bytes, usage) {
    if (this.#scratch?.key !== key) {
      if (this.#scratch !== undefined) {
        for (const [name, entry] of Object.entries(this.#scratch.buffers)) {
          entry.buffer.destroy();
          noteDestroy(this.device, entry.bytes, name);
        }
      }
      this.#scratch = { key, buffers: {} };
      this.#bindGroups.clear();
    }
    const found = this.#scratch.buffers[label];
    if (found !== undefined) return { buffer: found.buffer };
    const size = Math.ceil(bytes / 4) * 4;
    noteAllocation(this.device, label, size);
    const buffer = this.device.createBuffer({ label, size, usage });
    this.#scratch.buffers[label] = { buffer, bytes: size };
    return { buffer };
  }

  /**
   * @param {Float32Array} act tokens * channels
   * @param {Float32Array} cond tokens * condChannels
   * @param {Float32Array} pairCond tokens * tokens * pairChannels
   * @param {Float32Array} mask tokens
   * @param {number} tokens
   * @param {object} weights channels, condChannels, heads, dimension,
   *   transitionFactor, blocksPerSuperBlock, pairInputLayerNormScale, superBlocks
   */
  async run(act, cond, pairCond, mask, tokens, weights, options = {}) {
    try {
      return await this.#runBlocks(act, cond, pairCond, mask, tokens, weights, options);
    } catch (error) {
      if (!(error instanceof GpuMemoryBudgetError) || !this.residentWeights) throw error;
      // The pairformer's reasoning exactly; see the note on its run(). The
      // refusal arrives with some blocks already resident and their command
      // buffers in flight, so the call is abandoned rather than patched up.
      await this.device.queue.onSubmittedWorkDone();
      const reclaimed = releaseResidentWeights(this.device);
      this.residentWeights = false;
      noteResidencyRefused(this.device);
      this.degradedTo = `uploading weights per call (${(reclaimed / (1024 * 1024)).toFixed(0)}`
        + ` MiB reclaimed): ${error.message}`;
      return await this.#runBlocks(act, cond, pairCond, mask, tokens, weights, options);
    }
  }

  /**
   * Everything about a run that depends only on the shape: the tiles, the
   * shaders and their pipelines.
   *
   * 🔴 SEPARATED BECAUSE IT WAS DOING PER-CALL WORK THAT DEPENDS ON NOTHING.
   * It read the packing offsets by calling `packBlockWeights` on a sample block
   * - concatenating 31.5 MiB, and in f16 converting it, to obtain a dozen
   * numbers that are a running sum of lengths - and it did that on EVERY
   * denoiser call, eight times a fold. `blockWeightOffsets` is the same numbers
   * without the buffer: a steady call goes 86-89 ms to 83-85, its transformer
   * 43 to 40.
   *
   * Compiling this early was tried too and is not worth keeping: measured from
   * a fold, the pipelines are ready in 6 ms, so compilation is not what makes
   * the first call several times a steady one. That is the weight conversion,
   * and docs/AF3.md records why it cannot be moved either.
   */
  /**
   * Compile this stack without running it, so the trunk can pay for it.
   *
   * 🔴 THE HEAD'S CONSTRUCTOR COMPILES NOTHING, WHICH fold.js BELIEVED IT DID.
   * Its comment read "building one compiles its pipelines - 730 ms - and doing
   * that here overlaps the compile with the trunk"; the constructor stores a
   * device, an allocator and a cache. Every pipeline was built inside the
   * FIRST denoiser call, which is why the page's "Folding" band is 311 ms at
   * 58 residues before a single step reports, and 4.9 s of a 27 s fold at 150.
   * There is nothing to overlap it with at that point - the trunk is over.
   */
  async warm(tokens, weights) {
    await this.#compile(tokens, weights);
  }

  /**
   * 🔴 THE PROMISE, NOT THE RESULT. Storing the settled value means two
   * concurrent callers both miss and both build the largest WGSL in the model -
   * which is exactly what warming during the trunk creates, since the sampler
   * asks for the same shape while the warm may still be in flight. The pipeline
   * cache stores promises for the same reason.
   */
  //
  // 🔴 AND THE SPLIT RULE IS PART OF THE KEY, BECAUSE IT IS READ INSIDE THE
  // THING BEING MEMOISED. `#buildCompile` takes the geometry from
  // `deviceTuning(this.device).diffusionSplitK`, which a caller can change with
  // `setDeviceTuning` between two runs on ONE instance - and this memo, keyed
  // on (tokens, weights) alone, would hand back the shaders compiled for the
  // previous rule while the dispatch used the new one. Nothing in a fold does
  // that today (the tool sets the tuning once, before folding), so this is a
  // trap rather than a live bug; a sweep that reuses an instance across arms
  // would walk straight into it and read the first arm's numbers for every arm.
  #splitKeyFor() {
    const rule = deviceTuning(this.device).diffusionSplitK;
    return rule === null || rule === undefined ? "none" : JSON.stringify(rule);
  }

  async #compile(tokens, weights) {
    const splitKey = this.#splitKeyFor();
    if (this.#compiled?.tokens === tokens && this.#compiled.weights === weights
        && this.#compiled.splitKey === splitKey) {
      return this.#compiled.promise;
    }
    const promise = this.#buildCompile(tokens, weights);
    this.#compiled = { tokens, weights, splitKey, promise };
    return promise;
  }

  async #buildCompile(tokens, weights) {
    const channels = weights.channels;
    const condChannels = weights.condChannels;
    const pairChannels = weights.pairChannels;
    const heads = weights.heads;
    const dimension = weights.dimension;
    const perSuper = weights.blocksPerSuperBlock;
    const width = heads * dimension;
    const pairs = tokens * tokens;
    const sampleOffsets = blockWeightOffsets(weights.superBlocks[0].blocks[0]);
    // 🔴 FOUR AND TWO ARE MEASURED, AND MORE IS WORSE. Each kernel holds its
    // tile of activations in workgroup storage - the projection and the
    // widening keep `channels` floats a token, the way back `intermediate`,
    // twice as many - so a bigger tile buys weight traffic and costs
    // workgroups, and past here the workgroups are worth more:
    //
    //     tile     2   4   4   4   4   6   8      outTile 1 2 3 4 for tile 4
    //     outTile  2   1   2   3   4   2   2
    //     ms      91  75  74  79  85  79  83
    //
    // 🔴 SO RAISING maxComputeWorkgroupStorageSize BOUGHT NOTHING. It is asked
    // for in src/runtime/device.js and it does lift the ceiling from four
    // tokens to ten - but the ceiling was never what bound this, occupancy was.
    // The limit stays requested because it costs nothing and the cap below is
    // then real rather than notional; the numbers above are why the defaults do
    // not use the room.
    //
    // 🔴 AND THEY ARE RESOLVED BEFORE THE SHAPE, NOT AFTER. Leaving them to
    // default a second time inside the shader factory meant the SHADERS tiled
    // by four while the DISPATCH divided the token count by eight: every second
    // tile of tokens was simply never projected, and the bench reported it as a
    // 30% speedup. One resolution, passed down.
    const workgroupStorage = this.device.limits?.maxComputeWorkgroupStorageSize ?? 16384;
    const intermediate = channels * weights.transitionFactor;
    const fits = (perToken) => Math.max(1, Math.floor(workgroupStorage / (perToken * 4)));
    // 🔴 FOUR, AND RE-MEASURED AFTER THE CONDITIONING WAS STAGED. Eight used to
    // lose partly because two kernels' zero-gate loops were TILE global reads a
    // step; staged, those loops cost the same whatever the tile, so the reason
    // for the old answer had gone even though the answer had not. Re-swept on
    // the whole transformer at 150 tokens: 4 -> 103, 105 ms; 8 -> 106, 113.
    // 🔴 AND THE DEVICE GETS A SAY, because the sweep above is an M2's. See
    // src/runtime/device-profile.js: null there means this rule, unchanged.
    const deviceTile = deviceTuning(this.device).diffusionTokenTile;
    // 🔴 A TILE THAT DOES NOT MOVE WITH THE TOKEN COUNT IS WRONG AT ONE END
    // OR THE OTHER. The tile trades weight traffic - proportional to
    // tokens/tile - against workgroups, and which side binds depends on how
    // many tokens there are. See the sweep in src/runtime/device-profile.js.
    const wantedTile = deviceTile === null ? Math.min(4, fits(channels))
      : tokens < deviceTile.crossover ? deviceTile.below : deviceTile.atOrAbove;
    // 🔴 SPLITTING K AND THE TOKEN TILE ARE ONE DECISION. Below the crossover
    // the tile alone loses - it halves the workgroups and this device has 108
    // multiprocessors to fill - so the split is what makes a bigger tile
    // affordable, and asking for one without the other is asking for the
    // slower of the two arms. Above it the tile stands on its own and the
    // split's partial traffic is pure cost.
    const splitRule = weights.splitK === undefined
      ? deviceTuning(this.device).diffusionSplitK : weights.splitK;
    const splitting = splitRule !== null && splitRule !== undefined
      && tokens < splitRule.crossover && channels % splitRule.splits === 0;
    const kSplits = splitting ? splitRule.splits : 1;
    // 🔴 THE SHARED TILE IS UNCHANGED BY THE SPLIT. Only qkvg has a K split
    // behind it, so only qkvg can afford the bigger tile; adaln, ffw-adaln and
    // the rest measured 0.54x-0.82x when the raised tile reached them, which is
    // what makes this two numbers rather than one.
    const tile = weights.tile ?? Math.min(wantedTile, fits(channels));
    const qkvgTile = splitting
      ? Math.min(weights.qkvgTile ?? splitRule.tile, fits(channels)) : tile;
    const wideTile = splitting
      ? Math.min(weights.wideTile ?? splitRule.tile, fits(channels)) : tile;
    // ffw-out's inner extent is the intermediate, so it gets its own part
    // count - and its epilogue is a projection, so the reduction is most of
    // the original kernel. Same crossover.
    const splits = weights.splits ?? 2;
    // 🔴 THE CHUNK IS WHAT `ffw-out` STAGES, NOT THE WHOLE INTERMEDIATE, AND
    // THIS RULE STILL SAID OTHERWISE. It read `Math.min(2, tile,
    // fits(intermediate))`: a hard cap of two, under a sizing term that assumed
    // a workgroup held outTile x 1536 floats. Chunking removed that - it holds
    // outTile x outChunk, 6 KiB at four - and the kernel's own comment says so
    // ("staging a CHUNK of the intermediate instead unties them"), but the cap
    // was never lifted, so the measurement that set it (outTile 4 at 85 ms
    // against 2's 74) was left standing against a kernel that no longer
    // existed. Re-measured on the whole transformer with the chunking in place,
    // as medians of repeated runs: at 150 tokens **2 -> 138 ms, 4 -> 128, 8 ->
    // 132**; at 59 tokens they tie (63-65 either way). Four it is.
    const outChunk = resolveOutChunk(intermediate, weights.outChunk);
    // 🔴 AFTER outChunk, BECAUSE IT DEPENDS ON IT. ffw-out's inner extent is
    // the intermediate and a part walks its span in chunks, so only a part
    // count whose span the chunk divides is legal - the shader factory throws
    // on the rest. Placed before outChunk this read it in its temporal dead
    // zone, which failed every run that asked for a split and none that did
    // not.
    const wantedOutK = weights.outKSplits ?? (splitting ? (splitRule.outSplits ?? 4) : 1);
    const outKSplits = (wantedOutK > 1 && intermediate % wantedOutK === 0
      && (intermediate / wantedOutK) % outChunk === 0) ? wantedOutK : 1;
    // attention-output's K is the width, staged whole rather than in chunks, so
    // any divisor works.
    const wantedAttnK = weights.attnKSplits ?? (splitting ? (splitRule.attnSplits ?? 4) : 1);
    const attnKSplits = (wantedAttnK > 1 && width % wantedAttnK === 0) ? wantedAttnK : 1;
    // 🔴 AFTER THE SPLIT COUNTS, BECAUSE BOTH TILES NOW DEPEND ON THEM - the
    // same temporal-dead-zone trap the note on outChunk above records, and it
    // failed the same way: "Cannot access 'outKSplits' before initialization".
    //
    // 🔴 ffw-out HAS A K SPLIT, SO BY THIS FILE'S OWN RULE IT SHOULD NOT BE
    // CAPPED BY THE SHARED TILE - and it was: `Math.min(wantedTile, tile, ...)`
    // pins it to 1 wherever the shared tile is 1, which on this device is
    // everywhere below 175 tokens. Its own sweep above chose FOUR ("at 150
    // tokens 2 -> 138 ms, 4 -> 128, 8 -> 132"), and that conclusion has been
    // overridden by the shared tile ever since the tile went to 1 for occupancy.
    // The split is what pays for the tile; qkvg and ffw-wide already read
    // splitRule.tile for exactly this reason.
    const outTile = weights.outTile ?? (splitting && outKSplits > 1
      ? Math.min(splitRule.outTile ?? splitRule.tile, fits(outChunk))
      : Math.min(wantedTile, tile, fits(outChunk)));
    // ...and the same for attention-output, whose staging is the width, or one
    // part's span when it is split.
    const attnOutTile = weights.attnOutTile ?? (splitting && attnKSplits > 1
      ? Math.min(splitRule.attnTile ?? splitRule.tile, fits(width / attnKSplits))
      : tile);
    const wantedNormK = weights.normKSplits ?? (splitting ? (splitRule.normSplits ?? 4) : 1);
    const normKSplits = (wantedNormK > 1 && condChannels % wantedNormK === 0) ? wantedNormK : 1;
    // 🔴 THE BATCHED ZERO GATE NEEDS THE BLOCKS TO BE RESIDENT, because it
    // builds one concatenated buffer over all of them and memoises it against
    // the weights object. Under the budget fallback the blocks are uploaded and
    // released per call, and a duplicate of weights that are NOT being kept is
    // exactly the trade that fallback exists to refuse.
    const batchedGates = (weights.batchedGates
      ?? deviceTuning(this.device).diffusionBatchedGates) === true;
    // See the note in the shader factory: this kernel wants the OPPOSITE of
    // what every other kernel here wants, because the block axis already fills
    // the device and weight bandwidth is what binds it.
    // 🔴 CLAMPED BY WORKGROUP STORAGE, HERE, WHERE THE LIMITS ARE. The staging
    // array is `gateTile * C_COND` floats, so 16 tiles of a 384-wide
    // conditioning is 24 KiB against a 16 KiB default limit - which fails
    // pipeline creation rather than running slowly. `fits` is the same helper
    // the other tiles use.
    const wantedGateTile = weights.gateTile
      ?? deviceTuning(this.device).diffusionGateTile ?? 8;
    const gateTileRoom = fits(condChannels);
    const gateTile = Math.max(1, Math.min(wantedGateTile, gateTileRoom >= 4
      ? gateTileRoom - (gateTileRoom % 4) : (gateTileRoom >= 2 ? 2 : 1)));
    // 🔴 f16 WHEREVER THE DEVICE HAS IT, FOR THE MEMORY. See the note in the
    // shader factory: this is the largest resident tensor a fold holds.
    const weightPrecision = weights.weightPrecision
      ?? (halfPrecisionAvailable(this.device) ? "f16" : "f32");
    // 🔴 THE FEATURE IS CHECKED HERE AND NOT IN THE SHADER FACTORY, which has
    // no device. A shader asking for subgroups on an adapter without them
    // fails pipeline creation, which is fatal rather than slow.
    const info = this.device.adapterInfo ?? {};
    const attendSubgroups = (weights.attendSubgroups
        ?? deviceTuning(this.device).diffusionAttendSubgroups) === true
      && this.device.features.has("subgroups")
      && this.device.features.has("subgroup-size-control")
      && info.subgroupMinSize === info.subgroupMaxSize;
    // The same three conditions attendSubgroups needs, asked for the norm's
    // reductions separately - see the note on reduce_sum.
    const normSubgroups = (weights.normSubgroups
        ?? deviceTuning(this.device).diffusionNormSubgroups) === true
      && this.device.features.has("subgroups")
      && this.device.features.has("subgroup-size-control")
      && info.subgroupMinSize === info.subgroupMaxSize;
    const samples = Math.max(1, weights.samples ?? 1);
    const shape = { tokens, channels, condChannels, pairChannels, heads, dimension, samples,
                    attendSubgroups, normSubgroups,
                    normSplit: (weights.normSplit
                      ?? deviceTuning(this.device).diffusionNormSplit) === true,
                    subgroupSize: info.subgroupMaxSize ?? 32,
                    // 🔴 THE KEY CHUNK, WHICH WAS A HARDCODED 64 NO DEVICE COULD
                    // MOVE. It sizes `k_tile`, and the shipped default assumed a
                    // 16 KiB workgroup-storage limit; this card reports 49152.
                    // A bigger chunk is fewer staging barriers AND more lanes in
                    // the key dot loop, against fewer workgroups resident.
                    attendKeyChunk: weights.attendKeyChunk
                      ?? deviceTuning(this.device).diffusionAttendKeyChunk ?? undefined,
                    attendStageKeys: weights.attendStageKeys
                      ?? deviceTuning(this.device).diffusionAttendStageKeys ?? undefined,
                    factor: weights.transitionFactor,
                    // 🔴 THE WORKGROUP WIDTH FOR EVERY KERNEL IN THIS STACK, and
                    // 256 is another number chosen on a device with a handful of
                    // cores. It sets the output split (range / lanes) and so the
                    // workgroup count, which is what has been binding all day.
                    lanes: weights.lanes ?? deviceTuning(this.device).diffusionLanes ?? undefined,
                    tile, splits, outTile, outChunk, weightPrecision, kSplits, qkvgTile,
                    wideTile, outKSplits, attnKSplits, normKSplits, batchedGates, gateTile,
                    attnOutTile, channelChunk: weights.channelChunk };
    const sources = createDiffusionTransformerShaders(shape, sampleOffsets);
    // 🔴 THE LANE COUNT IS PART OF THE KEY. It is baked into every one of these
    // sources as a workgroup size, so a cache that ignored it would hand a
    // later run the pipeline compiled for a different width.
    const base = `af3-difftx:${tokens}:${channels}:${condChannels}:${pairChannels}`
      + `:${heads}:${dimension}:${weights.transitionFactor}:${perSuper}`
      + `:${shape.lanes ?? "default"}:${attendSubgroups}:${normSubgroups}:${shape.normSplit}:${shape.attendStageKeys}:${shape.attendKeyChunk ?? "d"}:${tile}:${splits}:${outTile}:${outChunk}`
      + `:${weights.channelChunk ?? "d"}:${weightPrecision}:k${kSplits}:qt${qkvgTile}:wt${wideTile}`
      // 🔴 THESE THREE WERE MISSING, and a key that omits a split count hands a
      // later run the pipeline compiled for a different one. Harmless across
      // processes, which is how every sweep here was taken; a collision waiting
      // for two configurations in one.
      + `:ok${outKSplits}:ak${attnKSplits}:nk${normKSplits}:s${samples}:bg${batchedGates}:gt${gateTile ?? "d"}:aot${attnOutTile}`;
    // 🔴 AWAITED TOGETHER, NOT ONE AT A TIME. `createComputePipelineAsync`
    // compiles off the main thread, so a loop that awaits each one in turn
    // serialises eleven compilations that could overlap - and this stack's
    // shaders are the largest in the model. It is paid once per process and
    // lands inside the FIRST denoiser call, which bench-head.js reports at 606
    // ms against a steady 86. The cache stores the promise, not the pipeline,
    // so asking for the same key twice is still one compilation.
    const compiled = { pairLogits: [] };
    const pending = [];
    for (const [name, source] of Object.entries(sources)) {
      // ...the factory also returns the split counts the dispatch needs, which
      // are numbers rather than shaders.
      if (name === "pairLogitsFor" || typeof source !== "string") continue;
      pending.push(this.pipelines.get(`${base}:${name}`, source)
        .then((pipeline) => { compiled[name] = pipeline; }));
    }
    for (let inner = 0; inner < perSuper; inner += 1) {
      const at = inner;
      pending.push(this.pipelines.get(
        `${base}:pair-logits:${at}`, sources.pairLogitsFor(at, perSuper))
        .then((pipeline) => { compiled.pairLogits[at] = pipeline; }));
    }
    await Promise.all(pending);
    return { channels, condChannels, pairChannels, heads, dimension, perSuper,
             width, pairs, shape, sources, compiled, tile, splits, outTile, outChunk,
             weightPrecision, kSplits, qkvgTile, wideTile, outKSplits, attnKSplits,
             // 🔴 THE FACTORY'S VALUE, NOT THIS SCOPE'S. The factory forces
             // normKSplits to 1 when the projection is batched, and the
             // dispatch below decides whether to run a reduce pass from this
             // number - so taking the unforced one would encode a
             // "reduce" that the shader never compiled.
             normKSplits: sources.normKSplits, samples,
             batchedGates: sources.batchedGates,
             attnOutTile: sources.attnOutTile };
  }

  async #runBlocks(act, cond, pairCond, mask, tokens, weights, options = {}) {
    // The caller keeps the stack's output as a buffer, and hands in the scope
    // it will settle. See the head's #chain.
    const keepOnDevice = options.keepOnDevice === true;
    const {
      channels, condChannels, pairChannels, heads, dimension, perSuper,
      width, pairs, shape, sources, compiled, tile, splits, outTile, outChunk,
      weightPrecision, kSplits, qkvgTile, wideTile, outKSplits, attnKSplits, normKSplits,
      samples, batchedGates, attnOutTile,
    } = await this.#compile(tokens, weights);
    // 🔴 EVERY ROW-PARALLEL DISPATCH AND BUFFER IS PER ROW, NOT PER TOKEN. The
    // same number at one sample; above one it is what gives the widened
    // kernels the work they were widened for.
    const rows = samples * tokens;
    // The activation is per ROW; the conditioning and the pair are per token
    // and shared across samples, which is the whole reason batching pays.
    if (!(act instanceof GPUBuffer) && act.length !== rows * channels) {
      throw new Error(`act has ${act.length} elements; expected ${rows * channels}`);
    }

    const storage = GPUBufferUsage.STORAGE;
    const allocations = [];
    const keep = (allocation) => { allocations.push(allocation); return allocation; };
    try {
      // See #scratch. `act` and `cond` are written into their buffers rather
      // than uploaded into new ones; `mask` is the shape's and is written for
      // the same price as testing whether it changed.
      const shapeKey = `${tokens}:${channels}:${condChannels}:${pairChannels}`
        + `:${width}:${weights.transitionFactor}:${heads}:${samples}`;
      const scratch = (label, bytes, usage = storage) =>
        this.#scratchBuffer(shapeKey, label, bytes, usage);
      // 🔴 EITHER A HOST ARRAY OR A DEVICE BUFFER, FOR THE TWO THAT MOVE. The
      // stack's input and its conditioning are produced on the GPU one stage
      // earlier by the diffusion head, which keeps them there rather than
      // draining the pipeline twice a step to copy them out and back.
      const write = (allocation, data) => this.device.queue.writeBuffer(
        allocation.buffer, 0, data.buffer, data.byteOffset, data.byteLength);
      const given = (value, label, bytes) => {
        if (value instanceof GPUBuffer) return { buffer: value };
        const allocation = scratch(label, bytes,
          storage | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
        write(allocation, value instanceof Float32Array ? value : Float32Array.from(value));
        return allocation;
      };
      // 🔴 THE ACTIVATION IS PER ROW AND THE CONDITIONING IS NOT. cond is one
      // tensor for every sample - that asymmetry IS the batching win, since a
      // weight read and a conditioning read now serve S rows instead of one.
      const actBuffer = given(act, "difftx.act", rows * channels * 4);
      const condBuffer = given(cond, "difftx.cond", tokens * condChannels * 4);
      const maskBuffer = scratch("difftx.mask", tokens * 4,
        storage | GPUBufferUsage.COPY_DST);
      write(maskBuffer, mask);
      // See #pairNorm: everything on this line and the two below it is the
      // trunk's, not the step's, and is skipped outright when the caller keeps
      // this instance across a schedule.
      const normBytes = pairs * pairChannels * 4;
      const buildPairNorm = this.#pairNorm?.pairCond !== pairCond
        || this.#pairNorm?.bytes !== normBytes;
      if (buildPairNorm) {
        // 🔴 THE PAIR NORM ALONE, NOT dispose(). The scratch set was created
        // three lines above and its buffers are already bound into this call's
        // command encoder; dropping it here destroyed a buffer the submit then
        // used, which WebGPU reports at the end of the stack rather than here.
        this.#releasePairNorm();
        noteAllocation(this.device, "difftx.pair-norm", normBytes);
        // 🔴 `pairCond` IS NOT SET UNTIL THE PASS THAT FILLS THIS HAS BEEN
        // SUBMITTED. An allocation between here and there can refuse on
        // budget, and run() retries the whole call - which would find a cache
        // that matches and skip the norm, handing twenty-four blocks an
        // uninitialised buffer.
        this.#pairNorm = {
          pairCond: undefined, bytes: normBytes,
          buffer: this.device.createBuffer({
            label: "difftx.pair-norm", size: normBytes, usage: storage,
          }),
        };
      }
      const normalized = { buffer: this.#pairNorm.buffer };
      const pairBuffer = buildPairNorm
        ? keep(this.allocator.upload("difftx.pair", pairCond, storage)) : undefined;
      const pairScale = buildPairNorm
        ? keep(this.allocator.upload("difftx.pair-scale",
                                     weights.pairInputLayerNormScale, storage))
        : undefined;
      // See #pairLogits: one buffer a block, held for the schedule, because
      // nothing the pair-logits pass reads moves with the noise level.
      const logitsBytes = heads * pairs * 4;
      const blockCount = weights.superBlocks
        .reduce((count, group) => count + group.blocks.length, 0);
      // 🔴 AS MANY BLOCKS AS FIT, NOT ALL OR NONE. A block's logits are
      // 64 x tokens^2 bytes, so caching all twenty-four costs 61 MiB at 200
      // tokens and 246 at 400 - and length is exactly where memory binds. The
      // saving is per BLOCK and so is the cost, so keeping the first
      // PAIR_LOGITS_CACHE_BYTES worth of them buys that fraction of the time
      // for a bounded amount of memory: every block at 208 tokens or fewer,
      // a quarter of them at 400.
      const cached = Math.max(0,
        Math.min(blockCount, Math.floor(PAIR_LOGITS_CACHE_BYTES / logitsBytes)));
      const buildPairLogits = buildPairNorm
        || this.#pairLogits?.ready !== true
        || this.#pairLogits.bytes !== logitsBytes
        || this.#pairLogits.buffers.length !== cached;
      if (buildPairLogits) {
        this.#releasePairLogits();
        const buffers = [];
        for (let at = 0; at < cached; at += 1) {
          noteAllocation(this.device, "difftx.pair-logits", logitsBytes);
          buffers.push(this.device.createBuffer({
            label: `difftx.pair-logits.${at}`, size: logitsBytes, usage: storage,
          }));
        }
        this.#pairLogits = { ready: false, bytes: logitsBytes, buffers };
      }
      // ...the blocks past the cache share one scratch buffer and recompute,
      // which is what every block did before.
      const spare = cached < blockCount
        ? scratch("difftx.logits", logitsBytes) : undefined;
      const logitsFor = (at) =>
        (at < cached ? { buffer: this.#pairLogits.buffers[at] } : spare);
      // The AdaLN pass hands the projection its input through this.
      const xBuffer = scratch("difftx.x", rows * channels * 4);
      const gatedBuffer = scratch("difftx.gated",
        rows * channels * weights.transitionFactor * 4);
      const q = scratch("difftx.q", rows * width * 4);
      const k = scratch("difftx.k", rows * width * 4);
      const v = scratch("difftx.v", rows * width * 4);
      const gate = scratch("difftx.gate", rows * width * 4);
      const gathered = scratch("difftx.gathered", rows * width * 4);
      // 🔴 THE PARTIALS ARE THE PRICE OF SPLITTING K, and they are why the
      // split is gated to small token counts rather than always on: kSplits x
      // four outputs x the whole q/k/v/gate tensor. At 68 tokens and eight
      // parts that is 6.7 MiB; at 384 it would be 37.7, for an arm that is
      // slower there anyway.
      // 🔴 ONE PARTIALS BUFFER FOR BOTH SPLIT KERNELS. qkvg's partials are
      // consumed by its reduction before `attend` runs, and ffw-wide's are
      // written afterwards, so the two never overlap in a block - and at these
      // shapes they are the same size to the float (kSplits x 4 x tokens x
      // width against kSplits x 2 x tokens x intermediate, and intermediate is
      // twice width). Two buffers cost 13 MiB of L2 that the passes AROUND
      // them were paying for: ffw-out and attention-output read 0.91x with two
      // and 1.00x with one, while neither split kernel noticed.
      const partials = kSplits > 1
        ? scratch("difftx.split-partials",
          Math.max(sources.qkvgPartialFloats, sources.widePartialFloats) * 4)
        : undefined;
      const qkvgPartials = partials;
      const widePartials = partials;
      // ffw-out's partials are live across its own two passes only, but those
      // sit AFTER ffw-wide's reduction has consumed the shared buffer - so this
      // could share it too. It does not, because ffw-out is the last pass of a
      // block and the next block's qkvg writes the shared one immediately.
      const outPartials = outKSplits > 1
        ? scratch("difftx.ffw-out-partials", sources.outPartialFloats * 4) : undefined;
      const attnPartials = attnKSplits > 1
        ? scratch("difftx.attn-out-partials", sources.attnPartialFloats * 4) : undefined;
      // 🔴 ONE TENSOR FOR THE WHOLE CALL. tokens x C_COND, 102 KiB at 68
      // tokens, against 144 workgroup-sets deriving it with two reductions
      // each.
      const condNormalised = scratch("difftx.cond-normalised",
        tokens * condChannels * 4);
      const normPartials = normKSplits > 1
        ? scratch("difftx.norm-partials", sources.normPartialFloats * 4) : undefined;
      // 🔴 NOT ALLOCATED WHEN THE CALLER IS KEEPING THE ANSWER ON THE DEVICE.
      // The head's next act is a LayerNorm and then the atom decoder, both on
      // the GPU; reading tokens x 768 floats out and writing them straight back
      // is a pipeline drain for nothing.
      const readback = keepOnDevice ? undefined
        : scratch("difftx.readback", tokens * channels * 4,
                  GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);

      const start = performance.now();
      // 🔴 ONE VALIDATION SCOPE PER BLOCK, NONE OF THEM AWAITED IN THE LOOP.
      // See src/runtime/validation.js: `await popErrorScope()` between blocks
      // puts a host-device synchronisation in the middle of the stack, which is
      // what the pairformer found first. This loop also awaited
      // `onSubmittedWorkDone()` per block on top of that - two round trips a
      // block, twenty-four blocks, every denoiser call - and a denoiser call
      // happens up to 200 times a fold.
      // The caller's scope when it has one, so it can settle every stage of a
      // denoiser step at the one boundary that already synchronises.
      const validation = options.validation
        ?? new DeferredValidation(this.device, "diffusion transformer");
      // The shared pair LayerNorm, once for the whole stack - and once for the
      // whole SCHEDULE, since nothing it reads moves with the noise level.
      if (buildPairNorm) {
        validation.begin();
        const encoder = this.device.createCommandEncoder({ label: "difftx.pair-norm" });
        const pass = encoder.beginComputePass({ label: "pair-norm" });
        pass.setPipeline(compiled.normalisePair);
        pass.setBindGroup(0, this.device.createBindGroup({
          layout: compiled.normalisePair.getBindGroupLayout(0),
          entries: [pairBuffer, pairScale, normalized].map((allocation, binding) => ({
            binding, resource: { buffer: allocation.buffer },
          })),
        }));
        const groups = Math.ceil(pairs / 64);
        pass.dispatchWorkgroups(Math.min(groups, GRID_WIDTH), Math.ceil(groups / GRID_WIDTH));
        pass.end();
        this.device.queue.submit([encoder.finish()]);
        validation.end("pair layer norm");
        this.#pairNorm.pairCond = pairCond;
      }

      // 🔴 AND THE CONDITIONING'S NORM, ONCE FOR THE WHOLE CALL, for the same
      // reason the pair norm above is: nothing it reads moves with the block.
      // Only the SCALE is per block, so what is shared is
      // `(cond - mean) * inverse` - which 144 workgroup-sets were deriving
      // separately, two barrier-tree reductions each. Unlike the pair norm this
      // one does move with the noise level, because the conditioning does, so
      // it runs per call rather than per schedule.
      {
        validation.begin();
        const encoder = this.device.createCommandEncoder({ label: "difftx.cond-norm" });
        const pass = encoder.beginComputePass({ label: "cond-norm" });
        pass.setPipeline(compiled.normaliseCond);
        pass.setBindGroup(0, this.device.createBindGroup({
          layout: compiled.normaliseCond.getBindGroupLayout(0),
          entries: [condBuffer, condNormalised].map((allocation, binding) => ({
            binding, resource: { buffer: allocation.buffer },
          })),
        }));
        pass.dispatchWorkgroups(Math.min(tokens, GRID_WIDTH), Math.ceil(tokens / GRID_WIDTH));
        pass.end();
        this.device.queue.submit([encoder.finish()]);
        validation.end("conditioning layer norm");
      }

      // 🔴 AND EVERY BLOCK'S ZERO-INIT GATE, ONCE, FOR THE SAME REASON THE TWO
      // NORMS ABOVE ARE HOISTED: nothing it reads moves with the block. What is
      // new here is that the BLOCKS ARE THE z AXIS, so this is not twenty-four
      // hoisted passes but one - 17 x 3 x 24 workgroups at 68 tokens against
      // the 204 the epilogue it replaces ran, twenty-four times a step.
      const allBlocks = batchedGates
        ? weights.superBlocks.flatMap((group) => group.blocks) : [];
      // 🔴 ON THE DEVICE FIRST, BECAUSE THE HOST PACK IS 496 ms OF AN AF3 FIRST
      // FOLD. Measured with `residentPackStats`: one call, 81.2 MiB, 496 ms -
      // the largest single item in a fold, and larger than every other host
      // packer put together. It is a plain concatenation of twenty-four blocks'
      // twelve gate tensors, so the decoder takes it whole; what it needed was
      // an order entry that can name its OWN holder, since this is one buffer
      // over twenty-four SOURCES maps. See src/af3/device-weights.js.
      const zeroGateOnDevice = batchedGates && weightPrecision === "f16"
        ? await residentPackedOnDevice(this.device, {
            key: weights, label: "difftx.zerogate.resident", variant: weightPrecision,
            order: allBlocks.flatMap((block) =>
              ZERO_GATE_ORDER.map((name) => ({ name, weights: block }))),
            weights: allBlocks[0],
          })
        : undefined;
      const zeroGateWeights = batchedGates
        ? { buffer: zeroGateOnDevice ?? residentWeightBuffer(
              this.device, weights, "difftx.zerogate.resident",
              () => packZeroGateWeights(allBlocks, weightPrecision), weightPrecision) }
        : undefined;
      // Per token and per block; the gate does not read the sample.
      const zeroGateBytes = tokens * channels * 4;
      const zeroGateOut = batchedGates
        ? scratch("difftx.zero-gates", allBlocks.length * zeroGateBytes) : undefined;
      // 🔴 A SECOND BUFFER RATHER THAN TWO HALVES OF ONE, so the kernel needs no
      // block COUNT to find ffw-out's half - it writes both at the same index
      // and the block arrives as z.
      const ffwGateOut = batchedGates
        ? scratch("difftx.ffw-zero-gates", allBlocks.length * zeroGateBytes) : undefined;
      // ...and the two conditioned norms' scale AND shift, so twice as wide.
      const scaleShiftBytes = tokens * 2 * channels * 4;
      const adalnScaleShift = batchedGates
        ? scratch("difftx.adaln-scale-shift", allBlocks.length * scaleShiftBytes) : undefined;
      const ffwAdalnScaleShift = batchedGates
        ? scratch("difftx.ffw-adaln-scale-shift", allBlocks.length * scaleShiftBytes)
        : undefined;
      if (batchedGates) {
        validation.begin();
        const encoder = this.device.createCommandEncoder({ label: "difftx.zero-gates" });
        const pass = encoder.beginComputePass({ label: "zero-gates" });
        pass.setPipeline(compiled.zeroGates);
        pass.setBindGroup(0, this.device.createBindGroup({
          layout: compiled.zeroGates.getBindGroupLayout(0),
          entries: [condBuffer, condNormalised, zeroGateWeights,
                    zeroGateOut, ffwGateOut, adalnScaleShift, ffwAdalnScaleShift]
            .map((allocation, binding) => ({
              binding, resource: { buffer: allocation.buffer },
            })),
        }));
        // 🔴 sources.gateTile, NOT `tile`. This kernel is the one built under its
        // own tiling, and the file's own warning above says what dividing by a
        // different one costs: "the SHADERS tiled by four while the DISPATCH
        // divided the token count by eight". Here it was the benign direction -
        // 68 groups launched where 9 do the work, the other 59 returning at the
        // bounds check - but benign by luck of a guard is not a reason to keep
        // it, and it is 7.5x the launches.
        pass.dispatchWorkgroups(Math.ceil(tokens / sources.gateTile),
            channels / (shape.lanes ?? 256), allBlocks.length);
        pass.end();
        this.device.queue.submit([encoder.finish()]);
        validation.end("batched zero gate");
      }

      // 🔴 ONE ENCODER AND ONE SUBMIT FOR ALL TWENTY-FOUR BLOCKS. Every block
      // used to finish and submit its own command buffer, which at eight tokens
      // - where the matmuls are nothing at all - was most of what the stack
      // cost. The blocks are strictly sequential on the same buffers and WebGPU
      // orders passes within an encoder, so batching them changes nothing about
      // what runs, only how many times the CPU asks the driver to run it.
      validation.begin();
      let encoder = this.device.createCommandEncoder({ label: "difftx.stack" });
      // See #bindGroups: `key` names the pass, and the entry is rebuilt if the
      // buffers behind it are not the ones it was made from.
      // 🔴 AN ENTRY MAY BE A WINDOW ONTO A BUFFER, NOT ONLY A BUFFER. The
      // batched zero gate is one allocation holding every block's slice, and
      // the kernel reading it is compiled once for all of them - so the block
      // index has to arrive as a bind-group offset. Pass
      // `{allocation, offset, size}` for that and a plain allocation otherwise.
      const bind = (key, pipeline, buffers) => {
        const resources = buffers.map((entry) => (entry.allocation === undefined
          ? { buffer: entry.buffer }
          : { buffer: entry.allocation.buffer, offset: entry.offset, size: entry.size }));
        const held = resources.map((resource) => resource.buffer);
        const found = this.#bindGroups.get(key);
        if (found !== undefined && found.pipeline === pipeline
            && found.buffers.length === held.length
            && found.buffers.every((buffer, at) => buffer === held[at])
            // ...and the same window, or a cached group would hand this block
            // its predecessor's slice.
            && found.offsets.every((at, index) => at === (resources[index].offset ?? 0))) {
          return found.group;
        }
        const group = this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: resources.map((resource, binding) => ({ binding, resource })),
        });
        this.#bindGroups.set(key, { pipeline, buffers: held, group,
                                    offsets: resources.map((r) => r.offset ?? 0) });
        return group;
      };
      // 🔴 ONE PASS, NOT ONE A DISPATCH. WebGPU orders dispatches inside a
      // compute pass and makes each one's writes visible to the next, which is
      // what AF2's evoformer stack has always relied on - see CLAUDE.md, where
      // "profile.js cannot see into AF2" is that same choice. A denoiser call
      // is 318 passes at 59 tokens and a fold is two hundred of them; measured
      // on AF3 at 200 steps over two interleaved rounds, the warm fold is
      // 3.03-3.11 s batched against 3.12-3.16 per dispatch, pLDDT 84.2848
      // either way.
      //
      // 🔴 AND `batchComputePasses: false` IS WHAT `profileDevice` SETS, because
      // one pass a dispatch is the only shape `tools/gpu/profile.js` can
      // attribute. So the profiled number is 2.5% slower than the shipped one -
      // which is this file's own rule about the profiler costing something,
      // written down where the next person will hit it.
      const batchPasses = deviceTuning(this.device).batchComputePasses !== false;
      let openPass = null;
      const passFor = (label) => {
        if (!batchPasses) return encoder.beginComputePass({ label });
        if (openPass === null) openPass = encoder.beginComputePass({ label: "difftx.blocks" });
        return openPass;
      };
      const endPass = () => {
        if (openPass !== null) { openPass.end(); openPass = null; }
      };
      const run = (label, pipeline, buffers, x, y = 1, key = label, z = 1) => {
        const pass = passFor(label);
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bind(key, pipeline, buffers));
        pass.dispatchWorkgroups(x, y, z);
        if (!batchPasses) pass.end();
      };
      // This block's window onto the batched ffw gate, or the raw conditioning
      // when it was not batched - whichever the kernel was compiled to read.
      const ffwGateSlice = (at) => (batchedGates
        ? { allocation: ffwGateOut, offset: at * zeroGateBytes, size: zeroGateBytes }
        : condBuffer);
      const gateSlice = (at) => ({
        allocation: zeroGateOut, offset: at * zeroGateBytes, size: zeroGateBytes });
      const scaleShiftSlice = (buffer) => (at) => ({
        allocation: buffer, offset: at * scaleShiftBytes, size: scaleShiftBytes });
      const adalnSlice = batchedGates ? scaleShiftSlice(adalnScaleShift) : null;
      const ffwAdalnSlice = batchedGates ? scaleShiftSlice(ffwAdalnScaleShift) : null;
      const projections = [];
      // The uploads a super-block owns, released once its commands are queued.
      const pending = [];
      let submits = 0;
      /**
       * Queue what has been encoded and let its weight uploads be reused.
       *
       * 🔴 RELEASING AFTER THE SUBMIT IS SAFE, AND RELEASING BEFORE IT IS NOT.
       * The allocator RECYCLES a released buffer, so handing one back while a
       * later block's pass is still being encoded against it would give that
       * block the same memory. Once the commands are queued the ordering does
       * the rest: allocator.upload writes through device.queue.writeBuffer,
       * which is ordered against work already submitted, so a recycled buffer
       * is only overwritten after the passes reading it have run. This is the
       * pairformer stack's idiom, for the same reason.
       */
      const flush = (label) => {
        endPass();
        this.device.queue.submit([encoder.finish()]);
        validation.end(label);
        for (let at = pending.length - 1; at >= 0; at -= 1) pending[at].release();
        pending.length = 0;
        submits += 1;
        validation.begin();
        encoder = this.device.createCommandEncoder({ label: "difftx.stack" });
      };
      for (const [groupIndex, group] of weights.superBlocks.entries()) {
        // ...and no upload at all on a call that reuses the logits, since the
        // projection is read by that one pass and by nothing else.
        const firstInGroup = groupIndex * group.blocks.length;
        const projection = buildPairLogits || firstInGroup + group.blocks.length > cached
          ? this.allocator.upload("difftx.pair-projection",
              group.pairLogitsProjection, storage)
          : undefined;
        if (projection !== undefined) {
          projections.push(projection);
          if (!this.residentWeights) pending.push(projection);
        }
        for (let inner = 0; inner < group.blocks.length; inner += 1) {
          const block = group.blocks[inner];
          // 🔴 THE SAME TRADE AS THE PAIRFORMER'S, AND THE SAME ANSWER: TRY IT
          // AND LET THE BUDGET DECIDE. The 24 blocks are about 630 MB resident,
          // which is what makes a 200-step fold affordable - the upload alone
          // was 174 ms a call at eight tokens - but it is also half of what a
          // fold holds on the device. Over budget, this drops to uploading the
          // block per call, which is slow and finishes, rather than a
          // createBuffer the driver accepts on its way to freezing the machine.
          let blockWeights;
          if (this.residentWeights) {
            // A refusal here is not caught: run() restarts the call.
            // 🔴 THE DEVICE PATH FIRST, THE HOST ONE WHEN IT CANNOT. See
            // residentBlockOnDevice: 437 ms of host packing becomes 119 of
            // compute, bit for bit, and a store or manifest that cannot supply
            // the codes falls straight through.
            const onDevice = await residentBlockOnDevice(this.device, block, weightPrecision);
            blockWeights = {
              buffer: onDevice ?? residentBlockBuffer(
                this.device, block, () => packBlockWeights(block, weightPrecision),
                weightPrecision),
            };
            // ...and the host's float32 goes: every buffer this block needs is
            // on the device for the model's lifetime. A lazily loaded weight
            // object decodes again if anything reads it after this.
            releaseWeights(block);
          } else {
            // ...held only until this super-block is submitted; see flush().
            blockWeights = keep(this.allocator.upload("difftx.block",
              packBlockWeights(block, weightPrecision).data, storage));
            pending.push(blockWeights);
          }
          // 🔴 THE BIND GROUP CACHE IS KEYED BY BLOCK, NOT BY LABEL. Every
          // block runs the same seven labels against its OWN weights, so one
          // entry per label would be a cache that misses every time and
          // rebuilds - or, worse, hits and binds block 0's weights.
          const at = groupIndex * perSuper + inner;
          const runBlock = (label, pipeline, buffers, x, y = 1, z = 1) =>
            run(label, pipeline, buffers, x, y, `${label}:${at}`, z);
          const logits = logitsFor(at);
          if (buildPairLogits || at >= cached) {
            const pairGroups = Math.ceil(pairs / 64);
            runBlock("pair-logits", compiled.pairLogits[inner],
                [normalized, projection, logits],
                Math.min(pairGroups, GRID_WIDTH), Math.ceil(pairGroups / GRID_WIDTH));
          }
          if (normKSplits > 1) {
            runBlock("adaln", compiled.adaln, [condNormalised, blockWeights, normPartials],
                Math.ceil(rows / tile), sources.normSplits, normKSplits);
            runBlock("adaln-reduce", compiled.adalnReduce,
                [actBuffer, blockWeights, normPartials, xBuffer],
                Math.ceil(rows / tile), sources.normSplits);
          } else {
            // 🔴 THREE BINDINGS WHEN BATCHED, and act first in both forms. The
            // projection was this kernel's only read of the block weights, so
            // an auto layout drops that binding once it is hoisted.
            runBlock("adaln", compiled.adaln,
                batchedGates
                  ? [actBuffer, adalnSlice(at), xBuffer]
                  : [actBuffer, condNormalised, blockWeights, xBuffer],
                Math.ceil(rows / tile), sources.normSplits);
          }
          if (kSplits > 1) {
            runBlock("qkvg", compiled.qkvg, [xBuffer, blockWeights, qkvgPartials],
                Math.ceil(rows / qkvgTile), sources.qkvgSplits, kSplits);
            const reduceGroups = Math.ceil((rows * width) / (shape.lanes ?? 256));
            runBlock("qkvg-reduce", compiled.qkvgReduce,
                [qkvgPartials, blockWeights, q, k, v, gate],
                Math.min(reduceGroups, GRID_WIDTH), Math.ceil(reduceGroups / GRID_WIDTH));
          } else {
            runBlock("qkvg", compiled.qkvg, [xBuffer, blockWeights, q, k, v, gate],
                Math.ceil(rows / tile), sources.qkvgSplits);
          }
          const slots = rows * heads;
          runBlock("attend", compiled.attend, [q, k, v, logits, maskBuffer, gathered],
              Math.min(slots, GRID_WIDTH), Math.ceil(slots / GRID_WIDTH));
          if (attnKSplits > 1) {
            runBlock("attention-output", compiled.attentionOutput,
                [gathered, gate, blockWeights, attnPartials],
                Math.ceil(rows / attnOutTile), sources.outSplits, attnKSplits);
            runBlock("attention-output-reduce", compiled.attentionOutputReduce,
                batchedGates
                  ? [attnPartials, gateSlice(at), actBuffer]
                  : [attnPartials, condBuffer, blockWeights, actBuffer],
                Math.ceil(rows / attnOutTile), sources.outSplits);
          } else {
            // 🔴 THIS BLOCK'S WINDOW ONTO THE BATCHED GATE, or the raw
            // conditioning when it was not batched - the kernel is compiled for
            // one or the other and binding 2 is whichever it reads.
            runBlock("attention-output", compiled.attentionOutput,
                [gathered, gate, batchedGates ? gateSlice(at) : condBuffer,
                 blockWeights, actBuffer],
                Math.ceil(rows / attnOutTile), sources.outSplits);
          }
          if (normKSplits > 1) {
            runBlock("ffw-adaln", compiled.ffwAdaln, [condNormalised, blockWeights, normPartials],
                Math.ceil(rows / tile), sources.normSplits, normKSplits);
            runBlock("ffw-adaln-reduce", compiled.ffwAdalnReduce,
                [actBuffer, blockWeights, normPartials, xBuffer],
                Math.ceil(rows / tile), sources.normSplits);
          } else {
            runBlock("ffw-adaln", compiled.ffwAdaln,
                batchedGates
                  ? [actBuffer, ffwAdalnSlice(at), xBuffer]
                  : [condNormalised, blockWeights, actBuffer, xBuffer],
                Math.ceil(rows / tile), sources.normSplits);
          }
          if (kSplits > 1) {
            runBlock("ffw-wide", compiled.ffwWide, [xBuffer, blockWeights, widePartials],
                Math.ceil(rows / wideTile), sources.wideSplits, kSplits);
            const wideGroups = Math.ceil(
              (rows * channels * weights.transitionFactor) / (shape.lanes ?? 256));
            runBlock("ffw-wide-reduce", compiled.ffwWideReduce, [widePartials, gatedBuffer],
                Math.min(wideGroups, GRID_WIDTH), Math.ceil(wideGroups / GRID_WIDTH));
          } else {
            runBlock("ffw-wide", compiled.ffwWide, [xBuffer, blockWeights, gatedBuffer],
                Math.ceil(rows / tile), sources.wideSplits);
          }
          if (outKSplits > 1) {
            runBlock("ffw-out", compiled.ffwOut,
                [gatedBuffer, blockWeights, outPartials],
                Math.ceil(rows / outTile), sources.outSplits, outKSplits);
            // ...and the reduce drops `weights` entirely once the gate is
            // precomputed, so the bind group is three long, not four.
            runBlock("ffw-out-reduce", compiled.ffwOutReduce,
                batchedGates
                  ? [outPartials, ffwGateSlice(at), actBuffer]
                  : [outPartials, condBuffer, blockWeights, actBuffer],
                Math.ceil(rows / outTile), sources.outSplits);
          } else {
            runBlock("ffw-out", compiled.ffwOut,
                [gatedBuffer, ffwGateSlice(at), blockWeights, actBuffer],
                Math.ceil(rows / outTile), sources.outSplits);
          }
        }
        // 🔴 ONE SUBMIT FOR THE WHOLE STACK WHEN THE WEIGHTS ARE RESIDENT, AND
        // ONE PER SUPER-BLOCK WHEN THEY ARE NOT. Batching all twenty-four
        // blocks into one command buffer is worth a lot at small token counts,
        // where the driver call is most of what the stack costs - but it also
        // means no upload can be released until the end, and the uploading path
        // then holds all twenty-four at once: 24 x 31.5 MiB is 756 MiB, MORE
        // than the 630 of residency it was called in to avoid. Four blocks at a
        // time bounds that at a super-block, which is the whole point of
        // falling back. Resident runs are untouched and still submit once.
        if (!this.residentWeights) flush(`super-block ${groupIndex}`);
      }
      endPass();
      // ...and the readback rides the same submit, when there is one.
      if (!keepOnDevice) {
        encoder.copyBufferToBuffer(actBuffer.buffer, 0, readback.buffer, 0, tokens * channels * 4);
      }
      this.device.queue.submit([encoder.finish()]);
      validation.end(submits === 0 ? "block stack" : "readback");
      // ...only now, because a refusal anywhere above restarts the call and a
      // cache marked ready before it is filled is one attend reads as noise.
      this.#pairLogits.ready = true;
      // 🔴 THE PAIR PROJECTIONS ARE RELEASED AFTER THE SUBMIT, NOT INSIDE THE
      // LOOP. They are pooled, so releasing one while a later block's encoded
      // pass still refers to it would hand the same buffer to that block's
      // upload.
      for (const projection of projections) projection.release();
      if (keepOnDevice) {
        return {
          output: undefined, outputBuffer: actBuffer.buffer,
          elapsedMilliseconds: performance.now() - start,
          memory: this.allocator.snapshot(),
        };
      }
      // ...the boundary that already synchronises, so the deferred scopes cost
      // nothing to read here.
      await validation.settle();
      await readback.buffer.mapAsync(GPUMapMode.READ);
      const result = new Float32Array(readback.buffer.getMappedRange().slice(0));
      readback.buffer.unmap();
      return {
        output: result,
        elapsedMilliseconds: performance.now() - start,
        memory: this.allocator.snapshot(),
      };
    } finally {
      for (let index = allocations.length - 1; index >= 0; index -= 1) allocations[index].release();
    }
  }
}
