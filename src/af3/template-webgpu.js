/**
 * AF3's template embedder on the GPU - the empty-template path.
 *
 * 🔴 THIS IS THE MODULE EVERYONE SKIPS AND NOBODY SHOULD. With FOUR EMPTY
 * template slots its output measures std 13.1 against a pair whose own std is
 * 55 - about a quarter of what enters the MSA stack. Nine features are summed
 * into the embedding and only six are template geometry; the other three are
 * the query's own aatype (once per axis) and the query pair representation
 * itself. With no template the geometry vanishes and those three do not, so the
 * module becomes a learned transform of the query - and then runs it through
 * two pairformer blocks. AF2-multimer had the identical trap and it cost this
 * project a week there.
 *
 * 🔴 THE SIX GEOMETRY FEATURES ARE COMPUTED ON THE HOST, NOT IN A SHADER, and
 * that is a choice rather than an omission. They are O(tokens^2) arithmetic
 * over coordinates - a distogram bin, two masks and a unit vector per pair -
 * and src/af3/template-features.js already computes them, is held to AF3 by
 * tools/oracle/check_af3_template_geometry.js, and is where the one real bug
 * in them was found. Writing them again in WGSL would mean a second
 * implementation of a thing that took an oracle to get right, for work that
 * does not scale with the model: 300 tokens is 90k pairs, once per fold,
 * against a trunk that runs 48 blocks over the same pairs 4 times.
 *
 * What goes to the device is the RESULT, six floats a pair - see
 * packTemplateGeometry.
 *
 * 🔴 THE SUM IS DIVIDED BY THE SLOT COUNT, NOT BY HOW MANY SLOTS ARE REAL. Four
 * empty slots produce the same embedding four times, so the division puts it
 * back and the module behaves as though there were exactly one template.
 * Dividing by the number of REAL templates would be a division by zero here -
 * and it means one real template among four slots is worth a QUARTER of what
 * it would be alone, which is AF3's arithmetic and not an oversight.
 *
 * The two template blocks are the shared pair track at 64 channels with a
 * factor-2 transition; see src/af3/pair-track-gpu.js.
 */
import { GpuBufferAllocator } from "../runtime/allocator.js";
import { storageBytes } from "../runtime/storage.js";
import { pipelineCacheForDevice } from "../runtime/pipeline-cache.js";
import {
  GRID_WIDTH, PAIR_SCRATCH_COUNT, UNPACKED_PAIR_SCRATCH, compilePairTrack, encodePairTrack,
  packPairTrackWeights,
} from "./pair-track-gpu.js";
import {
  GEOMETRY_STRIDE, coverageOf, multichainMaskFor, packTemplateGeometry, templateGeometry,
} from "./template-features.js";

// ...re-exported from where they used to live, because the packing is shared
// with AF2 now and the geometry module is where both models reach for it.
export { GEOMETRY_STRIDE, packTemplateGeometry };

const CHANNELS = 64;
const RESTYPES = 31;

const ORDER = [
  "queryEmbeddingNormScale", "queryEmbeddingNormOffset", "templatePairEmbedding8",
  "templatePairEmbedding2", "templatePairEmbedding3",
  // The six geometry projections. Four of them are [64] rather than a matrix:
  // AF3 builds them with `num_input_dims=0`, so the feature is a SCALAR per
  // pair and the weight a per-channel scale. See template-features.js.
  "templatePairEmbedding0", "templatePairEmbedding1", "templatePairEmbedding4",
  "templatePairEmbedding5", "templatePairEmbedding6", "templatePairEmbedding7",
  "outputLayerNormScale", "outputLayerNormOffset", "outputLinear",
];

export function packTemplateWeights(weights) {
  const offsets = {};
  let total = 0;
  for (const name of ORDER) {
    if (weights[name] === undefined) throw new Error(`template weights missing ${name}`);
    offsets[name] = total;
    total += weights[name].length;
  }
  const data = new Float32Array(total);
  for (const name of ORDER) data.set(weights[name], offsets[name]);
  return { data, offsets };
}

export function createTemplateShaders(shape, offsets, epsilon, variance) {
  const { tokens, queryChannels, templates } = shape;
  const pairs = tokens * tokens;

  const common = `
const TOKENS: u32 = ${tokens}u;
const PAIRS: u32 = ${pairs}u;
const QUERY_CHANNELS: u32 = ${queryChannels}u;
const CHANNELS: u32 = ${CHANNELS}u;
const RESTYPES: u32 = ${RESTYPES}u;
const GRID_WIDTH: u32 = ${GRID_WIDTH}u;
const EPSILON: f32 = ${epsilon};
// 🔴 1/(slots), NOT slots/(slots) - AND IT USED TO BE THE SECOND. While every
// slot produced the same embedding the shader computed ONE of them, so
// "sum four and divide by four" collapsed to a multiply by
// templates/(1e-7 + templates), which is 1 to within a rounding error. The
// sum is real now, so the scale is the division alone. Leaving the old
// expression would have made a one-template fold four times too strong and an
// empty fold unchanged, which is the shape of bug that passes every existing
// check.
//
// It is the SLOT count and not the real-template count: four empty slots each
// produce the same embedding and the division puts it back, so the module
// behaves as though there were exactly one template whatever the slot count.
const TEMPLATE_SCALE: f32 = ${1 / (1e-7 + templates)};
const W_QUERY_SCALE: u32 = ${offsets.queryEmbeddingNormScale}u;
const W_QUERY_OFFSET: u32 = ${offsets.queryEmbeddingNormOffset}u;
const W_EMBED8: u32 = ${offsets.templatePairEmbedding8}u;
const W_EMBED2: u32 = ${offsets.templatePairEmbedding2}u;
const W_EMBED3: u32 = ${offsets.templatePairEmbedding3}u;
const W_EMBED0: u32 = ${offsets.templatePairEmbedding0}u;
const W_EMBED1: u32 = ${offsets.templatePairEmbedding1}u;
const W_EMBED4: u32 = ${offsets.templatePairEmbedding4}u;
const W_EMBED5: u32 = ${offsets.templatePairEmbedding5}u;
const W_EMBED6: u32 = ${offsets.templatePairEmbedding6}u;
const W_EMBED7: u32 = ${offsets.templatePairEmbedding7}u;
const GEOMETRY_STRIDE: u32 = ${GEOMETRY_STRIDE}u;
const W_OUT_SCALE: u32 = ${offsets.outputLayerNormScale}u;
const W_OUT_OFFSET: u32 = ${offsets.outputLayerNormOffset}u;
const W_OUT: u32 = ${offsets.outputLinear}u;
`;

  const varianceCode = (count, read) => variance === "fast"
    ? `let variance = squares / f32(${count}) - mean * mean;`
    : `var variance = 0.0;
  for (var c = 0u; c < ${count}; c += 1u) {
    let d = ${read} - mean;
    variance += d * d;
  }
  variance /= f32(${count});`;

  // The query pair representation, normalised and projected to 64, plus the
  // template aatype along each axis. An empty slot carries type 0, so those two
  // contribute ROW 0 of each weight rather than nothing.
  const embed = `${common}
@group(0) @binding(0) var<storage, read> pair: array<f32>;
@group(0) @binding(1) var<storage, read> aatype: array<i32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> act: array<f32>;
// 🔴 ZEROES MEAN AN EMPTY SLOT AND THE SHADER NEED NOT KNOW WHICH. The
// distogram bin is stored PLUS ONE, so 0 is "no bin" - which is what both an
// empty slot and a pair closer than 3.25 A have. One pipeline serves both.
@group(0) @binding(4) var<storage, read> geometry: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x + id.y * GRID_WIDTH * 64u;
  if (row >= PAIRS) { return; }
  let i = row / TOKENS;
  let j = row % TOKENS;
  let base = row * QUERY_CHANNELS;

  var total = 0.0;
  var squares = 0.0;
  for (var c = 0u; c < QUERY_CHANNELS; c += 1u) {
    let value = pair[base + c];
    total += value;
    squares += value * value;
  }
  let mean = total / f32(QUERY_CHANNELS);
  ${varianceCode("QUERY_CHANNELS", "pair[base + c]")}
  let inverse_std = inverseSqrt(variance + EPSILON);

  // 🔴 FEATURE 2 VARIES ALONG j AND FEATURE 3 ALONG i. AF3 writes them as
  // aatype[None, :, :] and aatype[:, None, :]; swapping them transposes a term
  // that nothing downstream complains about.
  let code_row = aatype[j];
  let code_column = aatype[i];
  for (var e = 0u; e < CHANNELS; e += 1u) {
    var value = 0.0;
    for (var c = 0u; c < QUERY_CHANNELS; c += 1u) {
      let normalized = (pair[base + c] - mean) * inverse_std * weights[W_QUERY_SCALE + c]
        + weights[W_QUERY_OFFSET + c];
      value += normalized * weights[W_EMBED8 + c * CHANNELS + e];
    }
    if (code_row >= 0 && u32(code_row) < RESTYPES) {
      value += weights[W_EMBED2 + u32(code_row) * CHANNELS + e];
    }
    if (code_column >= 0 && u32(code_column) < RESTYPES) {
      value += weights[W_EMBED3 + u32(code_column) * CHANNELS + e];
    }

    // Features 0, 1, 4, 5, 6 and 7: the template geometry, computed on the
    // host and packed by packTemplateGeometry.
    let g = row * GEOMETRY_STRIDE;
    let bin = u32(geometry[g]);
    if (bin > 0u) {
      value += weights[W_EMBED0 + (bin - 1u) * CHANNELS + e];
    }
    value += geometry[g + 1u] * weights[W_EMBED1 + e];
    value += geometry[g + 2u] * weights[W_EMBED4 + e];
    value += geometry[g + 3u] * weights[W_EMBED5 + e];
    value += geometry[g + 4u] * weights[W_EMBED6 + e];
    value += geometry[g + 5u] * weights[W_EMBED7 + e];

    act[row * CHANNELS + e] = value;
  }
}`;

  // 🔴 THESE WERE ONE SHADER AND COULD NOT STAY ONE. It fused the LayerNorm,
  // the slot-count scaling, the relu and the projection, which is correct only
  // when every slot produces the SAME embedding - true while the only path was
  // four empty slots and false the moment one carries a template. The
  // LayerNorm and the summation are per slot; the scale, the relu and the
  // projection happen once, on the sum.
  const accumulate = `${common}
@group(0) @binding(0) var<storage, read> act: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> summed: array<f32>;
// 🔴 HOW MANY IDENTICAL SLOTS THIS PASS STANDS FOR. Empty slots all produce
// the SAME embedding, so running four of them is four times the work for an
// answer that is one of them times four. See the note in run().
@group(0) @binding(3) var<storage, read> repeat: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x + id.y * GRID_WIDTH * 64u;
  if (row >= PAIRS) { return; }
  let base = row * CHANNELS;

  var total = 0.0;
  var squares = 0.0;
  for (var c = 0u; c < CHANNELS; c += 1u) {
    let value = act[base + c];
    total += value;
    squares += value * value;
  }
  let mean = total / f32(CHANNELS);
  ${varianceCode("CHANNELS", "act[base + c]")}
  let inverse_std = inverseSqrt(variance + EPSILON);

  for (var c = 0u; c < CHANNELS; c += 1u) {
    summed[base + c] += ((act[base + c] - mean) * inverse_std * weights[W_OUT_SCALE + c]
      + weights[W_OUT_OFFSET + c]) * repeat[0];
  }
}`;

  // The slot-count scaling, a relu, and the projection back up.
  const output = `${common}
@group(0) @binding(0) var<storage, read> summed: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x + id.y * GRID_WIDTH * 64u;
  if (row >= PAIRS) { return; }
  let base = row * CHANNELS;

  for (var f = 0u; f < QUERY_CHANNELS; f += 1u) {
    var value = 0.0;
    for (var c = 0u; c < CHANNELS; c += 1u) {
      // ...relu BEFORE the projection, so the module can only add along a
      // non-negative combination of output_linear's directions.
      let scaled = max(summed[base + c] * TEMPLATE_SCALE, 0.0);
      value += scaled * weights[W_OUT + c * QUERY_CHANNELS + f];
    }
    output[row * QUERY_CHANNELS + f] = value;
  }
}`;

  return { embed, accumulate, output };
}

export class Af3TemplateEmbedderGpu {
  constructor(device) {
    this.device = device;
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
  }

  /**
   * @param {{pair: Float32Array, pairMask: Float32Array, tokens: number,
   *          templates: number, slots?: (object|undefined)[],
   *          multichainMask2d?: ArrayLike<number>}} input `slots` holds one
   *   entry per OCCUPIED slot - `{aatype, atomPositions, atomMask}` in AF3's
   *   dense-24 layout - with `undefined` for an empty one. Absent, every slot
   *   is empty, which is what a de novo fold has and is still not a no-op.
   * @param {object} weights the tensors in ORDER, `blocks`, `queryChannels`
   * @param {{swapTransposedBias: boolean}} dialect
   */
  async run(input, weights, dialect, options = {}) {
    const { tokens, templates } = input;
    const slots = input.slots ?? [];
    if (slots.length > templates) {
      throw new RangeError(`${slots.length} templates for ${templates} slots`);
    }
    // 🔴 THE OLD FLAG STILL REFUSES, RATHER THAN BEING IGNORED. Callers wrote
    // `templateOccupied: <does the dump have a template>` to fail loudly when
    // one appeared, back when this path could not handle it. Now that it can,
    // dropping the flag would turn that deliberate noise into silence: a dump
    // WITH a template would be folded WITHOUT one and simply score worse.
    if (input.templateOccupied === true && slots.filter(Boolean).length === 0) {
      throw new Error("templateOccupied is true but no slots were given:"
        + " pass `slots` with {aatype, atomPositions, atomMask} per template");
    }
    if (dialect?.swapTransposedBias === undefined) {
      throw new Error("dialect.swapTransposedBias has no default");
    }
    const queryChannels = weights.queryChannels;
    const pairs = tokens * tokens;
    const epsilon = options.epsilon ?? 1e-5;
    const variance = options.variance ?? "fast";

    const packed = packTemplateWeights(weights);
    const sources = createTemplateShaders(
      { tokens, queryChannels, templates }, packed.offsets, epsilon, variance);
    const base = `af3-template:${tokens}:${queryChannels}:${templates}:${epsilon}`
      + `:${variance}:${dialect.swapTransposedBias}`;
    const compiled = {};
    for (const [name, source] of Object.entries(sources)) {
      compiled[name] = await this.pipelines.get(`${base}:${name}`, source);
    }
    // The template stack: the shared pair track at 64 channels, factor 2.
    const trackPipelines = await compilePairTrack(this.pipelines, {
      scratchStorage: UNPACKED_PAIR_SCRATCH,
      n: tokens, channels: CHANNELS, transitionFactor: 2,
      sample: weights.blocks[0], epsilon, variance, dialect, base: `${base}:track`,
    });

    const gridHeads = weights.blocks[0].pairAttention1.heads;
    const storage = GPUBufferUsage.STORAGE;
    const allocations = [];
    const keep = (allocation) => { allocations.push(allocation); return allocation; };
    try {
      const pair = keep(this.allocator.upload("af3-template.pair", input.pair, storage));
      const pairMask = keep(this.allocator.upload("af3-template.mask", input.pairMask, storage));
      const weightBuffer = keep(this.allocator.upload("af3-template.weights", packed.data, storage));
      // 🔴 ONE BUFFER PER SLOT, AND REUSING ONE IS THE BUG THAT LOOKS LIKE A
      // WRONG KERNEL. `queue.writeBuffer` is ordered against SUBMITS, not
      // against the recording of a command encoder - so writing slot 0's data,
      // recording its passes, writing slot 1's over the top, recording those,
      // and submitting once at the end runs every slot against the LAST
      // slot's data. Measured: with one occupied slot of four the whole module
      // computed the all-empty answer, which differs by only the real slot's
      // quarter share and scored relRMS 2.1e-2 - small enough to read as a
      // precision problem and wrong enough to lose the template entirely.
      //
      // The aatype and the geometry are the only things that differ between
      // slots; the query pair, the masks and every weight are shared and are
      // uploaded once. Four geometry buffers is 6 floats a pair per slot -
      // 8.6 MiB at 300 tokens, against a trunk that holds hundreds.
      const empty = new Float32Array(pairs * GEOMETRY_STRIDE);
      const EMPTY_MASK = new Float32Array(pairs);
  // 🔴 THE MASK IS PER SLOT AND IS NOT ALLOWED TO DEFAULT TO "EVERYTHING". It
      // did, and a two-chain query with a template on each chain then scored
      // relRMS 1.09 against AF3 - the cross-chain geometry is most of the module's
      // answer, so a permissive default is not a small error. It went unnoticed
      // because every check had a ONE-CHAIN query, where all-ones and per-chain
      // are the same array.
      const chainMaskFor = (template) => {
        if (input.multichainMask2d !== undefined) return input.multichainMask2d;
        if (input.asymId === undefined) {
          if (template === undefined || template === null) {
            // An empty slot has no geometry to mask, so the mask is unread.
            return EMPTY_MASK;
          }
          throw new Error("a template needs `asymId` (or `multichainMask2d`):"
            + " AF3 masks the geometry features across chains, and assuming one"
            + " chain silently lets a template speak about pairs it has never"
            + " seen in one coordinate frame");
        }
        return multichainMaskFor(input.asymId, tokens, {
          coverage: coverageOf(template, tokens),
          // ...opt in, and only where one structure covered both chains. See
          // multichainMaskFor.
          spanChains: template.spanChains === true,
        });
      };

      // 🔴 THE EMPTY SLOTS ARE RUN ONCE BETWEEN THEM, NOT ONCE EACH. They
      // produce the same embedding by construction - same all-ALA aatype, same
      // zero geometry, same query pair, same weights - so four of them is four
      // times the work for one answer counted four times. The old code got
      // this for free by never having a real slot to run; measured on the
      // trunk checker, running all four cost 150 ms against 40 for one, on
      // every de novo fold, for an identical result.
      //
      // So each PASS carries how many slots it stands for, and the accumulate
      // shader multiplies by it. A fold with no templates runs one pass, which
      // is what it always did.
      const passes = [];
      let emptySlots = 0;
      for (let slot = 0; slot < templates; slot += 1) {
        if (slots[slot] === undefined || slots[slot] === null) emptySlots += 1;
        else passes.push({ template: slots[slot], repeat: 1 });
      }
      if (emptySlots > 0) passes.push({ template: undefined, repeat: emptySlots });

      const slotBuffers = [];
      for (const { template, repeat } of passes) {
        const slot = slotBuffers.length;
        const aatypeData = new Int32Array(tokens);
        // An empty slot carries type 0 - ALA - which contributes ROW 0 of each
        // aatype weight rather than nothing. That is half of why an empty slot
        // is not a no-op; see the note at the top of this file.
        if (template !== undefined && template !== null) {
          for (let t = 0; t < tokens; t += 1) aatypeData[t] = template.aatype[t];
        }
        slotBuffers.push({
          repeat: keep(this.allocator.upload(
            `af3-template.repeat.${slot}`, Float32Array.from([repeat]), storage)),
          aatype: keep(this.allocator.upload(
            `af3-template.aatype.${slot}`, aatypeData, storage)),
          // ...and an empty slot's geometry is zeros, which the shader reads as
          // "no bin, no mask, no direction" with no branch of its own.
          geometry: keep(this.allocator.upload(
            `af3-template.geometry.${slot}`,
            template !== undefined && template !== null
              ? packTemplateGeometry(
                templateGeometry(template, chainMaskFor(template), tokens), tokens)
              : empty,
            storage)),
        });
      }

      const act = keep(this.allocator.allocate("af3-template.act", pairs * CHANNELS * 4, storage));
      // The running sum over slots, which the projection reads once at the end.
      const summed = keep(this.allocator.allocate(
        "af3-template.summed", pairs * CHANNELS * 4, storage | GPUBufferUsage.COPY_DST));
      // 🔴 SEVEN PAIR-SIZED SCRATCH BUFFERS, AND THIS STACK KEEPS THEM WHOLE.
      // See UNPACKED_PAIR_SCRATCH: packing them costs this embedder 150x its
      // agreement with AF3 and saves 26 MiB on a stage that is not the trunk's
      // peak. The allocation and the shaders read the same array, because a
      // buffer that disagrees with a shader about its element is not something
      // WebGPU can catch.
      const scratch = [];
      for (let index = 0; index < PAIR_SCRATCH_COUNT; index += 1) {
        scratch.push(keep(this.allocator.allocate(
          `af3-template.scratch${index}`,
          storageBytes(pairs * CHANNELS, UNPACKED_PAIR_SCRATCH[index]), storage)));
      }
      const biasBuffer = keep(this.allocator.allocate(
        "af3-template.bias", gridHeads * pairs * 4, storage));
      const output = keep(this.allocator.allocate(
        "af3-template.output", pairs * queryChannels * 4, storage | GPUBufferUsage.COPY_SRC));
      const readback = keep(this.allocator.allocate(
        "af3-template.readback", pairs * queryChannels * 4,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));

      const blockAllocations = [];
      const upload = (label, data) => {
        const allocation = this.allocator.upload(label, data, storage);
        blockAllocations.push(allocation);
        return allocation;
      };

      this.device.pushErrorScope("validation");
      const encoder = this.device.createCommandEncoder({ label: "af3-template" });
      const run = (label, pipeline, buffers, x, y = 1, z = 1) => {
        const pass = encoder.beginComputePass({ label });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: buffers.map((allocation, binding) => ({
            binding, resource: { buffer: allocation.buffer },
          })),
        }));
        pass.dispatchWorkgroups(x, y, z);
        pass.end();
      };
      const spread = (groups) => [Math.min(groups, GRID_WIDTH), Math.ceil(groups / GRID_WIDTH)];
      const linear = spread(Math.ceil(pairs / 64));

      // 🔴 THE BLOCK WEIGHTS ARE PACKED AND UPLOADED ONCE, OUTSIDE THE SLOT
      // LOOP. Every slot runs the SAME two pairformer blocks, so packing them
      // per slot would repack 1.4 MiB four times for four identical buffers -
      // and the release below would then have to know which upload belonged to
      // which pass.
      const blockWeights = weights.blocks.map((block, index) => {
        const packedTrack = packPairTrackWeights(block, CHANNELS);
        return {
          outgoing: upload(`w.tri.out.${index}`, packedTrack.outgoing),
          incoming: upload(`w.tri.in.${index}`, packedTrack.incoming),
          grid1: upload(`w.grid1.${index}`, packedTrack.grid1),
          grid2: upload(`w.grid2.${index}`, packedTrack.grid2),
          transition: upload(`w.transition.${index}`, packedTrack.transition),
        };
      });

      for (let slot = 0; slot < slotBuffers.length; slot += 1) {
        run(`template.embed.${slot}`, compiled.embed,
            [pair, slotBuffers[slot].aatype, weightBuffer, act,
             slotBuffers[slot].geometry], linear[0], linear[1]);
        for (let index = 0; index < blockWeights.length; index += 1) {
          encodePairTrack({
            run, pipelines: trackPipelines, n: tokens, channels: CHANNELS, gridHeads,
            pair: act, pairMask, scratch, biasBuffer, weights: blockWeights[index],
          });
        }
        run(`template.accumulate.${slot}`, compiled.accumulate,
            [act, weightBuffer, summed, slotBuffers[slot].repeat], linear[0], linear[1]);
      }
      run("template.output", compiled.output, [summed, weightBuffer, output],
          linear[0], linear[1]);
      encoder.copyBufferToBuffer(output.buffer, 0, readback.buffer, 0, pairs * queryChannels * 4);

      const start = performance.now();
      this.device.queue.submit([encoder.finish()]);
      const error = await this.device.popErrorScope();
      await this.device.queue.onSubmittedWorkDone();
      for (let index = blockAllocations.length - 1; index >= 0; index -= 1) {
        blockAllocations[index].release();
      }
      if (error !== null) throw new Error(`WebGPU validation failed: ${error.message}`);
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
