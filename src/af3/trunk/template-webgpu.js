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
 * and src/af3/featurise/template-features.js already computes them, is held to AF3 by
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
 * factor-2 transition; see src/af3/trunk/pair-track-gpu.js.
 */
import { templateTransitionFactor } from "./template-reference.js";
import { deviceTuning, shapedKnob } from "../../runtime/device-profile.js";
import { GpuBufferAllocator } from "../../runtime/allocator.js";
import { residentWeightBuffer } from "../../runtime/resident.js";
import { residencyAllowed } from "../../runtime/device-memory.js";
import { storageBytes } from "../../runtime/storage.js";
import { pipelineCacheForDevice, settleAll } from "../../runtime/pipeline-cache.js";
import {
  GRID_WIDTH, PAIR_SCRATCH_COUNT, UNPACKED_PAIR_SCRATCH, compilePairTrack, createAddShader,
  encodePairTrack, packPairTrackWeights,
} from "./pair-track-gpu.js";
import { residentPairTrackOnDevice } from "../weights/pair-track-device-weights.js";
import { allocateGridProjectMatrix, gridProjectMatrixConfig }
  from "./grid-project-matrix.js";
import {
  DGRAM_BINS, GEOMETRY_STRIDE, boltz2TemplateFeatures, rosettafold3TemplateFeatures,
  coverageOf, multichainMaskFor,
  packTemplateGeometry,
  templateGeometry,
} from "../featurise/template-features.js";

// ...re-exported from where they used to live, because the packing is shared
// with AF2 now and the geometry module is where both models reach for it.
export { GEOMETRY_STRIDE, packTemplateGeometry };

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

// 🔴 THE FUSED EMBEDDER'S TENSORS, WHICH ARE A DIFFERENT SET AND NOT A SUBSET.
// protenix2 and boltz2 run boltz2's module: one `a_proj` over the 108
// concatenated feature columns where AF3 sums nine projections, and `z_norm` /
// `z_proj` where AF3 has `query_embedding_norm` / `template_pair_embedding_8`.
// weights.js maps the last two onto the AF3 names already, and maps `v_norm` /
// `u_proj` onto the output pair - so only the INPUT stage differs here.
const FUSED_ORDER = [
  "queryEmbeddingNormScale", "queryEmbeddingNormOffset",
  "zProjection", "aProjection",
  "outputLayerNormScale", "outputLayerNormOffset", "outputLinear",
];

/** Pair rows a `template.embed` workgroup stages; the dispatch divides by it. */
const TEMPLATE_EMBED_ROWS = 4;

/**
 * The fused embedder's 108 feature columns for ONE slot.
 *
 * 🔴 EMPTY SLOTS ONLY, AND IT SAYS SO RATHER THAN GUESSING. A de novo fold has
 * four empty slots and their columns are constant: every geometry feature is
 * exactly zero and both restype blocks are one-hot at GAP. That is measured -
 * `EMPTY=1 tools/oracle/dump_af3_template.py protenix2` - not inferred from the
 * featuriser, whose empty-slot convention differs per vendor (protenix fills the
 * first slot with GAP, opendde all four, intellifold2 deliberately uses 0).
 *
 * A slot WITH a template needs the real featuriser: Boltz's frame convention,
 * the 39 bin edges, the 32-class remap and the multichain masking, all specified
 * in docs/AF3.md and gated by nothing yet. Building it from that specification
 * and checking it against a reference built the same way would prove nothing.
 */
export function fusedTemplateFeatures(template, tokens, width, dialect,
                                      multichainMask2d = undefined, useGap = true) {
  if (template !== undefined && template !== null) {
    // 🔴 THE 108 COLUMNS ARE THE NINE-PROJECTION EMBEDDER'S OWN FEATURES,
    // CONCATENATED. 39 distogram + 1 pseudo-beta mask + 32 restype_i + 32
    // restype_j + 3 unit vector + 1 backbone frame mask = 108, and
    // `templateGeometry` already computes four of the six for AF3's path -
    // measured against af3-any-model's `our_features`, the distogram and both
    // masks are EXACT and the unit vector is 3.64e-7. So this is a
    // concatenation of things this port has had all along, which is why the
    // refusal that stood here was costing more than it protected: boltz2 and
    // protenix2 could not take a template at all while their forward scored
    // 1.52e-7 against the oracle.
    //
    // 🔴 AND `restype_i` VARIES ALONG j, `restype_j` ALONG i. The name says
    // which index the tensor varies along in the reference's own naming, not
    // which one selects its value; built the other way both columns score 1.36
    // and the fold is plausible. The aatype needs NO remap - the table
    // recovered from the dump is the identity - which is worth stating because
    // docs/AF3.md's "32-class remap" reads as though it does.
    // See tools/gpu/check-fused-template-features.js.
    // 🔴 BOLTZ-2's 109 ARE A DIFFERENT CONSTRUCTION, NOT WIDER ONES. 38 bins on
    // different edges, a unit vector that is a SIGN, a restype vocabulary
    // shifted by two over 33 classes, and `restype_i` varying along i where
    // protenix2's varies along j. See boltz2TemplateFeatures.
    if (dialect?.boltz2TemplateFeatures === true) {
      return boltz2TemplateFeatures(template, multichainMask2d, tokens);
    }
    // 🔴 AND RoseTTAFold3's 66 ARE NOT A TEMPLATE IN THE OTHER TWO'S SENSE.
    // They are a CA-CA distance histogram, a coverage flag and a noise level -
    // distance-distribution conditioning rather than a geometry embedding -
    // riding the identical weight scopes, which is why only `a_proj`'s first
    // dimension tells the three apart: 66 against 108 and 109. See
    // rosettafold3TemplateFeatures.
    if (dialect?.rosettafold3TemplateFeatures === true) {
      return rosettafold3TemplateFeatures(template, multichainMask2d, tokens);
    }
    const columnsFor = dialect?.fusedTemplateLayout;
    if (columnsFor === undefined || columnsFor === null) {
      throw new Error("dialect.fusedTemplateLayout has no default: protenix2's"
        + " 108 columns are 39/1/32/32/3/1 and boltz2's 109 are not the same"
        + " widths, and guessing the bin count is guessing the model");
    }
    const { distogramBins, restypes } = columnsFor;
    const geometry = templateGeometry(template, multichainMask2d, tokens);
    if (distogramBins !== DGRAM_BINS) {
      throw new Error(`this dialect wants ${distogramBins} distogram bins and`
        + ` templateGeometry computes ${DGRAM_BINS}: the bin edges are not the`
        + " same feature and nothing here has measured the other set");
    }
    const features = new Float32Array(tokens * tokens * width);
    const restypeI = distogramBins + 1;
    const restypeJ = restypeI + restypes;
    const vectorAt = restypeJ + restypes;
    for (let i = 0; i < tokens; i += 1) {
      for (let j = 0; j < tokens; j += 1) {
        const pair = i * tokens + j;
        const base = pair * width;
        for (let bin = 0; bin < distogramBins; bin += 1) {
          features[base + bin] = geometry.distogram[pair * distogramBins + bin];
        }
        features[base + distogramBins] = geometry.pseudoBetaMask2d[pair];
        const ci = template.aatype[j], cj = template.aatype[i];
        if (ci >= 0 && ci < restypes) features[base + restypeI + ci] = 1;
        if (cj >= 0 && cj < restypes) features[base + restypeJ + cj] = 1;
        for (let axis = 0; axis < 3; axis += 1) {
          features[base + vectorAt + axis] = geometry.unitVector[pair * 3 + axis];
        }
        features[base + vectorAt + 3] = geometry.backboneMask2d[pair];
      }
    }
    return features;
  }
  // 🔴 WHICH COLUMNS AN EMPTY SLOT SETS IS THE DIALECT'S, AND THE TWO FUSED
  // MODELS DISAGREE. protenix2's 108 columns carry GAP in both restype blocks;
  // boltz2's 109 are all zero. Its whole feature ORDER is different too -
  // distogram 38 against 39, restypes 33 against 32 - which is why `a_proj`
  // refused a 108-wide build with "wants 109" rather than folding something
  // plausible.
  // 🔴 AND THE GAP GOES IN EVERY EMPTY SLOT HERE, NOT ONLY THE FIRST - WHICH IS
  // THE OPPOSITE OF THE NINE-PROJECTION PATH. protenix2's `template_aatype` is
  // 21 in slot 0 and 0 in slots 1..3, exactly like OpenDDE's, so gating the gap
  // to the first empty slot is the obvious symmetry - and MEASURED it makes the
  // trunk's `z_after_template` seam WORSE, 3.89e-3 to 4.55e-3. The fused
  // embedder does not consume `template_aatype`; it consumes 108 columns that
  // protenix's own featuriser builds, and those are not the same array. Left as
  // it is, on the measurement rather than on the symmetry. `useGap` is kept as
  // the arm for re-running that comparison.
  // 🔴 AND A PADDED SLOT IS NOT AN EMPTY ONE. protenix2's featuriser fills its
  // ONE empty template with the GAP restype and zero-pads the rest - but
  // `template_aatype = 0` is zero-padding of the AATYPE, and `one_hot(0, 32)`
  // is NOT a zero row: it sets restype column 0. So the four slots the trunk
  // runs are [gap, restype-0, restype-0, restype-0], not [gap, 0, 0, 0] and
  // not four gaps.
  //
  // All four measured against af3-any-model's own `evoformer/template_embedding`
  // on 6MRR, which has no template: four gaps read 7.39e-3, gap-then-ZERO read
  // worse still, and this reads what is below. Written the wrong way twice
  // before the batch dump was read carefully enough to notice that a one-hot of
  // zero is a one.
  const layout = dialect?.fusedTemplateLayout;
  const gapColumns = dialect?.emptyTemplateRestypeColumns;
  const columns = useGap ? gapColumns
    : (layout === undefined || layout === null || gapColumns === null ? []
      : [layout.distogramBins + 1, layout.distogramBins + 1 + layout.restypes]);
  if (columns === undefined || columns === null) {
    throw new Error("dialect.emptyTemplateRestypeColumns has no default: an "
      + "empty template slot carries GAP under protenix2 and zeros under "
      + "boltz2, and guessing either is a different model");
  }
  const pairs = tokens * tokens;
  const features = new Float32Array(pairs * width);
  for (let index = 0; index < pairs; index += 1) {
    for (const column of columns) features[index * width + column] = 1;
  }
  return features;
}

export function packTemplateWeights(weights) {
  const offsets = {};
  let total = 0;
  for (const name of (weights.fused ? FUSED_ORDER : ORDER)) {
    if (weights[name] === undefined) throw new Error(`template weights missing ${name}`);
    offsets[name] = total;
    total += weights[name].length;
  }
  const data = new Float32Array(total);
  for (const name of (weights.fused ? FUSED_ORDER : ORDER)) {
    data.set(weights[name], offsets[name]);
  }
  return { data, offsets };
}

export function createTemplateShaders(shape, offsets, epsilon, variance) {
  // 🔴 `channels` IS THE STACK'S OWN WIDTH AND IS NOT 64. See stackChannels in
  // template-reference.js: five checkpoints say 64 and IntelliFold-2 says 256.
  const { tokens, queryChannels, templates, channels: CHANNELS,
          fused = false, featureWidth = 0 } = shape;
  if (!Number.isInteger(CHANNELS)) {
    throw new Error("createTemplateShaders needs shape.channels, the template "
      + "stack's own width, which is read off the bundle rather than assumed");
  }
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
const W_QUERY_SCALE: u32 = ${offsets.queryEmbeddingNormScale ?? 0}u;
const W_QUERY_OFFSET: u32 = ${offsets.queryEmbeddingNormOffset ?? 0}u;
const W_EMBED8: u32 = ${offsets.templatePairEmbedding8 ?? 0}u;
const W_EMBED2: u32 = ${offsets.templatePairEmbedding2 ?? 0}u;
const W_EMBED3: u32 = ${offsets.templatePairEmbedding3 ?? 0}u;
const W_EMBED0: u32 = ${offsets.templatePairEmbedding0 ?? 0}u;
const W_EMBED1: u32 = ${offsets.templatePairEmbedding1 ?? 0}u;
const W_EMBED4: u32 = ${offsets.templatePairEmbedding4 ?? 0}u;
const W_EMBED5: u32 = ${offsets.templatePairEmbedding5 ?? 0}u;
const W_EMBED6: u32 = ${offsets.templatePairEmbedding6 ?? 0}u;
const W_EMBED7: u32 = ${offsets.templatePairEmbedding7 ?? 0}u;
const GEOMETRY_STRIDE: u32 = ${GEOMETRY_STRIDE}u;
const W_OUT_SCALE: u32 = ${offsets.outputLayerNormScale ?? 0}u;
const W_OUT_OFFSET: u32 = ${offsets.outputLayerNormOffset ?? 0}u;
const W_OUT: u32 = ${offsets.outputLinear ?? 0}u;
`;

  const varianceCode = (count, read) => variance === "fast"
    ? `let variance = squares / f32(${count}) - mean * mean;`
    : `var variance = 0.0;
  for (var c = 0u; c < ${count}; c += 1u) {
    let d = ${read} - mean;
    variance += d * d;
  }
  variance /= f32(${count});`;

  // 🔴 A WORKGROUP OF EMBED_ROWS PAIR ROWS, NOT A THREAD A ROW. A thread a row
  // read its row of the query pair from global memory with a stride of
  // QUERY_CHANNELS floats between lanes - every lane its own cache line - and
  // recomputed each normalised value once per output channel: 14 ms a slot at
  // OpenDDE's 384 channels and 255 tokens. Staged, a row is read once and
  // coalesced, normalised once, and each lane owns output channels. The row's
  // statistics are the same sequential sums and every normalised value the same
  // expression, and each output sums over c in the same order, so the answer is
  // bit-identical.
  const EMBED_ROWS = TEMPLATE_EMBED_ROWS;
  const embedPrelude = `
const EMBED_ROWS: u32 = ${EMBED_ROWS}u;
var<workgroup> normalized_rows: array<f32, ${EMBED_ROWS * queryChannels}>;
fn stage_rows(first: u32, lane: u32) {
  for (var at = lane; at < EMBED_ROWS * QUERY_CHANNELS; at += 64u) {
    let row = first + at / QUERY_CHANNELS;
    normalized_rows[at] = select(0.0, pair[min(row, PAIRS - 1u) * QUERY_CHANNELS
                                          + at % QUERY_CHANNELS], row < PAIRS);
  }
  workgroupBarrier();
  var mean = 0.0;
  var inverse_std = 0.0;
  if (lane < EMBED_ROWS) {
    let base = lane * QUERY_CHANNELS;
    var total = 0.0;
    var squares = 0.0;
    for (var c = 0u; c < QUERY_CHANNELS; c += 1u) {
      let value = normalized_rows[base + c];
      total += value;
      squares += value * value;
    }
    mean = total / f32(QUERY_CHANNELS);
    ${varianceCode("QUERY_CHANNELS", "normalized_rows[base + c]")}
    inverse_std = inverseSqrt(variance + EPSILON);
    row_mean[lane] = mean;
    row_inverse_std[lane] = inverse_std;
  }
  workgroupBarrier();
  for (var at = lane; at < EMBED_ROWS * QUERY_CHANNELS; at += 64u) {
    let r = at / QUERY_CHANNELS;
    let c = at % QUERY_CHANNELS;
    normalized_rows[at] = (normalized_rows[at] - row_mean[r]) * row_inverse_std[r]
      * weights[W_QUERY_SCALE + c] + weights[W_QUERY_OFFSET + c];
  }
  workgroupBarrier();
}
var<workgroup> row_mean: array<f32, ${EMBED_ROWS}>;
var<workgroup> row_inverse_std: array<f32, ${EMBED_ROWS}>;
`;
  // Each lane's projection of the staged rows onto output channel e, one
  // accumulator a row, summed over c in ascending order.
  const projectRows = (weight) => `
    var values: array<f32, ${EMBED_ROWS}>;
    for (var r = 0u; r < EMBED_ROWS; r += 1u) { values[r] = 0.0; }
    for (var c = 0u; c < QUERY_CHANNELS; c += 1u) {
      let w = weights[${weight} + c * CHANNELS + e];
      for (var r = 0u; r < EMBED_ROWS; r += 1u) {
        values[r] += normalized_rows[r * QUERY_CHANNELS + c] * w;
      }
    }`;

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
${embedPrelude}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_index) lane: u32) {
  let first = (group.x + group.y * GRID_WIDTH) * EMBED_ROWS;
  if (first >= PAIRS) { return; }
  stage_rows(first, lane);
  for (var e = lane; e < CHANNELS; e += 64u) {
${projectRows("W_EMBED8")}
    for (var r = 0u; r < EMBED_ROWS; r += 1u) {
      let row = first + r;
      if (row >= PAIRS) { break; }
      let i = row / TOKENS;
      let j = row % TOKENS;
      var value = values[r];
      // 🔴 FEATURE 2 VARIES ALONG j AND FEATURE 3 ALONG i. AF3 writes them as
      // aatype[None, :, :] and aatype[:, None, :]; swapping them transposes a
      // term that nothing downstream complains about.
      let code_row = aatype[j];
      let code_column = aatype[i];
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
  }
}`;

  // 🔴 THESE WERE ONE SHADER AND COULD NOT STAY ONE. It fused the LayerNorm,
  // the slot-count scaling, the relu and the projection, which is correct only
  // when every slot produces the SAME embedding - true while the only path was
  // four empty slots and false the moment one carries a template. The
  // LayerNorm and the summation are per slot; the scale, the relu and the
  // projection happen once, on the sum.
  // 🔴 AND boltz2 WRAPS THE WHOLE STACK IN A RESIDUAL. `before` is the
  // activation as it entered the two pairformer blocks, added back here rather
  // than in a pass of its own: this shader already reads `act` row by row, so
  // the add is free and needs no second dispatch. Under every other dialect the
  // binding is absent and not a line of this changes.
  const outerResidual = shape.templateStackOuterResidual === true;
  const accumulate = `${common}
@group(0) @binding(0) var<storage, read> act: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> summed: array<f32>;
// 🔴 HOW MANY IDENTICAL SLOTS THIS PASS STANDS FOR. Empty slots all produce
// the SAME embedding, so running four of them is four times the work for an
// answer that is one of them times four. See the note in run().
@group(0) @binding(3) var<storage, read> repeat: array<f32>;
${outerResidual ? "@group(0) @binding(4) var<storage, read> before: array<f32>;" : ""}

fn value_at(index: u32) -> f32 {
${outerResidual ? "  return act[index] + before[index];" : "  return act[index];"}
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x + id.y * GRID_WIDTH * 64u;
  if (row >= PAIRS) { return; }
  let base = row * CHANNELS;

  var total = 0.0;
  var squares = 0.0;
  for (var c = 0u; c < CHANNELS; c += 1u) {
    let value = value_at(base + c);
    total += value;
    squares += value * value;
  }
  let mean = total / f32(CHANNELS);
  ${varianceCode("CHANNELS", "value_at(base + c)")}
  let inverse_std = inverseSqrt(variance + EPSILON);

  for (var c = 0u; c < CHANNELS; c += 1u) {
    summed[base + c] += ((value_at(base + c) - mean) * inverse_std * weights[W_OUT_SCALE + c]
      + weights[W_OUT_OFFSET + c]) * repeat[0];
  }
}`;

  // The slot-count scaling, a relu, and the projection back up.
  const output = `${common}
@group(0) @binding(0) var<storage, read> summed: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// 🔴 FOUR PAIR ROWS A WORKGROUP AND A LANE AN OUTPUT, NOT A THREAD A ROW.
// A thread a row walked every output channel over its own row - 1017
// workgroups at 255 tokens, strided reads, and 62.8 ms a trunk pass at
// IntelliFold-2's 512 channels, ~34 GFLOP/s. Staged, the scaled rows are read
// once and each lane sums its channel over c in the same order from zero, so
// the answer is bit-identical. Same shape as template.embed above.
const OUT_ROWS: u32 = ${TEMPLATE_EMBED_ROWS}u;
var<workgroup> scaled_rows: array<f32, ${TEMPLATE_EMBED_ROWS * CHANNELS}>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_index) lane: u32) {
  let first = (group.x + group.y * GRID_WIDTH) * OUT_ROWS;
  if (first >= PAIRS) { return; }
  for (var at = lane; at < OUT_ROWS * CHANNELS; at += 64u) {
    let row = first + at / CHANNELS;
    // ...relu BEFORE the projection, so the module can only add along a
    // non-negative combination of output_linear's directions.
    scaled_rows[at] = select(0.0,
      max(summed[min(row, PAIRS - 1u) * CHANNELS + at % CHANNELS] * TEMPLATE_SCALE, 0.0),
      row < PAIRS);
  }
  workgroupBarrier();
  for (var f = lane; f < QUERY_CHANNELS; f += 64u) {
    var values: array<f32, ${TEMPLATE_EMBED_ROWS}>;
    for (var r = 0u; r < OUT_ROWS; r += 1u) { values[r] = 0.0; }
    for (var c = 0u; c < CHANNELS; c += 1u) {
      let w = weights[W_OUT + c * QUERY_CHANNELS + f];
      for (var r = 0u; r < OUT_ROWS; r += 1u) {
        values[r] += scaled_rows[r * CHANNELS + c] * w;
      }
    }
    for (var r = 0u; r < OUT_ROWS; r += 1u) {
      let row = first + r;
      if (row < PAIRS) { output[row * QUERY_CHANNELS + f] = values[r]; }
    }
  }
}`;

  // 🔴 THE FUSED INPUT STAGE. `v = z_proj(z_norm(z)) + a_proj(a)`, where `a` is
  // the 108 concatenated feature columns per pair. Everything after it - the two
  // pairformer blocks, the accumulate and the output projection - is the same
  // code, because the fused module differs only in how the stack's input is
  // built. See docs/AF3.md and src/af3/trunk/template-reference.js, which this
  // mirrors and which is held to af3-any-model at 1.5e-7 on both a templated
  // and a templateless dump.
  //
  // 🔴 THE FEATURES ARE PER PAIR AND THE HOST BUILDS THEM. For a de novo fold
  // every slot is empty and the 108 columns are CONSTANT - measured, not
  // assumed: zero everywhere except restype_i and restype_j one-hot at column
  // 31, which is GAP. The buffer is therefore mostly zeros and could be one
  // vector; it is per pair so the shape does not have to change when the
  // featuriser lands.
  const fusedEmbed = `${common}
const FEATURE_WIDTH: u32 = ${featureWidth}u;
const W_Z_PROJECTION: u32 = ${offsets.zProjection ?? 0}u;
const W_A_PROJECTION: u32 = ${offsets.aProjection ?? 0}u;

@group(0) @binding(0) var<storage, read> pair: array<f32>;
@group(0) @binding(1) var<storage, read> features: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> act: array<f32>;
${embedPrelude}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_index) lane: u32) {
  let first = (group.x + group.y * GRID_WIDTH) * EMBED_ROWS;
  if (first >= PAIRS) { return; }
  stage_rows(first, lane);
  for (var e = lane; e < CHANNELS; e += 64u) {
${projectRows("W_Z_PROJECTION")}
    for (var r = 0u; r < EMBED_ROWS; r += 1u) {
      let row = first + r;
      if (row >= PAIRS) { break; }
      let feature_base = row * FEATURE_WIDTH;
      var value = values[r];
      for (var c = 0u; c < FEATURE_WIDTH; c += 1u) {
        let f = features[feature_base + c];
        if (f != 0.0) { value += f * weights[W_A_PROJECTION + c * CHANNELS + e]; }
      }
      act[row * CHANNELS + e] = value;
    }
  }
}`;

  return { embed: fused ? fusedEmbed : embed, accumulate, output };
}

export class Af3TemplateEmbedderGpu {
  constructor(device, options = {}) {
    this.device = device;
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
    // 🔴 KEPT, BECAUSE THE TRUNK PINS THROUGH THE CONSTRUCTOR. Af3TrunkGpu
    // hands `this.options` to all three stacks' constructors and its run-time
    // options to two of their `run`s, so a stack reading only `run`'s could not
    // see `pairMatrixKernels` set the way the pairformer sees it.
    this.options = options;
    // The same default and the same escape the pairformer takes.
    this.residentWeights = (options.residentWeights ?? true) && residencyAllowed(device);
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
    // 🔴 THE FUSED EMBEDDER IS A DIFFERENT INPUT STAGE, AND THE KEY SAYS SO.
    // Its dimensions are otherwise identical to AF3's, so a cache indexed on
    // those alone would hand one model the other's kernel.
    const fused = weights.fused === true;
    const featureWidth = fused ? weights.featureWidth : 0;
    const outerResidual = dialect.templateStackOuterResidual === true;
    // 🔴 THE STACK'S WIDTH, OFF THE BUNDLE. It was a module constant of 64,
    // which is right for five checkpoints and wrong for IntelliFold-2's 256 -
    // and the failure was not a wrong answer but `splitInterleaved` refusing
    // the triangle weights, because the pack's width and the shader's are the
    // same number in two places. See stackChannels in template-reference.js.
    const CHANNELS = weights.channels;
    if (!Number.isInteger(CHANNELS)) {
      throw new Error("template weights carry no `channels`: the stack's width "
        + "is read from output_layer_norm/scale (or v_norm/scale), not assumed");
    }
    const sources = createTemplateShaders(
      { tokens, queryChannels, templates, channels: CHANNELS, fused, featureWidth,
        templateStackOuterResidual: outerResidual },
      packed.offsets, epsilon, variance);
    // ...and it is in the KEY, because two bundles differing only in the stack
    // width would otherwise share every one of these pipelines.
    const base = `af3-template:${tokens}:${queryChannels}:${templates}:${epsilon}`
      + `:${variance}:${dialect.swapTransposedBias}:c${CHANNELS}`
      + `:${fused ? `fused${featureWidth}` : ""}${outerResidual ? ":or" : ""}`;
    // 🔴 THE TRUNK'S PAIR, UPDATED IN PLACE, WHEN IT HANDS ONE OVER - the
    // pairformer's `pairBuffer` convention. The term is added on the device
    // (the same f32 add the trunk used to do on the host) and nothing is read
    // back, so the stage needs no drain and costs no bus crossing.
    const pairBuffer = options.pairBuffer;
    // 🔴 `compileOnly` BUILDS THE PIPELINES AND RETURNS, for Af3TrunkGpu.warm,
    // as the trunk runs it: with a pair buffer, so the add is compiled too.
    const compileOnly = options.compileOnly === true;
    if (!compileOnly && pairBuffer !== undefined && options.validation === undefined) {
      throw new Error("pairBuffer needs options.validation: nothing here awaits the scope");
    }
    // Compiled together, not one after another: the browser builds them in
    // parallel, and a serial loop put every one of them on a cold fold's path.
    // ...and the three groups below are asked for together too.
    const compiling = Object.fromEntries(Object.entries(sources).map(
      ([name, source]) => [name, this.pipelines.get(`${base}:${name}`, source)]));
    if (pairBuffer !== undefined || compileOnly) {
      compiling.addPair = this.pipelines.get(
        `af3-template:add-pair:${pairs * queryChannels}`, createAddShader(pairs * queryChannels));
    }
    // The template stack: the shared pair track at the stack's own width.
    // ...one variable for the shader and the packing; see the note in
    // msa-stack-webgpu.js for what their disagreeing costs.
    const pairWeightPrecision = options.pairWeightPrecision ?? "f32";
    const trackCompiling = compilePairTrack(this.pipelines, {
      triangleProjectTile: shapedKnob(deviceTuning(this.device).trianglePairProjectTile),
      triangleProjectOutColumns: deviceTuning(this.device).triangleProjectOutColumns,
      scratchStorage: UNPACKED_PAIR_SCRATCH,
      // ...derived, not 2: boltz2's template transition is a factor of 4. See
      // templateTransitionFactor in template-reference.js.
      n: tokens, channels: CHANNELS,
      transitionFactor: templateTransitionFactor(weights.blocks[0].pairTransition,
                                                 CHANNELS),
      weightPrecision: pairWeightPrecision,
      sample: weights.blocks[0], epsilon, variance, dialect, base: `${base}:track`,
      // ...the same pair track, so the same kernel choice. Four of an AF3
      // trunk's 108 `grid.project` passes are this stack's.
      // 🔴 THE MATRIX PAIR KERNEL IS A PRECISION AXIS; see the note in
      // pairformer-block-webgpu.js. This stack has only the one of the four.
      gridProjectMatrix:
        (options.pairMatrixKernels ?? this.options?.pairMatrixKernels) !== false
        && gridProjectMatrixConfig(this.device),
      maxComputeWorkgroupStorageSize: this.device.limits.maxComputeWorkgroupStorageSize,
    });
    const [compiled, trackPipelines] = await Promise.all([settleAll(compiling), trackCompiling]);
    if (compileOnly) return undefined;
    const gridProjectMatrix = trackPipelines.gridProjectMatrix === undefined ? undefined
      : allocateGridProjectMatrix(this.allocator, {
        ...trackPipelines.gridProjectMatrix, label: "af3-template.grid-project",
      }, (a) => a);

    const gridHeads = weights.blocks[0].pairAttention1.heads;
    const storage = GPUBufferUsage.STORAGE;
    const allocations = [];
    const keep = (allocation) => { allocations.push(allocation); return allocation; };
    try {
      const pair = pairBuffer !== undefined ? { buffer: pairBuffer }
        : keep(this.allocator.upload("af3-template.pair", input.pair, storage));
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
      // 🔴 RoseTTAFold3 AVERAGES THE FEATURES AND RUNS ONE PASS, WHERE EVERY
      // OTHER FAMILY RUNS A PASS PER SLOT AND AVERAGES THE OUTPUTS. Its
      // reference is explicit - `a_tij = einsum('t,tijc->ijc', present, feats)
      // / clip(present.sum(), 1)` and then a single forward with "no
      // per-template loop and no template gating". Diluting one real template
      // over four SLOTS instead of dividing by the one PRESENT template makes
      // its whole term a quarter of what the checkpoint expects, and a quarter
      // of a distance-distribution conditioning is a template that does almost
      // nothing.
      //
      // 🔴 AND NO MODULE CHECK COULD SEE IT: `check-fused-template-features.js`
      // passes `templates: 1`, where a slot mean and a present mean are the
      // same number, so rf3 read 0.061 there (inside its int5 bundle's floor -
      // boltz2's int5 reads 0.167 where its f32 reads 8.25e-7) while a FOLD
      // with a perfect self-template moved 17.949 A to 17.771. AlphaFold 3 and
      // IntelliFold-2 take the identical input to 0.281 and 0.254.
      // The empty case is untouched, which is why rf3's trunk seam was exact.
      if (dialect.templateFeatureMeanOnePass === true) {
        if (!fused) {
          throw new Error("templateFeatureMeanOnePass is a FUSED convention:"
            + " the nine-projection path has no single feature tensor to average");
        }
        // Present is "this slot has an atom", the reference's own test - not
        // "a slot object was passed", because a covered-nothing slot is absent
        // to the reference and present to a null check.
        const present = [];
        for (let slot = 0; slot < templates; slot += 1) {
          const here = slots[slot];
          if (here === undefined || here === null) continue;
          if (here.atomMask.some === undefined
            ? Array.prototype.some.call(here.atomMask, (v) => v > 0)
            : here.atomMask.some((v) => v > 0)) present.push(here);
        }
        let features;
        if (present.length === 0) {
          // Zero features, one pass - byte-identical to the empty path below,
          // which is what kept rf3's no-template trunk seam exact.
          features = fusedTemplateFeatures(undefined, tokens, featureWidth, dialect,
                                           undefined, false);
        } else {
          features = fusedTemplateFeatures(present[0], tokens, featureWidth, dialect,
                                           chainMaskFor(present[0]), false);
          for (let extra = 1; extra < present.length; extra += 1) {
            const more = fusedTemplateFeatures(present[extra], tokens, featureWidth,
                                               dialect, chainMaskFor(present[extra]), false);
            for (let index = 0; index < features.length; index += 1) features[index] += more[index];
          }
          if (present.length > 1) {
            for (let index = 0; index < features.length; index += 1) {
              features[index] /= present.length;
            }
          }
        }
        // `repeat` cancels TEMPLATE_SCALE, which is 1/slots: this module
        // contributes its one forward at full strength, whatever the slot count.
        passes.push({ template: undefined, repeat: templates, features });
      } else {
      let emptySlots = 0;
      for (let slot = 0; slot < templates; slot += 1) {
        if (slots[slot] === undefined || slots[slot] === null) emptySlots += 1;
        else passes.push({ template: slots[slot], repeat: 1 });
      }
      // 🔴 AND THE EMPTY SLOTS ARE NOT ALL THE SAME SLOT UNDER EVERY DIALECT,
      // WHICH IS WHAT BROKE THE COLLAPSE ABOVE. OpenDDE and protenix2 take
      // protenix's featuriser, which fills its ONE empty template with the GAP
      // restype and zero-pads the rest - so `template_aatype` on a query with
      // no template is 21 across slot 0 and 0 across slots 1..3, and the four
      // empty slots are two distinct embeddings, not one. Folding them into a
      // single pass with repeat 4 put row 0 (ALA) where row 21 belongs and was
      // the whole of OpenDDE's `z_after_template` 2.07e-2 - a trunk defect that
      // needed no template to appear. Where there is no gap the split does not
      // happen and this is the one pass it always was.
      const gap = dialect.emptyTemplateAatype ?? null;
      if (emptySlots > 0 && gap !== null) {
        passes.push({ template: undefined, repeat: 1, emptyAatype: gap });
        if (emptySlots > 1) {
          passes.push({ template: undefined, repeat: emptySlots - 1, emptyAatype: 0 });
        }
      } else if (emptySlots > 0) {
        passes.push({ template: undefined, repeat: emptySlots, emptyAatype: 0 });
      }
      }

      const slotBuffers = [];
      for (const { template, repeat, emptyAatype, features: given } of passes) {
        const slot = slotBuffers.length;
        // An empty slot contributes a ROW of each aatype weight rather than
        // nothing - that is half of why an empty slot is not a no-op - and
        // WHICH row is the dialect's. See the note on `gap` above.
        const aatypeData = new Int32Array(tokens).fill(emptyAatype ?? 0);
        if (template !== undefined && template !== null) {
          for (let t = 0; t < tokens; t += 1) aatypeData[t] = template.aatype[t];
        }
        // 🔴 AN EMPTY SLOT'S WEIGHT IS ZERO UNDER TEMPLATE_VISIBILITY_BY_COVERAGE.
        // `repeat` already scales a slot's contribution into the sum, so the
        // convention costs no branch and no pass: boltz2 masks by what the
        // template covers, and with none supplied its whole term is zero -
        // af3-any-model returns rms 0.0000 there.
        const covered = !(dialect.templateVisibilityByCoverage === true
          && (template === undefined || template === null));
        slotBuffers.push({
          repeat: keep(this.allocator.upload(
            `af3-template.repeat.${slot}`,
            Float32Array.from([covered ? repeat : 0]), storage)),
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
          // 🔴 THE FUSED EMBEDDER'S 108 COLUMNS. For an EMPTY slot they are
          // measured rather than assumed - zero everywhere except restype_i and
          // restype_j one-hot at column 31, which is GAP; see
          // oracle-dumps/af3-oracle-template-protenix2-empty.json and the note
          // in template-reference.js. A slot WITH a template needs the
          // featuriser, which is not written: the frame convention, the bin
          // edges and the multichain masking are specified in docs/AF3.md and
          // gated by nothing, so this refuses rather than guessing.
          features: fused ? keep(this.allocator.upload(
            `af3-template.features.${slot}`,
            given ?? fusedTemplateFeatures(template, tokens, featureWidth, dialect,
                                  template === undefined || template === null
                                    ? undefined : chainMaskFor(template),
                                  (emptyAatype ?? 0) !== 0),
            storage)) : undefined,
        });
      }

      // ...COPY_SRC only where the residual needs to snapshot it; see below.
      const act = keep(this.allocator.allocate("af3-template.act", pairs * CHANNELS * 4,
        outerResidual ? storage | GPUBufferUsage.COPY_SRC : storage));
      // The running sum over slots, which the projection reads once at the end.
      const summed = keep(this.allocator.allocate(
        "af3-template.summed", pairs * CHANNELS * 4, storage | GPUBufferUsage.COPY_DST));
      // 🔴 FIVE PAIR-SIZED SCRATCH BUFFERS, AND THIS STACK KEEPS THEM WHOLE.
      // See UNPACKED_PAIR_SCRATCH: packing them costs this embedder 150x its
      // agreement with AF3 and saves 26 MiB on a stage that is not the trunk's
      // peak. The allocation and the shaders read the same array, because a
      // buffer that disagrees with a shader about its element is not something
      // WebGPU can catch.
      // 🔴 SIZED BY THE ATTENTION'S WIDTH, NOT BY THE CHANNEL COUNT. The grid
      // projection writes `heads * dimension` per pair per role, and in every
      // other stack in every model that equals the channel width - AF3's trunk
      // is 4 x 32 = 128 channels, OpenDDE's 12 x 32 = 384, protenix2's template
      // 2 x 32 = 64, AF3's template 4 x 16 = 64. **boltz2's TEMPLATE stack is
      // 4 x 32 = 128 out of 64 channels**, the only place anywhere the two
      // differ, so scratch sized by CHANNELS held half of what the projection
      // wrote. Nothing raises: the buffer is simply too short and the kernel's
      // own bounds check drops the tail.
      //
      // Measured: the whole fused path 0.748 from its own CPU reference where
      // protenix2's was 3.9e-5, with the EMBED exact at 2.2e-7 either side - so
      // it was always the two pairformer blocks - and boltz2's self-template
      // fold of 5CAJ at 3.859 A where AF3 and protenix2 reach 0.2.
      const attentionWidth = Math.max(CHANNELS,
        gridHeads * (weights.blocks[0]?.pairAttention1?.dimension ?? 0));
      const scratch = [];
      for (let index = 0; index < trackPipelines.pairScratchCount; index += 1) {
        scratch.push(keep(this.allocator.allocate(
          `af3-template.scratch${index}`,
          trackPipelines.pairScratchBytes(index,
  storageBytes(pairs * attentionWidth, UNPACKED_PAIR_SCRATCH[index])), storage)));
      }
      const biasBuffer = keep(this.allocator.allocate(
        "af3-template.bias", gridHeads * pairs * 4, storage));
      // 🔴 `output` AND `readback` ARE ALLOCATED AFTER THE BLOCK LOOP HAS RUN,
      // WHICH IS WORTH A THIRD OF AN AF3 TRUNK'S PEAK. This stage was the
      // largest holder of device memory in the whole trunk - 508 MiB of a 575
      // MiB peak at 400 tokens, against the pairformer's own 66 - because it
      // allocated everything up front and released it in one `finally`. Its
      // five pair-sized scratch buffers are DEAD once the last slot has
      // accumulated, and these two are not needed until after that; taking
      // them in that order, with a submit between so the encoded passes are
      // done with the scratch, means the two sets never coexist.
      //
      // 🔴 THE SUBMIT IS WHAT MAKES THE RELEASE LEGAL. A released allocation is
      // destroyed (this allocator does not pool), and destroying a buffer an
      // encoded-but-unsubmitted pass still references is a use-after-free that
      // WebGPU reports as a validation error at submit time and not before.

      const blockAllocations = [];
      const upload = (label, data) => {
        const allocation = this.allocator.upload(label, data, storage);
        blockAllocations.push(allocation);
        return allocation;
      };

      if (pairBuffer !== undefined) options.validation.begin();
      else this.device.pushErrorScope("validation");
      const encoder = this.device.createCommandEncoder({ label: "af3-template" });
      const run = (label, pipeline, buffers, x, y = 1, z = 1) => {
        const pass = encoder.beginComputePass({ label });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          // byteOffset and byteSize honoured, as the other stacks do: the grid
          // attention's chunks are slices of the scratch.
          entries: buffers.map((allocation, binding) => ({
            binding,
            resource: allocation.byteOffset === undefined
              ? { buffer: allocation.buffer }
              : { buffer: allocation.buffer,
                  offset: allocation.byteOffset, size: allocation.byteSize },
          })),
        }));
        pass.dispatchWorkgroups(x, y, z);
        pass.end();
      };
      const spread = (groups) => [Math.min(groups, GRID_WIDTH), Math.ceil(groups / GRID_WIDTH)];
      const linear = spread(Math.ceil(pairs / 64));
      // A workgroup of EMBED_ROWS pair rows; see createTemplateShaders.
      const perEmbed = spread(Math.ceil(pairs / TEMPLATE_EMBED_ROWS));

      // 🔴 THE BLOCK WEIGHTS ARE PACKED AND UPLOADED ONCE, OUTSIDE THE SLOT
      // LOOP. Every slot runs the SAME two pairformer blocks, so packing them
      // per slot would repack 1.4 MiB four times for four identical buffers -
      // and the release below would then have to know which upload belonged to
      // which pass.
      // 🔴 PACKED ON DEMAND AND UPLOADED ONCE, EVER, the way the pairformer and
      // the MSA stack do it. This stack's blocks are re-encoded once per
      // template SLOT as well as once per pass, so the same weights were being
      // packed and written four times over on a four-template job.
      const resident = this.residentWeights
        ? (label, key, pack, variant) => ({
          buffer: residentWeightBuffer(this.device, key, label, pack, variant),
        })
        : (label, key, pack) => upload(label, pack());
      const blockWeights = [];
      for (const block of weights.blocks) {
        // The four the device can pack itself; see pair-track-device-weights.js.
        // eslint-disable-next-line no-await-in-loop
        const onDevice = await residentPairTrackOnDevice(this.device, block, {
          channels: CHANNELS, pairWeightPrecision, resident: this.residentWeights,
        });
        // Lazily, and not held: residentWeightBuffer calls pack() only on a
        // miss, so after the first encode nothing reads these arrays again.
        let packed;
        const packedFor = () => (packed ??= packPairTrackWeights(
          block, CHANNELS, pairWeightPrecision, true, "blocked", onDevice.want));
        // 🔴 THE LABEL DROPS THE BLOCK INDEX AND THE CACHE KEY IS THE BLOCK
        // OBJECT, which is the right way round: a device-memory breakdown reads
        // one row per tensor instead of one per block, and two blocks cannot
        // share a buffer because they are not the same object.
        const w = pairWeightPrecision;
        blockWeights.push({
          outgoing: onDevice.buffers.outgoing
            ?? resident("w.tri.out", block, () => packedFor().outgoing, w),
          incoming: onDevice.buffers.incoming
            ?? resident("w.tri.in", block, () => packedFor().incoming, w),
          grid1: onDevice.buffers.grid1 ?? resident("w.grid1", block, () => packedFor().grid1),
          grid2: onDevice.buffers.grid2 ?? resident("w.grid2", block, () => packedFor().grid2),
          transition: resident("w.transition", block, () => packedFor().transition, w),
        });
      }

      // 🔴 THE STACK'S INPUT, KEPT FOR THE RESIDUAL. A copy rather than a
      // second `act`: the pair track writes `act` in place across both blocks,
      // so the only way to add the input back afterwards is to have kept it.
      const beforeStack = outerResidual
        ? keep(this.allocator.allocate("af3-template.before", pairs * CHANNELS * 4,
                                       storage | GPUBufferUsage.COPY_DST))
        : undefined;
      for (let slot = 0; slot < slotBuffers.length; slot += 1) {
        run(`template.embed.${slot}`, compiled.embed,
            fused
              ? [pair, slotBuffers[slot].features, weightBuffer, act]
              : [pair, slotBuffers[slot].aatype, weightBuffer, act,
                 slotBuffers[slot].geometry], perEmbed[0], perEmbed[1]);
        if (outerResidual) {
          encoder.copyBufferToBuffer(act.buffer, 0, beforeStack.buffer, 0,
                                     pairs * CHANNELS * 4);
        }
        // 🔴 A STOP POINT, so the EMBED can be compared on its own. boltz2's
        // whole fused path is 0.748 from its own CPU reference where
        // protenix2's is 3.9e-5, and `act` here - `z_proj(z_norm(z)) +
        // a_proj(a)`, before the two pairformer blocks touch it - is the one
        // seam that separates a wrong projection from a wrong pair track.
        for (let index = 0; index < blockWeights.length; index += 1) {
          if (options?.stopAfterEmbed === true) break;
          encodePairTrack({
            run, pipelines: trackPipelines, n: tokens, channels: CHANNELS, gridHeads,
            pair: act, pairMask, scratch, biasBuffer, weights: blockWeights[index],
            gridProjectMatrix,
          });
        }
        run(`template.accumulate.${slot}`, compiled.accumulate,
            outerResidual
              ? [act, weightBuffer, summed, slotBuffers[slot].repeat, beforeStack]
              : [act, weightBuffer, summed, slotBuffers[slot].repeat],
            linear[0], linear[1]);
      }
      const start = performance.now();
      this.device.queue.submit([encoder.finish()]);
      // 🔴 NO DRAIN ON THE DEVICE PATH. Releasing after a submit is safe on
      // queue ordering - the pairformer has released its per-block weights
      // that way for as long as it has pipelined - and the drain existed only
      // to hold the device still for a readback this path never takes.
      let error = null;
      if (pairBuffer !== undefined) {
        options.validation.end("template");
      } else {
        error = await this.device.popErrorScope();
        await this.device.queue.onSubmittedWorkDone();
      }
      for (let index = blockAllocations.length - 1; index >= 0; index -= 1) {
        blockAllocations[index].release();
      }
      // ...the five pair-sized scratch tensors, now that the device is done
      // with them and before the two output-sized ones exist. See above.
      for (let index = scratch.length - 1; index >= 0; index -= 1) {
        scratch[index].release();
        allocations.splice(allocations.indexOf(scratch[index]), 1);
      }
      biasBuffer.release();
      allocations.splice(allocations.indexOf(biasBuffer), 1);
      if (error !== null) throw new Error(`WebGPU validation failed: ${error.message}`);

      const output = keep(this.allocator.allocate(
        "af3-template.output", pairs * queryChannels * 4, storage | GPUBufferUsage.COPY_SRC));
      const readback = pairBuffer !== undefined ? undefined : keep(this.allocator.allocate(
        "af3-template.readback", pairs * queryChannels * 4,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));
      if (pairBuffer !== undefined) options.validation.begin();
      else this.device.pushErrorScope("validation");
      const finish = this.device.createCommandEncoder({ label: "af3-template-output" });
      const pass = finish.beginComputePass({ label: "template.output" });
      pass.setPipeline(compiled.output);
      pass.setBindGroup(0, this.device.createBindGroup({
        layout: compiled.output.getBindGroupLayout(0),
        entries: [summed, weightBuffer, output].map((allocation, binding) => ({
          binding, resource: { buffer: allocation.buffer },
        })),
      }));
      // Four pair rows a workgroup; see the shader.
      pass.dispatchWorkgroups(perEmbed[0], perEmbed[1]);
      pass.end();
      if (pairBuffer !== undefined) {
        // z += template(z), the add the trunk did on the host.
        const add = finish.beginComputePass({ label: "template.add-pair" });
        add.setPipeline(compiled.addPair);
        add.setBindGroup(0, this.device.createBindGroup({
          layout: compiled.addPair.getBindGroupLayout(0),
          entries: [pair, output].map((allocation, binding) => ({
            binding, resource: { buffer: allocation.buffer },
          })),
        }));
        // ELEMENTWISE over pairs x channels, not `linear`, which is per PAIR.
        const addGrid = spread(Math.ceil(pairs * queryChannels / 64));
        add.dispatchWorkgroups(addGrid[0], addGrid[1]);
        add.end();
        this.device.queue.submit([finish.finish()]);
        options.validation.end("template output");
        return { elapsedMilliseconds: performance.now() - start, memory: this.allocator.snapshot() };
      }
      finish.copyBufferToBuffer(output.buffer, 0, readback.buffer, 0, pairs * queryChannels * 4);
      this.device.queue.submit([finish.finish()]);
      const outputError = await this.device.popErrorScope();
      if (outputError !== null) {
        throw new Error(`WebGPU validation failed: ${outputError.message}`);
      }
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

/**
 * The old name, kept because three tools import it.
 *
 * 🔴 IT IS NO LONGER ONLY THE EMPTY CASE. `fusedTemplateFeatures` builds a
 * SUPPLIED template's 108 columns now - see the note on it - and the name
 * `emptyFusedFeatures` described the limitation rather than the job.
 */
export const emptyFusedFeatures = fusedTemplateFeatures;
