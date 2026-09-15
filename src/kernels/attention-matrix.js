/**
 * Flash attention on the subgroup matrix units.
 *
 * 🔴 THE UNITS DO THE QUERY-KEY REDUCTION IN HARDWARE, WHICH IS THE WHOLE POINT.
 * The register-resident kernel this competes with gives one invocation one query
 * and has it read every key itself, so a staged key vector is read once per
 * lane; docs/A100.md prices those workgroup reads at 8.7 ms of 20.8 and calls
 * them the kernel's cost. A matrix multiply contracts sixteen queries against
 * sixteen keys in one instruction, so a staged key tile is read once for sixteen
 * queries instead of once each, and the dot product's reduction never becomes
 * lane traffic.
 *
 * 🔴 AND THIS IS NOT THE SHAPE A GEMM BENCHMARK PREDICTS. `head_dim` is 32, and
 * tools/gpu/bench-evoformer-linear.js --shape=headdim measures the staged matrix
 * kernel at 6.5 TFLOP/s against the vector kernel's 9.3 at exactly that K - from
 * which docs/AF2.md once concluded that the units could not help here. A plain
 * GEMM at K = 32 amortises its staged panel over two k steps and nothing else.
 * A flash attention stages the QUERY tile once and reuses it across every key in
 * the sequence, and each key tile across sixteen queries: the reuse that pays
 * for the staging is in the loop, not in K. Measure this in the stack.
 *
 * 🔴 THE ONLINE SOFTMAX CANNOT LIVE IN AN ACCUMULATOR. Its correction is per
 * QUERY ROW and the extension's only scalar multiply takes one uniform value, so
 * `P V` accumulates into a freshly zeroed result each key tile, is stored to
 * workgroup memory, and the running output is rescaled there in plain f32.
 *
 * 🔴 WHERE THE TIME GOES, BY ABLATION - each of these computes the WRONG
 * answer and exists only to price one term, measured as an 825-residue block
 * whose four attentions are 69.0 ms of 263.3:
 *
 *     the per-key pair bias read, collapsed to one address    -13.0 ms
 *     exp(), replaced by its argument                          -0.3
 *     the running output's rescale multiply, removed           -0.1
 *
 * So the softmax arithmetic this kernel is shaped around is FREE, and the one
 * identifiable cost left is a global read the register kernel pays too. The
 * other 56 ms is diffuse - staging, the multiplies and workgroup traffic - and
 * matches a latency-bound kernel at about 12% occupancy rather than any one
 * term. The bias index is (head * queries + q) * queries + k, so lanes of
 * adjacent query rows are `queries` floats apart: 32 sectors requested per warp
 * instruction where four would do. Half of the 13 ms is the traffic itself
 * (about 29 GB a block, L2-resident at 10.9-21.8 MB a tensor) and half is that
 * pattern. Coalescing it means giving a whole subgroup one query ROW at a time
 * - 32 lanes over 32 consecutive keys - which turns the two shuffles into
 * sixteen subgroup reductions and needs the rescale and the running sum in a
 * 512-byte workgroup array, because the accumulate would then index them
 * dynamically. Worth about 6 ms of 264. Nobody has built it.
 *
 * 🔴 AND NOTHING IS LOADED TRANSPOSED. `subgroupMatrixLoad` takes a column-major
 * flag that this repository has never exercised - tools/gpu/check-subgroup-
 * matrix.js passes `false` in both of its arms - so the key tile is STAGED
 * transposed instead, which is one index in a loop that already runs. A flag
 * whose meaning is not pinned by a checker is not a thing to build a kernel on.
 */

import { deviceProfile } from "../runtime/device-profile.js";

// 🔴 THE GEOMETRY MOVED TO src/kernels/attention-geometry.js to break the
// af3 <-> evoformer cycle; see the note there. Imported AND re-exported,
// because this module uses every one of them and a bare `export ... from`
// puts a name in the exports without putting it in scope - which
// test/module-references.test.js caught the last time.
import {
  ATTENTION_MATRIX_SUBGROUP_SIZE, ATTENTION_MATRIX_UNIT, attentionMatrixGeometry,
  attentionMatrixStorageBytes, paddedHead, strides, supportsAttentionMatrix,
} from "./attention-geometry.js";
export {
  ATTENTION_MATRIX_SUBGROUP_SIZE, ATTENTION_MATRIX_UNIT, attentionMatrixGeometry,
  attentionMatrixStorageBytes, paddedHead, strides, supportsAttentionMatrix,
};
const UNIT = ATTENTION_MATRIX_UNIT;

const COMMON = `
struct Parameters {
  batch: u32, queries: u32, channels: u32, heads: u32,
  head_dim: u32, transpose: u32, has_pair_bias: u32,
  query_weight: u32, key_weight: u32, value_weight: u32,
  gating_weight: u32, gating_bias: u32, output_weight: u32,
  output_bias: u32, pair_weight: u32, pair_channels: u32,
};
`;

/**
 * @param {number} headDim channels a head, 4 to 32 and a multiple of four.
 * @param {{input?: "f32"|"f16", value?: "f32"|"f16", output?: "f32"|"f16"}} [storage]
 *   how the projected tensors are held; the same three the register kernel takes.
 */
export function createAttentionMatrixFlashShader(headDim, storage = {}, geometry,
                                                 options = {}) {
  if (!Number.isSafeInteger(headDim) || headDim % 4 !== 0 || headDim > 32 || headDim < 4) {
    throw new RangeError(`matrix attention takes a head of 4 to 32 channels; got ${headDim}`);
  }
  const inputStorage = storage.input ?? "f32";
  const valueStorage = storage.value ?? inputStorage;
  const outputStorage = storage.output ?? "f32";
  const packIn = inputStorage === "f16";
  const packValue = valueStorage === "f16";
  const packOut = outputStorage === "f16";
  const vec4In = packIn ? "vec2<u32>" : "vec4<f32>";
  const vec4Value = packValue ? "vec2<u32>" : "vec4<f32>";
  const vec4Out = packOut ? "vec2<u32>" : "vec4<f32>";
  const read4 = (array, index) => (packIn ? `load4(${array}[${index}])` : `${array}[${index}]`);
  const readValue = (index) => (packValue
    ? `${packIn ? "load4" : "load4v"}(value[${index}])` : `value[${index}]`);

  const g = attentionMatrixGeometry(geometry);
  const HEAD = paddedHead(headDim);
  const ROWS = g.rows;
  const KEYS = g.keyTile;
  const LANES = g.lanes;
  const s = strides(headDim, g);
  const vectors = headDim / 4;
  const headSteps = HEAD / UNIT;
  const keyTiles = KEYS / UNIT;
  const channelTiles = HEAD / UNIT;
  // Two lanes share a query row and take half the keys each. This holds for
  // every geometry - a subgroup is 32 lanes and owns 16 rows - so the softmax
  // reduction needs nothing said about it, but say it anyway.
  const perHalf = KEYS / 2;
  if (ROWS * 2 !== LANES) throw new RangeError("two lanes a row is what the reduction assumes");
  // 🔴 THE RUNNING OUTPUT LIVES IN REGISTERS, FOUR CHANNELS TO A SLOT. It is
  // ROWS x HEAD floats and there are LANES lanes, so a lane owns exactly
  // HEAD / 8 vec4s whatever the geometry - four at a head of 32, two at a
  // padded head of 16 - and the count is a constant the compiler can unroll.
  const accVectors = (ROWS * HEAD) / (4 * LANES);
  // 🔴 AND THE BARRIERS ARE NOT WORTH CHASING, WHICH WAS MEASURED AND NOT
  // ASSUMED. Three of the five are between a subgroup and its OWN rows -
  // subgroup sg stores its scores to rows sg*16..sg*16+15, reads exactly those
  // in the softmax, writes exactly those probabilities and reads exactly those
  // back - so a subgroup-scoped barrier would do. There is not one:
  // "unresolved call target 'subgroupBarrier'" on this Dawn. Omitting them
  // outright, which is a data race and shipped nowhere, prices the whole
  // restructure at 293.3 -> 289.6 ms of block at 6x16 and 300.0 -> 287.0 at
  // 4x16. One to four percent, for a race - so the scalar softmax below is
  // where the time is, and that is where the work went.
  const own = "workgroupBarrier();";
  const lines = (n, body) => Array.from({ length: n }, (_, i) => body(i)).join("\n");

  // 🔴 THE STAGING LOOP'S TRIP COUNT IS NOT UNIFORM, AND THAT HAS COST A
  // MEASUREMENT ONCE ALREADY. `KEYS * HD4` is 256 at a head of 32 and 128 at a
  // head of 16 - two whole passes and one, for every one of the ${LANES} lanes
  // - but an AF2 fold also compiles this kernel at a head of EIGHT, where it is
  // 64 and only the first half of the workgroup enters the loop at all. So:
  //   - a `workgroupBarrier()` inside this loop is in NON-UNIFORM control flow,
  //     which is undefined behaviour and not a slow path;
  //   - and unrolling it to a fixed two iterations without the guard writes
  //     `ki` up to 63 into arrays that hold KEYS = ${KEYS}.
  // docs/A100.md records "AF2's matrix flash attention has a race" from a
  // bisection that did the second of those. Both reproduce as a fold that
  // differs run to run, and neither is a race in the shipped kernel.
  const steps = Math.ceil((KEYS * vectors) / LANES);
  const stageGuard = (n, body) => `    {
      let i = local + ${n * LANES}u;
${KEYS * vectors <= n * LANES + LANES && KEYS * vectors % LANES === 0
    ? `      {
${body}
      }` : `      if (i < KEYS * HD4) {
${body}
      }`}
    }`;
  // Reading the tile into registers BEFORE the barrier puts its memory latency
  // underneath the tail of the previous tile's compute; the barrier still
  // stands between the read and the workgroup write it has to protect, and it
  // stands at the tile body's top level, where every lane reaches it.
  const prefetch = options.prefetch === true;
  const readTile = (checked) => lines(steps, (n) => stageGuard(n, `        let ki = i / HD4;
        let c4 = i % HD4;
        let k_index = k0 + ki;
        ${checked ? "if (k_index < p.queries) {" : "{"}
          let at = ((batch_index * p.queries + k_index) * p.heads + head) * HD4 + c4;
          kv[${n}u] = ${read4("key", "at")};
          vv[${n}u] = ${readValue("at")};
        }`));
  const writeTile = () => lines(steps, (n) => stageGuard(n, `        let ki = i / HD4;
        let c4 = i % HD4;
        // The key goes in transposed and the value does not.
        staged_kt[(c4 * 4u + 0u) * S_KEY + ki] = f16(kv[${n}u].x);
        staged_kt[(c4 * 4u + 1u) * S_KEY + ki] = f16(kv[${n}u].y);
        staged_kt[(c4 * 4u + 2u) * S_KEY + ki] = f16(kv[${n}u].z);
        staged_kt[(c4 * 4u + 3u) * S_KEY + ki] = f16(kv[${n}u].w);
        staged_v[ki * S_HEAD + c4 * 4u + 0u] = f16(vv[${n}u].x);
        staged_v[ki * S_HEAD + c4 * 4u + 1u] = f16(vv[${n}u].y);
        staged_v[ki * S_HEAD + c4 * 4u + 2u] = f16(vv[${n}u].z);
        staged_v[ki * S_HEAD + c4 * 4u + 3u] = f16(vv[${n}u].w);`));

  // One pass over KEYS keys. The checked form is the last, partial tile; every
  // other one knows its keys are real, which is worth three instructions a key.
  const tileBody = (checked) => `    for (var n = 0u; n < ${steps}u; n += 1u) {
      kv[n] = vec4<f32>(0.0); vv[n] = vec4<f32>(0.0);
    }
${prefetch ? "" : "    workgroupBarrier();"}
${readTile(checked)}
${prefetch ? "    workgroupBarrier();" : ""}
${writeTile()}
    // 🔴 THE MASK IS PER KEY AND WAS BEING READ PER QUERY ROW. Every one of the
    // ${ROWS} rows in this workgroup read the same global float for the same
    // key, and turned it into the same additive term. Stage it once: ${KEYS}
    // floats, ${KEYS * 4} bytes, against ${ROWS} redundant loads a key.
    for (var i = local; i < KEYS; i += ${LANES}u) {
      let k_index = k0 + i;
      ${checked
        ? "mask_add[i] = select(0.0, 1e9 * (mask[mask_index(batch_index, k_index)] - 1.0), k_index < p.queries);"
        : "mask_add[i] = 1e9 * (mask[mask_index(batch_index, k_index)] - 1.0);"}
    }
${HEAD === headDim ? "" : `    for (var i = local; i < ${HEAD - headDim}u * KEYS; i += ${LANES}u) {
      staged_kt[(${headDim}u + i / KEYS) * S_KEY + i % KEYS] = f16(0.0);
      staged_v[(i % KEYS) * S_HEAD + ${headDim}u + i / KEYS] = f16(0.0);
    }`}
    workgroupBarrier();

    // S = Q K^T, one subgroup's sixteen queries against every staged key.
${lines(keyTiles, (t) => `    {
      var acc = subgroup_matrix_result<f32, ${UNIT}, ${UNIT}>();
${lines(headSteps, (step) => `      {
        let l = subgroupMatrixLoad<subgroup_matrix_left<f16, ${UNIT}, ${UNIT}>>(
          &staged_q, sg * ${UNIT}u * S_HEAD + ${step * UNIT}u, false, S_HEAD);
        let r = subgroupMatrixLoad<subgroup_matrix_right<f16, ${UNIT}, ${UNIT}>>(
          &staged_kt, ${step * UNIT}u * S_KEY + ${t * UNIT}u, false, S_KEY);
        acc = subgroupMatrixMultiplyAccumulate(l, r, acc);
      }`)}
      subgroupMatrixStore(&scores, sg * ${UNIT}u * S_SCORE + ${t * UNIT}u, acc, false, S_SCORE);
    }`)}
    ${own}

    // The mask, the bias and the clamp go in on the pass that takes the max, so
    // the exponential pass reads each entry once more from a register and the
    // pair bias is read from global exactly once.
    let bias_row = bias_base + k0;
    var m = -1e30;
    for (var j = 0u; j < ${perHalf}u; j += 1u) {
      let slot = half * ${perHalf}u + j;
      ${checked ? `var logit = -1e30;
      if (k0 + slot < p.queries) {` : "{"}
        var scored = scores[row * S_SCORE + slot] + mask_add[slot];
        if (has_bias) { scored += pair_bias[bias_row + slot]; }
        ${checked ? "logit = clamp(scored, -1e8, 1e8);" : "let logit = clamp(scored, -1e8, 1e8);"}
        ${checked ? "" : `logits[j] = logit;
        m = max(m, logit);`}
      }
      ${checked ? `logits[j] = logit;
      m = max(m, logit);` : ""}
    }
    let tile_max = max(m, subgroupShuffleXor(m, 1u));
    let new_max = max(running_max, tile_max);
    let rescale = exp(running_max - new_max);
    ${checked ? `// 🔴 A KEY PAST THE END WEIGHS NOTHING, and it is not enough to have scored
    // it -1e30: if a whole tile is past the end then new_max is -1e30 too and
    // exp(logit - new_max) is exp(0), which is one. The register kernel selects
    // the same case out; a fully MASKED key is left alone, because that one is
    // -1e8 and both kernels then agree on a row of equal weights.` : ""}
    var partial = 0.0;
    for (var j = 0u; j < ${perHalf}u; j += 1u) {
      let slot = half * ${perHalf}u + j;
      ${checked
        ? "let weight = select(0.0, exp(logits[j] - new_max), k0 + slot < p.queries);"
        : "let weight = exp(logits[j] - new_max);"}
      probs[row * S_KEY + slot] = f16(weight);
      partial += weight;
    }
    running_sum = running_sum * rescale + partial + subgroupShuffleXor(partial, 1u);
    running_max = new_max;
    ${own}

    // O_tile = P V, into the array the scores have finished with.
${lines(channelTiles, (c) => `    {
      var acc = subgroup_matrix_result<f32, ${UNIT}, ${UNIT}>();
${lines(keyTiles, (step) => `      {
        let l = subgroupMatrixLoad<subgroup_matrix_left<f16, ${UNIT}, ${UNIT}>>(
          &probs, sg * ${UNIT}u * S_KEY + ${step * UNIT}u, false, S_KEY);
        let r = subgroupMatrixLoad<subgroup_matrix_right<f16, ${UNIT}, ${UNIT}>>(
          &staged_v, ${step * UNIT}u * S_HEAD + ${c * UNIT}u, false, S_HEAD);
        acc = subgroupMatrixMultiplyAccumulate(l, r, acc);
      }`)}
      subgroupMatrixStore(&scores, sg * ${UNIT}u * S_SCORE + ${c * UNIT}u, acc, false, S_SCORE);
    }`)}
    ${own}
    // 🔴 A LANE ACCUMULATES ITS OWN ROW, WHICH IS WHY THE RESCALE IS A REGISTER.
    // The obvious split of a ROWS x HEAD tile over LANES lanes gives each lane
    // a slice of SOME row, so every lane needs every row's rescale and the
    // three row statistics have to live in workgroup memory. Splitting it the
    // other way - the two lanes of one row take half the channels each - gives
    // the same HEAD / 8 vec4s a lane and asks only for the row this lane
    // already owns. The three arrays disappear, and so does a barrier.
    for (var t = 0u; t < ${accVectors}u; t += 1u) {
      let at = row * S_SCORE + half * ${HEAD / 2}u + t * 4u;
      out_acc[t] = out_acc[t] * rescale
        + vec4<f32>(scores[at], scores[at + 1u], scores[at + 2u], scores[at + 3u]);
    }`;

  return `enable f16;
enable subgroups;
enable chromium_experimental_subgroup_matrix;
${COMMON}
const HD4: u32 = ${vectors}u;
const HEAD: u32 = ${HEAD}u;
const ROWS: u32 = ${ROWS}u;
const KEYS: u32 = ${KEYS}u;
const S_HEAD: u32 = ${s.head}u;
const S_KEY: u32 = ${s.key}u;
const S_SCORE: u32 = ${s.score}u;

@group(0) @binding(0) var<storage, read> query: array<${vec4In}>;
@group(0) @binding(1) var<storage, read> key: array<${vec4In}>;
@group(0) @binding(2) var<storage, read> value: array<${vec4Value}>;
@group(0) @binding(3) var<storage, read> gate: array<${vec4In}>;
@group(0) @binding(4) var<storage, read> mask: array<f32>;
@group(0) @binding(5) var<storage, read> pair_bias: array<f32>;
@group(0) @binding(6) var<uniform> p: Parameters;
@group(0) @binding(7) var<storage, read_write> output: array<${vec4Out}>;

var<workgroup> staged_q: array<f16, ${ROWS * s.head}>;
// ...TRANSPOSED, so the right operand loads row-major; see the note on the flag.
var<workgroup> staged_kt: array<f16, ${HEAD * s.key}>;
var<workgroup> staged_v: array<f16, ${KEYS * s.head}>;
// The scores, and then the P V result: the second is written only after every
// lane has finished reading the first, so one array serves both.
var<workgroup> scores: array<f32, ${ROWS * s.score}>;
var<workgroup> probs: array<f16, ${ROWS * s.key}>;
var<workgroup> mask_add: array<f32, ${KEYS}>;
${packIn || packValue ? `
fn load4(w: vec2<u32>) -> vec4<f32> {
  let lo = unpack2x16float(w.x);
  let hi = unpack2x16float(w.y);
  return vec4<f32>(lo.x, lo.y, hi.x, hi.y);
}` : ""}${packValue && !packIn ? `
fn load4v(w: vec2<u32>) -> vec4<f32> {
  let lo = unpack2x16float(w.x);
  let hi = unpack2x16float(w.y);
  return vec4<f32>(lo.x, lo.y, hi.x, hi.y);
}` : ""}${packOut ? `
fn store4(v: vec4<f32>) -> vec2<u32> {
  return vec2<u32>(pack2x16float(v.xy), pack2x16float(v.zw));
}` : ""}

fn mask_index(batch: u32, key_index: u32) -> u32 {
  if (p.transpose == 0u) { return batch * p.queries + key_index; }
  return key_index * p.batch + batch;
}

@compute @workgroup_size(${LANES})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        // 🔴 NOT local_id.x / 32. subgroupMatrixStore needs a SUBGROUP-UNIFORM
        // offset, and Tint cannot prove a value derived from
        // local_invocation_id is one - "requires argument 1 to be uniform".
        // The builtin is uniform by construction; src/kernels/matrix-linear.js
        // takes it the same way, and the note at its top records that this
        // Dawn accepts it in a store offset at all.
        @builtin(subgroup_id) sg: u32) {
  let local = local_id.x;
  let batch_index = group.y;
  let head = group.z;
  if (batch_index >= p.batch || head >= p.heads) { return; }
  let q_base = group.x * ROWS;

  // ...the query tile once, for every key in the sequence.
  for (var i = local; i < ROWS * HD4; i += ${LANES}u) {
    let qi = i / HD4;
    let c4 = i % HD4;
    let q_index = q_base + qi;
    var v = vec4<f32>(0.0);
    if (q_index < p.queries) {
      v = ${read4("query", "((batch_index * p.queries + q_index) * p.heads + head) * HD4 + c4")};
    }
    staged_q[qi * S_HEAD + c4 * 4u + 0u] = f16(v.x);
    staged_q[qi * S_HEAD + c4 * 4u + 1u] = f16(v.y);
    staged_q[qi * S_HEAD + c4 * 4u + 2u] = f16(v.z);
    staged_q[qi * S_HEAD + c4 * 4u + 3u] = f16(v.w);
  }
${HEAD === headDim ? "" : `  // ...and the padding past the head, which the contraction still reaches.
  // Written out only when there IS padding: HEAD minus the head width is a
  // constant, and dividing by a constant zero is a WGSL compile error even in
  // dead code.
  for (var i = local; i < ROWS * ${HEAD - headDim}u; i += ${LANES}u) {
    staged_q[(i / ${HEAD - headDim}u) * S_HEAD + ${headDim}u + i % ${HEAD - headDim}u] = f16(0.0);
  }
`}  // 🔴 REGISTERS, NOT WORKGROUP MEMORY - THE SWEEP SAID THIS IS THE WHOLE GAME.
  // Five geometries ranked EXACTLY by workgroup bytes a lane: 242 for 4x32 at
  // 324.6 ms, 276 for 2x32 at 341.0, 438 for 2x64 at 441.1, 762 for 2x128 at
  // 568.8. This kernel is latency-bound and its occupancy is capped by shared
  // memory, so a byte a lane is worth more than an arithmetic saving - which
  // is why the tile that amortises the most fixed cost is also the slowest.
  // The running output was ROWS x HEAD floats of that, 64 bytes a lane, and it
  // is the one array that is never a matrix operand, so it can be registers.
  var out_acc: array<vec4<f32>, ${accVectors}>;
  for (var t = 0u; t < ${accVectors}u; t += 1u) { out_acc[t] = vec4<f32>(0.0); }

  // 🔴 TWO LANES A ROW, AND THE PAIR IS ADJACENT - so the online softmax's
  // running max and sum need no workgroup memory and no barrier either. Both
  // lanes keep the same copy and a shuffle across the low bit combines their
  // halves; what used to be four small arrays, two barriers and four
  // round trips a key tile is now two instructions. Small arrays are not free
  // here: with occupancy capped by shared memory, LANES + 3 x ROWS floats was
  // 1280 bytes of the 4x32 tile, which is the whole distance between two
  // workgroups an SM and three.
  let row = local / 2u;
  let half = local % 2u;
  var running_max = -1e30;
  var running_sum = 0.0;
  // ...and the corrected logits this lane owns, which the two softmax passes
  // used to hand each other through the score array: 8 bytes a key a lane of
  // workgroup traffic, for a value that never leaves the lane that made it.
  var logits: array<f32, ${perHalf}>;
  // ...and the key tile on its way from global memory into workgroup memory.
  // The step count is a constant, so these unroll into registers; see the note
  // above on why the loop they replace could not simply be unrolled.
  var kv: array<vec4<f32>, ${steps}>;
  var vv: array<vec4<f32>, ${steps}>;

  // 🔴 EVERYTHING THAT DOES NOT DEPEND ON THE KEY COMES OUT OF THE KEY LOOP.
  // The pair bias index was (head * queries + q) * queries + k, computed once a
  // key a lane - TWO integer multiplies for a base that is the same for every
  // key this lane will ever look at, on a row that never changes.
  let q_index = q_base + row;
  let q_live = q_index < p.queries;
  let has_bias = p.has_pair_bias != 0u && q_live;
  let bias_base = select(0u, (head * p.queries + q_index) * p.queries, q_live);
  // 🔴 AND THE BOUNDS CHECK COMES OUT OF EVERY TILE BUT THE LAST. Asking
  // whether the key index is past the end was three instructions a key - the
  // compare, the select in the exponential pass and the -1e30 it selects - on
  // tiles where it is true by construction. Two copies of the body, one of them
  // entered at most once.
  let full = (p.queries / KEYS) * KEYS;
  for (var k0 = 0u; k0 < full; k0 += KEYS) {
${tileBody(false)}
  }
  if (full < p.queries) {
    let k0 = full;
${tileBody(true)}
  }
  // 🔴 AND STRAIGHT OUT: no barrier, no round trip through the score array.
  // A lane holds whole vec4s of one query row, which is exactly the grouping
  // the gate and the output want, and running_sum is the divisor for the row
  // it holds. The padded channels past a narrow head belong to the lanes whose
  // channel group runs off the end of the real head, and those write nothing.
  let inverse = 1.0 / running_sum;
  if (q_live) {
    for (var t = 0u; t < ${accVectors}u; t += 1u) {
      let c4 = half * ${HEAD / 8}u + t;
      if (c4 < HD4) {
        let at = ((batch_index * p.queries + q_index) * p.heads + head) * HD4 + c4;
        let gated = out_acc[t] * inverse * ${read4("gate", "at")};
        output[at] = ${packOut ? "store4(gated)" : "gated"};
      }
    }
  }
}`;
}
