/**
 * AF3's `grid.attend` on the subgroup matrix units.
 *
 * 🔴 IT IS THE LARGEST KERNEL IN THE TRUNK AND THE ONLY CUBIC ONE. docs/A100.md
 * measures `grid.attend` at 21.7 ms of a 59-token pairformer and 318 of a
 * 150-token one; everything around it is quadratic, so it leads by more on
 * every longer chain. The scalar kernel it replaces gives one invocation one
 * query and has that invocation read every key itself - its own docstring
 * names its cost as "the 2 * dimension/4 vectors a key needs" - and a matrix
 * multiply contracts sixteen queries against sixteen keys in one instruction,
 * so a staged key tile is read once for sixteen queries instead of once each.
 *
 * 🔴 AND IT IS ONE KERNEL FOR TWO MODELS. OpenDDE runs AF3's pairformer, so
 * this is on both paths; ESMFold2 imports the same pair track and gets it too.
 *
 * 🔴 THE RULE THAT KEPT THIS UNBUILT WAS THE WRONG RULE. docs/A100.md's
 * "materialised, past a billion multiply-accumulates, and K deep" was applied
 * to a flash attention on the grounds that `dimension` is 32, and that
 * inference is wrong: a GEMM at K = 32 amortises its staged panel over two k
 * steps and nothing else, while a flash attention stages the QUERY tile once
 * and reuses it across the whole key sequence. The reuse that pays for the
 * staging is in the loop, not in K. AF2's `src/evoformer/attention-matrix.js`
 * measures 1.69x on exactly this shape.
 *
 * 🔴 WRITTEN BESIDE AF2'S KERNEL RATHER THAN SHARED WITH IT, WHICH IS THE SAME
 * CALL src/af3/grid-attention-webgpu.js ALREADY MADE AND RECORDED. AF2's takes
 * a uniform for its shape, folds the 1/sqrt(d) scale into the query projection
 * and multiplies by the gate itself; AF3 has all three as a compile-time
 * constant, an explicit multiply in this pass, and the output projection. What
 * IS shared is the shape arithmetic - the geometry, the bank-avoiding strides
 * and the support test - because those are properties of the units and not of
 * the model, and a second copy of them would be a second thing to re-derive.
 *
 * 🔴 THE MASK STAYS A CONDITIONAL SUBTRACTION, because the kernel this replaces
 * carries a measured warning against turning it into an additive penalty:
 * relRMS 2.24e-1 against the CPU reference where the conditional measures
 * 9.63e-7, deterministically, for an algebraically identical expression that
 * was never run down. Only the READ is hoisted here - the mask is per key and
 * the scalar kernel read it once per query row - and the conditional itself is
 * left exactly as it is written there. A trap that is recorded and not
 * explained is a trap to route around, not to re-enter.
 */

import {
  ATTENTION_MATRIX_SUBGROUP_SIZE, ATTENTION_MATRIX_UNIT,
  attentionMatrixGeometry, attentionMatrixStorageBytes,
  paddedHead, strides, supportsAttentionMatrix,
} from "../evoformer/attention-matrix.js";

const UNIT = ATTENTION_MATRIX_UNIT;

/**
 * 🔴 THE GEOMETRY IS SWEPT PER KERNEL AND NEVER INHERITED. AF2's optimum moved
 * from 6x16 to 4x32 when the scalar work in its body changed, with each 5-6%
 * worse at the other's tile, and this body is not that body: no gate, no
 * uniform, a bias that is always present and a mask that stays conditional.
 * tools/gpu/bench-grid-attend.js --matrix= is how to re-ask.
 */
export const GRID_ATTEND_MATRIX_DEFAULT_TILE = { subgroups: 4, keyTile: 32 };

/**
 * The one place the decision is made: the device's answer, the head width, and
 * the room, resolved to what `compilePairTrack` wants.
 *
 * 🔴 IT IS RESOLVED HERE AND NOT AT EACH CALLER. Three stacks compile this
 * track - the pairformer block, the MSA stack and ESMFold2's trunk - and a rule
 * written three times is a rule that will be three rules. CLAUDE.md's own
 * habit: when a kernel's shape comes from a device limit, resolve it once and
 * pass it down.
 *
 * @returns {false|true|string} false, or the geometry `compilePairTrack` takes.
 */
export function resolveGridAttendMatrix(device, dimension, tuning) {
  if (tuning?.gridAttendMatrix !== true) return false;
  const tile = tuning.gridAttendMatrixTile ?? undefined;
  if (!supportsGridAttendMatrix(device, dimension, tile)) return false;
  return tile ?? true;
}

/** Whether this device and this head width can run it at all. */
export function supportsGridAttendMatrix(device, dimension, geometry) {
  return supportsAttentionMatrix(device, dimension, geometry ?? GRID_ATTEND_MATRIX_DEFAULT_TILE);
}

/** The resolved geometry, so the dispatch and the shader divide by one number. */
export function gridAttendMatrixGeometry(requested) {
  return attentionMatrixGeometry(requested ?? GRID_ATTEND_MATRIX_DEFAULT_TILE);
}

/** Workgroup bytes it declares, for the caller that has to check the limit. */
export function gridAttendMatrixStorageBytes(dimension, geometry) {
  return attentionMatrixStorageBytes(dimension, gridAttendMatrixGeometry(geometry));
}

/**
 * The attend pass, with `createGridAttentionShaders`'s six bindings unchanged:
 * q, k, v, bias, mask, gathered. Only the dispatch's x extent moves, from
 * ceil(n / 64) to ceil(n / rows) - which is why the geometry travels back out
 * with the shaders as a tile and is not re-derived at the dispatch.
 *
 * @param {{n, heads, dimension, transpose}} shape
 * @param {{q?, k?, v?, gathered?}} store4 per-tensor storage, "f32" or "f16"
 * @param {object|string} [geometry] `{subgroups, keyTile}` or "4x32"
 */
export function createGridAttendMatrixShader(shape, store4 = {}, geometry) {
  const { n, heads, dimension, transpose } = shape;
  if (!Number.isSafeInteger(dimension) || dimension % 4 !== 0
      || dimension > 32 || dimension < 4) {
    throw new RangeError(`the matrix attend takes a head of 4 to 32 channels; got ${dimension}`);
  }
  const g = gridAttendMatrixGeometry(geometry);
  const HEAD = paddedHead(dimension);
  const ROWS = g.rows;
  const KEYS = g.keyTile;
  const LANES = g.lanes;
  const s = strides(dimension, g);
  const HD4 = dimension / 4;
  const headSteps = HEAD / UNIT;
  const keyTiles = KEYS / UNIT;
  const channelTiles = HEAD / UNIT;
  const perHalf = KEYS / 2;
  if (ROWS * 2 !== LANES) throw new RangeError("two lanes a row is what the reduction assumes");
  if (LANES / ATTENTION_MATRIX_SUBGROUP_SIZE !== g.subgroups) {
    throw new RangeError("a subgroup is 32 lanes and owns 16 rows");
  }
  // ROWS x HEAD floats over LANES lanes is HEAD / 8 vec4s a lane, whatever the
  // geometry - and a constant the compiler can unroll.
  const accVectors = (ROWS * HEAD) / (4 * LANES);

  const packQ = store4.q === "f16";
  const packK = store4.k === "f16";
  const packV = store4.v === "f16";
  const packGathered = store4.gathered === "f16";
  const array4 = (packed) => (packed ? "vec2<u32>" : "vec4<f32>");
  const read4 = (name, packed, index) => (packed ? `load4(${name}[${index}])` : `${name}[${index}]`);

  const lines = (count, body) => Array.from({ length: count }, (_, i) => body(i)).join("\n");
  // 🔴 N IS A COMPILE-TIME CONSTANT HERE, WHICH AF2'S UNIFORM IS NOT - so the
  // partial tile is a generation-time question and not a runtime branch. Where
  // the token count divides the key tile there is no checked body at all, and
  // where it does not there is exactly one entry into it.
  const fullKeys = Math.floor(n / KEYS) * KEYS;
  // ...and the same for the query rows a workgroup owns.
  const rowsExact = n % ROWS === 0;

  // The key's mask, transposed with the activation - mask[(j,row)] in the
  // column direction and mask[(row,j)] otherwise.
  const maskIndex = transpose ? "k_index * N + row" : "row * N + k_index";

  // 🔴 THE GLOBAL READS COME BEFORE THE BARRIER, NOT AFTER IT. This tile used to
  // be barrier, read global, write workgroup, barrier, compute - so every key
  // tile exposed a full memory latency with nothing of this workgroup's own to
  // cover it. Reading into REGISTERS first and moving the barrier between the
  // read and the write puts that latency underneath the TAIL OF THE PREVIOUS
  // TILE'S compute. The same change is worth 1.43x on the staged GEMM and 5% on
  // AF2's flash attention; see `prefetch` in src/runtime/matrix-linear.js.
  //
  // 🔴 AND THE HELD TILE IS UNROLLED INTO NAMED VARIABLES, because a dynamic
  // index into a WGSL array is addressable memory and not registers - the trap
  // CLAUDE.md records, and the one that made the GEMM's first prefetch SLOWER
  // than no prefetch at all.
  const stageCount = KEYS * HD4;
  const stageIterations = Math.ceil(stageCount / LANES);
  const stageExact = stageCount % LANES === 0;
  const maskIterations = Math.ceil(KEYS / LANES);
  const maskExact = KEYS % LANES === 0;
  const tileBody = (checked) => `${lines(stageIterations, (j) => `    var kv_${j} = vec4<f32>(0.0);
    var vv_${j} = vec4<f32>(0.0);
    {
      let i = local + ${j * LANES}u;
      ${stageExact ? "{" : `if (i < ${stageCount}u) {`}
        let ki = i / ${HD4}u;
        let c4 = i % ${HD4}u;
        let k_index = k0 + ki;
        ${checked ? "if (k_index < N) {" : "{"}
          let at = ((row * N + k_index) * HEADS + head) * ${HD4}u + c4;
          kv_${j} = ${read4("k", packK, "at")};
          vv_${j} = ${read4("v", packV, "at")};
        }
      }
    }`)}
${lines(maskIterations, (j) => `    var km_${j} = 0.0;
    {
      let i = local + ${j * LANES}u;
      ${maskExact ? "{" : `if (i < ${KEYS}u) {`}
        let k_index = k0 + i;
        ${checked
          ? `km_${j} = select(0.0, mask[${maskIndex}], k_index < N);`
          : `km_${j} = mask[${maskIndex}];`}
      }
    }`)}
    workgroupBarrier();
${lines(stageIterations, (j) => `    {
      let i = local + ${j * LANES}u;
      ${stageExact ? "{" : `if (i < ${stageCount}u) {`}
        let ki = i / ${HD4}u;
        let c4 = i % ${HD4}u;
        // The key goes in transposed and the value does not.
${lines(4, (c) => `        staged_kt[(c4 * 4u + ${c}u) * S_KEY + ki] = f16(kv_${j}${[".x", ".y", ".z", ".w"][c]});`)}
${lines(4, (c) => `        staged_v[ki * S_HEAD + c4 * 4u + ${c}u] = f16(vv_${j}${[".x", ".y", ".z", ".w"][c]});`)}
      }
    }`)}
    // 🔴 THE MASK IS PER KEY AND THE SCALAR KERNEL READ IT PER QUERY ROW. All
    // ${ROWS} rows of this workgroup want the same global float for the same
    // key. Stage it once - ${KEYS} floats against ${ROWS} redundant loads a key
    // - and leave the conditional it feeds alone; see the note at the top.
${lines(maskIterations, (j) => `    {
      let i = local + ${j * LANES}u;
      ${maskExact ? "{" : `if (i < ${KEYS}u) {`}
        key_mask[i] = km_${j};
      }
    }`)}
${HEAD === dimension ? "" : `    for (var i = local; i < ${HEAD - dimension}u * ${KEYS}u; i += ${LANES}u) {
      staged_kt[(${dimension}u + i / ${KEYS}u) * S_KEY + i % ${KEYS}u] = f16(0.0);
      staged_v[(i % ${KEYS}u) * S_HEAD + ${dimension}u + i / ${KEYS}u] = f16(0.0);
    }`}
    workgroupBarrier();

    // S = Q K^T: one subgroup's sixteen queries against every staged key.
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
    workgroupBarrier();

    // The scale, the bias and the mask all go in on the pass that takes the
    // maximum, so the exponential pass reads each logit back from a register
    // and the pair bias is read from global exactly once.
    let bias_row = bias_base + k0;
    var m = -1e30;
    for (var j = 0u; j < ${perHalf}u; j += 1u) {
      let slot = half * ${perHalf}u + j;
      ${checked ? `var logit = -1e30;
      if (k0 + slot < N) {` : "{"}
        // 🔴 SCALE MULTIPLIES THE SCORE, IT IS NOT FOLDED INTO THE STAGED
        // QUERY. Folding it would put a 1/sqrt(32) rounding step in front of
        // the f16 conversion, on a kernel whose whole differential bar is that
        // conversion; here it is one f32 multiply on a value the unit has
        // already reduced.
        var scored = scores[row_slot * S_SCORE + slot] * SCALE + bias[bias_row + slot];
        if (key_mask[slot] <= 0.0) { scored = scored - 1.0e9; }
        ${checked ? "logit = scored;" : "let logit = scored;"}
        ${checked ? "" : `logits[j] = logit;
        m = max(m, logit);`}
      }
      ${checked ? `logits[j] = logit;
      m = max(m, logit);` : ""}
    }
    // Two lanes share a query row and take half the keys each, and the pair is
    // adjacent - so the row reduction is a shuffle and not an array.
    let tile_max = max(m, subgroupShuffleXor(m, 1u));
    let new_max = max(running_max, tile_max);
    let rescale = exp(running_max - new_max);
    ${checked ? `// 🔴 A KEY PAST THE END WEIGHS NOTHING, and scoring it -1e30 is not enough:
    // a tile entirely past the end makes new_max -1e30 too, and exp(0) is one.` : ""}
    var partial = 0.0;
    for (var j = 0u; j < ${perHalf}u; j += 1u) {
      let slot = half * ${perHalf}u + j;
      ${checked
        ? "let weight = select(0.0, exp(logits[j] - new_max), k0 + slot < N);"
        : "let weight = exp(logits[j] - new_max);"}
      probs[row_slot * S_KEY + slot] = f16(weight);
      partial += weight;
    }
    running_sum = running_sum * rescale + partial + subgroupShuffleXor(partial, 1u);
    running_max = new_max;
    workgroupBarrier();

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
    workgroupBarrier();
    // 🔴 A LANE ACCUMULATES ITS OWN ROW, WHICH IS WHY THE RESCALE IS A
    // REGISTER. Splitting a ROWS x HEAD tile the obvious way gives each lane a
    // slice of SOME row, so every lane needs every row's rescale and the row
    // statistics have to live in workgroup memory. Splitting it the other way
    // - the two lanes of one row take half the channels each - gives the same
    // HEAD / 8 vec4s a lane and asks only for the row this lane already owns.
    for (var t = 0u; t < ${accVectors}u; t += 1u) {
      let at = row_slot * S_SCORE + half * ${HEAD / 2}u + t * 4u;
      out_acc[t] = out_acc[t] * rescale
        + vec4<f32>(scores[at], scores[at + 1u], scores[at + 2u], scores[at + 3u]);
    }`;

  return `enable f16;
enable subgroups;
enable chromium_experimental_subgroup_matrix;
const N: u32 = ${n}u;
const PAIRS: u32 = ${n * n}u;
const HEADS: u32 = ${heads}u;
const SCALE: f32 = ${1 / Math.sqrt(dimension)};
const S_HEAD: u32 = ${s.head}u;
const S_KEY: u32 = ${s.key}u;
const S_SCORE: u32 = ${s.score}u;

@group(0) @binding(0) var<storage, read> q: array<${array4(packQ)}>;
@group(0) @binding(1) var<storage, read> k: array<${array4(packK)}>;
@group(0) @binding(2) var<storage, read> v: array<${array4(packV)}>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> mask: array<f32>;
@group(0) @binding(5) var<storage, read_write> gathered: array<${array4(packGathered)}>;

var<workgroup> staged_q: array<f16, ${ROWS * s.head}>;
// ...TRANSPOSED, so the right operand loads row-major. subgroupMatrixLoad's
// column-major flag is passed as false by every checker in this repository, so
// its meaning is not pinned and a kernel is not the place to find out.
var<workgroup> staged_kt: array<f16, ${HEAD * s.key}>;
var<workgroup> staged_v: array<f16, ${KEYS * s.head}>;
// The scores, and then the P V result: the second is written only after every
// lane has finished reading the first, so one array serves both.
var<workgroup> scores: array<f32, ${ROWS * s.score}>;
var<workgroup> probs: array<f16, ${ROWS * s.key}>;
var<workgroup> key_mask: array<f32, ${KEYS}>;
${packQ || packK || packV ? `
fn load4(w: vec2<u32>) -> vec4<f32> {
  let lo = unpack2x16float(w.x);
  let hi = unpack2x16float(w.y);
  return vec4<f32>(lo.x, lo.y, hi.x, hi.y);
}` : ""}${packGathered ? `
fn store4(value: vec4<f32>) -> vec2<u32> {
  return vec2<u32>(pack2x16float(value.xy), pack2x16float(value.zw));
}` : ""}

@compute @workgroup_size(${LANES})
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        // 🔴 NOT local_id.x / 32. subgroupMatrixStore needs a SUBGROUP-UNIFORM
        // offset and Tint cannot prove a value derived from
        // local_invocation_id is one; the builtin is uniform by construction.
        @builtin(subgroup_id) sg: u32) {
  let local = local_id.x;
  // 🔴 THE WORKGROUP id, NOT THE GLOBAL ONE. row and head have to be visibly
  // workgroup-uniform: this kernel branches on them and the staging loop below
  // has barriers in it.
  let row = group.y;
  let head = group.z;
  if (row >= N || head >= HEADS) { return; }
  let q_base = group.x * ${ROWS}u;

  // ...the query tile once, for every key in the row.
  for (var i = local; i < ${ROWS}u * ${HD4}u; i += ${LANES}u) {
    let qi = i / ${HD4}u;
    let c4 = i % ${HD4}u;
    let q_index = q_base + qi;
    var value = vec4<f32>(0.0);
    ${rowsExact ? "{" : "if (q_index < N) {"}
      value = ${read4("q", packQ, `((row * N + q_index) * HEADS + head) * ${HD4}u + c4`)};
    }
${lines(4, (c) => `    staged_q[qi * S_HEAD + c4 * 4u + ${c}u] = f16(value${[".x", ".y", ".z", ".w"][c]});`)}
  }
${HEAD === dimension ? "" : `  // ...and the padding past the head, which the contraction still reaches.
  for (var i = local; i < ${ROWS}u * ${HEAD - dimension}u; i += ${LANES}u) {
    staged_q[(i / ${HEAD - dimension}u) * S_HEAD + ${dimension}u + i % ${HEAD - dimension}u] = f16(0.0);
  }
`}
  // 🔴 REGISTERS, NOT WORKGROUP MEMORY. AF2's sweep ranked five geometries
  // EXACTLY by workgroup bytes a lane, so on a kernel this latency-bound a byte
  // a lane is worth more than an arithmetic saving. The running output is
  // ROWS x HEAD floats of it and is the one array that is never a matrix
  // operand, so it is the one that can be registers.
  var out_acc: array<vec4<f32>, ${accVectors}>;
  for (var t = 0u; t < ${accVectors}u; t += 1u) { out_acc[t] = vec4<f32>(0.0); }

  let row_slot = local / 2u;
  let half = local % 2u;
  var running_max = -1e30;
  var running_sum = 0.0;
  // ...and the corrected logits this lane owns, which the two softmax passes
  // would otherwise hand each other through the score array.
  var logits: array<f32, ${perHalf}>;

  // 🔴 EVERYTHING THAT DOES NOT DEPEND ON THE KEY COMES OUT OF THE KEY LOOP.
  // The bias index is head * PAIRS + i * N + j: two integer multiplies for a
  // base that is the same for every key this lane will ever look at.
  let q_index = q_base + row_slot;
  let q_live = q_index < N;
  let bias_base = head * PAIRS + select(0u, q_index * N, q_live);
${[
  fullKeys > 0 ? `  for (var k0 = 0u; k0 < ${fullKeys}u; k0 += ${KEYS}u) {
${tileBody(false)}
  }` : "",
  fullKeys < n ? `  {
    let k0 = ${fullKeys}u;
${tileBody(true)}
  }` : "",
].filter(Boolean).join("\n")}

  // 🔴 AND STRAIGHT OUT: no barrier and no round trip through the score array.
  // A lane holds whole vec4s of one query row, and running_sum is that row's
  // divisor. The padded channels past a narrow head belong to the lanes whose
  // channel group runs off the end of the real head, and those write nothing.
  let inverse = 1.0 / running_sum;
  if (q_live) {
    for (var t = 0u; t < ${accVectors}u; t += 1u) {
      let c4 = half * ${HEAD / 8}u + t;
      if (c4 < ${HD4}u) {
        let at = ((row * N + q_index) * HEADS + head) * ${HD4}u + c4;
        gathered[at] = ${packGathered ? "store4(out_acc[t] * inverse)" : "out_acc[t] * inverse"};
      }
    }
  }
}`;
}
