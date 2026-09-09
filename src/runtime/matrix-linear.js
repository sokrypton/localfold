/**
 * A dense projection on the matrix units, with the operands STAGED IN
 * WORKGROUP MEMORY and shared by several subgroups.
 *
 * 🔴 chromium_experimental_subgroup_matrix IS NOT STANDARDS-COMPLIANT WGSL, and
 * AGENTS.md asks for standards compliance. This is here anyway on the same
 * terms as the subgroup attention kernels: a CAPABILITY-GATED FAST PATH that a
 * device without the feature never compiles. chooseLinearKernel is the gate,
 * and a device that fails it gets createLinearShader exactly as before.
 *
 * 🔴 THIS IS THE SHAPE gemm-matrix.js SAYS IS IMPOSSIBLE, AND IT WAS, ONCE.
 * That file is built around "one subgroup per workgroup, because the store must
 * be uniform" - WGSL's uniformity analysis works at workgroup scope, so an
 * offset derived from WHICH SUBGROUP YOU ARE could not be proven uniform even
 * when it was. On the Dawn this A100 runs, a subgroup_id builtin in a
 * subgroupMatrixStore offset COMPILES (tools/gpu/probe-staged-gemm-parts.js
 * grew out of checking that). Sharing staged operands between subgroups is the
 * whole difference between a memory-bound kernel and a compute-bound one.
 *
 * The arithmetic, at 128x128x16 with eight subgroups: one K panel costs 4096
 * halves of staging and feeds 8 subgroups x 8 accumulators = 64
 * multiply-accumulates. gemm-matrix.js's straight-line form reads its operands
 * from GLOBAL memory every k step, reaches about 16 flop per byte, and measures
 * 19 TFLOP/s against units that will do 310 - it is bound on operand traffic
 * and nothing else. Staging raises that by the number of subgroups sharing the
 * panel, and this measures 28.2.
 *
 * 🔴 THE PANEL IS ALWAYS f16 AND THE BUFFERS NEED NOT BE. The matrix units take
 * halves and nothing else on this device, but the conversion is free: it
 * happens in the staging copy, which every version of this kernel already
 * does. So `sourcePrecision`, `weightPrecision` and `outputPrecision` are
 * independent of the units, and this can stand in for createLinearShader
 * wherever that reads an f32 activation and writes an f16 hidden one. Without
 * that the kernel would only serve callers that already hold halves, which in
 * AF2's transition is the second pass and not the first.
 *
 * 🔴 THE EPILOGUE FLUSHES A ROW-BLOCK, NOT A TILE. One 16x16 tile is 16 f32 to
 * a row - 64 bytes, half a transaction - so flushing tiles one at a time writes
 * the output in half-width bursts; it measured 0.70 ms against 0.575 for the
 * row-block form. A region-sized result buffer is not an option either: 128x128
 * in f32 is 64 KiB against a 48 KiB limit. A row-block per subgroup is the
 * middle, and the bias and activation are applied by the lanes as they copy it
 * out.
 *
 * 🔴 EVERY EDGE IS ZERO-PADDED, AND NONE OF THEM SLIDES BACK. gemm-matrix.js
 * slides its last region back to end on the final row, because an
 * out-of-bounds subgroupMatrixLoad returns an entirely zero matrix rather than
 * a clamped one - so a region hanging over the edge cannot be masked
 * afterwards. That works, and it costs two restrictions this kernel cannot
 * afford: the shape must be at least one whole region on each axis, and a
 * RESIDUAL is wrong on a ragged shape because the overlap adds twice
 * (check-staged-matrix.js measured relRms 0.61).
 *
 * Because this kernel already copies its operands through workgroup memory, it
 * can pad instead. A staged element past the end is written as zero, so the
 * matrix load is always in range and always reads a real panel; the epilogue
 * bounds-checks the store. That makes rows, columns and K all the same case,
 * lets a 59-row tensor use a 128-row block, and makes the residual safe
 * everywhere - which is what a transition's third shader needs.
 */

/**
 * A staged GEMM block, as `blockRows x blockColumns x blockInner x subgroupRows
 * x subgroupColumns`.
 *
 * 🔴 IT IS ONE STRING BECAUSE `--tune=` SPLITS ITS ARGUMENT ON COMMAS and so
 * cannot carry an object - the same reason `attentionMatrixTile` is "4x32".
 * Four kernels now share one geometry and none of them has been swept at its
 * own shape; this is what makes that a knob rather than a recompile.
 */
export function stagedMatrixBlock(spec) {
  if (spec === null || spec === undefined) return {};
  if (typeof spec === "object") return spec;
  const parts = String(spec).split("x").map(Number);
  if (parts.length !== 5 || !parts.every((v) => Number.isSafeInteger(v) && v > 0)) {
    throw new RangeError(`a staged matrix block reads "BMxBNxBKxSRxSC"; got ${spec}`);
  }
  const [blockRows, blockColumns, blockInner, subgroupRows, subgroupColumns] = parts;
  return { blockRows, blockColumns, blockInner, subgroupRows, subgroupColumns };
}

/**
 * Whether a geometry's `directWeights` may actually be taken at this shape.
 *
 * 🔴 THE KNOB IS A REQUEST AND THIS IS THE ANSWER, because two of the three
 * conditions are the CALLER's and not the device's. The weight buffer has to
 * already hold the matrix element - a direct read has no staging copy left to
 * convert in - and the contracted extent has to divide by the K panel, because
 * the buffer is bound as [weights | bias] and a tail panel would read the bias
 * as a weight row. A caller that sets the knob and skips this gets a kernel
 * that returns a plausible tensor.
 */
export function directWeightsAllowed(geometry, shape) {
  if (geometry?.directWeights !== true) return false;
  if (geometry.weightsTransposed === true) return false;
  if (shape.weightPrecision !== (geometry.matrixElement ?? "f16")) return false;
  return shape.inner % (geometry.blockInner ?? 16) === 0;
}

/**
 * Whether a geometry can serve a shape at all.
 *
 * 🔴 IT ALWAYS CAN, NOW, and this stays as a named fact rather than being
 * deleted: the previous version refused a shape smaller than one region and
 * refused a residual on a ragged one, and both restrictions were real until
 * the edges became zero-padded rather than slid back. A caller that used to
 * consult this does not need to any more, and one reading the history should
 * see why.
 */
export function stagedMatrixFits(shape, geometry = {}) {
  void shape;
  void geometry;
  return true;
}

/**
 * How much workgroup storage a geometry needs, so a caller can refuse it
 * before the pipeline does.
 */
export function stagedMatrixStorage(geometry = {}) {
  const {
    blockRows = 128, blockColumns = 128, blockInner = 16,
    subgroupRows = 1, subgroupColumns = 8,
    tile = { M: 16, N: 16, K: 16 }, result = "f32",
  } = geometry;
  const panels = (blockRows * blockInner
    + (geometry.directWeights ? 0 : blockInner * blockColumns)) * 2;
  const scratch = subgroupRows * subgroupColumns * tile.M * (blockColumns / subgroupColumns)
    * (result === "f32" ? 4 : 2);
  return panels + scratch;
}

/**
 * @param {object} [options]
 * @param {number} [options.blockRows] rows of the output one workgroup owns.
 * @param {number} [options.blockColumns] columns of it.
 * @param {number} [options.blockInner] how much of K is staged at a time. 16
 *   beat 32 and 64 on every shape measured - a smaller panel is more barriers
 *   and less workgroup storage, and the storage buys occupancy.
 * @param {number} [options.subgroupRows] subgroups down.
 * @param {number} [options.subgroupColumns] subgroups across.
 * @param {number} [options.subgroupSize] lanes per subgroup; 32 on NVIDIA.
 * @param {{M: number, N: number, K: number}} [options.tile] the device's config.
 * @param {"f16"|"f32"} [options.matrixElement] what the units multiply in -
 *   the componentType of the device's chosen config. f16 everywhere on NVIDIA;
 *   an M2 also offers f32.
 * @param {"f16"|"f32"} [options.result] accumulator width. f32 is free on an
 *   A100 and strictly more accurate - see docs/A100.md.
 * @param {"f16"|"f32"} [options.sourcePrecision] the activation BUFFER's type.
 * @param {"f16"|"f32"} [options.weightPrecision] the weight BUFFER's type.
 * @param {"f16"|"f32"} [options.outputPrecision] what to write.
 * @param {boolean} [options.residual] add the existing output rather than
 *   replacing it, which is what the transition's third shader wants. It
 *   requires a block-aligned shape - see stagedMatrixFits.
 * @param {boolean} [options.vectorStaging] fetch four elements of global memory
 *   at a time. It was worth 1.24 -> 0.70 ms and needs every offset it forms to
 *   divide by four.
 *
 *   🔴 THAT INCLUDES `weight_offset`, WHICH IS A RUNTIME VALUE. A vec4 read at
 *   element `i` returns elements `i & ~3 .. i & ~3 + 3`, so an unaligned base
 *   silently shifts the whole weight panel. The caller knows the offset and
 *   this shader does not, so the caller decides - see createTransitionShaders,
 *   which asks per pass because a transition's two projections sit at
 *   different offsets in one buffer.
 */
export function createStagedMatrixShader(options = {}) {
  const BM = options.blockRows ?? 128;
  const BN = options.blockColumns ?? 128;
  const BK = options.blockInner ?? 16;
  const SGY = options.subgroupRows ?? 1;
  const SGX = options.subgroupColumns ?? 8;
  const lanes = options.subgroupSize ?? 32;
  const tile = options.tile ?? { M: 16, N: 16, K: 16 };
  const { M, N, K } = tile;
  const result = options.result ?? "f32";
  const sourcePrecision = options.sourcePrecision ?? "f16";
  const weightPrecision = options.weightPrecision ?? "f16";
  const outputPrecision = options.outputPrecision ?? "f32";
  const residual = options.residual ?? false;
  // 🔴 PER OPERAND, NOT PER KERNEL, BECAUSE A TRIANGLE CONTRACTION TRANSPOSES
  // EXACTLY ONE OF THE TWO. `true` means both, which is every caller that
  // predates this. A transposed operand walks the axis it is not stored on, so
  // its four consecutive elements are `columns` apart and it cannot be
  // vec4-read - but the OTHER one still can, and docs/A100.md prices the win as
  // almost entirely this.
  const staging = options.vectorStaging ?? false;
  const vectorSource = typeof staging === "object" ? staging.source === true : staging === true;
  const vectorWeights = typeof staging === "object" ? staging.weights === true : staging === true;
  const vectorStaging = vectorSource || vectorWeights;
  // 🔴 THREE OPTIONS THE OUTER PRODUCT MEAN NEEDS AND A DENSE PROJECTION DOES
  // NOT. Its contraction is a GEMM - `outer[(i,cl)][(j,cr)] = sum_s
  // left[s][(i,cl)] * right[s][(j,cr)]` - with the deepest K in the model, but
  // its left operand is stored sequence-major, it has no bias, and its output
  // is indexed by PAIR rather than by row-major (row, column). All three are
  // one line each here and a layout migration anywhere else.
  const sourceTransposed = options.sourceTransposed ?? false;
  // 🔴 THE ROW THE SOURCE IS READ FROM NEED NOT BE THE ROW BEING WRITTEN, and
  // AF3's grid attention is where that shows: the transposed direction reads
  // `(row % n) * n + row / n`, a pair transpose, and writes at `row`. That is
  // not `sourceTransposed`, which swaps the two AXES of a matrix; it is a
  // permutation of the row index alone. A WGSL expression in `r`; null is `r`.
  const sourceRowIndex = options.sourceRowIndex ?? null;
  // 🔴 AND THE WEIGHT'S INDEX CAN BE AN EXPRESSION TOO, WHICH IS AN INTERLEAVE
  // WITHOUT A REPACK. AF2's q/k/v/gate projection holds its four matrices
  // SEPARATELY and contiguously - `[role][k][out]` - where `outputGroup` wants
  // `[k][4 * out + role]`. AF3's triangle projection solved the same mismatch
  // by re-laying its weights at pack time; here the roles are one stride apart
  // in one buffer, so the mapping is arithmetic and the pack can stay as it is
  // - which keeps the four other shaders binding that buffer untouched.
  //
  // A WGSL expression in `$k` and `$c`, relative to `weight_offset`; null is
  // the plain row-major `$k * parameters.columns + $c`. It cannot be
  // vec4-staged, because four consecutive columns are no longer four
  // consecutive elements, and that is refused rather than made quietly wrong.
  const weightIndex = options.weightIndex ?? null;
  const sourceRow = (r) => (sourceRowIndex === null
    ? r : `(${sourceRowIndex.replaceAll("$row", r)})`);
  const bias = options.bias ?? true;
  const outputIndex = options.outputIndex ?? "row * parameters.columns + column";
  // 🔴 A PER-ROW SCALE APPLIED BEFORE THE RESIDUAL, which is what turns the
  // outer product mean's output projection into a GEMM. Its result is
  // `(bias + sum) / count` where the count depends on the PAIR - the row - so
  // without this the kernel would need an epilogue pass over the whole pair
  // track. `scaleIndex` is a WGSL expression in `row`; the binding is optional
  // and unread when it is absent, which matters because `layout: "auto"` drops
  // a binding a shader does not use.
  const scaleIndex = options.scaleIndex ?? null;
  // 🔴 AND THE SCALE HAS TO REACH THE OPERAND, NOT JUST THE RESULT, WHENEVER THE
  // OPERAND IS A SUM. The outer product mean's intermediate is a sum over the
  // WHOLE ALIGNMENT before its divide, so at 1024 sequences and 400 residues it
  // leaves f16's range and the staged copy becomes inf: the fold came back with
  // every coordinate NaN, at 512 sequences and at 200 residues it did not, and
  // that shape of threshold is always this. Scaling at the staging keeps the
  // operand O(1) whatever the depth, and the bias is scaled with it so the
  // result is the same `(bias + sum) * s` the vector kernel computes.
  //
  // `rowScaleOffset` is a WGSL expression added to the row; null is no scale.
  const rowScaleOffset = options.rowScaleOffset ?? null;
  if (rowScaleOffset !== null && scaleIndex !== null) {
    throw new RangeError("a row scale is applied at the operand OR the result, not both");
  }
  // 🔴 THE SWIGLU GATE, APPLIED AT THE OPERAND, WHICH IS WHAT REMOVES A WHOLE
  // TENSOR. A transition's second projection contracts `swish(wide[k]) *
  // wide[hidden + k]` - two columns of the widened activation, `hidden` apart.
  // Fusing that into the FIRST projection's epilogue is impossible at any block
  // width, because the two columns land in different workgroups; doing it as
  // its own elementwise pass costs a pass and a rows x hidden tensor. Doing it
  // HERE costs one extra vec4 read in a loop that already runs, because the
  // staging reads the operand element by element anyway.
  //
  // `{stride, offset}` are the widened row's stride and the distance to the
  // value half, and they are BAKED AS LITERALS rather than taken from
  // `parameters`: the shader is already generated per shape, `parameters.inner`
  // is the contracted extent (the hidden width) and not the row stride, and
  // adding two runtime fields would change the struct for every caller.
  // 🔴 A GROUP OF ADJACENT OUTPUT COLUMNS IS ONE OUTPUT CHANNEL, WHICH IS WHAT
  // A GATED PROJECTION IS. AF3's triangle projection contracts FOUR matrices
  // over one source - a, a's gate, b, b's gate - and writes
  // `mask * a * logistic(a_gate)` and the same for b, in CHANNEL-MAJOR order.
  // Fusing that needs column `4h` and column `4h+1` in the same lane's reach,
  // which is why the weights are packed interleaved for this path: the four
  // roles of one channel are four ADJACENT columns and therefore always inside
  // one subgroup's flush block, where the epilogue already reads them out of
  // workgroup memory.
  //
  // `{size, stores}`: `size` adjacent columns are one channel and each store is
  // a `[value, gate]` pair of indices within the group. Two stores means two
  // output bindings. `rowMask` adds a per-row multiplier read from its own
  // binding, which is the pair mask.
  const outputGroup = options.outputGroup ?? null;
  const rowMask = options.rowMask ?? false;
  if (outputGroup !== null) {
    if (!Number.isSafeInteger(outputGroup.size) || outputGroup.size < 2) {
      throw new RangeError("outputGroup.size wants an integer of at least two");
    }
    if (!Array.isArray(outputGroup.stores) || outputGroup.stores.length === 0) {
      throw new RangeError("outputGroup wants at least one store");
    }
    // 🔴 A STORE IS EITHER GATED OR PLAIN. `[value, gate]` writes
    // `value * logistic(gate)`, which is AF3's triangle projection; a bare
    // index writes that lane as it is, which is grid attention's q/k/v/gate -
    // four outputs off one source read, no gate applied here at all. Both are
    // the same epilogue over the same interleaved weight block.
    for (const [lane, expression] of Object.entries(outputGroup.laneBias ?? {})) {
      // 🔴 ONE LANE OF THE GROUP HAS A BIAS AND THE OTHERS DO NOT, which is
      // AF2's projection: q, k and v have none and the gate's rides in the w
      // lane. The generic `bias` adds `weights[bias_offset + column]` to EVERY
      // lane of the group, which here would add the gate's bias to the query.
      // A WGSL expression in `$channel`, per lane, read from the weight buffer.
      if (!(Number(lane) >= 0 && Number(lane) < outputGroup.size)
          || typeof expression !== "string") {
        throw new RangeError("outputGroup.laneBias maps a lane of the group to a WGSL expression");
      }
    }
    if (outputGroup.pack === true) {
      // 🔴 A PACKED OUTPUT IS TWO CHANNELS TO A WORD, AND ONE LANE HAS TO OWN
      // BOTH HALVES. Every other caller here refuses a packed target for
      // exactly that reason - this epilogue owns one channel of one row. It can
      // own two: make the GROUP twice as wide, so `size` adjacent columns are
      // TWO output channels of every role, and the lane that writes the word
      // holds both. Each store is then a PAIR of WGSL expressions, low half
      // first, and the binding is u32 rather than a float.
      if (outputGroup.size % 2 !== 0) {
        throw new RangeError("a packed output group holds two channels, so its size is even");
      }
      for (const store of outputGroup.stores) {
        if (!Array.isArray(store) || store.length !== 2
            || !store.every((e) => typeof e === "string")) {
          throw new RangeError("a packed outputGroup store is [low, high] WGSL expressions");
        }
      }
    } else
    for (const store of outputGroup.stores) {
      // 🔴 ...OR A WGSL EXPRESSION IN `v0..v{size-1}`, which is what AF2's
      // projection needs: its four lanes are the query TIMES 1/sqrt(head_dim),
      // the key and the value plain, and the gate through a logistic. Three
      // forms, one epilogue, because they are all "what to write from a group
      // of adjacent columns".
      const plain = Number.isSafeInteger(store) || typeof store === "string";
      if (!plain && !(Array.isArray(store) && store.length === 2)) {
        throw new RangeError(
          "an outputGroup store is a lane index, a WGSL expression, or a [value, gate] pair");
      }
    }
    if (residual) throw new RangeError("a grouped output does not add into its target");
    if (scaleIndex !== null) throw new RangeError("a grouped output takes no result scale");
  }
  // 🔴 A GATE COMPUTED BY A DIFFERENT GEMM, READ AT THE SAME OUTPUT CELL. AF3's
  // triangle OUTPUT projection is `logistic(z . Wg) * (x . Wz)` - two
  // contractions over two DIFFERENT sources, which is not one GEMM and cannot
  // be made into one by any column trick. Two staged sources with two
  // accumulator sets would fit the workgroup budget and then spend about 128
  // registers a lane on the accumulators; reading the FIRST pass's result in
  // the second pass's epilogue costs one global read an output cell and no
  // registers at all. The modulator is addressed by the same `outputIndex` the
  // store uses, so it is the same cell by construction.
  const outputModulate = options.outputModulate ?? false;
  // 🔴 A BATCH ON group.z, WHICH IS WHAT A TRIANGLE MULTIPLICATION IS. AF3's
  // contraction is `out[h][i][j] = sum_k a[h][i][k] * b[h][j][k]` - one n x n
  // by n x n product PER CHANNEL, 256 of them, all with the same shape and the
  // same stride. That is a batched GEMM and not a big one, so the batch index
  // is a dispatch dimension rather than 256 dispatches. The stride is in
  // ELEMENTS and is the same for all three tensors here; it is a compile-time
  // literal because the shader is generated per shape anyway.
  const batchStride = options.batchStride ?? null;
  if (batchStride !== null && !Number.isSafeInteger(batchStride)) {
    throw new RangeError("batchStride wants an element count");
  }
  // 🔴 AND THE RIGHT OPERAND CAN BE TRANSPOSED, which the OUTGOING direction
  // needs and the incoming one does not. `sourceTransposed` has existed for the
  // left operand since the outer product mean; this is its mirror, and the two
  // directions of one op need one each.
  // 🔴 STAGING THE RIGHT OPERAND BUYS NOTHING WHEN `subgroupRows` IS ONE, AND
  // IT IS TWO THIRDS OF THE STAGING. A workgroup stages BM x BK of the source
  // and BK x BN of the weights; at 64x128x16 that is 1024 halves against 2048.
  // The source is shared - with SGY = 1 every subgroup has sg_row 0 and reads
  // the same rows, so one staged panel feeds eight subgroups. The WEIGHTS are
  // not: subgroup `sg` owns columns `sg * BN/SGX ..`, which no other subgroup
  // touches, so the staged copy is read exactly once and the workgroup memory
  // is a detour. `subgroupMatrixLoad` takes a storage pointer, so the tile can
  // come straight out of the weight buffer at stride `parameters.columns` -
  // the same layout the staged panel has, at a different stride.
  //
  // What it costs: `tools/gpu/probe-staged-gemm-parts.js` prices the staging
  // loop at 3.10 ms of a 4.58 ms kernel and the matrix issue at 1.69.
  //
  // 🔴 IT NEEDS `inner % blockInner == 0` AND THE CALLER IS WHAT KNOWS THAT.
  // The weight buffer is bound as [weights | bias], so a tail panel reading
  // past the inner extent picks the bias up as a weight row - which is the
  // trap the zero-padded staging loop exists to avoid, and there is no padding
  // in a direct read. Every caller that sets this contracts a channel count,
  // which is a multiple of sixteen; the outer product mean's K is an alignment
  // depth and the triangle contraction's is a protein length, and neither may.
  const directWeights = options.directWeights ?? false;
  // 🔴 STAGE THE NEXT PANEL'S GLOBAL READS BEFORE THIS PANEL'S MULTIPLIES.
  // Without it every panel is barrier, read global, barrier, multiply - so the
  // global latency is exposed once per panel and covered only by whatever other
  // workgroups the SM happens to hold. The reads go into REGISTERS first, which
  // costs BM * BK / threads of them - four halves a lane at 64x128x16 with 256
  // threads, one vec4 - and the workgroup write moves after the multiply.
  const prefetch = options.prefetch ?? false;
  const weightsTransposed = options.weightsTransposed ?? false;
  if (directWeights) {
    if (weightsTransposed) throw new RangeError("a transposed right operand is staged, not direct");
    if (weightPrecision !== (options.matrixElement ?? "f16")) {
      throw new RangeError("a direct right operand is read as the matrix element, so the buffer must hold it");
    }
  }
  if (weightsTransposed && vectorWeights) {
    // A transposed read walks k, which is the slow axis of the stored operand,
    // so four consecutive elements of the vec4 are `columns` apart. Refused
    // rather than made quietly wrong.
    throw new RangeError("a transposed right operand cannot be vector-staged");
  }
  if (sourceTransposed && vectorSource) {
    throw new RangeError("a transposed left operand cannot be vector-staged");
  }
  // 🔴 A GATE IN ITS OWN BUFFER, AT THE SAME INDEX AS THE SOURCE. `sourceGate`
  // above is the SwiGLU's - two halves of one widened row, a fixed distance
  // apart. AF3's grid attention wants the other shape: `gathered[i] *
  // logistic(gate[i])`, where the gate is a separate tensor the projection
  // kernel binds. Applying it at the staging costs one extra read in a loop
  // that already runs and removes an elementwise pass over `rows x width`.
  const sourceModulate = options.sourceModulate ?? false;
  const sourceModulatePrecision = options.sourceModulatePrecision ?? "f32";
  const sourceGate = options.sourceGate ?? null;
  if (sourceGate !== null) {
    for (const key of ["stride", "offset"]) {
      if (!Number.isSafeInteger(sourceGate[key]) || sourceGate[key] <= 0) {
        throw new RangeError(`sourceGate.${key} wants a positive integer`);
      }
    }
    if (sourceTransposed) throw new RangeError("a gated operand is not transposed here");
    if (rowScaleOffset !== null) throw new RangeError("a gated operand takes no row scale");
    // A vec4 read at element i returns i & ~3 .. i & ~3 + 3, so an odd stride
    // or offset silently shifts the value half against the gate half.
    if (vectorStaging && (sourceGate.stride % 4 !== 0 || sourceGate.offset % 4 !== 0)) {
      throw new RangeError("vector staging needs the gate's stride and offset divisible by 4");
    }
  }
  const scaled = (value, row) => (rowScaleOffset === null
    ? value : `(${value}) * scale[${rowScaleOffset} + ${row}]`);

  const subgroups = SGY * SGX;
  const threads = subgroups * lanes;
  const rowsPerSubgroup = BM / SGY;
  const columnsPerSubgroup = BN / SGX;
  if (BM % (SGY * M) !== 0) throw new RangeError("blockRows must divide by subgroupRows * tile.M");
  if (BN % (SGX * N) !== 0) throw new RangeError("blockColumns must divide by subgroupColumns * tile.N");
  if (BK % K !== 0) throw new RangeError("blockInner must divide by tile.K");
  if (vectorStaging && (BK % 4 !== 0 || BN % 4 !== 0)) {
    throw new RangeError("vector staging needs blockInner and blockColumns divisible by four");
  }
  const accRows = rowsPerSubgroup / M;
  const accColumns = columnsPerSubgroup / N;
  const flushWidth = accColumns * N;
  // ...checked here rather than beside the option, because the flush width is
  // what a group has to fit inside and it is derived from the geometry.
  if (outputGroup !== null && flushWidth % outputGroup.size !== 0) {
    throw new RangeError(
      `an output group of ${outputGroup.size} does not divide the flush width ${flushWidth}`);
  }

  const stagedSource = BM * BK;
  const stagedWeights = BK * BN;
  const scratch = subgroups * M * flushWidth;

  // The half the units multiply in. The buffers are whatever the caller holds.
  // 🔴 THE PANEL'S ELEMENT IS THE DEVICE'S, NOT ALWAYS f16. This A100 offers no
  // f32 component type so f16 was the only choice and got written down as one;
  // an M2 offers BOTH at 8x8x8, and upstream measures the f32 tiles as EXACT -
  // the same error as the scalar kernel to the digit - where its f16 config
  // accumulates in f16. So a device that has f32 units should be able to ask
  // for them, and hardcoding the half made that untestable.
  const half = options.matrixElement ?? "f16";
  if (half !== "f16" && half !== "f32") {
    throw new RangeError(`unknown matrix element ${half}`);
  }
  const sourceElement = vectorSource ? `vec4<${sourcePrecision}>` : sourcePrecision;
  // A direct read needs the binding to BE the matrix element, so it cannot also
  // be a vec4 - and it has no staging loop left to vectorise.
  if (weightIndex !== null && vectorWeights) {
    throw new RangeError("an indexed right operand cannot be vector-staged");
  }
  if (weightIndex !== null && directWeights) {
    throw new RangeError("an indexed right operand is staged, not read directly");
  }
  const weightAt = (k, c) => (weightIndex === null
    ? `${k} * parameters.columns + ${c}`
    : `(${weightIndex.replaceAll("$k", `(${k})`).replaceAll("$c", `(${c})`)})`);
  const weightVector = vectorWeights && !directWeights;
  const weightElement = weightVector ? `vec4<${weightPrecision}>` : weightPrecision;

  const declarations = [];
  const stores = [];
  // ...the bias lives in the WEIGHT buffer, so its read follows that binding.
  const biasRead = weightVector
    ? "weights[(parameters.bias_offset + column) / 4u][(parameters.bias_offset + column) % 4u]"
    : "weights[parameters.bias_offset + column]";

  for (let m = 0; m < accRows; m += 1) {
    const block = [];
    for (let n = 0; n < accColumns; n += 1) {
      declarations.push(`  var acc_${m}_${n} = subgroup_matrix_result<${result}, ${N}, ${M}>();`);
      block.push(`  subgroupMatrixStore(&scratch, sg * ${M * flushWidth}u + ${n * N}u,`
        + ` acc_${m}_${n}, false, ${flushWidth}u);`);
    }
    if (outputGroup !== null) {
    stores.push(`${block.join("\n")}
    workgroupBarrier();
    for (var g = lane; g < ${M * flushWidth / (outputGroup?.size ?? 1)}u; g += ${lanes}u) {
      let i = g * ${outputGroup.size}u;
      let row = row_origin + sg_row + ${m * M}u + i / ${flushWidth}u;
      let column = column_origin + sg_column + i % ${flushWidth}u;
      if (row >= parameters.rows || column >= parameters.columns) { continue; }
      // The channel this group of ${outputGroup.size} columns is.
      let channel = column / ${outputGroup.size}u;
  ${Array.from({ length: outputGroup.size }, (_, r) => `    let v${r} = f32(scratch[sg * ${M * flushWidth}u + i + ${r}u])`
      + (outputGroup.laneBias?.[r] === undefined ? ""
        : ` + f32(weights[${outputGroup.laneBias[r].replaceAll("$channel", "channel")}])`)
      + (bias ? ` + f32(${weightVector
      ? `weights[(parameters.bias_offset + column + ${r}u) / 4u][(parameters.bias_offset + column + ${r}u) % 4u]`
      : `weights[parameters.bias_offset + column + ${r}u]`})` : "") + ";").join("\n")}
  ${rowMask ? "    let row_mask = mask[row];\n" : ""}    let at = ${outputIndex};
  ${outputGroup.stores.map((store, index) => (outputGroup.pack === true
      ? `    ${index === 0 ? "output" : `output${index + 1}`}[at] = `
        + `pack2x16float(vec2<f32>(${store[0]}, ${store[1]}));`
      : `    ${index === 0 ? "output" : `output${index + 1}`}[at] = `
      + `${outputPrecision}(${rowMask ? "row_mask * " : ""}${typeof store === "string"
        ? `(${store})`
        : Number.isSafeInteger(store)
          ? `v${store}`
          : `v${store[0]} * (1.0 / (1.0 + exp(-v${store[1]})))`});`)).join("\n")}
    }
    workgroupBarrier();`);
    } else {
    stores.push(`${block.join("\n")}
    workgroupBarrier();
    for (var i = lane; i < ${M * flushWidth}u; i += ${lanes}u) {
      let row = row_origin + sg_row + ${m * M}u + i / ${flushWidth}u;
      let column = column_origin + sg_column + i % ${flushWidth}u;
      if (row >= parameters.rows || column >= parameters.columns) { continue; }
      var value = f32(scratch[sg * ${M * flushWidth}u + i]);
  ${bias ? `    value += f32(${scaled(biasRead, "row")});\n` : ""}    if (parameters.activation == 1u) { value = max(value, 0.0); }
  ${scaleIndex === null ? "" : `    value *= scale[${scaleIndex}];\n`}    let at = ${outputIndex};
  ${outputModulate ? "    value *= 1.0 / (1.0 + exp(-f32(modulator[at])));\n" : ""}${residual ? `    value += f32(output[at]);\n` : ""}    output[at] = ${outputPrecision}(value);
    }
    workgroupBarrier();`);
    }
  }

  // The inner k step: each subgroup reads its own rows and columns out of the
  // staged panel. Both offsets are subgroup-derived, which is the point.
  const inner = [];
  for (let m = 0; m < accRows; m += 1) {
    inner.push(`      let left_${m} = subgroupMatrixLoad<subgroup_matrix_left<${half}, ${K}, ${M}>>(`
      + `&staged_source, (sg_row + ${m * M}u) * ${BK}u + kk, false, ${BK}u);`);
  }
  for (let n = 0; n < accColumns; n += 1) {
    inner.push(directWeights
      ? `      let right_${n} = subgroupMatrixLoad<subgroup_matrix_right<${half}, ${N}, ${K}>>(`
        + `&weights, parameters.weight_offset + ${batchStride === null ? "" : "batch_base + "}`
        + `(k0 + kk) * parameters.columns + column_origin + sg_column + ${n * N}u,`
        + ` false, parameters.columns);`
      : `      let right_${n} = subgroupMatrixLoad<subgroup_matrix_right<${half}, ${N}, ${K}>>(`
        + `&staged_weights, kk * ${BN}u + sg_column + ${n * N}u, false, ${BN}u);`);
  }
  for (let m = 0; m < accRows; m += 1) {
    for (let n = 0; n < accColumns; n += 1) {
      inner.push(`      acc_${m}_${n} = subgroupMatrixMultiplyAccumulate(left_${m}, right_${n}, acc_${m}_${n});`);
    }
  }

  // Consecutive threads take consecutive columns, which is the coalesced
  // direction in both tensors, and with vector staging each takes four.
  // Consecutive threads take consecutive columns, which is the coalesced
  // direction in both tensors, and with vector staging each takes four.
  // 🔴 THE TWO OPERANDS CHOOSE SEPARATELY - see vectorSource/vectorWeights.
  // Each loop's body is generated once and its DESTINATION varies: straight
  // into workgroup memory, or into a register array the next barrier writes
  // out. `i / threads` is the iteration number, because i starts at tid < threads
  // and steps by threads - so the register index needs no separate counter.
  const sourceCount = vectorSource ? stagedSource / 4 : stagedSource;
  const weightCount = vectorWeights ? stagedWeights / 4 : stagedWeights;
  const sourceIter = Math.ceil(sourceCount / threads);
  const weightIter = Math.ceil(weightCount / threads);
  // `logistic(gate[i]) *` where the source is modulated, nothing otherwise.
  const modulated = (value, index, wide) => (!sourceModulate ? value
    : wide
      ? `(${value}) * (${wide}(1.0) / (${wide}(1.0)`
        + ` + exp(-${wide}(source_gate[${index}]))))`
      : `(${value}) * (1.0 / (1.0 + exp(-f32(source_gate[${index}]))))`);
  const sourceValue = vectorSource
    ? `${sourceGate === null
        ? scaled(`select(vec4<f32>(0.0),
        ${modulated(`vec4<f32>(source[(${batchStride === null ? "" : "batch_base + "}${sourceRow("r")} * parameters.inner + k) / 4u])`,
          `(${sourceRow("r")} * parameters.inner + k) / 4u`, "vec4<f32>")}, k < parameters.inner && r < parameters.rows)`,
        "min(r, parameters.rows - 1u)")
        : `select(vec4<f32>(0.0), swish_gate4(
        vec4<f32>(source[(r * ${sourceGate.stride}u + k) / 4u]),
        vec4<f32>(source[(r * ${sourceGate.stride}u + ${sourceGate.offset}u + k) / 4u])),
        k < parameters.inner && r < parameters.rows)`}`
    : `select(${half}(0),
        ${half}(${sourceGate === null
          ? scaled(modulated(
            `f32(source[${batchStride === null ? "" : "batch_base + "}${sourceTransposed
              ? `k * parameters.rows + ${sourceRow("r")}` : `${sourceRow("r")} * parameters.inner + k`}])`,
            sourceTransposed
              ? `k * parameters.rows + ${sourceRow("r")}` : `${sourceRow("r")} * parameters.inner + k`),
          "min(r, parameters.rows - 1u)")
          : `swish_gate(f32(source[r * ${sourceGate.stride}u + k]),
             f32(source[r * ${sourceGate.stride}u + ${sourceGate.offset}u + k]))`}),
        k < parameters.inner && r < parameters.rows)`;
  const weightValue = vectorWeights
    ? `select(vec4<${weightPrecision}>(0),
        weights[(parameters.weight_offset + ${batchStride === null ? "" : "batch_base + "}k * parameters.columns + c) / 4u],
        k < parameters.inner && c < parameters.columns)`
    : `select(${half}(0),
        ${half}(weights[parameters.weight_offset + ${batchStride === null ? "" : "batch_base + "}${
          weightsTransposed ? "c * parameters.inner + k" : weightAt("k", "c")}]),
        k < parameters.inner && c < parameters.columns)`;

  // One staged element (or vec4), from global memory into `target` - a named
  // register when it is given, workgroup memory when it is null.
  const sourceBody = (panel, target) => {
    const head = `      let k = ${panel}${vectorSource ? ` + (i % ${BK / 4}u) * 4u` : ` + i % ${BK}u`};
      let r = row_origin + i / ${vectorSource ? BK / 4 : BK}u;
      let v = ${sourceValue};`;
    if (target !== null) {
      return `${head}\n      ${target} = ${vectorSource
        ? `vec4<${half}>(${half}(v.x), ${half}(v.y), ${half}(v.z), ${half}(v.w))` : "v"};`;
    }
    return `${head}\n${vectorSource ? `      let at = i * 4u;
      staged_source[at] = ${half}(v.x);
      staged_source[at + 1u] = ${half}(v.y);
      staged_source[at + 2u] = ${half}(v.z);
      staged_source[at + 3u] = ${half}(v.w);` : "      staged_source[i] = v;"}`;
  };
  const weightBody = (panel, target) => {
    const head = `      let k = ${panel} + i / ${vectorWeights ? BN / 4 : BN}u;
      let c = column_origin${vectorWeights ? ` + (i % ${BN / 4}u) * 4u` : ` + i % ${BN}u`};
      let v = ${weightValue};`;
    if (target !== null) {
      return `${head}\n      ${target} = ${vectorWeights
        ? `vec4<${half}>(${half}(v.x), ${half}(v.y), ${half}(v.z), ${half}(v.w))` : "v"};`;
    }
    return `${head}\n${vectorWeights ? `      let at = i * 4u;
      staged_weights[at] = ${half}(v.x);
      staged_weights[at + 1u] = ${half}(v.y);
      staged_weights[at + 2u] = ${half}(v.z);
      staged_weights[at + 3u] = ${half}(v.w);` : "      staged_weights[i] = v;"}`;
  };
  // `into` is "workgroup" or "registers"; `panel` is the WGSL expression for
  // the panel's first k.
  //
  // 🔴 THE REGISTER FORM IS UNROLLED AND THE REASON IS THIS FILE'S OWN TRAP. A
  // held panel written as `array<vec4<f16>, n>` indexed by `i / threads` is a
  // DYNAMIC index into an array, so WGSL puts it in addressable memory instead
  // of registers - the same fault CLAUDE.md records for `unpack2x16float(w)[at
  // & 1u]`, and it measured the same way: prefetching into an array took the
  // kernel 3.190 -> 4.202 ms, worse than not prefetching at all. Unrolled into
  // named variables it is registers, which is the whole point of holding it.
  const each = (count, body) => Array.from({ length: Math.ceil(count / threads) },
    (_, it) => `    {
      let i = tid + ${it * threads}u;
${count % threads === 0 && it < Math.floor(count / threads)
      ? body(it) : `      if (i < ${count}u) {\n${body(it)}\n      }`}
    }`).join("\n");
  const sourceLoop = (into, panel) => (into === "workgroup"
    ? `    for (var i = tid; i < ${sourceCount}u; i += ${threads}u) {
${sourceBody(panel, null)}
    }`
    : each(sourceCount, (it) => sourceBody(panel, `src_r${it}`)));
  const sourceFlush = () => each(sourceCount, (it) => (vectorSource
    ? `        let at${it} = i * 4u;
        staged_source[at${it}] = src_r${it}.x;
        staged_source[at${it} + 1u] = src_r${it}.y;
        staged_source[at${it} + 2u] = src_r${it}.z;
        staged_source[at${it} + 3u] = src_r${it}.w;`
    : `        staged_source[i] = src_r${it};`));
  const weightLoop = (into, panel) => (directWeights ? "" : (into === "workgroup"
    ? `\n    for (var i = tid; i < ${weightCount}u; i += ${threads}u) {
${weightBody(panel, null)}
    }`
    : `\n${each(weightCount, (it) => weightBody(panel, `w_r${it}`))}`));
  const weightFlush = () => (directWeights ? "" : `\n${each(weightCount, (it) => (vectorWeights
    ? `        let bt${it} = i * 4u;
        staged_weights[bt${it}] = w_r${it}.x;
        staged_weights[bt${it} + 1u] = w_r${it}.y;
        staged_weights[bt${it} + 2u] = w_r${it}.z;
        staged_weights[bt${it} + 3u] = w_r${it}.w;`
    : `        staged_weights[i] = w_r${it};`))}`);
  const registerDeclarations = !prefetch ? "" : [
    ...Array.from({ length: sourceIter }, (_, it) =>
      `  var src_r${it}: ${vectorSource ? `vec4<${half}>` : half};`),
    ...(directWeights ? [] : Array.from({ length: weightIter }, (_, it) =>
      `  var w_r${it}: ${vectorWeights ? `vec4<${half}>` : half};`)),
  ].join("\n");

  const stagingLoops = sourceLoop("workgroup", "k0") + weightLoop("workgroup", "k0");

  const usesHalf = [half, result, sourcePrecision, weightPrecision, outputPrecision]
    .includes("f16");
  return `${usesHalf ? "enable f16;\n" : ""}enable chromium_experimental_subgroup_matrix;
enable subgroups;
struct MatmulParameters {
  rows: u32,
  inner: u32,
  columns: u32,
  weight_offset: u32,
  bias_offset: u32,
  activation: u32,
  padding: vec2<u32>,
};
@group(0) @binding(0) var<storage, read> source: array<${sourceElement}>;
@group(0) @binding(1) var<storage, read> weights: array<${weightElement}>;
@group(0) @binding(2) var<uniform> parameters: MatmulParameters;
@group(0) @binding(3) var<storage, read_write> output: array<${outputPrecision}>;
${scaleIndex === null && rowScaleOffset === null
  ? "" : "@group(0) @binding(4) var<storage, read> scale: array<f32>;"}
${(() => {
  // 🔴 THE OPTIONAL BINDINGS ARE CONSECUTIVE, WITH NO GAP. `layout: "auto"`
  // would accept a hole at 4, but every caller in this repository binds by
  // mapping an ARRAY over its index - so a hole means the caller has to build a
  // sparse entry list, and the one that forgets binds the mask as the output.
  // The order is fixed and documented: scale, mask, then the extra outputs.
  let binding = 4;
  const lines = [];
  if (scaleIndex !== null || rowScaleOffset !== null) binding += 1;
  if (rowMask) {
    lines.push("// The pair mask, one float a ROW, multiplied into every gated store.");
    lines.push(`@group(0) @binding(${binding}) var<storage, read> mask: array<f32>;`);
    binding += 1;
  }
  if (outputModulate) {
    lines.push("// Another GEMM's result at the same cell, through a logistic.");
    lines.push(`@group(0) @binding(${binding}) var<storage, read> `
      + `modulator: array<${options.modulatePrecision ?? "f32"}>;`);
    binding += 1;
  }
  if (sourceModulate) {
    lines.push("// A gate at the SOURCE's own index, through a logistic - see sourceModulate.");
    lines.push(`@group(0) @binding(${binding}) var<storage, read> `
      + `source_gate: array<${vectorSource
        ? `vec4<${sourceModulatePrecision}>` : sourceModulatePrecision}>;`);
    binding += 1;
  }
  for (let index = 1; index < (outputGroup?.stores ?? []).length; index += 1) {
    lines.push(`@group(0) @binding(${binding}) var<storage, read_write> `
      + `output${index + 1}: array<${outputPrecision}>;`);
    binding += 1;
  }
  return lines.join("\n");
})()}

${sourceGate === null ? "" : `// swish(gate) * value, the transition's SwiGLU, computed where the operand is
// staged. In f32 whatever the buffer holds: the gate half leaves f16's range on
// a wide activation and the multiply is one instruction either way.
fn swish_gate(gate: f32, value: f32) -> f32 {
  return (gate / (1.0 + exp(-gate))) * value;
}
fn swish_gate4(gate: vec4<f32>, value: vec4<f32>) -> vec4<f32> {
  return (gate / (vec4<f32>(1.0) + exp(-gate))) * value;
}
`}
var<workgroup> staged_source: array<${half}, ${stagedSource}>;
${directWeights ? "// The right operand is read straight out of the weight buffer - see directWeights."
  : `var<workgroup> staged_weights: array<${half}, ${stagedWeights}>;`}
// One ${M}-row block per subgroup, which is all the result staging this needs.
var<workgroup> scratch: array<${result}, ${scratch}>;

@compute @workgroup_size(${threads})
fn main(
  @builtin(local_invocation_index) tid: u32,
  @builtin(subgroup_id) sg: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let lane = tid % ${lanes}u;
  // No slide-back: the edges are padded instead, so these are plain origins
  // and a partial region is normal rather than forbidden.
  let row_origin = group.y * ${BM}u;
  let column_origin = group.x * ${BN}u;
${batchStride === null ? "" : `  // The batch this workgroup belongs to, one channel of a triangle product.
  let batch_base = group.z * ${batchStride}u;`}
  let sg_row = (sg / ${SGX}u) * ${rowsPerSubgroup}u;
  let sg_column = (sg % ${SGX}u) * ${columnsPerSubgroup}u;

${declarations.join("\n")}

  // 🔴 THE LAST PANEL IS ZERO-PADDED RATHER THAN HANDLED SEPARATELY. A tail
  // loop that did read-modify-write on the output would DOUBLE-ADD wherever
  // the slide-back makes two regions overlap, and would apply the activation
  // twice. Staging a zero for k past the end costs one compare per staged
  // element and makes the ragged case the same code as the whole one.
  //
  // It also keeps k0 from ever reaching the BIAS: the weights are bound as
  // [weight | bias], and a panel reading past the inner extent would otherwise
  // pick the bias up as if it were a weight row.
  let panels = (parameters.inner + ${BK}u - 1u) / ${BK}u;
${prefetch ? `${registerDeclarations}
  // The first panel, then one panel ahead of the multiplies from there on.
${sourceLoop("registers", "0u")}${weightLoop("registers", "0u")}
  workgroupBarrier();
${sourceFlush()}${weightFlush()}
  workgroupBarrier();
  for (var p = 0u; p < panels; p += 1u) {
    let k0 = p * ${BK}u;
    // 🔴 THE NEXT PANEL'S GLOBAL READS ARE ISSUED BEFORE THIS PANEL'S
    // MULTIPLIES, so their latency is covered by work this workgroup has rather
    // than by whatever else the SM happens to hold.
    if (p + 1u < panels) {
${sourceLoop("registers", `(p + 1u) * ${BK}u`)}${weightLoop("registers", `(p + 1u) * ${BK}u`)}
    }
    for (var kk = 0u; kk < ${BK}u; kk += ${K}u) {
${inner.join("\n")}
    }
    workgroupBarrier();
    if (p + 1u < panels) {
${sourceFlush()}${weightFlush()}
    }
    workgroupBarrier();
  }` : `  for (var p = 0u; p < panels; p += 1u) {
    let k0 = p * ${BK}u;
    workgroupBarrier();
${stagingLoops}
    workgroupBarrier();
    for (var kk = 0u; kk < ${BK}u; kk += ${K}u) {
${inner.join("\n")}
    }
  }`}
  workgroupBarrier();
${stores.join("\n")}
}`;
}
