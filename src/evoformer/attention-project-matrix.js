/**
 * AF2's q/k/v/gate projection on the SUBGROUP MATRIX UNITS.
 *
 * 🔴 IT IS 20% OF AN EVOFORMER BLOCK AND THE LARGEST THING IN IT STILL ON THE
 * VECTOR PATH. Profiled per dispatch at 400 residues and 512 sequences, a block
 * is 78.01 ms: the four flash attentions are 18.75 of it and already on the
 * units, the transitions 14.81 and the outer product mean 10.75 likewise - and
 * `msa-row-attention.project` plus `msa-column-attention.project` are 12.73,
 * fused vector kernels at about 16.9 TFLOP/s where a staged GEMM now reaches
 * 39. As one packed GEMM of `rows x channels -> 4 * channels`,
 * probe-staged-gemm-parts.js prices the same work at 3.659 ms against 6.364.
 *
 * 🔴 AND THE FOUR MATRICES ARE NOT INTERLEAVED IN MEMORY, WHICH IS WHY THIS IS
 * AN INDEX AND NOT A REPACK. `packAttentionWeights` lays them out separately
 * and contiguously - query, key, value, gating, one stride apart - where
 * `outputGroup` wants the four roles of one output channel adjacent.
 * AF3's triangle projection solved that mismatch by re-laying its weights at
 * pack time; this buffer is bound by five shaders (the normalisation, this,
 * the pair bias, the output projection and the multimer's global attention),
 * so moving it is a five-shader change with a silent failure mode. The roles
 * are one stride apart, so `weightIndex` expresses the interleave
 * arithmetically and the pack stays as it is.
 *
 * 🔴 THE EPILOGUE IS FOUR DIFFERENT STORES OVER ONE GROUP. The query carries
 * the 1/sqrt(head_dim) scale, the key and value are plain, and the gate is a
 * logistic of its accumulator PLUS the only bias any of the four has - which
 * is `laneBias`, because the generic bias would add the gate's to the query.
 *
 * 🔴 AND IT REFUSES A PACKED OUTPUT, for the reason AF3's two projections do:
 * the vector kernel gives one lane a PAIR of adjacent channels so it owns the
 * whole word they share, and this epilogue owns one channel of one row.
 */
import {
  createStagedMatrixShader, stagedMatrixStorage,
} from "../runtime/matrix-linear.js";
import { deviceMatrixConfig, deviceTuning } from "../runtime/device-profile.js";

/** The block, from the profile; the same one the other staged GEMMs share. */
export const ATTENTION_PROJECT_MATRIX_GEOMETRY = {
  blockRows: 64, blockColumns: 128, blockInner: 16,
  subgroupRows: 1, subgroupColumns: 8,
};

/**
 * @param {{channels: number, heads: number}} shape `projected` is `channels`,
 *   which is what src/evoformer/attention.js asserts.
 * @param {{source?: "f32"|"f16", weight?: "f32"|"f16", output?: "f32"|"f16"}} storage
 * @param {object} [matrix] the device's tile and result type, plus a geometry.
 */
export function createAttentionProjectMatrixShader(shape, storage = {}, matrix = {}) {
  const { channels, heads } = shape;
  if (!(channels > 0) || !(heads > 0) || channels % heads !== 0) {
    throw new RangeError("attention project matrix wants channels divisible by heads");
  }
  const output = storage.output ?? "f32";
  if (output !== "f32" && output !== "f16") {
    throw new RangeError(`unknown attention project matrix output storage ${output}`);
  }
  // 🔴 A PACKED TARGET IS TWO CHANNELS TO A WORD, AND THAT DOUBLES THE GROUP.
  // The matrix flash attention reads q, k, v and the gate PACKED, so refusing
  // a packed output here would mean either keeping the vector projection or
  // doubling four of the largest tensors a block holds - 210 MiB more at 400
  // residues and 512 sequences, and twice the reads for the kernel after this
  // one. Instead the group is EIGHT columns: two output channels of all four
  // roles, so the lane that writes a word owns both of its halves. That is the
  // ownership rule the vector kernel states, arrived at from the other side.
  const packed = output === "f16";
  const projected = channels;
  if (packed && projected % 2 !== 0) {
    throw new RangeError("a packed projection needs an even output width");
  }
  const stride = channels * projected;
  const headDim = channels / heads;
  const geometry = { ...ATTENTION_PROJECT_MATRIX_GEOMETRY, ...matrix };
  return createStagedMatrixShader({
    ...geometry,
    // The source is row-major `[row][channel]`, so a vec4 read needs the
    // channel count divisible by four; every AF2 width is. The WEIGHTS cannot
    // be vector-staged at all here - see weightIndex.
    vectorStaging: { source: channels % 4 === 0 },
    sourcePrecision: storage.source ?? "f32",
    weightPrecision: storage.weight ?? "f32",
    outputPrecision: packed ? "u32" : output,
    bias: false,
    // role = column % 4 in the order the pack writes them: query, key, value,
    // gating. `weight_offset` is the QUERY matrix's base.
    weightIndex: `($c % 4u) * ${stride}u + ($k) * ${projected}u + ($c) / 4u`,
    // 🔴 THE SAME FOUR EXPRESSIONS THE VECTOR KERNEL STORES, and the scale is
    // a literal because `head_dim` is compile-time here where the vector kernel
    // reads it from its uniform. The gating bias is the only one of the four
    // that has one, which is `laneBias`: the generic `bias` would add it to the
    // query as well.
    outputGroup: packed ? {
      size: 8, pack: true,
      stores: [
        [`v0 * ${(1 / Math.sqrt(headDim)).toPrecision(9)}`,
         `v4 * ${(1 / Math.sqrt(headDim)).toPrecision(9)}`],
        ["v1", "v5"],
        ["v2", "v6"],
        ["1.0 / (1.0 + exp(-v3))", "1.0 / (1.0 + exp(-v7))"],
      ],
      laneBias: {
        3: "parameters.bias_offset + $channel * 2u",
        7: "parameters.bias_offset + $channel * 2u + 1u",
      },
    } : {
      size: 4,
      stores: [
        `v0 * ${(1 / Math.sqrt(headDim)).toPrecision(9)}`,
        "v1",
        "v2",
        "1.0 / (1.0 + exp(-v3))",
      ],
      laneBias: { 3: "parameters.bias_offset + $channel" },
    },
    // ...a word per pair of channels when packed, which is the index the
    // vector kernel writes: `row * (projected / 2) + hd / 2`.
    outputIndex: packed
      ? `row * ${projected / 2}u + channel` : `row * ${projected}u + channel`,
  });
}

/** Whether this device can stage the geometry at all. */
export function attentionProjectMatrixFits(matrix = {}, limitBytes = 49152) {
  return stagedMatrixStorage({ ...ATTENTION_PROJECT_MATRIX_GEOMETRY, ...matrix }) <= limitBytes;
}

/**
 * The dispatch, and the bindings in order. The tile travels with the shader.
 */
export function attentionProjectMatrixDispatch(shape, matrix = {}) {
  const geometry = { ...ATTENTION_PROJECT_MATRIX_GEOMETRY, ...matrix };
  return {
    x: Math.ceil(4 * shape.channels / geometry.blockColumns),
    y: Math.ceil(shape.rows / geometry.blockRows),
    bindings: ["source", "weights", "parameters", "query", "key", "value", "gate"],
  };
}

/** This device's answer for this kernel: a geometry, or false. */
export function attentionProjectMatrixConfig(device) {
  const tuning = deviceTuning(device);
  if (tuning.attentionProjectMatrix !== true) return false;
  const config = deviceMatrixConfig(device, { element: "f16" });
  if (config === null) return false;
  return {
    result: tuning.stagedMatrixResult ?? config.resultComponentType,
    matrixElement: config.componentType,
    tile: { M: config.M, N: config.N, K: config.K },
    prefetch: tuning.stagedMatrixPrefetch === true,
  };
}

/**
 * ...and the GATED OUTPUT projection, which is the same operation one kernel
 * later: `weighted @ outputWeight + outputBias`, optionally added into the
 * block's activation.
 *
 * 🔴 IT IS 8% OF A BLOCK AND THE PLAINEST GEMM IN AF2. Two of them are 5.36 ms
 * of a 67.29 ms block at 400 residues, and `probe-staged-gemm-parts.js` at that
 * shape - 204800 x 256 -> 256 - prices the staged kernel at 0.962 ms against
 * the vector one's 2.673. It needs no interleave, no group and no gate; the
 * only thing it needs that a plain projection does not is the store.
 *
 * 🔴 THE COLUMN ATTENTION WRITES ITS RESULT BACK TRANSPOSED, which is why this
 * could never be the transition's linear kernel. `output_row` is
 * `q * batch + b` where the input row is `b * queries + q`, and `queries` and
 * `batch` are runtime values - so they ride in MatmulParameters' two spare
 * words, the same place the outer product mean puts its pair base.
 */
export function createAttentionOutputMatrixShader(shape, storage = {}, matrix = {}, residual = false) {
  const { channels, transpose } = shape;
  if (!(channels > 0)) throw new RangeError("attention output matrix wants positive channels");
  const geometry = { ...ATTENTION_PROJECT_MATRIX_GEOMETRY, ...matrix };
  return createStagedMatrixShader({
    ...geometry,
    vectorStaging: { source: channels % 4 === 0, weights: channels % 4 === 0 },
    sourcePrecision: storage.source ?? "f32",
    weightPrecision: storage.weight ?? "f32",
    outputPrecision: "f32",
    bias: true,
    residual,
    // padding.x is `queries` and padding.y is `batch`.
    outputIndex: transpose
      ? "((row % parameters.padding.x) * parameters.padding.y"
        + " + row / parameters.padding.x) * parameters.columns + column"
      : "row * parameters.columns + column",
  });
}

/** The dispatch, and the bindings in order. */
export function attentionOutputMatrixDispatch(shape, matrix = {}) {
  const geometry = { ...ATTENTION_PROJECT_MATRIX_GEOMETRY, ...matrix };
  return {
    x: Math.ceil(shape.channels / geometry.blockColumns),
    y: Math.ceil(shape.rows / geometry.blockRows),
    bindings: ["weighted", "weights", "parameters", "output"],
  };
}
