import { concatenateAs, writeInto } from "../runtime/float16.js";

const ORDER = [
  "layerNormInWeight", "layerNormInBias",
  "linearAPWeight", "linearAPBias", "linearAGWeight", "linearAGBias",
  "linearBPWeight", "linearBPBias", "linearBGWeight", "linearBGBias",
  "layerNormOutWeight", "layerNormOutBias",
  "linearZWeight", "linearZBias", "linearGWeight", "linearGBias",
];

/**
 * The four projection matrices as ONE interleaved K x N block, for the matrix
 * path.
 *
 * 🔴 THE ROLES OF A CHANNEL HAVE TO BE ADJACENT COLUMNS, WHICH IS WHY THIS
 * EXISTS. `projectAB` contracts a, a's gate, b and b's gate over one source and
 * gates them pairwise in its epilogue. A staged matrix GEMM's epilogue can only
 * reach the columns inside its own subgroup's flush block, so the four roles of
 * channel `h` are written as columns `4h..4h+3`: always inside one block,
 * whatever the geometry, because four divides every flush width this uses.
 *
 * 🔴 AND IT ALSO TRANSPOSES. The four matrices are stored `[h][c]` - output
 * channel major, which is what the vector kernel indexes - and a staged matmul
 * reads `weights[k * columns + n]`. So this is a transpose and an interleave in
 * one pass, done once at pack time rather than per workgroup.
 *
 * 🔴 IT REPLACES THE FOUR RATHER THAN JOINING THEM, so a bundle costs no extra
 * bytes: the element count is the same 4 * cH * cZ either way and only one of
 * the two kernels is ever compiled for a given block.
 */
export const AB_INTERLEAVED = ["linearABWeight", "linearABBias"];

/**
 * 🔴 THE OUTPUT PROJECTION'S TWO MATRICES ARE STORED N x K AND A STAGED MATMUL
 * READS K x N. `projectOutput` indexes `linearZWeight[out_channel * CH + k]`,
 * which is output-channel-major, so the matrix path needs both transposed.
 * Same element count, so this is a reshape and not a cost, and it is done here
 * once rather than per workgroup. The BIASES are unchanged and keep their
 * names, because a bias is indexed by output channel either way.
 */
function transposeZG(weights, cHidden, cZ) {
  const flip = (source, rows, columns) => {
    const out = new Float32Array(rows * columns);
    for (let r = 0; r < rows; r += 1) {
      for (let c = 0; c < columns; c += 1) out[c * rows + r] = source[r * columns + c];
    }
    return out;
  };
  return {
    // (cZ out, cHidden in) -> (cHidden, cZ)
    linearZWeight: flip(weights.linearZWeight, cZ, cHidden),
    // (cZ out, cZ in) -> (cZ, cZ)
    linearGWeight: flip(weights.linearGWeight, cZ, cZ),
  };
}

function interleaveAB(weights, cHidden, cZ) {
  const roles = ["linearAPWeight", "linearAGWeight", "linearBPWeight", "linearBGWeight"];
  const biases = ["linearAPBias", "linearAGBias", "linearBPBias", "linearBGBias"];
  const columns = 4 * cHidden;
  const weight = new Float32Array(cZ * columns);
  for (let role = 0; role < 4; role += 1) {
    const source = weights[roles[role]];
    for (let h = 0; h < cHidden; h += 1) {
      for (let c = 0; c < cZ; c += 1) {
        weight[c * columns + h * 4 + role] = source[h * cZ + c];
      }
    }
  }
  const bias = new Float32Array(columns);
  for (let role = 0; role < 4; role += 1) {
    const source = weights[biases[role]];
    for (let h = 0; h < cHidden; h += 1) bias[h * 4 + role] = source[h];
  }
  return { linearABWeight: weight, linearABBias: bias };
}

/**
 * @param {{abLayout?: "blocked"|"interleaved", cHidden?: number, cZ?: number}} [options]
 *   `interleaved` swaps the four projection matrices for one transposed,
 *   interleaved block; it needs the two widths, because the layout is a
 *   reshape and the flat arrays do not state their shape.
 */
export function packWeights(weights, precision, options = {}) {
  const interleaved = options.abLayout === "interleaved";
  const transposed = options.zgLayout === "transposed";
  if (!interleaved && !transposed) return packOrder(weights, ORDER, precision);
  const { cHidden, cZ } = options;
  if (!(cHidden > 0) || !(cZ > 0)) {
    throw new RangeError("a matrix-path triangle pack needs cHidden and cZ");
  }
  let merged = weights;
  let order = ORDER;
  if (interleaved) {
    merged = { ...merged, ...interleaveAB(weights, cHidden, cZ) };
    order = order.filter((name) => !/^linear[AB][PG]/.test(name));
    order = [...order];
    order.splice(order.indexOf("layerNormOutWeight"), 0, ...AB_INTERLEAVED);
  }
  // ...names and offsets unchanged, contents transposed, so only the kernel
  // that reads them has to know. See transposeZG.
  if (transposed) merged = { ...merged, ...transposeZG(weights, cHidden, cZ) };
  return packOrder(merged, order, precision);
}

function packOrder(weights, order, precision) {
  const offsets = {};
  let elementCount = 0;
  for (const name of order) {
    if (weights[name] === undefined) throw new Error(`triangle weights missing ${name}`);
    offsets[name] = elementCount;
    elementCount += weights[name].length;
  }
  const data = concatenateAs(precision, elementCount, (target) => {
    for (const name of order) writeInto(target, weights[name], offsets[name]);
  });
  return { data, offsets };
}


export function expectedWeightElementCount(shape) {
  const { cZ, cHidden } = shape;
  return 2 * cZ + 4 * (cHidden * cZ + cHidden) + 2 * cHidden
    + (cZ * cHidden + cZ) + (cZ * cZ + cZ);
}
