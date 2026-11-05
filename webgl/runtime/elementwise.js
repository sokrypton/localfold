/**
 * The elementwise kernels, which are residuals and nothing else at this stage.
 *
 * `webgpu/runtime/elementwise.js` is the twin. Every one of these is one fetch
 * per operand per output texel - four floats a fragment, no loop - so they are
 * bandwidth and nothing else, and there is no geometry to tune.
 *
 * 🔴 THERE IS NO IN-PLACE ADD HERE, AND THAT IS NOT AN OMISSION. The WebGPU
 * side's `addInPlace` reads and writes one storage buffer; a fragment shader
 * cannot, because the texture it samples may not be the one its framebuffer is
 * attached to - "feedback loop" is undefined behaviour and ANGLE reports an
 * incomplete framebuffer for it. A residual therefore costs a destination, and
 * a stage that wants to accumulate ping-pongs between two tensors. Writing
 * `add(x, y, x)` would be the WebGPU habit silently producing undefined output.
 */
import { ONE_TARGET, PRELUDE } from "./execution.js";
/** @typedef {import("./storage.js").Tensor} Tensor */


const guard = (elements) => `
const int ELEMENTS = ${elements};
void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }`;

/** `a + b`, elementwise. */
export const addSource = (elements) => `${PRELUDE}${ONE_TARGET}
uniform sampler2D aTex;
uniform sampler2D bTex;
${guard(elements)}
  result = fetch4(aTex, texel) + fetch4(bTex, texel);
}`;

/** `a * b`, elementwise - a gate applied to an activation. */
export const multiplySource = (elements) => `${PRELUDE}${ONE_TARGET}
uniform sampler2D aTex;
uniform sampler2D bTex;
${guard(elements)}
  result = fetch4(aTex, texel) * fetch4(bTex, texel);
}`;

/** `a * scale`. */
export const scaleSource = (elements) => `${PRELUDE}${ONE_TARGET}
uniform sampler2D aTex;
uniform float scale;
${guard(elements)}
  result = fetch4(aTex, texel) * scale;
}`;

/**
 * @param {import("./execution.js").Execution} execution
 * @param {Tensor} a
 * @param {Tensor} b
 * @param {Tensor} [into] an existing tensor to draw into, instead of a new one
 * @returns {Tensor}
 */
export function add(execution, a, b, into) {
  if (a.elements !== b.elements) {
    throw new RangeError(`add: ${a.elements} and ${b.elements} elements do not match`);
  }
  const output = into ?? execution.allocate(a.elements, "add");
  if (output === a || output === b) {
    throw new Error("add: a fragment shader cannot read the texture it draws into");
  }
  return execution.dispatch({
    source: addSource(a.elements), output, inputs: { aTex: a, bTex: b }, label: "elementwise.add",
  });
}

/**
 * @param {import("./execution.js").Execution} execution
 * @param {Tensor} a
 * @param {Tensor} b
 * @param {Tensor} [into]
 * @returns {Tensor}
 */
export function multiply(execution, a, b, into) {
  if (a.elements !== b.elements) {
    throw new RangeError(`multiply: ${a.elements} and ${b.elements} elements do not match`);
  }
  const output = into ?? execution.allocate(a.elements, "multiply");
  if (output === a || output === b) {
    throw new Error("multiply: a fragment shader cannot read the texture it draws into");
  }
  return execution.dispatch({
    source: multiplySource(a.elements), output, inputs: { aTex: a, bTex: b },
    label: "elementwise.multiply",
  });
}

/**
 * @param {import("./execution.js").Execution} execution
 * @param {Tensor} a
 * @param {number} factor
 * @param {Tensor} [into]
 * @returns {Tensor}
 */
export function scale(execution, a, factor, into) {
  const output = into ?? execution.allocate(a.elements, "scale");
  if (output === a) {
    throw new Error("scale: a fragment shader cannot read the texture it draws into");
  }
  return execution.dispatch({
    source: scaleSource(a.elements), output, inputs: { aTex: a },
    floats: { scale: factor }, label: "elementwise.scale",
  });
}

/**
 * `out[row][c] = x[row][c] * mask[row]` - one mask value broadcast over a row.
 *
 * AF2 and AF3 both mask the pair track constantly (a padded residue contributes
 * nothing to a contraction), and the mask is one float a PAIR against `channels`
 * floats of activation, so it cannot be an ordinary elementwise multiply. The
 * WebGPU side folds this into whichever kernel is adjacent; here it is its own
 * draw until something measures that as a cost.
 */
export const maskRowsSource = (rows, channels) => `${PRELUDE}${ONE_TARGET}
const int ROWS = ${rows};
const int CH4 = ${channels / 4};
uniform sampler2D xTex;
uniform sampler2D maskTex;
void main() {
  int texel = linearTexel();
  if (texel >= ROWS * CH4) { result = vec4(0.0); return; }
  int row = texel / CH4;
  result = fetch4(xTex, texel) * fetch1(maskTex, row);
}`;

/**
 * @param {import("./execution.js").Execution} execution
 * @param {{x: Tensor, mask: Tensor, rows: number, channels: number}} call
 * @param {Tensor} [into]
 * @returns {Tensor}
 */
export function maskRows(execution, { x, mask, rows, channels }, into) {
  if (channels % 4 !== 0) {
    throw new RangeError(`maskRows: channels must be a multiple of 4, got ${channels}`);
  }
  if (x.elements !== rows * channels) {
    throw new RangeError(`maskRows: x holds ${x.elements} floats, not ${rows} x ${channels}`);
  }
  if (mask.elements !== rows) {
    throw new RangeError(`maskRows: mask holds ${mask.elements} floats, not ${rows}`);
  }
  const output = into ?? execution.allocate(rows * channels, "maskRows");
  if (output === x) {
    throw new Error("maskRows: a fragment shader cannot read the texture it draws into");
  }
  return execution.dispatch({
    source: maskRowsSource(rows, channels),
    output,
    inputs: { xTex: x, maskTex: mask },
    label: `elementwise.maskRows:${rows}x${channels}`,
  });
}
