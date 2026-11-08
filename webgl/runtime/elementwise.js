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
import { ONE_TARGET, PRELUDE, sized, part } from "./execution.js";
/** @typedef {import("./storage.js").Tensor} Tensor */


const guard = `
void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }`;

/**
 * `add`, `multiply`, `scale` and `relu` in ONE program, the operation a uniform: as four programs
 * they were four links of ~19 ms on Safari 16.3's page thread for the same few lines (the
 * one-input operations bind their input twice). The per-operation sources below stay for callers
 * that draw them directly.
 */
const OPS = { add: 0, multiply: 1, scale: 2, relu: 3, copy: 4 };
export const elementwiseSource = (elements) => part({ ELEMENTS: elements }, {
  uniforms: ["sampler2D aTex", "sampler2D bTex", "int OP", "float scale"],
  body: `  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  vec4 a = fetch4(aTex, texel);
  if (OP == ${OPS.add}) result = a + fetch4(bTex, texel);
  else if (OP == ${OPS.multiply}) result = a * fetch4(bTex, texel);
  else if (OP == ${OPS.scale}) result = a * scale;
  else if (OP == ${OPS.relu}) result = max(a, vec4(0.0));
  else result = a;`,
});

/**
 * The shared program as one operation, for a kernel module whose own `a + b`, `a * b` or copy was
 * otherwise its own program: OP travels with the sizes, so every draw sets it (a uniform keeps its
 * value between draws of a program, and other draws of this one set other operations). Bind both
 * `aTex` and `bTex` (the input twice for a copy).
 *
 * @param {"add" | "multiply" | "copy"} op
 * @param {number} elements
 */
export const elementwiseOpSource = (op, elements) => {
  const source = elementwiseSource(elements);
  return { ...source, sizes: { ...source.sizes, OP: OPS[op] } };
};

/** `a + b`, elementwise. */
export const addSource = (elements) => sized({ ELEMENTS: elements }, `${PRELUDE}${ONE_TARGET}
uniform sampler2D aTex;
uniform sampler2D bTex;
${guard}
  result = fetch4(aTex, texel) + fetch4(bTex, texel);
}`);

/** `a * b`, elementwise - a gate applied to an activation. */
export const multiplySource = (elements) => sized({ ELEMENTS: elements }, `${PRELUDE}${ONE_TARGET}
uniform sampler2D aTex;
uniform sampler2D bTex;
${guard}
  result = fetch4(aTex, texel) * fetch4(bTex, texel);
}`);

/**
 * `max(a, 0)`.
 *
 * 🔴 A STANDALONE ReLU, BECAUSE AF2's SIDECHAIN HEAD APPLIES IT TO A LINEAR'S
 * INPUT RATHER THAN ITS OUTPUT. Everywhere else in these models the activation
 * belongs to the projection that produced the value and `linear` folds it into
 * the epilogue for free; the torsion head is `Linear(relu(x))` repeatedly, where
 * `x` is the output of an ADD. Folding it into the previous linear would be
 * wrong - the add comes between them.
 */
export const reluSource = (elements) => sized({ ELEMENTS: elements }, `${PRELUDE}${ONE_TARGET}
uniform sampler2D aTex;
void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  result = max(fetch4(aTex, texel), vec4(0.0));
}`);

/**
 * @param {import("./execution.js").Execution} execution
 * @param {Tensor} a
 * @param {Tensor} [into]
 * @returns {Tensor}
 */
export function relu(execution, a, into) {
  const output = into ?? execution.allocate(a.elements, "relu");
  if (output === a) {
    throw new Error("relu: a fragment shader cannot read the texture it draws into");
  }
  return execution.dispatch({
    source: elementwiseSource(a.elements), output, inputs: { aTex: a, bTex: a },
    ints: { OP: OPS.relu }, label: "elementwise.relu",
  });
}

/** `a * scale`. */
export const scaleSource = (elements) => sized({ ELEMENTS: elements }, `${PRELUDE}${ONE_TARGET}
uniform sampler2D aTex;
uniform float scale;
${guard}
  result = fetch4(aTex, texel) * scale;
}`);

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
    source: elementwiseSource(a.elements), output, inputs: { aTex: a, bTex: b },
    ints: { OP: OPS.add }, label: "elementwise.add",
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
    source: elementwiseSource(a.elements), output, inputs: { aTex: a, bTex: b },
    ints: { OP: OPS.multiply }, label: "elementwise.multiply",
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
    source: elementwiseSource(a.elements), output, inputs: { aTex: a, bTex: a },
    ints: { OP: OPS.scale }, floats: { scale: factor }, label: "elementwise.scale",
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
export const maskRowsSource = (rows, channels) => part({ ROWS: rows, CH4: channels / 4 }, {
  uniforms: ["sampler2D xTex", "sampler2D maskTex"],
  body: `  int texel = linearTexel();
  if (texel >= ROWS * CH4) { result = vec4(0.0); return; }
  int row = texel / CH4;
  result = fetch4(xTex, texel) * fetch1(maskTex, row);`,
});

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

/**
 * `out[i] = x[i]` for the first `elements` of `x` - the leading prefix, copied.
 *
 * 🔴 IT EXISTS BECAUSE A TENSOR HERE CANNOT BE A VIEW OF ANOTHER. AF2's
 * structure module reads the trunk MSA's FIRST ROW, which on the WebGPU side is
 * `copyBufferToBuffer` of the leading `length * 256` floats out of a buffer that
 * holds every alignment row. A texture cannot be aliased with a smaller element
 * count and a fragment shader cannot sample the texture it draws into, so the
 * prefix is a draw.
 *
 * The copy is EXACT and not merely close: both tensors are addressed linearly
 * from element zero, so when the prefix is a multiple of four every destination
 * texel is one source texel fetched whole. A prefix that is not a multiple of
 * four would straddle, so it is refused rather than silently rounded - every
 * caller in this model has a channel count that is a multiple of four.
 */
export const prefixSource = (elements) => sized({ ELEMENTS: elements }, `${PRELUDE}${ONE_TARGET}
uniform sampler2D aTex;

void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }
  result = fetch4(aTex, texel);
}`);

/**
 * @param {import("./execution.js").Execution} execution
 * @param {Tensor} a
 * @param {number} elements how many leading floats to copy; a multiple of four
 * @param {Tensor} [into]
 * @returns {Tensor}
 */
export function prefix(execution, a, elements, into) {
  if (elements % 4 !== 0) {
    throw new RangeError(`prefix: elements must be a multiple of 4, got ${elements}`);
  }
  if (elements > a.elements) {
    throw new RangeError(`prefix: asked for ${elements} of a tensor holding ${a.elements}`);
  }
  const output = into ?? execution.allocate(elements, "prefix");
  if (output === a) {
    throw new Error("prefix: a fragment shader cannot read the texture it draws into");
  }
  return execution.dispatch({
    source: prefixSource(elements), output, inputs: { aTex: a },
    label: `elementwise.prefix:${elements}`,
  });
}
