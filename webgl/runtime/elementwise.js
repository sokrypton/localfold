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
import { PRELUDE } from "./execution.js";

const guard = (elements) => `
const int ELEMENTS = ${elements};
void main() {
  int texel = linearTexel();
  if (texel * 4 >= ELEMENTS) { result = vec4(0.0); return; }`;

/** `a + b`, elementwise. */
export const addSource = (elements) => `${PRELUDE}
uniform sampler2D aTex;
uniform sampler2D bTex;
${guard(elements)}
  result = fetch4(aTex, texel) + fetch4(bTex, texel);
}`;

/** `a * b`, elementwise - a gate applied to an activation. */
export const multiplySource = (elements) => `${PRELUDE}
uniform sampler2D aTex;
uniform sampler2D bTex;
${guard(elements)}
  result = fetch4(aTex, texel) * fetch4(bTex, texel);
}`;

/** `a * scale`. */
export const scaleSource = (elements) => `${PRELUDE}
uniform sampler2D aTex;
uniform float scale;
${guard(elements)}
  result = fetch4(aTex, texel) * scale;
}`;

/**
 * @param {import("./execution.js").Execution} execution
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
