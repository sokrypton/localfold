/**
 * Where the reference kernels' matrix multiply may be sent, when anything says so.
 *
 * Unset, which is every checker and every test of a kernel, these return undefined
 * and the reference's own loop runs. The CPU fold sets it (cpu/runtime/backend.js):
 * the kernels stay one copy, in cpu/, and only their inner loop moves.
 */
let backend;

/** @param {object | undefined} next a CpuBackend, or undefined for the plain loops */
export function setLinearBackend(next) {
  backend = next;
}

/** `(rows, inner) @ weights`, with weights (inner, outer) row-major, or undefined. */
export function fastLinear(input, rows, inner, weights, outer) {
  return backend?.linear(input, rows, inner, weights, outer);
}

/** 1 / (1 + exp(-x)) for every element of `input`, or undefined. */
export function fastLogistic(input) {
  return backend?.logistic(input);
}

/**
 * Both triangle contractions over (channel, n, n) planes, or undefined:
 * "outgoing" is out[c,i,j] = sum_k a[c,i,k] b[c,j,k]; "incoming" is sum_k a[c,k,j] b[c,k,i].
 */
export function fastTriangle(a, b, n, channels, direction) {
  return backend?.triangle(a, b, n, channels, direction);
}
