/**
 * The job a CpuBackend and its helpers share, as words in shared memory.
 *
 * A job is one call per chunk: `count` chunks, with each chunk's pointers advanced by
 * its stride. A projection strides over rows and shares one weight matrix; a triangle
 * strides over channel planes, each its own weight matrix; a logistic strides over
 * runs of elements and reads no weights.
 */
export const CONTROL = Object.freeze({
  offset: 1024, length: 16,
  sequence: 0, stop: 1, active: 2, next: 3, ready: 4,
  count: 5, rowsPerChunk: 6, inner: 7, outer: 8,
  out: 9, input: 10, weights: 11,
  outStride: 12, inputStride: 13, weightsStride: 14,
  kind: 15,
});

export const JOB_GEMM = 0;
export const JOB_LOGISTIC = 1;

/** Take chunks until there are none. Each is claimed once, by whoever gets there. */
export function runJobChunks(control, exports) {
  const { gemm, logistic } = exports;
  const count = control[CONTROL.count];
  const rows = control[CONTROL.rowsPerChunk];
  const inner = control[CONTROL.inner];
  const outer = control[CONTROL.outer];
  const out = control[CONTROL.out] >>> 0;
  const input = control[CONTROL.input] >>> 0;
  const weights = control[CONTROL.weights] >>> 0;
  const outStride = control[CONTROL.outStride] >>> 0;
  const inputStride = control[CONTROL.inputStride] >>> 0;
  const weightsStride = control[CONTROL.weightsStride] >>> 0;
  const kind = control[CONTROL.kind];
  for (;;) {
    const chunk = Atomics.add(control, CONTROL.next, 1);
    if (chunk >= count) return;
    if (kind === JOB_LOGISTIC) {
      logistic(out + chunk * outStride, input + chunk * inputStride, rows);
    } else {
      gemm(out + chunk * outStride, input + chunk * inputStride, weights + chunk * weightsStride,
           rows, inner, outer);
    }
  }
}
