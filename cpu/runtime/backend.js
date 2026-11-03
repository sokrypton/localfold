/**
 * Runs the reference kernels' matrix multiplies in SIMD, on every core offered.
 *
 * One WebAssembly memory holds the model's weights for good and the current
 * multiply's inputs and output for a moment. Copying in costs one pass over each
 * tensor against a multiply-add per weight on every element of it, so it does not
 * show. The weights are adopted once, by identity, which also lets the caller drop
 * its own copy: two gigabytes of tower is not worth holding twice.
 *
 * Threads need a SharedArrayBuffer, and so a cross-origin isolated page. Without one
 * this runs on the fold's own thread, which is slower and otherwise the same.
 */
import { CONTROL, JOB_GEMM, JOB_LOGISTIC, runJobChunks } from "./backend-job.js";
import { gemmModuleBytes, gemmSupported, relaxedSimdSupported } from "./gemm-wasm.js";
import { floats } from "./recycled-floats.js";

const PAGE = 65536;
const GIB = 1024 * 1024 * 1024;
/** Below this many multiply-adds a job is not worth waking anybody for. */
const PARALLEL_MIN_MACS = 1 << 20;

/** Up to a cache line, in arithmetic that stays unsigned past 2 GiB. */
const aligned = (bytes) => Math.ceil(bytes / 64) * 64;
/** Below this a call is not worth the copy, and the reference's own loop answers. */
const SIMD_MIN_MACS = 1 << 12;

async function spawnHelper(memory, module) {
  if (typeof Worker === "function") {
    const worker = new Worker(new URL("./gemm-helper.js", import.meta.url), { type: "module" });
    worker.postMessage({ memory, module });
    return worker;
  }
  const { Worker: NodeWorker } = await import("node:worker_threads");
  const worker = new NodeWorker(new URL("./gemm-helper.js", import.meta.url));
  worker.unref();
  worker.postMessage({ memory, module });
  return worker;
}

export class CpuBackend {
  #memory;
  #bytes;
  #float;
  #exports;
  #control;
  #helpers = [];
  /** Float32Array -> its adopted copy's byte offset. */
  #adopted = new WeakMap();
  #weightsEnd = CONTROL.offset + CONTROL.length * 4;
  #capacity;

  /** How many threads a multiply runs on, this one included. */
  threads = 1;

  /**
   * @param {{threads?: number, gibibytes?: number}} [options] `threads` above one
   *   needs SharedArrayBuffer and is lowered to one without it. A wasm32 memory tops
   *   out at 4 GiB, so offsets run past 2^31 here and a bitwise operator would turn
   *   them negative. The weights take the bottom of it, and what is left is the scratch
   *   that long inputs are chunked to fit.
   */
  static async create({ threads = 1, gibibytes = 4 } = {}) {
    if (!gemmSupported()) {
      throw new Error("This browser's WebAssembly has no SIMD, which the CPU fold needs.");
    }
    const shared = threads > 1 && typeof SharedArrayBuffer !== "undefined"
      && (typeof crossOriginIsolated === "undefined" || crossOriginIsolated === true);
    const pages = Math.floor(gibibytes * GIB / PAGE);
    let memory;
    try {
      memory = new WebAssembly.Memory({ initial: pages, maximum: pages, shared });
    } catch (error) {
      throw new Error(`The CPU fold could not reserve ${gibibytes} GiB of address space: ${error.message}`);
    }
    const module = await WebAssembly.compile(
      gemmModuleBytes({ shared, pages, relaxed: relaxedSimdSupported() }));
    const instance = await WebAssembly.instantiate(module, { env: { memory } });
    const backend = new CpuBackend();
    backend.#memory = memory;
    backend.#capacity = pages * PAGE;
    backend.#bytes = new Uint8Array(memory.buffer);
    backend.#float = new Float32Array(memory.buffer);
    backend.#exports = instance.exports;
    backend.#control = new Int32Array(memory.buffer, CONTROL.offset, CONTROL.length);
    if (shared) {
      backend.threads = threads;
      for (let helper = 1; helper < threads; helper += 1) {
        backend.#helpers.push(await spawnHelper(memory, module));
      }
      // ...up before the first job, because a helper counts every job it sees.
      for (;;) {
        const ready = Atomics.load(backend.#control, CONTROL.ready);
        if (ready >= backend.#helpers.length) break;
        await new Promise((resolve) => { setTimeout(resolve, 1); });
      }
    }
    return backend;
  }

  /**
   * Move a weight array into this backend and return the view that replaces it.
   *
   * @param {Float32Array} array
   * @returns {Float32Array} backed by this memory; the argument may be dropped
   */
  adopt(array) {
    if (array.buffer === this.#memory.buffer) return array;
    const known = this.#adopted.get(array);
    if (known !== undefined) return new Float32Array(this.#memory.buffer, known, array.length);
    const at = aligned(this.#weightsEnd);
    const end = at + array.byteLength;
    if (end > this.#capacity - (1 << 26)) {
      throw new Error("The CPU fold's weights do not fit its 4 GiB memory.");
    }
    this.#float.set(array, at / 4);
    this.#weightsEnd = end;
    this.#adopted.set(array, at);
    return new Float32Array(this.#memory.buffer, at, array.length);
  }

  /** Set aside address space no weight is in, so a test can reach the top of memory cheaply. */
  reserve(bytes) {
    this.#weightsEnd += bytes;
  }

  /** Walk a weight tree and adopt every Float32Array in it, in place. */
  adoptTree(tree) {
    if (tree === null || typeof tree !== "object") return tree;
    for (const key of Object.keys(tree)) {
      const value = tree[key];
      if (value instanceof Float32Array) tree[key] = this.adopt(value);
      else if (typeof value === "object") this.adoptTree(value);
    }
    return tree;
  }

  #scratchStart() {
    return aligned(this.#weightsEnd);
  }

  #run({ kind = JOB_GEMM, count, rowsPerChunk, inner = 0, outer = 0, out, input, weights = 0,
         outStride, inputStride, weightsStride = 0 }, macs) {
    if (this.threads === 1 || macs < PARALLEL_MIN_MACS || count < 2) {
      for (let chunk = 0; chunk < count; chunk += 1) {
        if (kind === JOB_LOGISTIC) {
          this.#exports.logistic(out + chunk * outStride, input + chunk * inputStride, rowsPerChunk);
        } else {
          this.#exports.gemm(out + chunk * outStride, input + chunk * inputStride,
                             weights + chunk * weightsStride, rowsPerChunk, inner, outer);
        }
      }
      return;
    }
    const control = this.#control;
    control[CONTROL.kind] = kind;
    control[CONTROL.count] = count;
    control[CONTROL.rowsPerChunk] = rowsPerChunk;
    control[CONTROL.inner] = inner;
    control[CONTROL.outer] = outer;
    control[CONTROL.out] = out;
    control[CONTROL.input] = input;
    control[CONTROL.weights] = weights;
    control[CONTROL.outStride] = outStride;
    control[CONTROL.inputStride] = inputStride;
    control[CONTROL.weightsStride] = weightsStride;
    Atomics.store(control, CONTROL.next, 0);
    Atomics.store(control, CONTROL.active, this.#helpers.length);
    Atomics.add(control, CONTROL.sequence, 1);
    Atomics.notify(control, CONTROL.sequence);
    runJobChunks(control, this.#exports);
    for (;;) {
      const active = Atomics.load(control, CONTROL.active);
      if (active === 0) return;
      Atomics.wait(control, CONTROL.active, active, 100);
    }
  }

  /**
   * `(rows, inner) @ weights` with weights (inner, outer) row-major.
   *
   * @returns {Float32Array | undefined} undefined when the call is too small to be
   *   worth moving, and the reference's loop should answer
   */
  linear(input, rows, inner, weights, outer) {
    if (rows * inner * outer < SIMD_MIN_MACS) return undefined;
    const adopted = this.adopt(weights);
    const weightsAt = adopted.byteOffset;
    const start = this.#scratchStart();
    const room = this.#capacity - start;
    const rowBytes = (inner + outer) * 4;
    const rowsAtOnce = Math.min(rows, Math.floor(room / rowBytes));
    if (rowsAtOnce < 1) throw new Error("The CPU fold's scratch memory is too small for one row.");
    const output = floats(rows * outer);
    for (let first = 0; first < rows; first += rowsAtOnce) {
      const here = Math.min(rowsAtOnce, rows - first);
      const inputAt = start;
      const outAt = start + here * inner * 4;
      this.#float.set(input.subarray(first * inner, (first + here) * inner), inputAt / 4);
      // Enough chunks for the threads to balance, not so many that claiming them shows.
      const perChunk = Math.max(1, Math.ceil(here / Math.max(1, this.threads * 8)));
      const count = Math.ceil(here / perChunk);
      const last = here - (count - 1) * perChunk;
      this.#run({ count: last === perChunk ? count : count - 1, rowsPerChunk: perChunk, inner, outer,
                  out: outAt, input: inputAt, weights: weightsAt,
                  outStride: perChunk * outer * 4, inputStride: perChunk * inner * 4 },
                here * inner * outer);
      if (last !== perChunk) {
        const done = (count - 1) * perChunk;
        this.#exports.gemm(outAt + done * outer * 4, inputAt + done * inner * 4, weightsAt, last, inner, outer);
      }
      output.set(this.#float.subarray(outAt / 4, outAt / 4 + here * outer), first * outer);
    }
    return output;
  }

  /**
   * Both triangle contractions, one n x n matrix product per channel.
   *
   * @param {Float32Array} a (channels, n, n)
   * @param {Float32Array} b (channels, n, n)
   * @returns {Float32Array | undefined} (channels, n, n)
   */
  triangle(a, b, n, channels, direction) {
    if (n * n * n * channels < SIMD_MIN_MACS) return undefined;
    const plane = n * n;
    const start = this.#scratchStart();
    const room = this.#capacity - start;
    const planesAtOnce = Math.min(channels, Math.floor(room / (plane * 12)));
    if (planesAtOnce < 1) throw new Error("The CPU fold's scratch memory is too small for one plane.");
    const output = floats(channels * plane);
    const float = this.#float;
    const outgoing = direction === "outgoing";
    for (let first = 0; first < channels; first += planesAtOnce) {
      const here = Math.min(planesAtOnce, channels - first);
      const leftAt = start / 4;
      const rightAt = leftAt + here * plane;
      const outAt = rightAt + here * plane;
      // outgoing: out = A @ B^T.  incoming: out = B^T @ A. Either way B goes in transposed.
      for (let c = 0; c < here; c += 1) {
        const from = (first + c) * plane;
        const transposedAt = (outgoing ? rightAt : leftAt) + c * plane;
        for (let i = 0; i < n; i += 1) {
          for (let j = 0; j < n; j += 1) float[transposedAt + j * n + i] = b[from + i * n + j];
        }
        float.set(a.subarray(from, from + plane), (outgoing ? leftAt : rightAt) + c * plane);
      }
      this.#run({ count: here, rowsPerChunk: n, inner: n, outer: n,
                  out: outAt * 4, input: leftAt * 4, weights: rightAt * 4,
                  outStride: plane * 4, inputStride: plane * 4, weightsStride: plane * 4 },
                here * plane * n);
      output.set(float.subarray(outAt, outAt + here * plane), first * plane);
    }
    return output;
  }

  /**
   * 1 / (1 + exp(-x)) for every element, or undefined when too few to be worth moving.
   *
   * @param {Float32Array} input
   * @returns {Float32Array | undefined}
   */
  logistic(input) {
    if (input.length < SIMD_MIN_MACS) return undefined;
    const start = this.#scratchStart();
    const room = this.#capacity - start;
    const atOnce = Math.min(input.length, Math.floor(room / 8) & ~3);
    if (atOnce < 4) throw new Error("The CPU fold's scratch memory is too small.");
    const output = floats(input.length);
    const float = this.#float;
    for (let first = 0; first < input.length; first += atOnce) {
      const here = Math.min(atOnce, input.length - first);
      const vectors = here & ~3;
      const inputAt = start;
      const outAt = start + here * 4;
      float.set(input.subarray(first, first + here), inputAt / 4);
      if (vectors > 0) {
        const perChunk = Math.max(4096, Math.ceil(vectors / Math.max(1, this.threads * 8) / 4) * 4);
        const count = Math.floor(vectors / perChunk);
        if (count > 0) {
          this.#run({ kind: JOB_LOGISTIC, count, rowsPerChunk: perChunk, out: outAt, input: inputAt,
                      outStride: perChunk * 4, inputStride: perChunk * 4 }, vectors * 16);
        }
        const done = count * perChunk;
        this.#exports.logistic(outAt + done * 4, inputAt + done * 4, vectors - done);
      }
      output.set(float.subarray(outAt / 4, outAt / 4 + vectors), first);
      for (let at = vectors; at < here; at += 1) {
        output[first + at] = 1 / (1 + Math.exp(-input[first + at]));
      }
    }
    return output;
  }

  /** Stop the helpers. The backend is finished after this. */
  close() {
    if (this.#helpers.length > 0) {
      Atomics.store(this.#control, CONTROL.stop, 1);
      Atomics.add(this.#control, CONTROL.sequence, 1);
      Atomics.notify(this.#control, CONTROL.sequence);
    }
    this.#helpers = [];
    this.threads = 1;
  }

  /** How much of its memory the weights hold, in bytes. */
  get weightBytes() {
    return this.#weightsEnd;
  }
}
