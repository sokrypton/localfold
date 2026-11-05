/**
 * Tensors in textures, addressed linearly - and the capacity that actually
 * bounds them, which is not the one docs/A100.md's headline number describes.
 *
 * 🔴 A TENSOR IS A LINEAR INDEX, NOT A SHAPE. Element `i` lives at texel
 * `i / 4` and component `i % 4`, and that texel is at `(t % W, t / W)` for a
 * fixed power-of-two `W`. Nothing about the tensor's logical shape reaches the
 * texture, so a pair tensor is as addressable as a vector and a kernel decodes
 * `i` into (i, j, channel) itself.
 *
 * 🔴 AND THAT IS WHY THE "181 TOKENS" CEILING IS NOT THIS BACKEND'S CEILING.
 * docs/A100.md measures a GEMM whose operand is a 2D texture of `(K/4) x M`
 * with `M = tokens^2`, so the pair COUNT has to fit one texture axis and stops
 * at `sqrt(MAX_TEXTURE_SIZE)` - 181 on that card, 128 on the M5 here. That is a
 * property of THAT LAYOUT. Laid out linearly the bound is on the total element
 * count instead:
 *
 *     capacity = W * MAX_TEXTURE_SIZE * 4 = 2048 * 16384 * 4 = 134,217,728 floats
 *
 * which at AF2's 128-channel pair track is **1024 tokens**, not 128 - past the
 * 2 GiB storage-binding ceiling the WebGPU port hits at 2047 residues being the
 * next wall rather than this one. `pairTokenCeiling` in ./limits.js is still
 * the right answer for a kernel that wants a 2D operand, and the two numbers
 * are not in conflict: they bound different layouts. This module is the linear
 * one, and the kernels here use it.
 *
 * An OUTPUT is bounded even more loosely, because a draw can be banded: see
 * ./execution.js, which writes rows `[row, row + band)` of a taller target.
 *
 * 🔴 EVERY ALLOCATION GOES THROUGH ONE OBJECT, which AGENTS.md requires of the
 * WebGPU side for the same reason - "so peak memory remains measurable". A
 * texture allocated behind this class's back is a texture no `stats()` can see,
 * and a backend whose memory cannot be measured cannot be compared with the one
 * it is a fallback for.
 */
import { CHANNELS_PER_TEXEL, TEXTURE_WIDTH, textureShape } from "./context.js";

/** Floats one texture can hold on this context, laid out linearly. */
export function tensorCapacity(gl) {
  return TEXTURE_WIDTH * gl.getParameter(gl.MAX_TEXTURE_SIZE) * CHANNELS_PER_TEXEL;
}

/** Bytes a tensor of `elements` floats occupies, padding to the texel rectangle. */
export function storageBytes(elements) {
  const { width, height } = textureShape(elements);
  return width * height * CHANNELS_PER_TEXEL * 4;
}

/**
 * @typedef {{texture: WebGLTexture, elements: number, width: number, height: number,
 *   label: string, bytes: number}} Tensor
 */

export class Storage {
  /** @param {WebGL2RenderingContext} gl */
  constructor(gl, options = {}) {
    this.gl = gl;
    this.capacity = tensorCapacity(gl);
    /** @type {Set<Tensor>} */
    this.live = new Set();
    this.bytes = 0;
    this.peakBytes = 0;
    this.allocations = 0;
    this.readback = null;
    /**
     * Released textures, by `width x height`, waiting for the next tensor of
     * that rectangle.
     *
     * 🔴 A FOLD RELEASES AND REALLOCATES THE SAME RECTANGLE THOUSANDS OF TIMES
     * AND `createTexture` IS NOT FREE. An AF2 monomer fold at 59 residues makes
     * **11,664** allocations, nearly all of them one of a handful of shapes -
     * the pair track's `3481 x 128`, a block's normalised rows - and each one
     * was a `createTexture` plus a `texImage2D` of 1.7 MiB of nothing followed
     * by a `deleteTexture`. The driver has to find, map and (on this backend)
     * zero that storage every time.
     *
     * 🔴 AND A POOLED TEXTURE KEEPS THE PREVIOUS TENSOR'S BYTES, WHICH IS THE
     * ONE WAY THIS CAN GO WRONG. `allocate` has always documented its result as
     * uninitialised and every kernel here guards on the element count, but
     * "documented" and "true" are different things: a kernel that accumulates
     * into its output, or reads a texel its own guard lets through, was reading
     * zeros before and reads a neighbour's leftovers now. That is the
     * first-touch difference CLAUDE.md records on the WebGPU side, where a
     * pooled buffer made a fold differ run to run and looked exactly like a
     * race. So `--no-pool` is a control arm on every WebGL fold driver and the
     * folds must agree to the LAST BIT, not to a tolerance.
     *
     * @type {Map<string, WebGLTexture[]>}
     */
    this.pool = new Map();
    this.pooledBytes = 0;
    this.reuses = 0;
    /**
     * Keep a released texture for the next tensor of its rectangle.
     *
     * 🔴 OFF BY DEFAULT, BECAUSE MEASURED END TO END IT IS A LOSS. It saves 1%
     * of an AF2 fold and costs 4% of its peak; on the fold that matters it is
     * worse on BOTH counts - bioemu2 at 68 tokens is **185.8 s and 1656 MiB
     * without it against 188.5 s and 2158 MiB with**, because a pooled texture
     * is held rather than freed and this backend's audience is machines with
     * 4-8 GiB. Texture creation is simply not what it waits for: the draws are
     * 97% of that fold.
     *
     * It stays because it is the instrument that found a real defect - a kernel
     * reading texels nothing wrote, which `texImage2D` with a null pointer
     * leaves undefined - and because a driver that allocates more slowly than
     * this one may want it. `--pool` turns it on.
     */
    this.pooling = options.pooling ?? false;
    /**
     * Reuse a pooled texture only for an allocation whose LABEL matches.
     *
     * The instrument for the paragraph above: a fold that changes its answer
     * when textures are reused has a kernel reading texels it never wrote, and
     * the label says which allocation's.
     *
     * @type {RegExp | null}
     */
    this.poolLabelPattern = options.poolLabelPattern ?? null;
    /** Allocations by label, so a bisect has a list to bisect over. */
    this.allocationLabels = new Map();
    /**
     * Where the host time in this module goes, and how many bytes moved.
     *
     * 🔴 BECAUSE THE GPU IS 2% OF A BIOEMU2 WebGL FOLD. Every draw in that fold
     * issued twice costs 565 ms of 28.6 seconds, so the arithmetic is not the
     * 246x gap against the WGSL backend - the host is. A readback is a
     * synchronous `readPixels` and an upload a `texSubImage2D`, and neither one
     * is visible in a kernel profile at all.
     */
    this.timing = {
      uploadMs: 0, uploadBytes: 0, uploads: 0,
      readMs: 0, readBytes: 0, reads: 0,
      createMs: 0, creates: 0,
    };
    /**
     * Fill every FRESH texture with this value instead of leaving it undefined.
     *
     * 🔴 `texImage2D` WITH A NULL POINTER LEAVES THE CONTENTS UNDEFINED, AND
     * THIS BACKEND'S WHOLE AUDIENCE IS DRIVERS NOBODY HAS TESTED. Chrome on
     * Metal happens to hand back zeros, so a kernel that reads a texel nothing
     * wrote has been reading 0.0 here and would read whatever the last tenant
     * left on a driver that does not clear - which is the same fault a texture
     * pool produces, arriving without a texture pool. Set this to NaN and any
     * such read turns the output non-finite, which `Execution`'s nanWatch names.
     *
     * @type {number | null}
     */
    this.fillFresh = options.fillFresh ?? null;
  }

  /**
   * An uninitialised tensor of `elements` floats.
   *
   * 🔴 THE REFUSAL NAMES THE REAL LIMIT AND WHAT WOULD FIT, because a texture
   * that is one row too tall fails inside `texImage2D` with an INVALID_VALUE
   * that names nothing, and the caller's question is always "how long a chain
   * does this hold".
   */
  allocate(elements, label = "tensor") {
    if (!Number.isInteger(elements) || elements <= 0) {
      throw new RangeError(`${label}: a tensor needs a positive integer element count, got ${elements}`);
    }
    if (elements > this.capacity) {
      throw new RangeError(
        `${label}: ${elements} floats is past this context's ${this.capacity} `
        + `(${TEXTURE_WIDTH} x MAX_TEXTURE_SIZE ${this.gl.getParameter(this.gl.MAX_TEXTURE_SIZE)} x `
        + `${CHANNELS_PER_TEXEL}); a 128-channel pair track reaches `
        + `${Math.floor(Math.sqrt(this.capacity / 128))} tokens here`);
    }
    const gl = this.gl;
    const { width, height } = textureShape(elements);
    this.allocationLabels.set(label, (this.allocationLabels.get(label) ?? 0) + 1);
    const free = this.poolLabelPattern === null || this.poolLabelPattern.test(label)
      ? this.pool.get(`${width}x${height}`) : undefined;
    const bytes = width * height * CHANNELS_PER_TEXEL * 4;
    let texture;
    if (free !== undefined && free.length > 0) {
      texture = free.pop();
      this.pooledBytes -= bytes;
      this.reuses += 1;
    } else {
      const createdAt = performance.now();
      texture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, null);
      // NEAREST and CLAMP_TO_EDGE: every read is a texelFetch, so filtering and
      // wrapping are never consulted - but an incomplete texture samples as
      // black, and a mip-filtered RGBA32F texture with no mips IS incomplete.
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      this.timing.createMs += performance.now() - createdAt;
      this.timing.creates += 1;
      if (this.fillFresh !== null) {
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, height, gl.RGBA, gl.FLOAT,
          new Float32Array(width * height * CHANNELS_PER_TEXEL).fill(this.fillFresh));
      }
    }
    /** @type {Tensor} */
    const tensor = { texture, elements, width, height, label, bytes };
    this.live.add(tensor);
    this.allocations += 1;
    this.bytes += bytes;
    this.peakBytes = Math.max(this.peakBytes, this.bytes + this.pooledBytes);
    return tensor;
  }

  /**
   * A tensor holding `values`.
   *
   * The upload is PADDED to the texel rectangle, because a texture is
   * width * height texels and `texImage2D` must be handed all of them. The
   * padding is zero and no kernel reads it - every one of them guards on the
   * element count - but it is uploaded, so `storageBytes` is what a tensor
   * costs rather than `values.length * 4`.
   */
  upload(values, label = "tensor") {
    const tensor = this.allocate(values.length, label);
    this.write(tensor, values);
    return tensor;
  }

  /**
   * A tensor of `elements` zeros.
   *
   * 🔴 BECAUSE `allocate` DOES NOT PROMISE ZEROS AND THREE CALLERS WERE
   * READING THEM. A first recycle pass has no previous activations, and this
   * tree expressed that by allocating the three tensors and handing them
   * straight to the embedder - correct only for as long as the driver happens
   * to clear a fresh texture, which the specification does not require of it.
   * `--nan-watch` on the fold drivers is what found them and is what finds the
   * next one; a caller that wants zeros asks for zeros.
   */
  zeros(elements, label = "zeros") {
    return this.upload(new Float32Array(elements), label);
  }

  /** Replace a tensor's contents, which must be exactly its element count. */
  write(tensor, values) {
    if (values.length !== tensor.elements) {
      throw new RangeError(
        `${tensor.label}: holds ${tensor.elements} floats, given ${values.length}`);
    }
    const gl = this.gl;
    const needed = tensor.width * tensor.height * CHANNELS_PER_TEXEL;
    const padded = values.length === needed ? values : (() => {
      const out = new Float32Array(needed);
      out.set(values);
      return out;
    })();
    const startedAt = performance.now();
    gl.bindTexture(gl.TEXTURE_2D, tensor.texture);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, tensor.width, tensor.height,
      gl.RGBA, gl.FLOAT, padded);
    this.timing.uploadMs += performance.now() - startedAt;
    this.timing.uploadBytes += padded.length * 4;
    this.timing.uploads += 1;
    return tensor;
  }

  /**
   * A tensor's contents, read back through a framebuffer.
   *
   * 🔴 ONE FRAMEBUFFER, REATTACHED. A fresh framebuffer per read leaks one per
   * readback in a fold that reads hundreds, and the attachment is the only
   * thing that differs between them.
   */
  read(tensor) {
    const gl = this.gl;
    if (this.readback === null) this.readback = gl.createFramebuffer();
    const previous = gl.getParameter(gl.FRAMEBUFFER_BINDING);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.readback);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D,
      tensor.texture, 0);
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error(`${tensor.label}: not readable as a framebuffer (0x${status.toString(16)})`);
    }
    const padded = new Float32Array(tensor.width * tensor.height * CHANNELS_PER_TEXEL);
    const startedAt = performance.now();
    gl.readPixels(0, 0, tensor.width, tensor.height, gl.RGBA, gl.FLOAT, padded);
    this.timing.readMs += performance.now() - startedAt;
    this.timing.readBytes += padded.length * 4;
    this.timing.reads += 1;
    gl.bindFramebuffer(gl.FRAMEBUFFER, previous);
    return padded.subarray(0, tensor.elements);
  }

  release(tensor) {
    if (!this.live.delete(tensor)) return;
    this.bytes -= tensor.bytes;
    if (!this.pooling) {
      this.gl.deleteTexture(tensor.texture);
      return;
    }
    const key = `${tensor.width}x${tensor.height}`;
    const free = this.pool.get(key);
    if (free === undefined) this.pool.set(key, [tensor.texture]);
    else free.push(tensor.texture);
    this.pooledBytes += tensor.bytes;
  }

  releaseAll() {
    for (const tensor of [...this.live]) this.release(tensor);
    this.drainPool();
  }

  /** Delete every pooled texture, which is what actually frees the memory. */
  drainPool() {
    for (const textures of this.pool.values()) {
      for (const texture of textures) this.gl.deleteTexture(texture);
    }
    this.pool.clear();
    this.pooledBytes = 0;
  }

  /** What this backend is holding, in the shape the WebGPU side reports it. */
  stats() {
    return {
      live: this.live.size,
      allocations: this.allocations,
      bytes: this.bytes,
      peakBytes: this.peakBytes,
      pooledBytes: this.pooledBytes,
      reuses: this.reuses,
      capacityFloats: this.capacity,
      allocationLabels: this.allocationLabels,
      timing: Object.fromEntries(
        Object.entries(this.timing).map(([key, value]) => [key, Math.round(value)])),
    };
  }
}
