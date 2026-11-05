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
  constructor(gl) {
    this.gl = gl;
    this.capacity = tensorCapacity(gl);
    /** @type {Set<Tensor>} */
    this.live = new Set();
    this.bytes = 0;
    this.peakBytes = 0;
    this.allocations = 0;
    this.readback = null;
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
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, null);
    // NEAREST and CLAMP_TO_EDGE: every read is a texelFetch, so filtering and
    // wrapping are never consulted - but an incomplete texture samples as
    // black, and a mip-filtered RGBA32F texture with no mips IS incomplete.
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const bytes = width * height * CHANNELS_PER_TEXEL * 4;
    /** @type {Tensor} */
    const tensor = { texture, elements, width, height, label, bytes };
    this.live.add(tensor);
    this.allocations += 1;
    this.bytes += bytes;
    this.peakBytes = Math.max(this.peakBytes, this.bytes);
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
    gl.bindTexture(gl.TEXTURE_2D, tensor.texture);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, tensor.width, tensor.height,
      gl.RGBA, gl.FLOAT, padded);
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
    gl.readPixels(0, 0, tensor.width, tensor.height, gl.RGBA, gl.FLOAT, padded);
    gl.bindFramebuffer(gl.FRAMEBUFFER, previous);
    return padded.subarray(0, tensor.elements);
  }

  release(tensor) {
    if (!this.live.delete(tensor)) return;
    this.gl.deleteTexture(tensor.texture);
    this.bytes -= tensor.bytes;
  }

  releaseAll() {
    for (const tensor of [...this.live]) this.release(tensor);
  }

  /** What this backend is holding, in the shape the WebGPU side reports it. */
  stats() {
    return {
      live: this.live.size,
      allocations: this.allocations,
      bytes: this.bytes,
      peakBytes: this.peakBytes,
      capacityFloats: this.capacity,
    };
  }
}
