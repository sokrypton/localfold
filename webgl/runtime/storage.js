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
import { float32ToFloat16Array } from "../../shared/weights/float16.js";
import { int5WeightSource } from "./int5.js";

/**
 * A weight of at least this many floats is held in float16 when
 * `Storage.halfWeights` is on.
 *
 * 🔴 THE FLOOR IS WHERE THE WEIGHT STOPS FITTING IN CACHE, which is the same
 * boundary `blockedApplies` turns on and for the same reason: a projection
 * whose weight is read from cache does not care what it costs to read, and one
 * whose weight streams from DRAM is bound by nothing else. 65,536 floats is
 * 256 KiB in f32. The sibling port (`sokrypton/rfd4-proteina-webgpu`) picks the
 * same number for the same decision.
 */
export const HALF_WEIGHT_FLOOR = 65536;

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
 *   label: string, bytes: number, half?: boolean, words?: boolean}} Tensor
 *
 * An int5 weight is NOT one of these: it is a pair of textures and a count, and
 * only `linear` knows how to read it. See `uploadInt5Weight`.
 *
 * @typedef {{int5: true, codes: Tensor, table: Tensor, elements: number,
 *   groups: number, decoded?: Tensor, free: () => void}} Int5Weight
 */

/**
 * A pooled texture's key: its rectangle AND its FORMAT.
 *
 * 🔴 THE FORMAT IS IN THE KEY BECAUSE A TEXTURE COMES BACK ONLY AS THE FORMAT
 * IT WAS MADE IN, AND THE THIRD FORMAT WAS MISSING FROM IT. `allocate` has said
 * so since f16 arrived - "handing an RGBA16F texture to an f32 tensor is a
 * silent halving of everything written into it" - and then `allocateWords`
 * added **RGBA32UI** for the int5 codes and the key never learned about it. So
 * `release` filed every words texture under `f32`, and the next float
 * allocation of that rectangle was handed an integer texture:
 *
 *     --pool: bioemu2.m0:1x68: GL error 0x502 after the draw
 *
 * which is `GL_INVALID_OPERATION` for drawing into RGBA32UI as though it were
 * RGBA32F. Reading one through a `sampler2D` rather than a `usampler2D` is
 * undefined too, so the quiet version of this bug is a wrong answer rather than
 * a refused draw - and `--no-check-errors` turns the loud one into the quiet
 * one. The rule was right, stated, and applied to two of the three formats.
 */
const poolKey = (width, height, format) => `${width}x${height}:${format}`;

/** Which format a tensor's texture actually is - see `poolKey`. */
const tensorFormat = (tensor) =>
  (tensor.words === true ? "u32" : tensor.half === true ? "f16" : "f32");

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
    /**
     * Hold a large WEIGHT in float16 (RGBA16F) rather than float32.
     *
     * 🔴 THE LAST FACTOR THE PROJECTION HAS, AND IT NEEDS NO KERNEL CHANGE AT
     * ALL. A big projection's time is its weight traffic (see
     * `webgl/kernels/linear.js`): bioemu2's 576 x 896 -> 1792 streams 460 MB of
     * weight per call even blocked eight rows deep, and halving the bytes
     * halves that. RGBA16F holds the SAME four values a texel in the SAME texel
     * order, and a `sampler2D` returns them as a `vec4` of floats - the
     * hardware converts on the read - so `fetch4` and every kernel built on it
     * are untouched. Only the texture's internal format and the bytes on the
     * bus change.
     *
     * 🔴 AND IT IS NOT FREE IN ACCURACY, WHICH IS WHY IT IS A FLAG AND NOT A
     * LAYOUT. A float16 weight has eleven bits of mantissa, so this is the
     * first optimisation in this backend that is not bit-identical - every
     * other one reorders a sum. The conversion is done HERE, on the host, by
     * `float32ToFloat16Array` (round to nearest, ties to even) rather than by
     * handing f32 bytes to an RGBA16F texture and letting the driver choose:
     * the sibling port records that ANGLE's Metal backend TRUNCATES on the way
     * in, and truncation's bias accumulated over a trunk pass cost it 6e-3
     * relative where rounding cost 4e-4.
     *
     * It is what the WGSL path does with the same tensors, which is why the
     * fold drivers' differentials pass `setHalfPrecision(device, false)` to
     * force the reference to f32 and compare like with like.
     */
    this.halfWeights = options.halfWeights ?? false;
    /** How many tensors were held in float16, and how many were too large for it. */
    this.halfWeightTensors = 0;
    this.halfRefusals = 0;
    /**
     * Leave a large int5 weight as CODES for the kernel to decode.
     *
     * Strictly better than `halfWeights` where it applies - the decoded value
     * is the float32 the host would have produced, and the codes are 0.875
     * bytes a weight against float16's two - so a fold turns both on and the
     * int5 path wins when a tensor qualifies. See `webgl/runtime/int5.js`.
     */
    this.int5Weights = options.int5Weights ?? false;
    this.int5WeightTensors = 0;
  }

  /**
   * An uninitialised tensor of `elements` floats.
   *
   * 🔴 THE REFUSAL NAMES THE REAL LIMIT AND WHAT WOULD FIT, because a texture
   * that is one row too tall fails inside `texImage2D` with an INVALID_VALUE
   * that names nothing, and the caller's question is always "how long a chain
   * does this hold".
   */
  /**
   * While true, tensors are VIRTUAL: their shape and no texture - writes skipped, reads zeros,
   * release a no-op. What a dry run (`Execution.collect`) allocates: it draws nothing, so the
   * textures it used to create and read back were ~0.2 s of a first fold's wait on Safari 16.3.
   */
  virtual = false;

  #virtualTensor(fields) {
    const tensor = { texture: null, bytes: 0, half: false, ...fields };
    this.live.add(tensor);
    return tensor;
  }

  allocate(elements, label = "tensor", half = false) {
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
    if (this.virtual) return this.#virtualTensor({ elements, width, height, label, half });
    this.allocationLabels.set(label, (this.allocationLabels.get(label) ?? 0) + 1);
    const key = poolKey(width, height, half ? "f16" : "f32");
    const free = this.poolLabelPattern === null || this.poolLabelPattern.test(label)
      ? this.pool.get(key) : undefined;
    const bytes = width * height * CHANNELS_PER_TEXEL * (half ? 2 : 4);
    let texture;
    if (free !== undefined && free.length > 0) {
      texture = free.pop();
      this.pooledBytes -= bytes;
      this.reuses += 1;
    } else {
      const createdAt = performance.now();
      texture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, half ? gl.RGBA16F : gl.RGBA32F, width, height, 0,
        gl.RGBA, half ? gl.HALF_FLOAT : gl.FLOAT, null);
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
    const tensor = { texture, elements, width, height, label, bytes, half };
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
  /** An uninitialised float16 tensor, for a kernel that writes rounded halves. */
  allocateHalf(elements, label = "half") {
    return this.allocate(elements, label, true);
  }

  zeros(elements, label = "zeros") {
    return this.upload(new Float32Array(elements), label);
  }

  /**
   * A WEIGHT tensor, in float16 when `halfWeights` is on and it is large
   * enough to be streaming from DRAM rather than sitting in cache.
   *
   * Weights are the only tensors this is offered for: an activation is written
   * by a kernel drawing into it, which would need an RGBA16F render target and
   * the rounding discipline that goes with one, and a weight is written once by
   * the host.
   */
  uploadWeight(values, label = "weight") {
    let half = this.halfWeights && values.length >= HALF_WEIGHT_FLOOR;
    if (half && !this.virtual) {
      // 🔴 AND A TENSOR THAT DOES NOT FIT IN float16 STAYS IN float32 RATHER
      // THAN BECOMING Inf. A checkpoint's weights are all well inside the
      // format's 65504, but this entry point is named by what the CALLER calls
      // a weight, and an index table or a precomputed bin edge is not a weight
      // in that sense. Checking the maximum costs one pass over a buffer that
      // is about to be converted anyway, and the alternative failure is a fold
      // full of infinities with nothing to point at.
      let largest = 0;
      for (let i = 0; i < values.length; i += 1) {
        const magnitude = Math.abs(values[i]);
        if (magnitude > largest) largest = magnitude;
      }
      if (!(largest <= 65504)) {
        half = false;
        this.halfRefusals += 1;
      }
    }
    if (!half) return this.upload(values, label);
    this.halfWeightTensors += 1;
    const tensor = this.allocate(values.length, label, true);
    this.write(tensor, values instanceof Float32Array ? values : Float32Array.from(values));
    return tensor;
  }

  /**
   * An int5 weight, left as CODES for the kernel to decode.
   *
   * 🔴 THE RESULT IS NOT A `Tensor`, IT IS A PAIR OF THEM PLUS A COUNT, and
   * that is deliberate: a caller that does not know about it cannot pass it to
   * a kernel that cannot read it. `linear` checks for `int5` by name and every
   * other path sees an object with no `.texture` and fails loudly rather than
   * sampling something plausible.
   *
   * Returns null when the tensor is not int5 at group 32, which is every
   * float32 and int3 bundle - the caller falls back to `uploadWeight`.
   *
   * @param {{record: object, buffer: ArrayBuffer, byteOffset: number}} source
   *   what `read.source(name)` / `HttpTensorStore.tensorSource(name)` returns
   */
  uploadInt5Weight(source, label = "weight") {
    if (!this.int5Weights) return null;
    const extracted = int5WeightSource(source.record, source.buffer, source.byteOffset);
    if (extracted === null) return null;
    const { words, scaleZero, groups, elements } = extracted;
    if (elements < HALF_WEIGHT_FLOOR) return null;
    const codes = this.allocateWords(words.length, `${label}.codes`);
    this.writeWords(codes, words);
    const table = this.upload(scaleZero, `${label}.scale-zero`);
    this.int5WeightTensors += 1;
    const held = { int5: true, codes, table, elements, groups };
    held.free = () => {
      this.release(codes);
      this.release(table);
      if (held.decoded !== undefined) this.release(held.decoded);
    };
    return held;
  }

  /**
   * A tensor of 32-bit WORDS (RGBA32UI), four to a texel.
   *
   * The same rectangle arithmetic as a float tensor, so `W` addresses it the
   * same way; only the format differs. Used for packed codes, which are not
   * numbers until a kernel decodes them.
   */
  allocateWords(words, label = "words") {
    const texels = Math.ceil(words / CHANNELS_PER_TEXEL);
    const width = TEXTURE_WIDTH;
    const height = Math.max(1, Math.ceil(texels / width));
    if (height > this.gl.getParameter(this.gl.MAX_TEXTURE_SIZE)) {
      throw new RangeError(`${label}: ${words} words is past this context's textures`);
    }
    if (this.virtual) return this.#virtualTensor({ elements: words, width, height, label, words: true });
    const gl = this.gl;
    const bytes = width * height * CHANNELS_PER_TEXEL * 4;
    // 🔴 AND IT TAKES FROM THE POOL, WHICH IT DID NOT, so a words texture was
    // released into the pool and never taken out of it - stranded bytes that
    // `pooledBytes` counts and nothing can use. Now it has its own `u32` key
    // and can only ever come back as what it is.
    const key = poolKey(width, height, "u32");
    const free = this.poolLabelPattern === null || this.poolLabelPattern.test(label)
      ? this.pool.get(key) : undefined;
    let texture;
    if (free !== undefined && free.length > 0) {
      texture = free.pop();
      this.pooledBytes -= bytes;
      this.reuses += 1;
      gl.bindTexture(gl.TEXTURE_2D, texture);
    } else {
      texture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32UI, width, height, 0,
        gl.RGBA_INTEGER, gl.UNSIGNED_INT, null);
      for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) {
        gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
      }
      for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T]) {
        gl.texParameteri(gl.TEXTURE_2D, p, gl.CLAMP_TO_EDGE);
      }
    }
    /** @type {Tensor} */
    const tensor = { texture, elements: words, width, height, label, bytes, words: true };
    this.live.add(tensor);
    this.allocations += 1;
    this.bytes += bytes;
    this.peakBytes = Math.max(this.peakBytes, this.bytes + this.pooledBytes);
    return tensor;
  }

  /** Fill a word tensor, padding to its rectangle. */
  writeWords(tensor, words) {
    if (tensor.texture === null) return tensor;
    const gl = this.gl;
    const needed = tensor.width * tensor.height * CHANNELS_PER_TEXEL;
    const padded = words.length === needed ? words : (() => {
      const out = new Uint32Array(needed);
      out.set(words);
      return out;
    })();
    const startedAt = performance.now();
    gl.bindTexture(gl.TEXTURE_2D, tensor.texture);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, tensor.width, tensor.height,
      gl.RGBA_INTEGER, gl.UNSIGNED_INT, padded);
    this.timing.uploadMs += performance.now() - startedAt;
    this.timing.uploadBytes += padded.length * 4;
    this.timing.uploads += 1;
    return tensor;
  }

  /** Replace a tensor's contents, which must be exactly its element count. */
  write(tensor, values) {
    if (values.length !== tensor.elements) {
      throw new RangeError(
        `${tensor.label}: holds ${tensor.elements} floats, given ${values.length}`);
    }
    if (tensor.texture === null) return tensor;
    const gl = this.gl;
    const needed = tensor.width * tensor.height * CHANNELS_PER_TEXEL;
    const padded = values.length === needed ? values : (() => {
      const out = new Float32Array(needed);
      out.set(values);
      return out;
    })();
    const startedAt = performance.now();
    gl.bindTexture(gl.TEXTURE_2D, tensor.texture);
    if (tensor.half) {
      // Rounded HERE, to nearest and ties to even, rather than by handing f32
      // bytes to an RGBA16F texture: see `halfWeights` for what truncation on
      // the way in cost the sibling port.
      const halves = float32ToFloat16Array(padded);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, tensor.width, tensor.height,
        gl.RGBA, gl.HALF_FLOAT, halves);
      this.timing.uploadBytes += halves.length * 2;
    } else {
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, tensor.width, tensor.height,
        gl.RGBA, gl.FLOAT, padded);
      this.timing.uploadBytes += padded.length * 4;
    }
    this.timing.uploadMs += performance.now() - startedAt;
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
    if (tensor.words) {
      throw new Error(`${tensor.label}: a packed-code tensor is not readable as floats`);
    }
    if (tensor.half) {
      // A weight is written once and never read back, and reading an RGBA16F
      // attachment as floats is another format question nothing here needs
      // answered. Refusing by name beats returning something plausible.
      throw new Error(`${tensor.label}: a float16 weight tensor is not readable`);
    }
    if (tensor.texture === null) return new Float32Array(tensor.elements);
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
    // 🔴 A DRY RUN MUST NOT FREE A REAL TEXTURE. Everything it allocates is virtual, so the only
    // tensors with a texture while `virtual` are the ones that existed BEFORE it - the weights and
    // the features - and a fold that releases one of its inputs at the end would delete it out
    // from under the real fold that follows. Its own virtual tensors are released as ever, or the
    // dry run ends with every tensor it ever named still live.
    if (this.virtual && tensor.texture !== null) return;
    if (!this.live.delete(tensor)) return;
    if (tensor.texture === null) return;
    this.bytes -= tensor.bytes;
    if (!this.pooling) {
      this.gl.deleteTexture(tensor.texture);
      return;
    }
    const key = poolKey(tensor.width, tensor.height, tensorFormat(tensor));
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
      halfWeightTensors: this.halfWeightTensors,
      halfRefusals: this.halfRefusals,
      int5WeightTensors: this.int5WeightTensors,
      timing: Object.fromEntries(
        Object.entries(this.timing).map(([key, value]) => [key, Math.round(value)])),
    };
  }
}
