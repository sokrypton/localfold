/**
 * What stands in for a compute dispatch here: one draw, one fragment per output
 * texel, into a tensor from ./storage.js.
 *
 * `webgpu/runtime/execution.js` is the shape this follows - allocate, upload,
 * dispatch, read, release, with the allocations and dispatches counted so a
 * fold's cost is measurable. The differences are forced by the API:
 *
 *   - There is no pipeline, only a linked PROGRAM, and no bind group: inputs are
 *     texture units and the kernel names a `sampler2D` per input.
 *   - A grid is a viewport. The WebGPU side folds a linear index across
 *     `(group.x, group.y)` because a dispatch is at most 65535 workgroups wide;
 *     here the output texture is `W` texels wide by construction and the
 *     fragment's own coordinates ARE the fold, so `linearTexel()` is the whole
 *     of it and there is no `linearGrid`/`rowGrid` to get wrong.
 *   - A tall output is drawn in BANDS, because a viewport is at most
 *     `MAX_VIEWPORT_DIMS` high while a texture may be `MAX_TEXTURE_SIZE`. Each
 *     band is one draw whose viewport ORIGIN is its first row, so
 *     `gl_FragCoord.y` is the absolute row and a kernel needs no row uniform.
 *
 * 🔴 THE PROGRAM CACHE IS KEYED BY SOURCE, NOT BY A NAME OR A HASH. The WebGPU
 * side learned this the expensive way twice - two models' kernels collided
 * under one key because the key named neither the channel widths nor the head
 * dimension the source embeds, and a cache that hands a caller somebody else's
 * kernel is a wrong answer with no error. Here every kernel is GENERATED with
 * its shapes baked in as `const int`, so the source is the only thing that
 * distinguishes two instances, and it is what the cache indexes.
 */
import { FULLSCREEN_TRIANGLE, TEXTURE_WIDTH, linkProgram } from "./context.js";
import { Storage } from "./storage.js";

/**
 * The declarations and helpers every kernel here opens with.
 *
 * `linearTexel()` is the fragment's own texel index in the output. `fetch4`
 * reads a whole texel of an input; `fetch1` reads one float, which costs a
 * texel fetch and a select and is for the ragged cases (a mask, a bias) rather
 * than for anything in an inner loop.
 */
export const PRELUDE = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
const int W = ${TEXTURE_WIDTH};

int linearTexel() { return int(gl_FragCoord.y) * W + int(gl_FragCoord.x); }

vec4 fetch4(sampler2D tex, int texel) {
  return texelFetch(tex, ivec2(texel % W, texel / W), 0);
}

float fetch1(sampler2D tex, int index) {
  vec4 texel = fetch4(tex, index >> 2);
  int component = index & 3;
  return component == 0 ? texel.x
       : component == 1 ? texel.y
       : component == 2 ? texel.z : texel.w;
}

// An exact f32 -> f16 -> f32 round trip, two channels to a word.
//
// HERE BECAUSE THE SHIPPED AF2 PATH KEEPS ITS INTERMEDIATES IN f16 AND NOTHING
// SWITCHES THAT OFF. webgpu/af2/evoformer/block.js sets normalizedStorage to
// f16 outright, asks its flash kernel for input and output f16, and sets the
// transition's hiddenStorage to f16 for every width that is a multiple of four
// - none of it gated by halfPrecisionAvailable or by a tuning knob. A port that
// keeps f32 at those points disagrees with the primary path by ~1e-4
// everywhere, which is what a whole-block comparison measured. This buys
// nothing here (the tensor is still an RGBA32F texture) and it is not an
// accuracy choice this backend is making: it is the one being matched, and a
// fallback whose folds cannot be compared with the primary path's is not much
// of a fallback. (No backticks in here - this comment is inside a JS template
// literal and one would end it. It has happened twice.)
// A relative dither applied before the rounding, zero unless a probe sets it.
// It exists to MEASURE the comparison's real floor, which is not an f32 step: two
// values 1e-7 apart that straddle an f16 boundary round 5e-4 apart, so a chain
// that quantises cannot be bounded by jittering its input. Set it to 1e-7 and
// the output moves by what two correct implementations of this chain differ by.
uniform float halfDither;

vec4 half4(vec4 v) {
  vec4 d = v * (1.0 + halfDither);
  return vec4(unpackHalf2x16(packHalf2x16(d.xy)), unpackHalf2x16(packHalf2x16(d.zw)));
}
`;

/** The single-target `result` declaration, which most kernels want. */
export const ONE_TARGET = "out vec4 result;\n";

/** `count` render targets named result0..result{count-1}, for a wide kernel. */
export const targetDeclarations = (count) => Array.from(
  { length: count }, (_, t) => `layout(location = ${t}) out vec4 result${t};`).join("\n") + "\n";

export class Execution {
  /** @param {WebGL2RenderingContext} gl */
  constructor(gl, options = {}) {
    this.gl = gl;
    this.storage = new Storage(gl, { pooling: options.pooling ?? false,
      poolLabelPattern: options.poolLabelPattern ?? null,
      fillFresh: options.fillFresh ?? null,
      halfWeights: options.halfWeights ?? false,
      int5Weights: options.int5Weights ?? false });
    /**
     * Read every output back and refuse at the first non-finite value.
     *
     * With `fillFresh: NaN` this names the kernel that reads a texel
     * nothing wrote - the FIRST one, which is the only useful one, since
     * a NaN reaches everything downstream.
     */
    this.nanWatch = options.nanWatch ?? false;
    /** Targets `nanWatch` could not read back, which are float16 or packed codes. */
    this.nanWatchSkipped = 0;
    /**
     * The control arm for the blocked projection (`webgl/kernels/linear.js`):
     * one output texel a fragment, as this backend did before it had a block.
     * A geometry is only worth what it is measured against.
     */
    this.noLinearBlock = options.noLinearBlock ?? false;
    /**
     * Draw every dispatch whose label matches TWICE, into the same output.
     *
     * 🔴 THE HONEST ATTRIBUTION INSTRUMENT, AND THE REASON `profile` IS NOT
     * ONE. A kernel timed by waiting for it is timed with the wait: every
     * label's share of a profiled AF2 fold came out within a few points of its
     * share of the DISPATCH COUNT, which is a measurement of the instrument.
     * Drawing a family a second time changes nothing it computes - these
     * kernels are pure, so the second draw writes the same texels - and the
     * fold's WALL grows by exactly that family's GPU time. Subtract the
     * baseline and that is the answer, with no sync anywhere.
     *
     * Only the draws are repeated, not the program and uniform setup, so what
     * it prices is the GPU work and not the host's.
     *
     * @type {RegExp | null}
     */
    this.doubleLabels = options.doubleLabels ?? null;
    /** Issue no draws at all; see the comment at the draw loop. */
    this.skipDraws = options.skipDraws ?? false;
    /**
     * The control arm for the transposed incoming contraction
     * (`webgl/kernels/triangle-contract.js`): read the column directly, as this
     * backend did before. Bit-identical, so this prices a LAYOUT and nothing
     * else.
     */
    this.noTriangleTranspose = options.noTriangleTranspose ?? false;
    /**
     * The control arm for the split contraction and centre norm
     * (`webgl/af3/trunk/triangle.js`): both in one draw, where a fragment
     * contracts every channel of its pair to normalise four of them.
     * Bit-identical, so this prices the pass count.
     */
    this.noTriangleSplit = options.noTriangleSplit ?? false;
    /**
     * The control arm for the coalesced outgoing contraction
     * (`webgl/af3/trunk/triangle.js`): one `fetch1` per k, which asks for the
     * same texel four times over.
     */
    this.noTriangleCoalesce = options.noTriangleCoalesce ?? false;
    /**
     * The control arm for the pair-blocked contraction
     * (`webgl/af3/trunk/triangle.js`): one pair a fragment, which reads both
     * operands' rows for every pair instead of sharing them across a block.
     */
    this.noContractBlock = options.noContractBlock ?? false;
    /**
     * The control arm for the row-block-fastest fragment order in the int5
     * projection (`webgl/kernels/linear.js`): ordered by output group, as the
     * float32 and float16 paths are.
     */
    this.noRowFastest = options.noRowFastest ?? false;
    /**
     * The control arm for x's padded row stride
     * (`webgl/runtime/pad-stride.js`): a row of `IN4` texels laid at its own
     * width, which wraps across texture rows whenever `IN4` does not divide the
     * texture width. Bit-identical, so this prices the ADDRESSING.
     */
    this.noStridePad = options.noStridePad ?? false;
    /**
     * The control arm for the pair-blocked contraction's fragment order
     * (`webgl/af3/trunk/triangle.js`): the CHANNEL group varying fastest, as it
     * did, rather than the pair block.
     *
     * 🔴 THE SAME PROPERTY `ROW_FASTEST_X_FLOATS` NAMES IN THE PROJECTION, in
     * the kernel where it is worth more. A contraction is `channels`
     * INDEPENDENT matmuls, one a channel, and one channel's operands at 68
     * tokens are 2 * 68 * 68 * 4 = 37 KiB - small enough to stay in cache for
     * the whole 68x68x68 product. With the channel varying fastest,
     * neighbouring fragments work on DIFFERENT channels, so 64 channel groups
     * are in flight together and none of their planes stays; with it varying
     * slowest they share one. Measured 1.49x at 68 tokens, bit-identical.
     */
    this.noContractChannelSlowest = options.noContractChannelSlowest ?? false;
    /**
     * The control arm for the pair block's SIZE CEILING
     * (`CONTRACT_BLOCK_MAX_PAIR_FLOATS`): block at every pair tensor size, as
     * it did, which is up to 4.8x the wrong kernel above the ceiling.
     */
    this.noContractCeiling = options.noContractCeiling ?? false;
    /**
     * The control arm for the contraction's PAIR AXES padded to a multiple of
     * four (`padPairPlanesSource`), which is what makes the coalesced reads and
     * the pair block legal at a length four does not divide - 66% of an
     * ESMFold2 fold at 75 tokens. Off, the odd lengths run the scalar loop.
     */
    this.noContractPad = options.noContractPad ?? false;
    /**
     * The control arm for the contraction's TILE-AWARE fragment mapping
     * (`blockedContractSource`): the channel group chosen from the texel index
     * rather than from the fragment's x range, so a rasterisation tile - a
     * square of (x, y), whose rows are W apart in the index - spans about a
     * hundred channel planes instead of one. Worth 3.29x, and bit-identical.
     */
    this.noContractTile = options.noContractTile ?? false;
    /**
     * The control arm for the in-major projection's ASPECT rule
     * (`blockedApplies`): the old `in * out >= 65536` threshold, which was
     * fitted with f32 weights where every fold uses float16.
     */
    this.noLinearAspect = options.noLinearAspect ?? false;
    this.noLinearTile = options.noLinearTile ?? false;
    this.linearTileWidth = options.linearTileWidth ?? 0;
    /**
     * The blocked projection's rows a fragment, forced - the control arm for
     * `blockRowsFor`'s fragment floor, and the way to sweep it. 0 leaves the
     * resolver alone.
     */
    this.linearBlockRows = options.linearBlockRows ?? 0;
    /**
     * The control arm for bioemu2's pair bias normalising a row ONCE
     * (`webgl/bioemu2/denoiser.js`): the statistics recomputed per head, at
     * 3584 texelFetch a fragment rather than 512. Bit-identical.
     */
    this.noPairBiasFast = options.noPairBiasFast ?? false;
    /**
     * The control arm for bioemu2's attentions reading their head dimension
     * four channels at a time (`headDot`): one `fetch1` a channel, twice over,
     * which is four times the texelFetch for the same numbers.
     */
    this.noAttendDot4 = options.noAttendDot4 ?? false;
    /**
     * Where the HOST time inside `dispatch` goes, when asked for.
     *
     * 🔴 BECAUSE THE GPU IS 3% OF AN AF2 WebGL FOLD AND NOTHING SAID SO.
     * `doubleLabels` over every label - every draw issued twice - costs 40 ms
     * of a 1400 ms fold, so the arithmetic is not what this backend is waiting
     * for and no amount of arithmetic intensity moves it. The rest is this
     * side of the API: generating a kernel's source, looking it up, binding,
     * and the host arithmetic between dispatches. `insideMs` against the fold's
     * own wall is what separates the last of those from the first three.
     */
    this.host = options.host ? {
      insideMs: 0, programMs: 0, bindMs: 0, drawMs: 0, sourceBytes: 0,
    } : null;
    /** @type {Map<string, {program: WebGLProgram, uniforms: Map<string, WebGLUniformLocation>}>} */
    this.programs = new Map();
    this.dispatches = 0;
    this.draws = 0;
    this.compiles = 0;
    /** Each draw checks for a GL error. Cheap, and off for a benchmark. */
    this.checkErrors = options.checkErrors ?? true;
    /**
     * A relative dither applied inside every f16 rounding; see `half4` in
     * PRELUDE. Zero in use - it is an instrument for measuring what a
     * quantising chain's agreement floor actually is.
     */
    this.halfDither = options.halfDither ?? 0;
    this.maxBandRows = Math.min(
      gl.getParameter(gl.MAX_VIEWPORT_DIMS)[1], gl.getParameter(gl.MAX_TEXTURE_SIZE));
    /**
     * How many `vec4`s one invocation can write, which is the whole reason
     * attention is affordable here.
     *
     * 🔴 THIS IS THE BUDGET docs/A100.md's WHOLE ARGUMENT TURNS ON. A fragment
     * writing one texel has to recompute every q.k dot product once per group
     * of four channels it owns - 4.5x the arithmetic at a 32-channel head. With
     * `MAX_DRAW_BUFFERS` attachments it computes each dot product exactly once.
     * Eight here, four guaranteed by the specification.
     */
    this.maxDrawBuffers = gl.getParameter(gl.MAX_DRAW_BUFFERS);

    /**
     * Per-label GPU time, in milliseconds, when a caller sets this to a `Map`.
     *
     * 🔴 A ONE-TEXEL READ IS THE ONLY TIMER THIS API HAS. There are no
     * timestamp queries in WebGL2 - `EXT_disjoint_timer_query_webgl2` was
     * removed from Chrome over a side channel - so a kernel's time is measured
     * by WAITING for it: a `readPixels` of one texel of the output cannot
     * return until that draw has run, and the wall clock either side of it is
     * the draw. That makes profiling INTRUSIVE (every kernel is a pipeline
     * flush, so a profiled fold is slower than the one it describes and the
     * kernels that would have overlapped no longer do) and it is why this is
     * off unless asked for. What it is good for is the RANKING, which is the
     * question "where does the gap live" needs answered.
     *
     * @type {Map<string, number> | null}
     */
    this.profile = options.profile ?? null;
    /** How many times each label was dispatched, counted whenever profiling. */
    this.counts = new Map();
    this.profileAt = 0;
    this.profilePixel = new Float32Array(4);

    this.vertices = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vertices);
    gl.bufferData(gl.ARRAY_BUFFER, FULLSCREEN_TRIANGLE, gl.STATIC_DRAW);
    this.vao = gl.createVertexArray();
    /**
     * Framebuffers by the textures attached to them.
     *
     * 🔴 ONE FRAMEBUFFER, REATTACHED EVERY DISPATCH, WAS THE SINGLE LARGEST
     * COST IN THIS BACKEND - AND IT IS WHY THE BLOCKED PROJECTION LOOKED SLOWER
     * THAN THE KERNEL IT BEATS. Each dispatch called `framebufferTexture2D` for
     * all eight attachment points and then `checkFramebufferStatus`, and
     * changing an attachment makes the driver revalidate the whole
     * framebuffer - a synchronous query. Measured through
     * tools/gpu/bench-webgl-linear.js: the blocked matmul alone is 2-3x the
     * one-texel kernel, its repack pass is a texel in and a texel out, and
     * the two together came out 2-7x SLOWER than the one kernel, because the
     * attachment set now alternated between four planes and one output and so
     * was revalidated twice a projection. Timed in isolation, where the
     * attachments never change between iterations, both halves were fast. The
     * configuration is the cost, not the arithmetic.
     *
     * A framebuffer per attachment SET, bound rather than rebuilt, is the whole
     * fix: a fold reuses a handful of them because `Storage` pools its textures,
     * so the key repeats. Capped, because a fold that never repeats a set would
     * otherwise leak one per dispatch.
     *
     * @type {Map<string, WebGLFramebuffer>}
     */
    this.framebuffers = new Map();
    /** Stable small ids for textures, so an attachment set has a short key. */
    this.textureIds = new WeakMap();
    this.nextTextureId = 1;
    this.framebufferBuilds = 0;
    /** How many attachment sets to keep. A fold uses tens; the cap is slack. */
    this.maxFramebuffers = options.maxFramebuffers ?? 256;
  }

  allocate(elements, label) { return this.storage.allocate(elements, label); }
  zeros(elements, label) { return this.storage.zeros(elements, label); }
  /** A weight, in float16 if this execution is holding large ones that way. */
  uploadWeight(values, label) { return this.storage.uploadWeight(values, label); }
  /** An int5 weight left as codes, or null if this one cannot be. */
  uploadInt5Weight(source, label) { return this.storage.uploadInt5Weight(source, label); }
  allocateHalf(elements, label) { return this.storage.allocateHalf(elements, label); }

  upload(values, label) { return this.storage.upload(values, label); }
  read(tensor) { return this.storage.read(tensor); }
  release(tensor) { return this.storage.release(tensor); }

  /** The program for this source, compiled once. */
  program(source) {
    const cached = this.programs.get(source);
    if (cached !== undefined) return cached;
    const gl = this.gl;
    const program = linkProgram(gl, source);
    // The attribute is bound once per program, into this execution's one VAO.
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vertices);
    const position = gl.getAttribLocation(program, "position");
    // `position` is the shared VERTEX shader's attribute, not the kernel's: a
    // fragment shader that declares it would be asking for a varying nothing
    // writes. A negative location means the vertex stage did not link.
    if (position < 0) throw new Error("the shared vertex shader lost its `position` attribute");
    gl.enableVertexAttribArray(position);
    gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
    const entry = { program, uniforms: new Map() };
    this.programs.set(source, entry);
    this.compiles += 1;
    return entry;
  }

  /**
   * Bind the framebuffer these targets are attached to, building it once.
   *
   * The attachment set and `drawBuffers` are both per-framebuffer state in
   * WebGL2, so a cached framebuffer needs neither re-attaching nor re-asking
   * for its completeness: binding it is one call.
   */
  #bindTargets(targets, label) {
    const gl = this.gl;
    let key = "";
    for (const tensor of targets) {
      let id = this.textureIds.get(tensor.texture);
      if (id === undefined) {
        id = this.nextTextureId;
        this.nextTextureId += 1;
        this.textureIds.set(tensor.texture, id);
      }
      key += `${id},`;
    }
    const cached = this.framebuffers.get(key);
    if (cached !== undefined) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, cached);
      return;
    }
    if (this.framebuffers.size >= this.maxFramebuffers) {
      // Oldest first: a Map keeps insertion order, and the set least recently
      // BUILT is the best guess at the one least likely to come back.
      const oldest = this.framebuffers.keys().next().value;
      gl.deleteFramebuffer(this.framebuffers.get(oldest));
      this.framebuffers.delete(oldest);
    }
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    for (let t = 0; t < targets.length; t += 1) {
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + t, gl.TEXTURE_2D,
        targets[t].texture, 0);
    }
    gl.drawBuffers(targets.map((_, t) => gl.COLOR_ATTACHMENT0 + t));
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error(`${label}: cannot draw into `
        + `${targets.map((tensor) => tensor.label).join(", ")} (0x${status.toString(16)})`);
    }
    this.framebuffers.set(key, framebuffer);
    this.framebufferBuilds += 1;
  }

  #uniform(entry, name) {
    if (!entry.uniforms.has(name)) {
      entry.uniforms.set(name, this.gl.getUniformLocation(entry.program, name));
    }
    return entry.uniforms.get(name);
  }

  /**
   * Run one kernel into `output`.
   *
   * `inputs` is an object of sampler name to tensor, bound to units in its own
   * key order. `ints` and `floats` are the scalar uniforms, kept APART rather
   * than inferred from the value: `bias: 0` is a float that looks like an int,
   * and `uniform1i` on a float uniform is a silent GL error on some drivers and
   * a wrong value on others.
   *
   * `outputs` draws into several tensors at once - what a kernel declares as
   * `layout(location = t) out vec4`. They must be the same shape, because they
   * share one viewport, and there may be at most `maxDrawBuffers` of them.
   *
   * @param {{source: string, output?: import("./storage.js").Tensor,
   *   outputs?: import("./storage.js").Tensor[],
   *   inputs?: Record<string, import("./storage.js").Tensor>,
   *   ints?: Record<string, number>, floats?: Record<string, number>,
   *   optionalInts?: Record<string, number>, label?: string}} call
   */
  dispatch(call) {
    const gl = this.gl;
    const {
      source, inputs = {}, ints = {}, floats = {}, optionalInts = {}, label = "kernel",
    } = call;
    const targets = call.outputs ?? [call.output];
    if (targets.length === 0 || targets.some((tensor) => tensor === undefined)) {
      throw new Error(`${label}: a dispatch needs an output`);
    }
    if (targets.length > this.maxDrawBuffers) {
      throw new RangeError(
        `${label}: ${targets.length} render targets, this context allows ${this.maxDrawBuffers}`);
    }
    const output = targets[0];
    for (const tensor of targets) {
      if (tensor.width !== output.width || tensor.height !== output.height) {
        throw new RangeError(
          `${label}: render targets share one viewport, so ${tensor.label} `
          + `(${tensor.width}x${tensor.height}) cannot join ${output.label} `
          + `(${output.width}x${output.height})`);
      }
    }
    // 🔴 A TENSOR BOUND BOTH WAYS IS UNDEFINED AND MUST BE REFUSED BY NAME. The
    // one-output path checks this at each kernel's door; with several targets the
    // chance of an alias rises and the symptom - a "feedback loop" framebuffer,
    // or silently stale texels - names nothing.
    const bound = new Set(Object.values(inputs));
    for (const tensor of targets) {
      if (bound.has(tensor)) {
        throw new Error(`${label}: ${tensor.label} is both an input and a render target`);
      }
    }
    // 🔴 AND A RELEASED TENSOR IS STILL A `Tensor` OBJECT WITH A TEXTURE FIELD,
    // WHICH IS WHY THIS HAS TO BE ASKED. Reading one back worked by accident for
    // as long as `release` deleted the texture: a deleted texture samples as
    // zeros rather than failing, so a stage that read past a release got a
    // plausible answer. The moment textures are POOLED the same read returns
    // another tensor's live contents, and the fold is wrong by an amount that
    // depends on allocation order. `Storage.live` is the only thing that knows,
    // so the dispatch asks it.
    if (this.checkErrors) {
      for (const [name, tensor] of Object.entries(inputs)) {
        if (!this.storage.live.has(tensor)) {
          throw new Error(`${label}: input \`${name}\` (${tensor.label}) has been released`);
        }
      }
      for (const tensor of targets) {
        if (!this.storage.live.has(tensor)) {
          throw new Error(`${label}: render target ${tensor.label} has been released`);
        }
      }
    }
    const enteredAt = this.host === null ? 0 : performance.now();
    const entry = this.program(source);
    const compiledAt = this.host === null ? 0 : performance.now();
    gl.useProgram(entry.program);
    gl.bindVertexArray(this.vao);
    this.#bindTargets(targets, label);

    let unit = 0;
    for (const [name, tensor] of Object.entries(inputs)) {
      // 🔴 AN UNDECLARED SAMPLER IS A REFUSAL, NOT A NO-OP. `getUniformLocation`
      // returns null for a name the program does not have - including one the
      // compiler removed as unused - and `uniform1i(null, ...)` is silently
      // ignored, so the kernel then reads whatever was bound to unit 0. That is
      // the "read what was bound from before" failure the triangle kernel's own
      // header records, and it returns a plausible wrong answer.
      const location = this.#uniform(entry, name);
      if (location === null) {
        throw new Error(`${label}: no sampler named \`${name}\` in this kernel`);
      }
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, tensor.texture);
      gl.uniform1i(location, unit);
      unit += 1;
    }
    for (const [name, value] of Object.entries(ints)) {
      const location = this.#uniform(entry, name);
      if (location === null) throw new Error(`${label}: no int uniform \`${name}\``);
      gl.uniform1i(location, value);
    }
    for (const [name, value] of Object.entries(floats)) {
      const location = this.#uniform(entry, name);
      if (location === null) throw new Error(`${label}: no float uniform \`${name}\``);
      gl.uniform1f(location, value);
    }
    // 🔴 A SHAPE UNIFORM MAY BE OPTIMISED AWAY, AND THAT IS NOT AN ERROR. The
    // shapes are uniforms rather than baked constants so that one program
    // serves every size (see `webgl/kernels/linear.js`) - but a kernel whose
    // branch does not read `ROWS` loses it in the compiler, `getUniformLocation`
    // returns null, and the strict check above would refuse a correct kernel.
    // Everything that is NOT a shape stays strict, because a silently ignored
    // uniform is how a kernel comes to read whatever was bound before it.
    for (const [name, value] of Object.entries(optionalInts)) {
      const location = this.#uniform(entry, name);
      if (location !== null) gl.uniform1i(location, value);
    }

    if (this.halfDither !== 0) {
      // Set where the kernel has it and skip where it does not: a kernel that
      // never rounds has no such uniform, and that is not an error.
      const location = this.#uniform(entry, "halfDither");
      if (location !== null) gl.uniform1f(location, this.halfDither);
    }

    this.dispatches += 1;
    if (this.profile !== null) {
      this.counts.set(label, (this.counts.get(label) ?? 0) + 1);
      // The clock starts where the PREVIOUS kernel's wait ended, so the whole
      // of the wall is attributed and a kernel carries its own host setup.
      if (this.profileAt === 0) this.profileAt = performance.now();
    }
    const boundAt = this.host === null ? 0 : performance.now();
    // 🔴 THE OTHER HALF OF `doubleLabels`, AND THE ONE THAT SETTLES AN ARGUMENT.
    // Doubling every draw said the GPU was 2% of a bioemu2 fold; profiling with
    // a barrier after each dispatch said it was 90%. They cannot both be right,
    // and the arm that decides is the fold with NO draws at all: whatever is
    // left is everything this backend does APART from drawing. The answer is a
    // fold of garbage, so it is a timing probe only.
    const passes = this.skipDraws ? 0
      : this.doubleLabels !== null && this.doubleLabels.test(label) ? 2 : 1;
    for (let pass = 0; pass < passes; pass += 1) {
    for (let row = 0; row < output.height; row += this.maxBandRows) {
      const rows = Math.min(this.maxBandRows, output.height - row);
      // The viewport's ORIGIN is the band's first row, so gl_FragCoord.y is
      // absolute and the band lands in the rows it belongs to - no row uniform
      // and no re-draw on readback.
      gl.viewport(0, row, output.width, rows);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      this.draws += 1;
    }
    }
    if (this.checkErrors) {
      const error = gl.getError();
      if (error !== gl.NO_ERROR) {
        throw new Error(`${label}: GL error 0x${error.toString(16)} after the draw`);
      }
    }
    if (this.host !== null) {
      const now = performance.now();
      this.host.programMs += compiledAt - enteredAt;
      this.host.bindMs += boundAt - compiledAt;
      this.host.drawMs += now - boundAt;
      this.host.insideMs += now - enteredAt;
      this.host.sourceBytes += source.length;
    }
    if (this.nanWatch) {
      for (const tensor of targets) {
        // 🔴 A float16 OR PACKED-CODE TARGET CANNOT BE READ BACK AS FLOATS, and
        // this instrument says how many it therefore did not check rather than
        // reporting a clean run it did not have. The only such target is the
        // one-off int5 decode (`webgl/kernels/linear.js`), whose own gate -
        // tools/gpu/check-webgl-int5-decode.js - compares every element of it
        // against `readTensor`, which is a stronger check than this one.
        if (tensor.half === true || tensor.words === true) {
          this.nanWatchSkipped += 1;
          continue;
        }
        const values = this.storage.read(tensor);
        for (let i = 0; i < values.length; i += 1) {
          if (!Number.isFinite(values[i])) {
            throw new Error(`${label}: ${tensor.label}[${i}] is ${values[i]} - `
              + "this kernel read a texel nothing wrote");
          }
        }
      }
    }
    if (this.profile !== null) {
      // The framebuffer is still bound with the targets attached, so one texel
      // of the first one is the cheapest thing that waits for the draw.
      gl.readBuffer(gl.COLOR_ATTACHMENT0);
      gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, this.profilePixel);
      const now = performance.now();
      this.profile.set(label, (this.profile.get(label) ?? 0) + now - this.profileAt);
      this.profileAt = now;
    }
    return output;
  }

  /**
   * The profile as a sorted table: the `top` slowest labels, each with its
   * dispatch count, and what they add up to.
   *
   * `kernel` is the label up to its first colon, which is how the kernels here
   * are named (`linear:59x128x128:none:out-major`), so the per-kernel rollup
   * answers "which KERNEL is the fold" where the per-label rows answer "at
   * which shape".
   */
  profileTable(top = 20) {
    if (this.profile === null) return null;
    const rows = [...this.profile.entries()]
      .map(([label, ms]) => ({ label, ms: Number(ms.toFixed(1)), calls: this.counts.get(label) ?? 0 }))
      .sort((a, b) => b.ms - a.ms);
    const byKernel = new Map();
    for (const row of rows) {
      const kernel = row.label.split(":")[0];
      const seen = byKernel.get(kernel) ?? { kernel, ms: 0, calls: 0 };
      byKernel.set(kernel, { kernel, ms: seen.ms + row.ms, calls: seen.calls + row.calls });
    }
    const totalMs = rows.reduce((sum, row) => sum + row.ms, 0);
    return {
      totalMs: Number(totalMs.toFixed(1)),
      kernels: [...byKernel.values()]
        .map((row) => ({ ...row, ms: Number(row.ms.toFixed(1)), percent: Number((100 * row.ms / totalMs).toFixed(1)) }))
        .sort((a, b) => b.ms - a.ms),
      labels: rows.slice(0, top),
    };
  }

  stats() {
    return {
      dispatches: this.dispatches, draws: this.draws, programs: this.programs.size,
      framebuffers: this.framebuffers.size, framebufferBuilds: this.framebufferBuilds,
      ...(this.nanWatch ? { nanWatchSkipped: this.nanWatchSkipped } : {}),
      ...(this.host === null ? {} : { host: Object.fromEntries(
        Object.entries(this.host).map(([k, v]) => [k, Math.round(v)])) }),
      maxBandRows: this.maxBandRows, maxDrawBuffers: this.maxDrawBuffers,
      ...this.storage.stats(),
    };
  }

  destroy() {
    const gl = this.gl;
    this.storage.releaseAll();
    for (const { program } of this.programs.values()) gl.deleteProgram(program);
    this.programs.clear();
    gl.deleteBuffer(this.vertices);
    gl.deleteVertexArray(this.vao);
    for (const framebuffer of this.framebuffers.values()) gl.deleteFramebuffer(framebuffer);
    this.framebuffers.clear();
  }
}
