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
import { FULLSCREEN_TRIANGLE, TEXTURE_WIDTH, finishProgram, linkProgram, startProgram } from "./context.js";
import { Storage } from "./storage.js";

/**
 * The declarations and helpers every kernel here opens with.
 *
 * `linearTexel()` is the fragment's own texel index in the output. `fetch4`
 * reads a whole texel of an input; `fetch1` reads one float, which costs a
 * texel fetch and a select and is for the ragged cases (a mask, a bias) rather
 * than for anything in an inner loop.
 */
/**
 * A kernel's source with its LENGTHS as uniforms rather than constants: `code` declares a
 * `uniform highp int` for each of `sizes` (right after the `#version` line, so put no `const`
 * of the same name in it), and `dispatch` sets them. The text no longer depends on the length, so
 * one compiled program serves every sequence length - a new length compiled ~140 programs again
 * while they were baked in, ~3 s on Safari 16.3, where each link holds the page's thread.
 *
 * Widths that do not change with the length (channels, heads) stay `const int`: the loops over
 * them keep their compile-time bounds.
 *
 * @param {Record<string, number>} sizes
 * @param {string} code
 * @returns {{code: string, sizes: Record<string, number>}}
 */
/**
 * A shader-source builder memoised on its arguments, so a repeated shape returns the SAME string (or source object).
 *
 * 🔴 A FOLD BUILDS ITS GLSL ONCE A DISPATCH, AND THEN LOOKS IT UP BY THE STRING. An AF3 fold is ~30,000 dispatches and
 * ~104 MB of source (`--host`'s `sourceBytes`): the unrolled builders cost 148 ms per 30,000 in V8 and the program
 * cache's lookup by a freshly built string 261 ms more, since a new string's hash is computed over all of it. The same
 * string object has its hash cached, so both go. Arguments are numbers, strings, booleans or plain shape objects.
 *
 * @template {(...args: any[]) => any} F
 * @param {F} build
 * @returns {F}
 */
export function memoSource(build) {
  const cache = new Map();
  return /** @type {F} */ ((...args) => {
    const key = args.every((arg) => arg === null || typeof arg !== "object")
      ? args.map((arg) => (typeof arg === "string" ? `s${arg}` : String(arg))).join("\u0001")
      : JSON.stringify(args);
    let built = cache.get(key);
    if (built === undefined) {
      built = build(...args);
      cache.set(key, built);
    }
    return built;
  });
}

export function sized(sizes, code) {
  const at = code.indexOf("\n") + 1;
  if (!code.startsWith("#version") || at === 0) throw new Error("sized: the code must start with #version");
  const declarations = Object.keys(sizes).map((name) => `uniform highp int ${name};\n`).join("");
  return { code: code.slice(0, at) + declarations + code.slice(at), sizes };
}

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

/**
 * A kernel as a PART: its uniforms, helpers and the body of its `main`, rather than a finished
 * text, so that the runtime can put several into ONE program (`Execution.combine`), each body a
 * function chosen by a `KERNEL` uniform. On its own a part assembles into exactly the program it
 * would have been. Safari 16.3 links a program in ~19 ms on the page's thread whatever its size -
 * 8 small kernels linked in 158-185 ms as 8 programs and 26-33 ms as one
 * (tools/gpu/probe-webgl-uber.js) - so the fold's light kernels are parts.
 *
 * A part has ONE target (`result`), declares every uniform it reads in `uniforms`
 * ("sampler2D xTex", "float scale"; its `sizes` are declared for it), no module-scope `const`
 * (a width is a size), and helpers named for the kernel, since parts share one namespace.
 *
 * @param {Record<string, number>} sizes the uniforms `dispatch` sets, as `sized`'s
 * @param {{uniforms: string[], helpers?: string, body: string}} definition `body` writes
 *   `result` and may `return`
 * @returns {{code: string, sizes: Record<string, number>, part: object}}
 */
export function part(sizes, { uniforms, helpers = "", body }) {
  const piece = Object.freeze({ sizes: Object.keys(sizes), uniforms, helpers, body });
  return { code: assembleParts([piece]), sizes, part: piece };
}

/**
 * One program from parts: their uniforms once each (the same name declared two ways is an error,
 * and those parts are not put together), their helpers, and each body as a function `main`
 * chooses by `KERNEL` - or, for one part, as `main` itself.
 */
export function assembleParts(pieces) {
  const declarations = new Map();
  const declare = (name, line) => {
    const held = declarations.get(name);
    if (held !== undefined && held !== line) throw new Error(`${name} is declared both "${held}" and "${line}"`);
    declarations.set(name, line);
  };
  for (const piece of pieces) {
    for (const name of piece.sizes) declare(name, `uniform highp int ${name};`);
    for (const uniform of piece.uniforms) declare(uniform.split(/\s+/).at(-1), `uniform ${uniform};`);
  }
  const head = `${PRELUDE}${ONE_TARGET}${[...declarations.values()].join("\n")}\n`;
  const helpers = pieces.map((piece) => piece.helpers).filter(Boolean).join("\n");
  if (pieces.length === 1) return `${head}${helpers}\nvoid main() {\n${pieces[0].body}\n}`;
  if (declarations.has("KERNEL")) throw new Error("a part may not declare KERNEL");
  return `${head}uniform highp int KERNEL;\n${helpers}\n`
    + pieces.map((piece, index) => `void kernel${index}() {\n${piece.body}\n}`).join("\n")
    + `\nvoid main() {\n${pieces.map((_, index) => `  ${index ? "else " : ""}if (KERNEL == ${index}) kernel${index}();`).join("\n")}\n}`;
}

/** The PRELUDE without its `#version` line, which is where `sized` injects its uniforms. */
const PRELUDE_BODY = PRELUDE.slice(PRELUDE.indexOf("\n") + 1);

/** A stable 32-bit digest, for naming one kernel's declarations apart from another's. */
function digestOf(text) {
  let hash = 2166136261;
  for (let at = 0; at < text.length; at += 1) hash = Math.imul(hash ^ text.charCodeAt(at), 16777619);
  return (hash >>> 0).toString(36);
}

/**
 * A kernel's module-scope declarations, split: each statement and each function definition, or
 * null if anything is left over that is neither.
 */
function headStatements(head) {
  const items = [];
  let depth = 0;
  let from = 0;
  for (let at = 0; at < head.length; at += 1) {
    const character = head[at];
    if (character === "{") depth += 1;
    else if (character === "}") {
      depth -= 1;
      if (depth < 0) return null;
      if (depth === 0) { items.push(head.slice(from, at + 1)); from = at + 1; }
    } else if (character === ";" && depth === 0) {
      items.push(head.slice(from, at + 1));
      from = at + 1;
    }
  }
  return depth === 0 && head.slice(from).trim() === ""
    ? items.map((item) => item.trim()).filter(Boolean) : null;
}

/**
 * The same kernel as a PART, read off a finished source - or null if it cannot be one.
 *
 * 🔴 BECAUSE THE BLOCKER WAS NEVER THE SHAPE OF `part`, IT WAS THAT PARTS SHARE ONE NAMESPACE.
 * Seven of this tree's ~150 kernels are written as parts; the rest are finished strings carrying
 * their widths as `const int` - which `sized` deliberately leaves as constants so a loop keeps its
 * compile-time bound - and their own helper functions, and two kernels cannot both declare `CH4`
 * or `coordinateOf` in one program. So each kernel's own declarations are RENAMED by a digest of
 * its source, along with every reference to them in its uniforms, helpers and body. The UNIFORMS
 * keep their names, which is what lets two kernels share a sampler unit. Nothing is rewritten by
 * hand and no kernel loses a bound.
 *
 * Refused, by design rather than as a limitation: a kernel with SEVERAL render targets, since a
 * part writes `result` - which is what keeps the heavy projections and attentions out - and one
 * whose module scope holds anything but uniforms, constants and functions (a struct, a global).
 *
 * @param {string} code a finished fragment source built on PRELUDE
 * @returns {{sizes: string[], uniforms: string[], helpers: string, body: string} | null}
 */
export function partFromSource(code) {
  const mainAt = code.indexOf("void main()");
  if (mainAt < 0 || !code.includes(PRELUDE_BODY)) return null;
  const head = code.slice(0, mainAt).replace(/^#version[^\n]*\n/, "").replace(PRELUDE_BODY, "")
    // GLSL has no strings, so the comments can come out before anything else is read.
    .replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/[^\n]*/g, "");
  const items = headStatements(head);
  if (items === null) return null;
  const uniforms = [];
  const declarations = [];
  const names = [];
  let targets = 0;
  for (const item of items) {
    if (item === "out vec4 result;") targets += 1;
    else if (item.startsWith("uniform ")) uniforms.push(item.slice("uniform ".length, -1));
    else if (item.startsWith("const ")) {
      const name = item.match(/^const\s+\S+\s+([A-Za-z_]\w*)/)?.[1];
      if (name === undefined) return null;
      names.push(name);
      declarations.push(item);
    } else if (item.endsWith("}")) {
      // A function definition, named by the first identifier that an open bracket follows.
      const name = item.match(/([A-Za-z_]\w*)\s*\(/)?.[1];
      if (name === undefined) return null;
      names.push(name);
      declarations.push(item);
    } else return null;
  }
  if (targets !== 1) return null;
  const opens = code.indexOf("{", mainAt);
  const closes = code.lastIndexOf("}");
  if (opens < 0 || closes < opens) return null;
  let rename = (text) => text;
  if (names.length > 0) {
    const prefix = `p${digestOf(code)}_`;
    // 🔴 NO LOOKBEHIND: Safari before 16.4 fails a whole MODULE that contains one at load, which
    // `npm run check:safari` exists to catch. The character before the name is matched and put
    // back instead, so that a field or a swizzle - `.x`, `.count` - is never renamed.
    const pattern = new RegExp(`(^|[^.\\w])(${names.join("|")})\\b`, "g");
    rename = (text) => text.replace(pattern, `$1${prefix}$2`);
  }
  return {
    sizes: [],
    uniforms: uniforms.map(rename),
    helpers: declarations.map(rename).join("\n"),
    body: rename(code.slice(opens + 1, closes)),
  };
}

/**
 * A part's samplers, each with whether it is an INTEGER one (`usampler2D`, for packed codes).
 *
 * 🔴 THE TYPE IS CARRIED BECAUSE ONE PLACEHOLDER CANNOT SERVE BOTH. A combined draw points the
 * samplers it does not use at a one-texel texture of its own, and two samplers of different types
 * on one texture unit is GL_INVALID_OPERATION - which arrived as
 * `bioemu2.fourier-add:68x768: GL error 0x502 after the draw` the moment a bin held an int5
 * kernel's `usampler2D` beside a float kernel's `sampler2D`.
 */
const samplersOf = (piece) => piece.uniforms.filter((uniform) => /sampler/.test(uniform))
  .map((uniform) => ({ name: uniform.split(/\s+/).at(-1), integer: /\busampler|\bisampler/.test(uniform) }));

/** `count` render targets named result0..result{count-1}, for a wide kernel. */
export const targetDeclarations = (count) => Array.from(
  { length: count }, (_, t) => `layout(location = ${t}) out vec4 result${t};`).join("\n") + "\n";

export class Execution {
  /** @param {WebGL2RenderingContext} gl */
  constructor(gl, options = {}) {
    this.gl = gl;
    this.storage = new Storage(gl, { pooling: options.pooling ?? false,
      poolLimitBytes: options.poolLimitBytes ?? Infinity,
      ...(options.halfWeightFloor === undefined ? {} : { halfWeightFloor: options.halfWeightFloor }),
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
     * Issue no draws for the dispatches whose label matches: `skipDraws` for one family.
     *
     * 🔴 THE ATTRIBUTION ARM THAT HIDDEN-SURFACE REMOVAL CANNOT FOOL. `doubleLabels` prices a family by drawing it
     * twice, and on a tile-based GPU the first of two identical draws into one target is discarded before it is
     * shaded - so doubling reads as free. Removing a family's draws cannot be discarded: the fold's wall shrinks by
     * that family's draw time (its GPU work and its render pass). The answer is garbage; it is a timing probe only.
     *
     * @type {RegExp | null}
     */
    this.skipLabels = options.skipLabels ?? null;
    /**
     * The control arm for the outer product mean contracting the sequences first (`outerProductFirst` in
     * kernels/outer-product-mean.js): the two-draw form through `intermediate[s][i][cr][z]`, as before.
     */
    this.noOpmOuterFirst = options.noOpmOuterFirst ?? false;
    this.noGridFusedProjection = options.noGridFusedProjection ?? false;
    /** The control arm for a page fold's pause one slice behind the GPU (`laggedPause` in runtime/steps.js). */
    this.noLaggedPause = options.noLaggedPause ?? false;
    /** The control arm for the triangle's operands written straight into the contraction's layout (`operandsLaidOutSource`). */
    this.noTriangleFusedLayout = options.noTriangleFusedLayout ?? false;
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
    // The 4 i x 8 j x 1 channel pair block - see `widePairBlockSource`.
    this.noContractWide = options.noContractWide ?? false;
    // The LayerNorm statistics in their own pass - see `layerNorm`.
    this.noNormSplit = options.noNormSplit ?? false;
    /**
     * The control arm for the in-major projection's ASPECT rule
     * (`blockedApplies`): the old `in * out >= 65536` threshold, which was
     * fitted with f32 weights where every fold uses float16.
     */
    this.noLinearAspect = options.noLinearAspect ?? false;
    this.noLinearTile = options.noLinearTile ?? false;
    // 🔴 OFF HERE AND ON IN THE FOLD DRIVERS, as `halfWeights` is - see
    // `padStride`, the only pass it reaches.
    this.halfActivations = options.halfActivations ?? false;
    // Narrow x on shapes whose row stride ALREADY divides the texture width,
    // where `padStride` would not otherwise run. A per-model trade - see
    // `linear`'s note and `webglExecutionOptions`.
    this.narrowAlignedX = options.narrowAlignedX ?? false;
    // 🔴 THE SAME NARROWING ON THE UNBLOCKED PATH, AND A SEPARATE OPTION
    // BECAUSE THE TWO WERE MEASURED TOGETHER AND ONLY ONE MAY PAY. Combined
    // they read 1.016x on a bioemu2 fold, which is near the run-to-run spread,
    // while the profile has `linear:4624x256x512` going 212 to 189 GFLOP/s and
    // `pad-stride` from 5284 calls to 9502 - so the unblocked half adds a pass
    // per call and may be a net loss. Split so each can be priced.
    this.narrowUnblockedX = options.narrowUnblockedX ?? false;
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
     * The control arm for the combined programs (`combine`, `part`, `partFromSource`): every
     * kernel its own program, as this backend did before it had any. A link costs ~40 ms on this
     * M2's Chrome whatever the program holds, so what it measures is the SIZE of that win and
     * whether sharing one program cost the draws anything.
     */
    this.noCombine = options.noCombine ?? false;
    /**
     * How much GLSL one combined program may hold, in characters.
     *
     * 🔴 BECAUSE A BIN IS NOT FREE TO FILL: on this M2's Chrome 53 AF2 programs link in 519 ms
     * and the same kernels in 36 programs take 1226, where ESMFold2's 82 in 5 bins go 719 to 274.
     * The fixed cost of a link is ~40 ms either way; what differs is the CONTENT, and a bin that
     * merges a dozen loop-heavy kernels costs the driver more than the twelve separately. The cap
     * is swept rather than argued - `--combine-bytes=`.
     */
    this.combineBytes = options.combineBytes ?? 0;
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
      insideMs: 0,
      programMs: 0,
      bindMs: 0,
      // 🔴 bindMs SPLIT THREE WAYS, because 4.4 s of a 22.4 s fold was in it
      // and caching `useProgram` and the VAO moved none of it. Guessing which
      // part had already cost three wrong hypotheses in one session.
      targetsMs: 0,
      inputsMs: 0,
      uniformsMs: 0,
      drawMs: 0,
      sourceBytes: 0,
    } : null;
    /** @type {Map<string, {program: WebGLProgram, uniforms: Map<string, WebGLUniformLocation>}>} */
    this.programs = new Map();
    /** While a Map, `dispatch` only records each kernel's source in it, with its label (see `prepare`). */
    this.collect = null;
    /** The parts (`part`) dispatch has met, by their own code, and those put in a combined program. */
    this.parts = new Map();
    this.combined = new Map();
    /** The one-texel textures a combined draw's unused samplers are pointed at, by type. */
    this.placeholders = {};
    /** Draws given back the zero an unset scalar uniform has on its own, by kernel and name. */
    this.unsetUniforms = new Map();

    this.dispatches = 0;
    this.draws = 0;
    this.compiles = 0;
    /** Each draw checks for a GL error. Cheap, and off for a benchmark. */
    this.checkErrors = options.checkErrors ?? true;
    /**
     * Flush the queued commands every this many dispatches; 0 never does.
     *
     * 🔴 `getError` AFTER EVERY DRAW WAS PACING THE GPU AS WELL AS CHECKING IT. It is a synchronous round trip - 1.4 s of
     * an AF3 fold's 1.6 s of host time with no draws issued at all - and taking it away left the fold no faster: the
     * trunk went 1.94 -> 1.41 s and the diffusion 3.44 -> 3.99, because a round trip also flushes, and without one the
     * GPU waits for whatever the driver decides to submit. A flush hands the work over without waiting for an answer.
     */
    this.flushEvery = options.flushEvery ?? 0;
    /**
     * `invalidateFramebuffer` on a dispatch's targets before it draws: their old contents are never read.
     *
     * 🔴 ON A TILE-BASED GPU A DRAW INTO A NEW TARGET IS A RENDER PASS, AND A PASS LOADS ITS TARGET UNLESS TOLD THE
     * CONTENTS DO NOT MATTER - a full read of every output before it is overwritten, ~30,000 times an AF3 fold.
     *
     * 🔴 AND IT IS A LOSS IN CHROME ON AN M2, SO IT IS OFF: an AF3 fold 5.6 s -> 6.3-6.5 s with it, bit-identical. The
     * likeliest reading is that the browser implements WebGL's invalidate with clears of its own. Kept as an arm,
     * because a driver that honours it as a load action is exactly where it would pay.
     */
    this.invalidateTargets = options.invalidateTargets ?? false;
    this.attachmentLists = Array.from({ length: 9 },
      (_, count) => Array.from({ length: count }, (__, t) => gl.COLOR_ATTACHMENT0 + t));
    this.sinceFlush = 0;
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
    /** @type {(WebGLTexture | null)[]} The texture on each sampler unit. */
    this.boundUnits = [];
    this.framebufferHits = 0;
    this.framebufferMisses = 0;
    /** One framebuffer per target count, re-attached - see `#bindTargets`. */
    /** @type {Map<number, WebGLFramebuffer>} */
    this.reusedFramebuffers = new Map();
    /** @type {Set<string>} */
    this.checkedShapes = new Set();
    this.noFramebufferReuse = options.noFramebufferReuse ?? false;
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
    return this.#adopt(source, linkProgram(this.gl, source));
  }

  /**
   * Compile these kernels' programs ahead of the draws that need them, without holding the
   * thread for long: links are started in batches of ~16 ms of the thread, a task each, and
   * then waited for (KHR_parallel_shader_compile's completion status, polled between tasks) and
   * adopted. A fold's first run links ~120 programs, ~2.2 s on Safari 16.3, which all fell
   * inside a few of the fold's slices; `foldAf3Async` collects them with a dry run (`collect`)
   * and calls this first. (Kernels take their lengths as uniforms - see `sized` - so a later fold
   * at another length finds them all compiled.)
   *
   * 🔴 THE BATCHES ARE TIMED, NOT COUNTED, because the extension does not mean the same thing
   * everywhere: Chrome links off the thread and starts all ~150 in one batch, while Safari 16.3
   * offers the extension and still links inside `linkProgram` (up to 61 ms each) - started all at
   * once there, they were one 3.1 s stall.
   *
   * @param {Iterable<string>} sources fragment sources
   */
  /**
   * Put the parts among these sources into as few programs as they fit in (`assembleParts`), for
   * draws of them to use from now on; a part already in one, or that cannot join any (a uniform
   * declared another way), stays as it is. Nothing is compiled here.
   *
   * @param {Iterable<string>} sources
   */
  combine(sources) {
    if (this.noCombine) return;
    // ...packed into programs of at most MAX_TEXTURE_IMAGE_UNITS samplers each (16 here: one
    // part alone, the per-atom conditioning, reads 11), a part that does not fit opening the next.
    // ...and one unit kept back for the placeholders the draws point their spare samplers at.
    const units = this.gl.getParameter(this.gl.MAX_TEXTURE_IMAGE_UNITS) - 2;
    const samplers = (pieces) => new Set(pieces.flatMap(samplersOf).map(({ name }) => name)).size;
    const bins = [];
    for (const source of new Set(sources)) {
      if (this.combined.has(source)) continue;
      // A kernel written as a part, else the same kernel read off its finished source.
      const piece = this.parts.get(source) ?? partFromSource(source);
      if (piece === undefined || piece === null) continue;
      const fits = (bin) => {
        const pieces = [...bin.map(([, held]) => held), piece];
        if (samplers(pieces) > units) return false;
        try {
          const assembled = assembleParts(pieces);
          return this.combineBytes === 0 || assembled.length <= this.combineBytes;
        } catch {
          return false; // declared differently from one already in
        }
      };
      const bin = bins.find(fits);
      if (bin === undefined) bins.push([[source, piece]]);
      else bin.push([source, piece]);
    }
    for (const bin of bins) {
      if (bin.length < 2) continue;
      const pieces = bin.map(([, piece]) => piece);
      const code = assembleParts(pieces);
      const held = new Map(pieces.flatMap(samplersOf).map((sampler) => [sampler.name, sampler]));
      const names = [...held.values()];
      bin.forEach(([source], index) => this.combined.set(source, { code, index, samplers: names,
        // ...and the SCALAR uniforms this part itself declares, with their type: a draw of it that
        // does not set one gets a zero rather than the last kernel's value (see `dispatch`).
        reads: bin[index][1].uniforms.filter((uniform) => !/sampler/.test(uniform))
          .map((uniform) => ({ name: uniform.split(/\s+/).at(-1),
                               integer: /\bu?int\b/.test(uniform) })) }));
    }
  }

  /**
   * The programs `run` would draw with, in the order it first draws with them, and nothing else:
   * a dry run on VIRTUAL tensors (`Storage.virtual`), so no texture is created, no draw is issued
   * and every read returns zeros. What `prepare` wants, for a model whose fold is one call.
   *
   * @param {() => void | Promise<void>} run the fold, or a stage of it; awaited if it returns a
   *   promise, since two of the four drivers here yield to the page
   * @returns {Promise<Map<string, string>>} source -> the label of the first dispatch that wanted it
   */
  async collectPrograms(run) {
    const held = this.collect;
    this.collect = new Map();
    this.storage.virtual = true;
    try {
      await run();
      return this.collect;
    } finally {
      this.collect = held;
      this.storage.virtual = false;
    }
  }

  /** The program a kernel's source is drawn with: the combined one it is in, or its own. */
  packaged(source) {
    return this.combined.get(source)?.code ?? source;
  }

  async prepare(sources) {
    const gl = this.gl;
    sources = [...sources];
    this.combine(sources);
    sources = sources.map((source) => this.packaged(source));
    const parallel = gl.getExtension("KHR_parallel_shader_compile");
    const nextTask = (ms) => new Promise((resolve) => { setTimeout(resolve, ms); });
    const pending = [];
    let batchFrom = performance.now();
    for (const source of new Set(sources)) {
      if (this.programs.has(source)) continue;
      pending.push([source, startProgram(gl, source)]);
      if (performance.now() - batchFrom > 16) {
        await nextTask(0);
        batchFrom = performance.now();
      }
    }
    if (parallel !== null) {
      while (pending.some(([, { program }]) =>
        !gl.getProgramParameter(program, parallel.COMPLETION_STATUS_KHR))) {
        await nextTask(8);
      }
    }
    // Without the extension the status is asked for one program a task (each waits for its link).
    for (const [source, started] of pending) {
      if (!this.programs.has(source)) this.#adopt(source, finishProgram(gl, started));
      if (parallel === null) await nextTask(0);
    }
  }

  /** A linked program into the cache, with its one attribute bound. */
  #adopt(source, program) {
    const gl = this.gl;
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
    // 🔴 A TEXTURE THAT IS BOUND TO A SAMPLER AND IS ALSO A RENDER TARGET IS A
    // FEEDBACK LOOP, AND WEBGL REFUSES THE DRAW - `GL error 0x502` after it,
    // naming the kernel. Nothing used to track what was bound, so every
    // dispatch rebound units 0..n-1 and the hazard only showed when a stale
    // texture above that count came back as a target. That is why `--pool`
    // errored (`bioemu2.m0:1x68: GL error 0x502`) and is on the "measured loss"
    // list, and why a binding cache fails the same way: both make a texture
    // linger on a unit. Clearing the conflict here fixes both at the source.
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
      this.framebufferHits += 1;
      gl.bindFramebuffer(gl.FRAMEBUFFER, cached);
      return;
    }
    this.framebufferMisses += 1;
    // 🔴 AND THE CACHE NEVER HITS, SO IT WAS COSTING WHAT IT WAS BUILT TO SAVE:
    // **0 hits and 54,381 misses** in a bioemu2 fold, which is every dispatch,
    // because the key is the texture OBJECTS and `Storage` only pools when
    // `--pool` is on - so every allocation is a fresh texture, a fresh key and
    // a fresh framebuffer. `--host` put `#bindTargets` at **4334 ms of a 22.4 s
    // fold**, 98.5% of `bindMs`, in createFramebuffer, the attachments,
    // `drawBuffers`, `checkFramebufferStatus` and a deleteFramebuffer at the
    // cap - per draw.
    //
    // So a miss RE-ATTACHES one framebuffer per target COUNT instead of
    // building a new one. The attachments and `drawBuffers` are per-framebuffer
    // state that has to be set either way; what this drops is the create, the
    // delete and the completeness query.
    // 🔴 EACH PATH SERVES THE CONFIGURATION IT IS BETTER FOR, AND BOTH ARE
    // REACHABLE. Without pooling every allocation is a fresh texture, so the
    // keyed cache cannot hit - 0 hits and 54,381 misses - and re-attaching one
    // framebuffer per target COUNT takes this call from 4285 ms to 92. With
    // pooling the objects repeat, the keyed cache hits **57,142 times against
    // 255 misses** and is the cheaper of the two (69 ms against 89), because a
    // hit is one `bindFramebuffer` where a reuse is that plus the attachments.
    // Gating on `pooling` rather than preferring one outright is what keeps the
    // keyed cache from being dead code with a comment claiming otherwise, which
    // is what it became when the reuse path first went in.
    if (!this.noFramebufferReuse && !this.storage.pooling) {
      let reused = this.reusedFramebuffers.get(targets.length);
      const fresh = reused === undefined;
      if (fresh) {
        reused = gl.createFramebuffer();
        this.reusedFramebuffers.set(targets.length, reused);
      }
      gl.bindFramebuffer(gl.FRAMEBUFFER, reused);
      for (let t = 0; t < targets.length; t += 1) {
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + t, gl.TEXTURE_2D,
          targets[t].texture, 0);
      }
      if (fresh) {
        gl.drawBuffers(targets.map((_, t) => gl.COLOR_ATTACHMENT0 + t));
      }
      // 🔴 THE COMPLETENESS QUERY ONLY ONCE PER COUNT AND RECTANGLE, because it
      // is the expensive call and the thing it checks - that these dimensions
      // and format can be drawn into - is a property of the SHAPE rather than
      // of the particular texture. A shape this backend has already drawn into
      // is not going to become incomplete.
      const shape = `${targets.length}:${targets[0].width}x${targets[0].height}`;
      if (!this.checkedShapes.has(shape)) {
        const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
        if (status !== gl.FRAMEBUFFER_COMPLETE) {
          throw new Error(`${label}: cannot draw into ${targets.length} target(s) of `
            + `${targets[0].width}x${targets[0].height} (status 0x${status.toString(16)})`);
        }
        this.checkedShapes.add(shape);
      }
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
      inputs = {}, ints = {}, floats = {}, label = "kernel",
    } = call;
    // A `sized` source: one program text at every length, and the lengths as uniforms.
    const source = typeof call.source === "string" ? call.source : call.source.code;
    const optionalInts = call.source.sizes === undefined ? { ...call.optionalInts }
      : { ...call.optionalInts, ...call.source.sizes };
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
    if (call.source.part !== undefined && !this.parts.has(source)) this.parts.set(source, call.source.part);
    // A dry run (`collect` a Set): the program this draw needs, and no draw.
    if (this.collect !== null) {
      this.collect.set(source, label);
      return output;
    }
    const enteredAt = this.host === null ? 0 : performance.now();
    // ...the combined program its part is in, choosing it by KERNEL.
    const combined = this.combined.get(source);
    const entry = this.program(combined?.code ?? source);
    if (combined !== undefined) optionalInts.KERNEL = combined.index;
    const compiledAt = this.host === null ? 0 : performance.now();
    // 🔴 AND CACHING ANY OF THIS IS NOT THE LEVER, MEASURED. `useProgram` and
    // the VAO cached was **1.004x** (22355 ms against 22434), and caching the
    // texture units broke the fold outright with `GL error 0x502` - a feedback
    // loop, because a cached binding lets a texture linger on a unit that a
    // later dispatch makes a render target. The split of `bindMs` says why it
    // could never have paid: of 4398 ms, `#bindTargets` is **4334** and the
    // input and uniform loops together are **64**. Reverted.
    gl.useProgram(entry.program);
    gl.bindVertexArray(this.vao);
    this.#bindTargets(targets, label);
    const targetsAt = this.host === null ? 0 : performance.now();

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
      // Recorded, not cached: `#bindTargets` needs to know what is bound to a
      // unit so it can release a texture that is about to be drawn into.
      this.boundUnits[unit] = tensor.texture;
      unit += 1;
    }
    // 🔴 A COMBINED PROGRAM'S OTHER SAMPLERS ARE STILL ITS SAMPLERS: they keep the units an earlier
    // draw gave them, and one holding this draw's target is a feedback loop (GL error 0x502) though
    // nothing reads it. They are pointed at a one-texel texture of their own.
    if (combined !== undefined) {
      const spare = combined.samplers.filter(({ name }) => !(name in inputs));
      // One placeholder unit per sampler TYPE in use, since a float and an integer sampler cannot
      // share a unit; `samplersOf` says why.
      for (const integer of [false, true]) {
        const named = spare.filter((sampler) => sampler.integer === integer);
        if (named.length === 0) continue;
        // ...on its own unit, made there: binding it while the last input's unit was active
        // replaced that input (every combined draw read a one-texel texture for its mask).
        gl.activeTexture(gl.TEXTURE0 + unit);
        const kind = integer ? "words" : "floats";
        if (this.placeholders[kind] === undefined) {
          const texture = gl.createTexture();
          gl.bindTexture(gl.TEXTURE_2D, texture);
          if (integer) {
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32UI, 1, 1, 0,
              gl.RGBA_INTEGER, gl.UNSIGNED_INT, null);
          } else {
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 1, 1, 0, gl.RGBA, gl.FLOAT, null);
          }
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
          this.placeholders[kind] = texture;
        }
        gl.bindTexture(gl.TEXTURE_2D, this.placeholders[kind]);
        this.boundUnits[unit] = this.placeholders[kind];
        for (const { name } of named) {
          const location = this.#uniform(entry, name);
          if (location !== null) gl.uniform1i(location, unit);
        }
        unit += 1;
      }
    }
    const inputsAt = this.host === null ? 0 : performance.now();
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

    // 🔴 A COMBINED PROGRAM'S SCALAR UNIFORMS PERSIST BETWEEN DRAWS, SO A KERNEL THAT LEAVES ONE
    // UNSET WOULD READ ANOTHER KERNEL'S VALUE. On its own a program's uniforms start at zero and a
    // kernel may lean on that - `triangle.contract` ends `result = sum + bias` and no caller has
    // ever set `bias` - so sharing a program takes that zero away, and the symptom is a wrong
    // number with nothing to point at. Measured over the four folds, 112 to 1159 draws a fold
    // leave one of their own scalars unset (mostly `scale`, which the shared elementwise part reads
    // in one branch of five), so this is not hypothetical: each is given its zero back. The folds
    // are identical either way, so it was latent - and latent by luck of the bin composition.
    // `halfDither` is excepted - it is this class's, and already set where it is not zero.
    if (combined !== undefined) {
      for (const { name, integer } of combined.reads) {
        if (name === "halfDither" || name in ints || name in floats || name in optionalInts) continue;
        const location = this.#uniform(entry, name);
        if (location === null) continue; // the compiler removed it: nothing in here reads it
        if (integer) gl.uniform1i(location, 0);
        else gl.uniform1f(location, 0);
        const key = `${label.split(":")[0]}:${name}`;
        this.unsetUniforms.set(key, (this.unsetUniforms.get(key) ?? 0) + 1);
      }
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
    const passes = this.skipDraws || (this.skipLabels !== null && this.skipLabels.test(label)) ? 0
      : this.doubleLabels !== null && this.doubleLabels.test(label) ? 2 : 1;
    // ...and its targets' old contents are dead: every kernel writes every texel of its output (the bands below cover
    // the whole texture), so the driver need not load them into the tile before drawing (`invalidateTargets`)
    if (this.invalidateTargets && passes > 0) {
      gl.invalidateFramebuffer(gl.FRAMEBUFFER, this.attachmentLists[targets.length]);
    }
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
    // ...and every `flushEvery` dispatches the queued commands go to the GPU (`flushEvery`'s note)
    if (this.flushEvery > 0 && ++this.sinceFlush >= this.flushEvery) {
      this.sinceFlush = 0;
      gl.flush();
    }
    if (this.checkErrors) {
      const error = gl.getError();
      if (error !== gl.NO_ERROR) {
        // 🔴 AND IT SAYS WHETHER AN INPUT IS ALSO A TARGET, because
        // GL_INVALID_OPERATION after a draw is almost always that - a texture
        // bound to a sampler while the framebuffer draws into it - and the
        // message without this named a kernel and left the reader to guess.
        // It is the pooling allocator's characteristic failure: a released
        // texture handed out for a new tensor while a kernel still names the
        // old one, which `--pool` is partly there to expose.
        const names = Object.entries(inputs)
          .filter(([, tensor]) => targets.some((t) => t.texture === tensor.texture))
          .map(([name]) => name);
        const stale = this.boundUnits
          .map((texture, u) => [texture, u])
          .filter(([texture, u]) => texture !== null && texture !== undefined
            && u >= unit && targets.some((t) => t.texture === texture))
          .map(([, u]) => u);
        if (names.length === 0 && stale.length > 0) {
          throw new Error(`${label}: GL error 0x${error.toString(16)} after the draw - a `
            + `target texture is still bound to sampler unit(s) ${stale.join(", ")} from an `
            + "EARLIER dispatch, which is a feedback loop this dispatch did not create");
        }
        throw new Error(`${label}: GL error 0x${error.toString(16)} after the draw`
          + (names.length === 0
            ? ` (no input shares a target's texture; ${targets.length} target(s), `
              + `inputs ${Object.keys(inputs).join(", ") || "none"})`
            : ` - the input(s) ${names.join(", ")} are drawn into by this same `
              + "dispatch, which is a feedback loop"));
      }
    }
    if (this.host !== null) {
      const now = performance.now();
      this.host.programMs += compiledAt - enteredAt;
      this.host.bindMs += boundAt - compiledAt;
      this.host.targetsMs += targetsAt - compiledAt;
      this.host.inputsMs += inputsAt - targetsAt;
      this.host.uniformsMs += boundAt - inputsAt;
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
      framebufferHits: this.framebufferHits,
      framebufferMisses: this.framebufferMisses,
      dispatches: this.dispatches, draws: this.draws, programs: this.programs.size,
      // Kernels sharing a program, and how many programs they share (`combine`): the two numbers
      // that say what the compile queue was spared, since a link costs the same whatever it holds.
      ...(this.unsetUniforms.size === 0 ? {}
        : { unsetUniforms: Object.fromEntries(this.unsetUniforms) }),
      combinedKernels: this.combined.size,
      combinedPrograms: new Set([...this.combined.values()].map(({ code }) => code)).size,
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
