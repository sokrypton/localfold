/**
 * The WebGL2 backend's diagnostic switches, read off a tool's own arguments.
 *
 * Every one of them is a CONTROL ARM or an INSTRUMENT, and they are here rather
 * than in each driver because a knob only one tool can reach is a knob nobody
 * turns - which is how `linearTallTile` came to be priced at zero on the WGSL
 * side for a configuration nobody had swept.
 *
 *   --pool / --no-pool    `Storage` keeps a released texture for the next
 *                         tensor of its rectangle instead of deleting it. OFF
 *                         by default because it is a measured loss (see
 *                         `Storage.pooling`), and it is also the arm that says
 *                         whether a kernel reads texels nothing wrote: a pooled
 *                         texture carries the last tensor's bytes, so a fold
 *                         with and without it must agree to the last bit.
 *   --no-int5-weights     Decode the bundle's five-bit weights on the HOST and
 *                         upload the result, as this backend used to. On by
 *                         default because it is strictly better where it
 *                         applies: the decoded value is the float32 the host
 *                         would have produced AND the codes are 0.875 bytes a
 *                         weight against float16's two. See
 *                         `webgl/runtime/int5.js`.
 *   --no-half-weights     Every weight in float32. A FOLD holds a weight of
 *                         65,536 floats or more in float16 (RGBA16F) by
 *                         default, which halves the bytes a big projection
 *                         streams - 1.4x and 1.95x less peak on bioemu2, with
 *                         its rmsd, TM, CA-CA, bond rms and contact count all
 *                         unchanged. It is the one optimisation here that is
 *                         not bit-identical (see `Storage.halfWeights`), so
 *                         this is the arm that compares f32 with f32: AF2's
 *                         differential against the WGSL fold reads 0.0291 A at
 *                         the worst atom with it and 0.0565 without.
 *
 *                         🔴 AND IT IS A DEFAULT OF THE FOLD DRIVERS, NOT OF
 *                         `Execution`, deliberately. The 47 stage gates build
 *                         their own weights through `upload*Weights` and hold
 *                         the kernels to f32 references at bars down to 1e-7;
 *                         a backend-wide default would have to re-bar every one
 *                         of them, which is the work it would take to make this
 *                         the shipped precision rather than the folding one.
 *   --no-half-x           x at float32 through `padStride`, where a FOLD now
 *                         narrows it to float16. x is the stream that binds the
 *                         dense projection - `probe-webgl-linear-operand.js`
 *                         measures 3.44x between the shipped kernel and one
 *                         whose x is in cache, against 1.09x for the weight
 *                         codes - and halving its bytes is **1.31x** on that
 *                         kernel alone and **2.56x** beside the tile-aware
 *                         order. It reaches only the shapes `padStride` runs
 *                         on, which is every one whose row width does not
 *                         divide the texture width.
 *
 *                         🔴 THE SECOND OPTIMISATION HERE THAT IS NOT
 *                         BIT-IDENTICAL, and like `--no-half-weights` it is a
 *                         default of the FOLD DRIVERS and not of `Execution`:
 *                         the stage gates build their own `new Execution(gl)`
 *                         and hold these kernels to f32 references at bars down
 *                         to 1e-7. It is also what the PRIMARY path does -
 *                         webgpu/af2/evoformer/block.js sets
 *                         `normalizedStorage` to f16 outright - so this is the
 *                         arm that compares f32 with f32.
 *   --no-narrow-aligned-x / --narrow-aligned-x
 *                         Whether x is narrowed to float16 on shapes whose row
 *                         stride ALREADY divides the texture width, where
 *                         `padStride` would not otherwise run - a pass whose
 *                         only purpose is the narrowing.
 *
 *                         🔴 IT IS A DEFAULT OF THE TWO DRIVERS IT PAYS FOR AND
 *                         NOT OF THE BACKEND, WHICH IS A MEASUREMENT.
 *                         bioemu2 **1.18x** (26.55 s to 22.45) and ESMFold2
 *                         **1.10x** (9672 ms to 8785); AlphaFold 3 **1.04x
 *                         SLOWER** (2460-2487 against 2365-2393 ms) with its
 *                         pLDDT moved, and AlphaFold 2 no faster at all while
 *                         its worst atom against the WGSL fold goes **0.0353 A
 *                         to 0.0904** - reproducible to the digit over two
 *                         rounds, and past the 0.0355 A dither floor, so a real
 *                         loss of agreement. Either flag overrides the driver.
 *   --no-framebuffer-reuse
 *                         Build a framebuffer per dispatch, as this backend
 *                         used to. Its keyed cache never hit - **0 hits and
 *                         54,381 misses** in a bioemu2 fold, because the key is
 *                         the texture OBJECTS and nothing pools them by default
 *                         - so every draw paid a createFramebuffer, its
 *                         attachments, a `drawBuffers`, a
 *                         `checkFramebufferStatus` and a deleteFramebuffer at
 *                         the cap. Reusing one framebuffer per target COUNT
 *                         takes `#bindTargets` from **4218 ms to 101**, 42x,
 *                         bit-identical.
 *
 *                         🔴 AND IT IS NOT A SPEEDUP: the fold is 22441 ms
 *                         against 22414. That host time was OVERLAPPED with the
 *                         GPU, which `--skip-draws` says is 16.5 s of the 22.4 -
 *                         so `bindMs` being 19.5% of the wall never meant it was
 *                         ON the wall. Kept because it removes 54,381
 *                         create/delete pairs and would matter on a slower CPU,
 *                         not because it folds faster.
 *   --nan-watch           Fill every fresh texture with NaN and read every
 *                         output back, refusing at the first non-finite value.
 *                         Names the kernel that reads a texel nothing wrote -
 *                         which `texImage2D` with a null pointer leaves
 *                         undefined, whatever this driver happens to do.
 *   --no-check-errors     Drop the `getError` after every draw. It is a
 *                         synchronous round trip and worth a few percent; a
 *                         benchmark that pays it measures the round trip.
 *   --no-linear-block     The dense projection one output texel a fragment, as
 *                         this backend did before it had a row block.
 *   --double=<regex>      Draw every dispatch whose label matches TWICE. The
 *                         fold's wall grows by that family's GPU time and by
 *                         nothing else, which is the only honest attribution
 *                         this API allows - see `Execution.doubleLabels`.
 *   --no-triangle-transpose
 *                         The incoming triangle contraction reading a COLUMN
 *                         directly, as it did before - a 69 KiB stride a step.
 *                         Bit-identical, so this prices a layout.
 *   --no-triangle-split   AF3's contraction and its centre norm in ONE draw,
 *                         as they were - which contracts every channel of a
 *                         pair to normalise four of them, sixty-four times
 *                         over. Bit-identical, so this prices a pass count.
 *   --no-triangle-coalesce
 *                         The outgoing contraction reading one float per
 *                         `fetch1`, which asks for the same texel four times.
 *   --no-contract-block   AF3's contraction one pair a fragment, which reads
 *                         both operands' rows for every pair rather than
 *                         sharing them across a block of two i and four j.
 *   --no-attend-dot4      bioemu2's two attentions walking their head
 *                         dimension with one fetch1 a channel, twice over -
 *                         four times the texelFetch for the same numbers.
 *   --no-combine          Every kernel its own program, where a fold packs the
 *                         light ones into combined programs (`Execution.combine`).
 *                         A link costs ~40 ms on this M2's Chrome whatever the
 *                         program holds, so this is the arm that prices them.
 *   --combine-bytes=<n>   How much GLSL one combined program may hold; 0 is no
 *                         cap. What a bin costs is its CONTENT, not the kernels
 *                         in it - see `Execution.combineBytes`.
 *   --no-pair-bias-fast   bioemu2's pair bias recomputing its LayerNorm
 *                         statistics once per HEAD, and reading every operand
 *                         through fetch1 - 3584 texelFetch a fragment against
 *                         512. Bit-identical.
 *   --block-rows=<n>      The blocked projection's rows a fragment, forced.
 *                         The resolver picks the largest power of two that
 *                         still leaves FRAGMENT_FLOOR fragments, because a
 *                         short projection is starved rather than
 *                         traffic-bound; this is how that is swept.
 *   --no-linear-aspect    The in-major projection blocked on the old
 *                         `in * out >= 65536` threshold rather than on the
 *                         aspect ratio - a threshold fitted with f32 weights
 *                         where every fold uses float16.
 *   --no-linear-tile      The int5 blocked projection's fragments in the 1D
 *                         `texel / RQ` order, so a rasterisation tile spans
 *                         every row block there is - 918 KiB of x under one
 *                         tile at bioemu2's largest projection - rather than
 *                         TILEW consecutive row blocks at one output group.
 *                         1.74-1.87x on that shape in
 *                         `probe-webgl-linear-operand.js`, which also measures
 *                         WHY: x binds at 3.44x and the weight codes at 1.09.
 *                         Bit-identical, so this prices the addressing.
 *   --linear-tile-width=<n>
 *                         That tile width, forced. The resolver takes the
 *                         NARROWEST inside a 2.5x plane-padding budget, because
 *                         the mapping pads the output group axis to `W / TILEW`
 *                         and the measured best width would pad AF2's 256
 *                         outputs sixteenfold; this is how it is swept.
 *   --no-norm-split       LayerNorm's row statistics computed by every fragment
 *                         that needs them rather than once a row. The fused
 *                         kernel has each fragment sweep its whole row and
 *                         there are `CH4` fragments a row, so the row is read
 *                         `CH4` times over - 192 at 768 channels, twice that
 *                         two-pass - and `bench-webgl-norm.js` prices
 *                         `576x768` at **968.8 us against the split's 200**.
 *                         Only at 768 channels and wider, which is the band
 *                         with no measured loss at any row count: the saving
 *                         scales with `CH4` and the cost does not, so a narrow
 *                         row pays an extra draw plus an under-occupied
 *                         statistics pass for nothing and reads **0.33x** at
 *                         128 channels.
 *   --no-contract-wide    AF3's contraction on the 2 i x 4 j x 4 channel pair
 *                         block rather than 4 i x 8 j x ONE channel. Channels a
 *                         fragment cost locality and buy no intensity: the
 *                         working set a rasterisation tile touches is
 *                         `rows * channels * LP`, 148 KiB at four channels and
 *                         **37 at one**, and the same 32-float budget then buys
 *                         a wider pair block at **21.3 flops a fetch4** against
 *                         10.7. Block and repack together, 1.33x at L=64, 2.01x
 *                         at 72 and 1.77x at 128. It also keeps the pair axes
 *                         padded to FOUR rather than eight, which is the
 *                         smaller tensor and the slower kernel.
 *   --no-contract-tile    The contraction's channel group taken from the texel
 *                         INDEX rather than from the fragment's x range, so a
 *                         rasterisation tile spans about a hundred channel
 *                         planes instead of one. 3.29x, bit-identical.
 *   --no-contract-pad     The contraction's pair axes left at an odd length,
 *                         so neither the coalesced reads nor the pair block is
 *                         legal and the scalar loop runs. 66% of an ESMFold2
 *                         fold at 75 tokens.
 *   --no-contract-ceiling The contraction pair-blocked at EVERY pair tensor
 *                         size, with no ceiling - which is up to 4.8x the wrong
 *                         kernel above 1.5M floats. Bit-identical.
 *   --no-contract-channel-slowest
 *                         The pair-blocked contraction's fragments ordered by
 *                         CHANNEL group rather than pair block, so neighbours
 *                         work on different channel planes and none of them
 *                         stays in cache. Bit-identical.
 *   --no-row-fastest      The int5 projection's fragments ordered by output
 *                         group rather than row block, so neighbours walk
 *                         different weight rows and the weight is streamed
 *                         again for every row block.
 *   --no-stride-pad       x laid at its own row width rather than padded to a
 *                         stride that divides the texture width. Worth 4.08x
 *                         on a shape whose width does not - the single largest
 *                         factor found in this backend after the row block.
 *                         Bit-identical, so it prices the addressing.
 *   --skip-draws          Issue no draws at all. The fold is garbage; what it
 *                         times is everything this backend does APART from
 *                         drawing, which is the arm that settles whether the
 *                         GPU is the cost.
 *   --host                Report where the host time inside `dispatch` goes.
 *   --profile[-top=N]     Per-label timings, by waiting for each kernel. READ
 *                         THE RANKING AND NOT THE TOTAL, and prefer `--double`:
 *                         the wait is most of what this measures.
 */
import { Execution } from "../../webgl/runtime/execution.js";

const flag = (args, name) => (args ?? []).includes(`--${name}`);
const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

/** @param {string[]} args */
export function webglExecutionOptions(args, defaults = {}) {
  const pattern = (name) => (option(args, name, null) === null
    ? null : new RegExp(option(args, name, "")));
  return {
    // A driver default too (the AF3 fold pools, bounded - fold-batch.js AF3_POOL_LIMIT_BYTES); `--no-pool` is its
    // control arm and `--pool` turns it on elsewhere
    pooling: flag(args, "pool") || ((defaults.pooling ?? false) && !flag(args, "no-pool")),
    // `--pool-mib=<n>` caps the pool (Storage.poolLimitBytes); the driver's cap, else unbounded, without it
    poolLimitBytes: option(args, "pool-mib", null) !== null ? Number(option(args, "pool-mib", "0")) * 2 ** 20
      : defaults.poolLimitBytes ?? Infinity,
    halfWeights: !flag(args, "no-half-weights"),
    // `--half-floor=<n>`: the smallest weight held in float16 (Storage.halfWeightFloor); the driver's, else 65536
    ...(option(args, "half-floor", null) !== null ? { halfWeightFloor: Number(option(args, "half-floor", "0")) }
      : defaults.halfWeightFloor === undefined ? {} : { halfWeightFloor: defaults.halfWeightFloor }),
    halfActivations: !flag(args, "no-half-x"),
    // 🔴 A DRIVER DEFAULT, BECAUSE THE MEASUREMENT DIFFERS BY MODEL. See
    // `--no-narrow-aligned-x` below; a driver opts in and either flag wins.
    narrowAlignedX: flag(args, "narrow-aligned-x")
      || ((defaults.narrowAlignedX ?? false) && !flag(args, "no-narrow-aligned-x")),
    narrowUnblockedX: flag(args, "narrow-unblocked-x")
      || ((defaults.narrowUnblockedX ?? false) && !flag(args, "no-narrow-unblocked-x")),
    int5Weights: !flag(args, "no-int5-weights"),
    checkErrors: !flag(args, "no-check-errors"),
    // `--invalidate` / `--no-invalidate`: Execution.invalidateTargets, a driver default too
    invalidateTargets: flag(args, "invalidate") || ((defaults.invalidateTargets ?? false) && !flag(args, "no-invalidate")),
    // `--flush-every=<n>`: gl.flush() every n dispatches (Execution.flushEvery); the driver's default without it
    flushEvery: Number(option(args, "flush-every", String(defaults.flushEvery ?? 0))),
    noLinearBlock: flag(args, "no-linear-block"),
    fillFresh: flag(args, "nan-watch") ? NaN : null,
    nanWatch: flag(args, "nan-watch"),
    host: flag(args, "host"),
    doubleLabels: pattern("double"),
    // `--skip=<regex>`: no draws for those labels - the family's draw time, where `--double` reads free (Execution.skipLabels)
    skipLabels: pattern("skip"),
    noOpmOuterFirst: flag(args, "no-opm-outer-first"),
    noTriangleFusedLayout: flag(args, "no-triangle-fused-layout"),
    skipDraws: flag(args, "skip-draws"),
    noTriangleTranspose: flag(args, "no-triangle-transpose"),
    noTriangleSplit: flag(args, "no-triangle-split"),
    noTriangleCoalesce: flag(args, "no-triangle-coalesce"),
    noContractBlock: flag(args, "no-contract-block"),
    noRowFastest: flag(args, "no-row-fastest"),
    noStridePad: flag(args, "no-stride-pad"),
    noContractChannelSlowest: flag(args, "no-contract-channel-slowest"),
    noContractCeiling: flag(args, "no-contract-ceiling"),
    noContractPad: flag(args, "no-contract-pad"),
    noContractTile: flag(args, "no-contract-tile"),
    noContractWide: flag(args, "no-contract-wide"),
    noNormSplit: flag(args, "no-norm-split"),
    noLinearAspect: flag(args, "no-linear-aspect"),
    noFramebufferReuse: flag(args, "no-framebuffer-reuse"),
    noLinearTile: flag(args, "no-linear-tile"),
    linearTileWidth: Number(option(args, "linear-tile-width", "0")),
    linearBlockRows: Number(option(args, "block-rows", "0")),
    noPairBiasFast: flag(args, "no-pair-bias-fast"),
    noAttendDot4: flag(args, "no-attend-dot4"),
    noCombine: flag(args, "no-combine"),
    combineBytes: Number(option(args, "combine-bytes", "0")),
    poolLabelPattern: pattern("pool-labels"),
    profile: flag(args, "profile") ? new Map() : null,
  };
}

/**
 * An `Execution` on `gl` configured by the arguments above.
 *
 * @param {WebGL2RenderingContext} gl
 * @param {string[]} args
 */
export function webglExecution(gl, args, defaults = {}) {
  return new Execution(gl, webglExecutionOptions(args, defaults));
}

/** The profile table a `--profile` run produced, or null. */
export function webglProfile(execution, args) {
  return execution.profileTable(Number(option(args, "profile-top", "24")));
}
