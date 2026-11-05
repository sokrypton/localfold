/**
 * The WebGL2 backend's diagnostic switches, read off a tool's own arguments.
 *
 * Every one of them is a CONTROL ARM or an INSTRUMENT, and they are here rather
 * than in each driver because a knob only one tool can reach is a knob nobody
 * turns - which is how `linearTallTile` came to be priced at zero on the WGSL
 * side for a configuration nobody had swept.
 *
 *   --pool                `Storage` keeps a released texture for the next
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
export function webglExecutionOptions(args) {
  const pattern = (name) => (option(args, name, null) === null
    ? null : new RegExp(option(args, name, "")));
  return {
    pooling: flag(args, "pool"),
    halfWeights: !flag(args, "no-half-weights"),
    int5Weights: !flag(args, "no-int5-weights"),
    checkErrors: !flag(args, "no-check-errors"),
    noLinearBlock: flag(args, "no-linear-block"),
    fillFresh: flag(args, "nan-watch") ? NaN : null,
    nanWatch: flag(args, "nan-watch"),
    host: flag(args, "host"),
    doubleLabels: pattern("double"),
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
    noLinearAspect: flag(args, "no-linear-aspect"),
    linearBlockRows: Number(option(args, "block-rows", "0")),
    noPairBiasFast: flag(args, "no-pair-bias-fast"),
    noAttendDot4: flag(args, "no-attend-dot4"),
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
export function webglExecution(gl, args) {
  return new Execution(gl, webglExecutionOptions(args));
}

/** The profile table a `--profile` run produced, or null. */
export function webglProfile(execution, args) {
  return execution.profileTable(Number(option(args, "profile-top", "24")));
}
