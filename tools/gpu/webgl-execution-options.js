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
    checkErrors: !flag(args, "no-check-errors"),
    noLinearBlock: flag(args, "no-linear-block"),
    fillFresh: flag(args, "nan-watch") ? NaN : null,
    nanWatch: flag(args, "nan-watch"),
    host: flag(args, "host"),
    doubleLabels: pattern("double"),
    skipDraws: flag(args, "skip-draws"),
    noTriangleTranspose: flag(args, "no-triangle-transpose"),
    noTriangleSplit: flag(args, "no-triangle-split"),
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
