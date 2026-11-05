/**
 * The triangle contraction as a BENCHMARK: prepare once, dispatch many times.
 *
 * WHY THIS EXISTS. Profiling puts 80% of an Evoformer block in two kernels, and
 * `triangle.*.contract` is the simpler of them - 22% of block time at 221
 * residues, and the one operation in the model that already has a CPU reference
 * to check against. If a WebGL2 port of the Evoformer is worth doing at all,
 * this is the kernel that says so, and it says it in a day rather than a week.
 *
 * 🔴 AND IT IS A DRIVER NOW, NOT A SECOND COPY OF THE KERNEL. The arithmetic is
 * `./triangle-contract.js`, shared with the stage path; this file is the
 * prepare/dispatch SPLIT that a timing needs. The inputs to this kernel in a
 * real pipeline are the output of the pass before it - already on the GPU, never
 * travelling - so a measurement that re-uploads them each time is timing 16 MiB
 * of transfer against a WebGPU path whose buffers were created once. Preparing
 * and dispatching separately is what makes the two comparable.
 *
 * 🔴 WEBGL2 HAS NO COMPUTE SHADERS. Everything about both files follows from
 * that: the work is done by a FRAGMENT shader, one draw covers a rectangle of
 * output texels and every fragment computes one output element; there are no
 * storage buffers, so tensors live in textures and every linear index has to be
 * decoded into a texel coordinate; and there is no workgroup shared memory, so
 * the tiling the WGSL kernel uses - 8x8 staging in `var<workgroup>` with two
 * barriers a step - has no equivalent. This reads straight from texture memory
 * and leans on the texture cache instead. That is the honest shape of the
 * comparison: not the same algorithm in two languages, but the same MATHS under
 * two different sets of constraints.
 *
 * The texture layout - a power-of-two row width, four floats to a texel - is
 * `webgl/runtime/context.js`, shared with every kernel here.
 *
 * 🔴 NOTHING ON THE PAGE SELECTS THIS BACKEND YET, WHICH IS DELIBERATE AND NOT
 * DEAD CODE. This file and its sibling were written as SPIKES - two kernels
 * ported to answer whether a WebGL2 fallback is possible at all - and for a
 * long while they sat in `webgpu/kernels/` with a `webgl2` in their names and
 * zero importers, where a dead-code sweep finds them immediately. They are part
 * of `webgl/` now: `webgl/README.md` is what this backend is and is not.
 *
 * What they bought is worth more than they cost. WebGL2 has no compute shaders,
 * so it has no workgroup memory and no barriers, and porting the attention
 * kernel to it is **what exposed how much the WGSL kernel was paying in
 * barriers - the 11.4x**. On a fair fight they are about 1.4x behind on the
 * contraction and about 1.55x ahead on attention, computing the same numbers.
 *
 * Delete them and the next person to ask "would a barrier-free formulation
 * help?" starts from nothing.
 */
import { contractSource, toChannelMajor } from "./triangle-contract.js";

export { toChannelMajor };

export class TriangleContractWebGL2 {
  /** @param {import("../runtime/execution.js").Execution} execution */
  constructor(execution) {
    this.execution = execution;
  }

  /** Upload the operands and allocate the target; no arithmetic runs. */
  prepare(input) {
    const { a, b, length, channels } = input;
    const direction = input.direction ?? "outgoing";
    const layout = input.layout ?? "channel-minor";
    const elements = length * length * channels;
    if (a.length !== elements || b.length !== elements) {
      throw new RangeError(`triangle contract expects ${elements} elements in a and b`);
    }
    if (channels % 4 !== 0) {
      throw new RangeError(`channels must be a multiple of 4, got ${channels}`);
    }
    const pack = (values) => (layout === "channel-major"
      ? toChannelMajor(values, length * length, channels) : values);
    return {
      source: contractSource(length, channels, direction, layout),
      a: this.execution.upload(pack(a), "triangle.a"),
      b: this.execution.upload(pack(b), "triangle.b"),
      output: this.execution.allocate(elements, "triangle.contract"),
      elements, direction, length, channels, layout,
    };
  }

  /** Draw the contraction into the prepared target. Nothing is read back. */
  dispatch(state, bias = 0) {
    return this.execution.dispatch({
      source: state.source,
      output: state.output,
      inputs: { aTex: state.a, bTex: state.b },
      floats: { bias },
      label: `triangle.contract:${state.direction}:${state.length}x${state.channels}`,
    });
  }

  /** Draw, then read the target back as a Float32Array. */
  read(state) {
    this.dispatch(state, 0);
    return this.execution.read(state.output);
  }

  release(state) {
    for (const tensor of [state.a, state.b, state.output]) this.execution.release(tensor);
  }

  /** Prepare, dispatch, read and release - the whole thing, for a single call. */
  run(input) {
    const state = this.prepare(input);
    try { return this.read(state).slice(); } finally { this.release(state); }
  }
}
