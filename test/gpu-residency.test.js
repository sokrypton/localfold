import { describe, expect, it } from "./harness.js";
import { WebGpuExecution } from "../src/runtime/execution.js";

/**
 * KEEPING A TENSOR ON THE DEVICE IS A LIFETIME QUESTION, not an arithmetic one.
 *
 * The trunk hands one pair buffer from the input embedder through the extra-MSA
 * stack into the main stack - 25 MiB at 221 residues that used to make nine
 * crossings of the bus per pass. Each stack releases its own scratch on the way
 * out, and the ONE thing that must survive that is the buffer it was handed.
 * Get the checkpoint wrong by a single allocation and the pair is freed under
 * the next stage, which on a real device is silent corruption rather than an
 * error: pooling hands the same memory straight back out for something else.
 *
 * None of that needs a GPU to check. It needs an allocator, a checkpoint and a
 * record of what was destroyed, which is what the stub below is.
 */
function stubDevice() {
  const buffers = [];
  return {
    buffers,
    createBuffer({ label, size }) {
      const buffer = { label, size, destroyed: false, destroy() { this.destroyed = true; } };
      buffers.push(buffer);
      return buffer;
    },
    queue: { writeBuffer() {} },
    // ...WebGpuExecution reads these to work out how large a transition chunk
    // may be. Real devices always report them; a stub has to say so too.
    limits: {
      maxStorageBufferBindingSize: 128 * 1024 * 1024,
      minStorageBufferOffsetAlignment: 256,
    },
  };
}

const STORAGE = 0x0080;
// The one constant the allocator reaches for on its own, when it ORs COPY_DST
// into an upload. Node has no WebGPU globals, and these are fixed by the spec.
globalThis.GPUBufferUsage ??= { COPY_DST: 0x0008, COPY_SRC: 0x0004, STORAGE, MAP_READ: 0x0001 };

describe("holding a tensor across stages", () => {
  it("leaves a caller's tensor alone when a stage releases its own scratch", () => {
    const device = stubDevice();
    const execution = new WebGpuExecution(device);
    // ...the trunk: the caller allocates the pair, then a stage runs inside it
    const pair = execution.allocate("trunk.pair", 16, STORAGE);
    const entry = execution.checkpoint();
    const scratch = execution.allocate("stack.scratch", 4, STORAGE);
    const mask = execution.upload("stack.mask", new Float32Array(4), STORAGE);
    execution.releaseSince(entry);

    expect(scratch.allocation.buffer.destroyed).toBe(false);   // pooled, not destroyed
    expect(mask.allocation.buffer.destroyed).toBe(false);
    expect(pair.allocation.buffer.destroyed).toBe(false);
    // 🔴 THE PAIR IS STILL THE PAIR. Pooling hands a released buffer straight
    // back out, so the test that matters is not "was it destroyed" but "can it
    // be handed to something else" - and it must not be.
    const next = execution.allocate("next-stage.scratch", 4, STORAGE);
    expect(next.allocation.buffer === pair.allocation.buffer).toBe(false);
    expect(execution.snapshot().currentBytes).toBe(16 * 4 + 4 * 4);
  });

  it("reuses what a stage gave back, which is the point of sharing one execution", () => {
    const device = stubDevice();
    const execution = new WebGpuExecution(device);
    const entry = execution.checkpoint();
    const first = execution.allocate("extra-stack.msa", 8, STORAGE);
    const buffer = first.allocation.buffer;
    execution.releaseSince(entry);
    const second = execution.allocate("stack.msa", 8, STORAGE);
    expect(second.allocation.buffer === buffer).toBe(true);
    expect(device.buffers.length).toBe(1);
  });

  it("nests, so a stage inside a stage releases only its own", () => {
    const device = stubDevice();
    const execution = new WebGpuExecution(device);
    const outer = execution.checkpoint();
    const held = execution.allocate("outer.held", 4, STORAGE);
    const inner = execution.checkpoint();
    execution.allocate("inner.scratch", 4, STORAGE);
    execution.releaseSince(inner);
    expect(execution.snapshot().currentBytes).toBe(4 * 4);
    execution.releaseSince(outer);
    expect(execution.snapshot().currentBytes).toBe(0);
    expect(held.allocation.buffer.destroyed).toBe(false);
  });

  it("refuses a checkpoint that is not one", () => {
    const execution = new WebGpuExecution(stubDevice());
    execution.allocate("a", 4, STORAGE);
    expect(() => execution.releaseSince(2)).toThrow();
    expect(() => execution.releaseSince(-1)).toThrow();
  });

  it("destroys the pool only when the execution itself is done", () => {
    const device = stubDevice();
    const execution = new WebGpuExecution(device);
    execution.allocate("trunk.pair", 4, STORAGE);
    execution.release();
    expect(device.buffers.every((buffer) => buffer.destroyed)).toBe(true);
    expect(execution.snapshot().currentBytes).toBe(0);
  });
});
