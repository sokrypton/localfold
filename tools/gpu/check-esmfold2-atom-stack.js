// Does the sliding-window atom transformer compute what its reference does?
//
//     node tools/gpu-chrome.mjs tools/gpu/check-esmfold2-atom-stack.js
//
// 🔴 THIS IS THE ONE KERNEL IN THE PORT WITH NO AF3 ANALOGUE AND, UNTIL THIS
// FILE, NO GATE OF ITS OWN. It was checked only through the whole denoiser -
// twelve token blocks and two atom stacks at once - which says "somewhere in
// three hundred dispatches" and cannot separate the attention from the
// modulation from the pooling. The reference is
// src/esmfold2/atom-encoder-reference.js, which is itself checked against the
// native module at 7.0e-8 with the attention in float32.
//
// 🔴 IT SYNTHESISES ITS OWN WEIGHTS AND ITS OWN MOLECULE, DELIBERATELY. What is
// under test is the kernels, not the checkpoint - the checkpoint has
// tools/gpu/check-esmfold2-diffusion-gpu.js - and a synthetic input can be made
// HARSHER than a real one: ragged token sizes, a masked tail, and enough atoms
// that the window is smaller than the molecule, which is the case where the
// rank-resolved bounds actually do something.
//
// 🔴 AND THE PRECISION IS AN AXIS WITH A DISCRIMINATING CONTROL. The module
// downcasts q, k and v to bfloat16 whatever the model's dtype, so the f32 arm
// must agree to ~1e-6 and the bf16 arm must NOT - if the two agree, the
// downcast has stopped reaching the kernel and no bound would say so.
import { GpuBufferAllocator } from "../../src/runtime/allocator.js";
import { pipelineCacheForDevice } from "../../src/runtime/pipeline-cache.js";
import { buildRope, swaBlock } from "../../src/esmfold2/atom-encoder-reference.js";
import {
  atomStackScratch, atomWindows, compileAtomStack, encodeAtomStack, widestWindow,
} from "../../src/esmfold2/atom-transformer-webgpu.js";

const option = (args, name, fallback) => {
  const prefix = `--${name}=`;
  return (args ?? []).find((a) => a.startsWith(prefix))?.slice(prefix.length) ?? fallback;
};

/** Reproducible noise, so two arms see the same input. */
function deterministic(count, seed) {
  const out = new Float32Array(count);
  let state = (seed >>> 0) || 1;
  for (let i = 0; i < count; i += 1) {
    state ^= state << 13; state >>>= 0;
    state ^= state >> 17;
    state ^= state << 5; state >>>= 0;
    out[i] = (state / 4294967296) * 2 - 1;
  }
  return out;
}

const relative = (got, want) => {
  let error = 0, total = 0;
  for (let i = 0; i < want.length; i += 1) {
    const d = got[i] - want[i];
    error += d * d;
    total += want[i] * want[i];
  }
  return Math.sqrt(error / total);
};

export async function main(device, args = []) {
  const atoms = Number(option(args, "atoms", "320"));
  const channels = Number(option(args, "channels", "128"));
  const heads = Number(option(args, "heads", "4"));
  const blocks = Number(option(args, "blocks", "3"));
  const window = Number(option(args, "window", "128"));
  const hidden = channels * 2;
  const halfWindow = window >> 1;
  const bound = Number(option(args, "bound", "3e-6"));
  const precisions = option(args, "precision", "f32,bf16").split(",");

  // 🔴 A MASKED TAIL, BECAUSE THE WINDOW IS OVER RANK AND NOT OVER INDEX. With
  // every atom live the two are the same number and the whole rank machinery is
  // untested; masking the last twelfth makes them differ for every query whose
  // window reaches the tail. The diagonal is allowed for a masked atom too,
  // which is what keeps its softmax finite.
  const mask = new Float32Array(atoms).fill(1);
  for (let atom = Math.floor(atoms * 11 / 12); atom < atoms; atom += 1) mask[atom] = 0;
  const bounds = atomWindows(mask, atoms, halfWindow);
  const stagedWindow = widestWindow(bounds, atoms);

  const refPos = deterministic(atoms * 3, 17);
  for (let i = 0; i < refPos.length; i += 1) refPos[i] *= 12;
  const spaceUid = new Float32Array(atoms);
  for (let atom = 0; atom < atoms; atom += 1) spaceUid[atom] = Math.floor(atom / 8);
  const rope = buildRope(refPos, spaceUid, atoms, channels / heads);

  const activation = deterministic(atoms * channels, 101);
  const conditioning = deterministic(atoms * channels, 211);
  const weights = [];
  for (let block = 0; block < blocks; block += 1) {
    const scale = (values, by) => {
      for (let i = 0; i < values.length; i += 1) values[i] *= by;
      return values;
    };
    weights.push({
      // ...small, because adaLN's shift and scale are added to a normalised
      // activation and a modulation of order one is what the trained values
      // look like. Uniform noise at full scale would make every arm's error a
      // property of the draw rather than of the kernel.
      adaln: scale(deterministic(channels * channels * 6, 301 + block), 0.05),
      qkv: scale(deterministic(channels * channels * 3, 401 + block), 0.1),
      attnGate: scale(deterministic(channels * channels, 501 + block), 0.1),
      attnOut: scale(deterministic(channels * channels, 601 + block), 0.1),
      ffnUp: scale(deterministic(channels * hidden * 2, 701 + block), 0.1),
      ffnDown: scale(deterministic(hidden * channels, 801 + block), 0.1),
    });
  }

  const allocator = new GpuBufferAllocator(device);
  const cache = pipelineCacheForDevice(device);
  const storage = GPUBufferUsage.STORAGE;
  const results = [];
  let failures = 0;
  for (const precision of precisions) {
    // The reference, block by block, with the conditioning held fixed.
    let want = Float32Array.from(activation);
    for (let block = 0; block < blocks; block += 1) {
      want = swaBlock(want, conditioning, atoms, channels, heads, weights[block],
                      rope, mask, halfWindow, hidden, precision);
    }

    const pipelines = await compileAtomStack(cache, {
      atoms, tokens: 1, channels, heads, blocks, hidden,
      window: stagedWindow, precision,
    });
    const held = [];
    const keep = (allocation) => { held.push(allocation); return allocation; };
    const state = {
      activation: keep(allocator.upload("check.act", activation,
        storage | GPUBufferUsage.COPY_SRC)),
      conditioning: keep(allocator.upload("check.cond", conditioning, storage)),
      cos: keep(allocator.upload("check.cos", rope.cos, storage)),
      sin: keep(allocator.upload("check.sin", rope.sin, storage)),
      bounds: keep(allocator.upload("check.bounds", bounds, storage)),
      valid: keep(allocator.upload("check.valid", mask, storage)),
    };
    const scratch = {};
    for (const [name, elements] of Object.entries(
      atomStackScratch({ atoms, channels, heads, hidden }))) {
      scratch[name] = keep(allocator.allocate(`check.${name}`,
        Math.max(16, elements * 4), storage));
    }
    const blockBuffers = weights.map((block, index) => {
      const at = (leaf) => keep(allocator.upload(`check.w${index}.${leaf}`,
        block[leaf], storage));
      return { adaln: at("adaln"), qkv: at("qkv"), attnGate: at("attnGate"),
               attnOut: at("attnOut"), ffnUp: at("ffnUp"), ffnDown: at("ffnDown") };
    });

    const encoder = device.createCommandEncoder({ label: "check.atom-stack" });
    const run = (label, pipeline, buffers, x, y = 1) => {
      const pass = encoder.beginComputePass({ label });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: buffers.map((entry, binding) =>
          ({ binding, resource: { buffer: entry.buffer } })),
      }));
      pass.dispatchWorkgroups(x, y, 1);
      pass.end();
    };
    encodeAtomStack({ run, pipelines, state, scratch, weights: blockBuffers });
    const readback = keep(allocator.allocate("check.readback", atoms * channels * 4,
      GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST));
    encoder.copyBufferToBuffer(state.activation.buffer, 0, readback.buffer, 0,
                               atoms * channels * 4);
    device.queue.submit([encoder.finish()]);
    await readback.buffer.mapAsync(GPUMapMode.READ);
    const got = new Float32Array(readback.buffer.getMappedRange().slice(0));
    readback.buffer.unmap();
    for (let at = held.length - 1; at >= 0; at -= 1) held[at].release();

    const score = relative(got, want);
    const ok = score <= bound;
    if (!ok) failures += 1;
    results.push({ precision, relRMS: score, ok });
    console.log(`  ${precision.padEnd(5)} against its own reference   `
      + `relRMS ${score.toExponential(3)}   ${ok ? "ok" : "FAILED"}`);
  }

  // 🔴 THE CONTROL: THE TWO ARMS MUST DISAGREE. Both agreeing with their own
  // reference says the GPU matches the CPU; it does not say the bfloat16
  // downcast reached either. Scoring the bf16 arm against the f32 REFERENCE is
  // what separates them, and it must be far outside the bound.
  let separation;
  if (precisions.includes("f32") && precisions.includes("bf16")) {
    let wide = Float32Array.from(activation);
    for (let block = 0; block < blocks; block += 1) {
      wide = swaBlock(wide, conditioning, atoms, channels, heads, weights[block],
                      rope, mask, halfWindow, hidden, "f32");
    }
    let narrow = Float32Array.from(activation);
    for (let block = 0; block < blocks; block += 1) {
      narrow = swaBlock(narrow, conditioning, atoms, channels, heads, weights[block],
                        rope, mask, halfWindow, hidden, "bf16");
    }
    separation = relative(narrow, wide);
    // 🔴 AGAINST THE ARMS' OWN ERRORS, NOT AGAINST THE BOUND. A fixed multiple
    // of the bound is a number about the checker rather than about the model:
    // it passes or fails depending on how tight the bound happens to be. What
    // has to be true is that the two arms differ by far MORE than either
    // differs from its own reference, and ten times is not close.
    const worst = Math.max(...results.map((arm) => arm.relRMS));
    const ok = separation > worst * 10;
    if (!ok) failures += 1;
    console.log(`  control: bf16 against the f32 reference   `
      + `relRMS ${separation.toExponential(3)}, ${(separation / worst).toFixed(0)}x `
      + `the worst arm   ${ok ? "ok" : "FAILED - the downcast reaches neither arm"}`);
    // 🔴 AND IT IS SMALLER HERE THAN IN THE REAL MODULE, WHICH IS THE INPUT AND
    // NOT THE KERNEL. These weights are scaled to 0.1 so the modulation is of
    // order one, which makes the logits small - and a bfloat16 error lands in a
    // LOGIT, where `exp` turns it into a relative weight error. The native
    // module measures 2.4e-3 on one block's attention against this checker's
    // 2.8e-5. Read this number as "the downcast is reaching the kernel", not as
    // what half precision costs a fold.
  }

  return {
    atoms, channels, heads, blocks, window, stagedWindow,
    maskedAtoms: atoms - mask.reduce((total, value) => total + value, 0),
    bound, failures, separation, results,
    message: failures === 0
      ? "the sliding-window atom transformer computes its reference"
      : `${failures} arm(s) out of bound`,
  };
}
