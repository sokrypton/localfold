/**
 * RoseTTAFold3's chirality gradient: the GPU kernel against the CPU reference.
 *
 *     node tools/gpu-chrome.mjs tools/gpu/check-chiral-gradient.js
 *
 * 🔴 THE ONE CHECK A NEW KERNEL HERE OWES. The featuriser half is exact against
 * af3-any-model's own batch (213 centres, 0 elements differing, gated in
 * tools/check-atom-windows.js) and the CPU gradient is property-tested - it
 * descends, and it tells a structure from its mirror. Neither says the WGSL
 * computes the same thing, and the term's effect on 6MRR and 5K9P is inside the
 * seed noise, so a fold cannot say either.
 *
 * It drives the SHIPPED kernel out of `createAtomEncoderShaders`, not a copy.
 */
import { createAtomEncoderShaders } from "../../src/af3/atom-encoder-webgpu.js";
import { chiralCentres } from "../../src/af3/template-features.js";
import { chiralPositionGradients } from "../../src/af3/chiral-gradient.js";
import { featuriseProtein } from "../../src/af3/featurise.js";
import { relativeRms } from "./relative-rms.js";

const SEQUENCE = "GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE";

export async function main(device, args) {
  const bound = Number((args.find((a) => a.startsWith("--bound=")) ?? "--bound=2e-3").slice(8));
  const batch = featuriseProtein(SEQUENCE, { dropTerminalAtoms: true, paddedAtomKeys: true });
  const { tokens } = batch;
  const dense = batch.dense ?? 24;
  const atoms = tokens * dense;
  const { centers, angles, count } = chiralCentres(
    batch.aatype, batch.predDenseAtomMask, tokens);

  // 🔴 NOT THE IDEAL CONFORMERS. At the ideal geometry every centre already
  // sits at its target and the gradient is ZERO, which both sides would agree
  // on while computing nothing. The sampler's coordinates are noisy, so this
  // perturbs them - which is also the only regime the term exists for.
  const positions = Float32Array.from(batch.refPos);
  let state = 20260914 >>> 0;
  for (let at = 0; at < positions.length; at += 1) {
    state = (state * 1664525 + 1013904223) >>> 0;
    positions[at] += ((state / 2 ** 32) * 2 - 1) * 0.7;
  }

  const expected = chiralPositionGradients(positions, centers, angles, atoms);

  const shape = { chiralGradients: true, tokens, dense, subsets: batch.shape.subsets,
                  queries: 32, keys: batch.shape.keys, channels: 128, pairChannels: 16,
                  heads: 4, dimension: 32, perTokenChannels: 768,
                  trunkSingleChannels: 384, trunkPairChannels: 128, blocks: 3 };
  const sources = createAtomEncoderShaders(shape, {}, 1e-5, "fast");
  if (typeof sources.chiralGrad !== "string") {
    throw new Error("the chirality kernel was not generated; shape.chiralGradients is the switch");
  }
  const pipeline = await device.createComputePipelineAsync({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: sources.chiralGrad }), entryPoint: "main" },
  });

  // The same inverted index the encoder builds; see its note on why.
  const counts = new Uint32Array(atoms + 1);
  for (let at = 0; at < centers.length; at += 1) counts[centers[at] + 1] += 1;
  for (let at = 0; at < atoms; at += 1) counts[at + 1] += counts[at];
  const offsets = Uint32Array.from(counts);
  const entries = new Uint32Array(centers.length);
  const cursor = Uint32Array.from(counts);
  for (let centre = 0; centre < count; centre += 1) {
    for (let corner = 0; corner < 4; corner += 1) {
      const atom = centers[centre * 4 + corner];
      entries[cursor[atom]] = (centre << 2) | corner;
      cursor[atom] += 1;
    }
  }

  const upload = (data) => {
    const buffer = device.createBuffer({
      size: Math.max(data.byteLength, 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  };
  const output = device.createBuffer({
    size: atoms * 3 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readback = device.createBuffer({
    size: atoms * 3 * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const buffers = [upload(positions), upload(Int32Array.from(centers)),
                   upload(Float32Array.from(angles)), upload(offsets), upload(entries), output];

  device.pushErrorScope("validation");
  const encoder = device.createCommandEncoder({ label: "chiral-grad" });
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
  }));
  pass.dispatchWorkgroups(Math.ceil(atoms / 64));
  pass.end();
  encoder.copyBufferToBuffer(output, 0, readback, 0, atoms * 3 * 4);
  device.queue.submit([encoder.finish()]);
  const error = await device.popErrorScope();
  if (error !== null) throw new Error(`WebGPU validation failed: ${error.message}`);
  await readback.mapAsync(GPUMapMode.READ);
  const actual = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();

  const relRms = relativeRms(actual, expected);
  // 🔴 AND THE REFERENCE MUST NOT BE ZERO, or a kernel that writes nothing
  // scores perfectly. This is the control the relative-rms note warns about.
  const magnitude = Math.sqrt(
    expected.reduce((sum, v) => sum + v * v, 0) / Math.max(expected.length, 1));
  const nonzero = [...expected].filter((v) => v !== 0).length;
  if (!(magnitude > 1e-6) || nonzero < 100) {
    throw new Error(`the CPU gradient is ~zero (rms ${magnitude}, ${nonzero} nonzero), `
      + "so this comparison would pass against a kernel that writes nothing");
  }
  if (!(relRms < bound)) throw new Error(`chiral gradient: relRMS ${relRms} against ${bound}`);
  return { centres: count, atoms, nonzeroComponents: nonzero,
           cpuRms: magnitude, relRms, bound };
}
