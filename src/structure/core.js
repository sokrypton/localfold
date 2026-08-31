import {
  InvariantPointAttentionGpu,

} from "./ipa.js";
import {
  StructurePostAttentionGpu,

} from "./iteration.js";
import { throwIfAborted, withAbort } from "../runtime/abort.js";

export class StructureCoreGpu {
  device;
  constructor(device) { this.device = device; }

  async run(input) {
    throwIfAborted(input.signal);
    let activations = input.activations;
    let affine = input.affine;
    const start = performance.now();
    const iterations = input.iterations ?? 8;
    const ipa = new InvariantPointAttentionGpu(this.device);
    const post = new StructurePostAttentionGpu(this.device);
    const geometry = {
      pair: input.pair,
      mask: input.mask,
      length: input.length,
      channels: input.channels,
      pairChannels: input.pairChannels,
      heads: 12,
      scalarQk: 16,
      scalarV: 16,
      pointQk: 4,
      pointV: 8,
      weights: input.ipaWeights,
    };
    // 🔴 THE PAIR REPRESENTATION IS UPLOADED ONCE, NOT EIGHT TIMES.
    //
    // Every iteration reads the same pair track and normalises it with the same
    // weights, so eight of them used to push L*L*128 floats across the bus and
    // layer-normalise them to produce a buffer identical to the last. At 221
    // residues that is 25 MiB a time, 200 MiB a pass, plus seven normalisations
    // whose only output was a copy. Prepared once here, it is resident for all
    // eight; only the activations and the frames still travel per iteration.
    const prepared = await withAbort(ipa.prepare(geometry), input.signal);
    try {
      for (let iteration = 0; iteration < iterations; iteration += 1) {
        throwIfAborted(input.signal);
        const attention = await withAbort(ipa.run({ ...geometry, activations, affine, prepared }), input.signal);
        throwIfAborted(input.signal);
        const update = await withAbort(post.run({
          activations,
          attentionUpdate: attention.output,
          affine,
          length: input.length,
          channels: input.channels,
          weights: input.postAttentionWeights,
        }), input.signal);
        activations = update.activations;
        affine = update.affine;
        throwIfAborted(input.signal);
        // ...REPORTED HERE AND NOWHERE ELSE. Both calls above already await real
        // GPU work, so this rides on a boundary that existed - it adds a function
        // call per iteration and no synchronisation of its own.
        input.onIteration?.(iteration + 1, iterations);
      }
    } finally {
      prepared.release();
    }
    throwIfAborted(input.signal);
    return { activations, affine, elapsedMilliseconds: performance.now() - start };
  }
}
