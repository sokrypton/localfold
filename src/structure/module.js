import { AtomGeometryGpu } from "./geometry.js";
import { StructureCoreGpu } from "./core.js";
import { StructureInitializeGpu } from "./initialize.js";

import { SidechainAnglesGpu } from "./sidechain.js";
import { throwIfAborted, withAbort } from "../runtime/abort.js";

/**
 * @typedef {object} StructureModuleResult
 * @property {Float32Array} atom14              dense per-residue atom frame
 * @property {Float32Array} atom37              sparse all-atom coordinates
 * @property {Float32Array} atom37Mask          which of the 37 slots are real
 * @property {Float32Array} finalRepresentation
 * @property {Float32Array} affine              per-residue backbone frames
 * @property {Float32Array} angles              normalized sidechain torsions
 * @property {Float32Array} unnormalizedAngles
 * @property {number} elapsedMilliseconds
 */

/** Complete eight-iteration AlphaFold structure module. All learned operations and atom geometry execute in WGSL. */
export class StructureModuleGpu {
  device;
  constructor(device) { this.device = device; }

  async run(input) {
    throwIfAborted(input.signal);
    const start = performance.now();
    const msaChannels = input.msaChannels ?? 256;
    const structureChannels = input.structureChannels ?? 384;
    const pairChannels = input.pairChannels ?? 128;
    // Four stages, the second of which runs eight iterations in one command
    // buffer. That stage is the longest single thing a fold does at any real
    // length, and the bar does sit still through it - 27 ms at 59 residues and
    // 119 ms at 221, which is the price of not fencing eight times to have
    // something to report.
    const step = input.onStep ?? (() => {});
    const initialized = await withAbort(new StructureInitializeGpu(this.device).run(
      input.msaFirstRow, input.length, msaChannels, structureChannels, input.weights.initialize,
    ), input.signal);
    throwIfAborted(input.signal);
    step("initialize");
    const core = await withAbort(new StructureCoreGpu(this.device).run({
      activations: initialized.activations,
      pair: input.pair,
      mask: input.mask,
      affine: initialized.affine,
      length: input.length,
      channels: structureChannels,
      pairChannels,
      ipaWeights: input.weights.ipa,
      postAttentionWeights: input.weights.postAttention,
      signal: input.signal,
      // 🔴 ONE STEP, ON COMPLETION, LIKE EVERY OTHER STAGE HERE. This used to
      // report `iteration ${done}/${total}` eight times, because each iteration
      // ended in a submit and a readback that the bar could ride on. The eight
      // now share one command buffer - see structure/core.js - so there is no
      // per-iteration boundary left to observe, and reporting one anyway would
      // be counting encodes rather than work, which is the failure the
      // pairformer's counter already had.
      onIteration: () => step("iterations"),
    }), input.signal);
    throwIfAborted(input.signal);
    const sidechain = await withAbort(new SidechainAnglesGpu(this.device).run(
      core.activations, initialized.initialRepresentation, input.length, structureChannels, 128, input.weights.sidechain,
    ), input.signal);
    throwIfAborted(input.signal);
    step("sidechains");
    const geometry = await withAbort(new AtomGeometryGpu(this.device).run({
      affine: core.affine,
      angles: sidechain.angles,
      aatype: input.aatype,
      atom37ToAtom14: input.atom37ToAtom14,
      atom37Mask: input.atom37Mask,
      length: input.length,
      tables: input.geometry,
    }), input.signal);
    throwIfAborted(input.signal);
    step("geometry");
    return {
      atom14: geometry.atom14,
      atom37: geometry.atom37,
      atom37Mask: input.atom37Mask,
      finalRepresentation: core.activations,
      affine: core.affine,
      angles: sidechain.angles,
      unnormalizedAngles: sidechain.unnormalizedAngles,
      elapsedMilliseconds: performance.now() - start,
    };
  }
}
