import { AtomGeometryGpu } from "./geometry.js";
import { StructureCoreGpu } from "./core.js";
import { StructureInitializeGpu } from "./initialize.js";

import { SidechainAnglesGpu } from "./sidechain.js";

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
    const start = performance.now();
    const msaChannels = input.msaChannels ?? 256;
    const structureChannels = input.structureChannels ?? 384;
    const pairChannels = input.pairChannels ?? 128;
    // Four stages and eight iterations inside the second of them: the structure
    // module is the longest single thing a fold does at any real length, and as
    // one progress step it left the bar sitting still through all of it.
    const step = input.onStep ?? (() => {});
    const initialized = await new StructureInitializeGpu(this.device).run(
      input.msaFirstRow, input.length, msaChannels, structureChannels, input.weights.initialize,
    );
    step("initialize");
    const core = await new StructureCoreGpu(this.device).run({
      activations: initialized.activations,
      pair: input.pair,
      mask: input.mask,
      affine: initialized.affine,
      length: input.length,
      channels: structureChannels,
      pairChannels,
      ipaWeights: input.weights.ipa,
      postAttentionWeights: input.weights.postAttention,
      onIteration: (done, total) => step(`iteration ${done}/${total}`),
    });
    const sidechain = await new SidechainAnglesGpu(this.device).run(
      core.activations, initialized.initialRepresentation, input.length, structureChannels, 128, input.weights.sidechain,
    );
    step("sidechains");
    const geometry = await new AtomGeometryGpu(this.device).run({
      affine: core.affine,
      angles: sidechain.angles,
      aatype: input.aatype,
      atom37ToAtom14: input.atom37ToAtom14,
      atom37Mask: input.atom37Mask,
      length: input.length,
      tables: input.geometry,
    });
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
