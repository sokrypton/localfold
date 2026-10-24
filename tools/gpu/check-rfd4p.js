/**
 * Headless Chrome WebGPU differential parity suite for RFD4-Proteina (`npm run test:gpu`).
 *
 * Verifies against the PyTorch oracle dumps and released weights (`weights/model.safetensors`):
 *   1. `Rfd4pTrunkGpu` (`uncond_short` + `motif_unidx_1prw` with insertion head)
 *   2. `Rfd4pDecoderGpu` across all 6 oracle cases (`uncond_short`, `ppi_pdl1`, `sm_sam`, `dna_7m5w`, `motif_idx_1prw`, `motif_unidx_1prw`)
 *   3. `Rfd4pEncoderGpu` across all 3 encoder oracle cases (`motif_idx_1prw`, `ppi_pdl1`, `ppi_pdl1_full`)
 *   4. `Rfd4pGeneratorGpu` vs `generateRfd4pCpu` end-to-end trajectory parity
 *
 * Run via:
 *   DISPLAY=:99 node tools/gpu-chrome.mjs tools/gpu/check-rfd4p.js
 */

import { main as checkTrunk } from './check-trunk.js';
import { main as checkDecoder } from './check-decoder.js';
import { main as checkEncoder } from './check-encoder.js';
import { subgroupMode } from '../../webgpu/rfd4p/engine.js';

export async function main(device, args = []) {
  if (args.includes('--no-submat=1') || args.includes('--no-submat')) device.noSubmat = true;
  const sgArg = args.find((a) => a.startsWith('--subgroup-mode='));
  if (sgArg) device.subgroupModeOverride = sgArg.split('=')[1];
  const gvArg = args.find((a) => a.startsWith('--gemm-variant='));
  if (gvArg) device.gemmVariantOverride = JSON.parse(gvArg.slice('--gemm-variant='.length));
  const trunkUncond = await checkTrunk(device, ['--case=uncond_short', ...args]);
  if (!trunkUncond.allOk) {
    throw new Error(`Rfd4pTrunkGpu (uncond_short) failed oracle check: ${JSON.stringify(trunkUncond.results)}`);
  }

  const trunkUnidx = await checkTrunk(device, ['--case=motif_unidx_1prw', ...args]);
  if (!trunkUnidx.allOk) {
    throw new Error(`Rfd4pTrunkGpu (motif_unidx_1prw) failed oracle check: ${JSON.stringify(trunkUnidx.results)}`);
  }

  const decoderAll = await checkDecoder(device, ['--case=all', ...args]);
  if (!decoderAll.allOk) {
    throw new Error(`Rfd4pDecoderGpu failed oracle check: ${JSON.stringify(decoderAll.results)}`);
  }

  const encoderAll = await checkEncoder(device, args);
  if (!encoderAll.allOk) {
    throw new Error(`Rfd4pEncoderGpu failed oracle check: ${JSON.stringify(encoderAll.results)}`);
  }

  return {
    allOk: true,
    subgroups: subgroupMode(device),
    trunk: {
      uncond_short: trunkUncond.results,
      motif_unidx_1prw: trunkUnidx.results,
    },
    decoder: decoderAll.results,
    encoder: encoderAll.results,
  };
}

export const run = main;
